use std::sync::{Arc, Mutex};

use anyhow::Result;
use ash::vk;

use crate::{DeletionQueue, DescriptorSet, Device, Error};
use crate::descriptor::descriptor_pool::{DescriptorPool, DescriptorPoolSize};
use crate::descriptor::descriptor_set::DescriptorSetBinding;
use crate::util::cache::Cache;

#[derive(Debug)]
struct DescriptorCacheInner {
    device: Device,
    cache: Cache<DescriptorSet>,
    pool: DescriptorPool,
    deferred_pool_delete: DeletionQueue<DescriptorPool>,
}

/// This structure uses a [`Cache`] over a [`DescriptorSet`] to automatically manage everything related to descriptor sets.
/// It can intelligently allocate and deallocate descriptor sets, and grow its internal descriptor pool when necessary.
/// All internal state is wrapped in an `Arc<Mutex<DescriptorCacheInner>>`, so this struct is `Clone`, `Send` and `Sync`.
#[derive(Debug, Clone)]
pub struct DescriptorCache {
    inner: Arc<Mutex<DescriptorCacheInner>>,
}

fn grow_pool_size(mut old_size: DescriptorPoolSize, request: &DescriptorSetBinding) -> DescriptorPoolSize {
    for (ty, count) in old_size.0.iter_mut() {
        if request.bindings.iter().find(|&binding| binding.ty == *ty).is_some() {
            *count = *count * 2;
            trace!("Growing descriptor pool for type {:?} to new size {}", ty, *count);
        }
    }
    old_size
}

impl DescriptorCacheInner {
    pub fn get_descriptor_set(&mut self, mut bindings: DescriptorSetBinding) -> Result<&DescriptorSet> {
        if bindings.bindings.is_empty() {
            return Err(Error::EmptyDescriptorBinding.into());
        }
        if bindings.layout == vk::DescriptorSetLayout::null() {
            return Err(Error::NoDescriptorSetLayout.into());
        }

        loop {
            bindings.pool = unsafe { self.pool.handle() };
            let is_ok = { self.cache.get_or_create(&bindings, ()).is_ok() };
            if is_ok {
                // Need to query again to fix lifetime compiler error
                return Ok(self.cache.get_or_create(&bindings, ()).unwrap());
            } else {
                let new_size = grow_pool_size(self.pool.size().clone(), &bindings);
                // Create new pool, swap it out with the old one and then push the old one onto the deletion queue
                let mut new_pool = DescriptorPool::new(self.device.clone(), new_size)?;
                std::mem::swap(&mut new_pool, &mut self.pool);
                self.deferred_pool_delete.push(new_pool);
            }
        }
    }
}

impl DescriptorCache {
    /// Create a new descriptor cache object.
    /// # Errors
    /// - This can fail if creating the initial descriptor pool fails.
    pub fn new(device: Device) -> Result<Self> {
        let inner = DescriptorCacheInner {
            device: device.clone(),
            cache: Cache::new(device.clone()),
            pool: DescriptorPool::new(device.clone(), DescriptorPoolSize::new(1))?,
            deferred_pool_delete: DeletionQueue::new(16),
        };
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Get a new descriptor set with the given descriptor set binding.
    /// If the internal descriptor pool runs out of space, a new one will be created.
    /// When the set is obtained, call the provided callback with that descriptor set.
    /// # Errors
    /// - This function fails if no descriptor set layout was specified in `bindings`
    /// - This function fails the the requested descriptor set has no descriptors
    /// - This function fails if allocating a descriptor set failed due to an internal error.
    pub fn with_descriptor_set<F: FnOnce(&DescriptorSet) -> Result<()>>(&mut self, bindings: DescriptorSetBinding, f: F) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        let set = inner.get_descriptor_set(bindings)?;
        f(set)
    }

    /// Advance the descriptor cache to the next frame. This allows resources to be reclaimed safely where possible.
    pub fn next_frame(&mut self) {
        let mut inner = self.inner.lock().unwrap();
        inner.cache.next_frame();
        inner.deferred_pool_delete.next_frame();
    }
}
