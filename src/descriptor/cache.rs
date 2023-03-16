use std::sync::{Arc, Mutex};
use crate::descriptor::descriptor_pool::{DescriptorPool, DescriptorPoolSize};
use crate::{DeletionQueue, DescriptorSet, Device, Error};

use anyhow::Result;
use ash::vk;
use crate::descriptor::descriptor_set::DescriptorSetBinding;
use crate::util::cache::Cache;

/// This structure uses a [`Cache`] over a [`DescriptorSet`] to automatically manage everything related to descriptor sets.
/// It can intelligently allocate and deallocate descriptor sets, and grow its internal descriptor pool when necessary.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct DescriptorCache {
    #[derivative(Debug="ignore")]
    device: Arc<Device>,
    cache: Cache<DescriptorSet>,
    pool: DescriptorPool,
    deferred_pool_delete: DeletionQueue<DescriptorPool>,
}



impl DescriptorCache {
    /// Create a new descriptor cache object.
    /// # Errors
    /// - This can fail if creating the initial descriptor pool fails.
    pub fn new(device: Arc<Device>) -> Result<Arc<Mutex<Self>>> {
        Ok(Arc::new(Mutex::new(Self {
            device: device.clone(),
            cache: Cache::new(device.clone()),
            pool: DescriptorPool::new(device.clone(), DescriptorPoolSize::new(1))?,
            deferred_pool_delete: DeletionQueue::new(16),
        })))
    }

    fn grow_pool_size(mut old_size: DescriptorPoolSize, request: &DescriptorSetBinding) -> DescriptorPoolSize {
        for (ty, count) in old_size.0.iter_mut() {
            if request.bindings.iter().find(|&binding| binding.ty == *ty).is_some() {
                *count = *count * 2;
            }
        }
        old_size
    }

    /// Get a new descriptor set with the given descriptor set binding.
    /// If the internal descriptor pool runs out of space, a new one will be created.
    /// # Errors
    /// - This function fails if no descriptor set layout was specified in `bindings`
    /// - This function fails the the requested descriptor set has no descriptors
    /// - This function fails if allocating a descriptor set failed due to an internal error.
    pub fn get_descriptor_set(&mut self, mut bindings: DescriptorSetBinding) -> Result<&DescriptorSet> {
        if bindings.bindings.is_empty() { return Err(anyhow::Error::from(Error::EmptyDescriptorBinding)); }
        if bindings.layout == vk::DescriptorSetLayout::null() { return Err(anyhow::Error::from(Error::NoDescriptorSetLayout)); }

        loop {
            bindings.pool = self.pool.handle;
            let set = self.cache.get_or_create(&bindings, ());
            match set {
                // Need to query again to fix lifetime compiler error
                Ok(_) => { return Ok(self.cache.get_or_create(&bindings, ()).unwrap()); }
                Err(_) => {
                    let new_size = Self::grow_pool_size(self.pool.size.clone(), &bindings);
                    // Create new pool, swap it out with the old one and then push the old one onto the deletion queue
                    let mut new_pool = DescriptorPool::new(self.device.clone(), new_size)?;
                    std::mem::swap(&mut new_pool, &mut self.pool);
                    self.deferred_pool_delete.push(new_pool);
                }
            }
        }
    }

    /// Advance the descriptor cache to the next frame. This allows resources to be reclaimed safely where possible.
    pub fn next_frame(&mut self) {
        self.cache.next_frame();
        self.deferred_pool_delete.next_frame();
    }
}