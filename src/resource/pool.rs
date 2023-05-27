use std::hash::Hash;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use ash::vk;
use multimap::{Entry, MultiMap};

use crate::allocator::scratch_allocator::ScratchAllocatorCreateInfo;
use crate::{
    Allocator, BufferView, DefaultAllocator, DescriptorCache, Device, Fence, PipelineCache,
    ScratchAllocator,
};

/// Indicates that this object can be pooled in a [`Pool`](crate::pool::Pool)
pub trait Poolable {
    /// Key used to identify this pooled object in the object pool.
    type Key: Clone + Hash + PartialEq + Eq;

    /// Called right before the object is released back to the pool, can be used to reset internal state.
    fn on_release(&mut self);

    /// Create a new pooled object from a pool.
    fn new_in_pool(pool: &Pool<Self>, key: &Self::Key) -> Result<Pooled<Self>>
    where
        Self: Sized, {
        let item = pool.with(|pool| pool.get(key))?;
        Ok(Pooled::from_pool(pool.clone(), key.clone(), item))
    }

    /// Move this item into the pool when it is dropped
    fn into_pooled(self, pool: &Pool<Self>, key: Self::Key) -> Pooled<Self>
    where
        Self: Sized, {
        Pooled::from_pool(pool.clone(), key, self)
    }
}

/// Represents a pooled object. When this is dropped, it's released back to the pool where it can
/// be reused immediately.
pub struct Pooled<P: Poolable> {
    item: Option<P>,
    pool: Pool<P>,
    key: Option<P::Key>,
}

type BoxedCreateFunc<P> = Box<dyn FnMut(&<P as Poolable>::Key) -> Result<P>>;

struct PoolInner<P: Poolable> {
    items: MultiMap<P::Key, P>,
    create_fn: BoxedCreateFunc<P>,
}

/// Represents an object pool that can be allocated from
pub struct Pool<P: Poolable> {
    inner: Arc<Mutex<PoolInner<P>>>,
}

/// Acts as a global resource pool that can safely be shared everywhere.
#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct ResourcePool<A: Allocator = DefaultAllocator> {
    /// Pipeline cache used to create pipelines on demand
    pub pipelines: PipelineCache<A>,
    /// Descriptor cache used to create descriptor sets on demand
    pub descriptors: DescriptorCache,
    /// Scratch allocator pool used to easily create scratch buffers anywhere
    #[derivative(Debug = "ignore")]
    pub allocators: Pool<ScratchAllocator<A>>,
    /// Fence pool to reuse fences where possible
    #[derivative(Debug = "ignore")]
    pub fences: Pool<Fence<()>>,
}

pub struct ResourcePoolCreateInfo<A: Allocator = DefaultAllocator> {
    pub device: Device,
    pub allocator: A,
    pub scratch_size: u64,
}

/// A local pool that will release its resources back to the main resource pool when it goes out of scope.
/// Such a scope could be a frame context, or a task spawned on a background thread.
pub struct LocalPool<A: Allocator = DefaultAllocator> {
    pool: ResourcePool<A>,
    vertex_allocator: Pooled<ScratchAllocator<A>>,
    index_allocator: Pooled<ScratchAllocator<A>>,
    uniform_allocator: Pooled<ScratchAllocator<A>>,
    storage_allocator: Pooled<ScratchAllocator<A>>,
}

impl<P: Poolable> Clone for Pool<P> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<P: Poolable> Pooled<P> {
    /// Create a new pooled object from a pool and an item
    fn from_pool(pool: Pool<P>, key: P::Key, item: P) -> Self {
        Self {
            item: Some(item),
            pool,
            key: Some(key),
        }
    }

    /// Moves the inner value into the closure, which should then return
    /// a new inner value.
    pub fn replace<F: FnOnce(P) -> P>(&mut self, f: F) {
        let item = self.item.take().unwrap();
        self.item = Some(f(item));
    }
}

impl<P: Poolable> Deref for Pooled<P> {
    type Target = P;

    fn deref(&self) -> &Self::Target {
        self.item.as_ref().unwrap()
    }
}

impl<P: Poolable> DerefMut for Pooled<P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item.as_mut().unwrap()
    }
}

impl<P: Poolable> Drop for Pooled<P> {
    fn drop(&mut self) {
        // Take the item out of self and release it back to the pool
        let mut item = self.item.take().unwrap();
        let key = self.key.take().unwrap();
        item.on_release();
        self.pool.with(|pool| pool.take(item, key));
    }
}

impl<P: Poolable> PoolInner<P> {
    /// Release an object back into the pool
    fn take(&mut self, item: P, key: P::Key) {
        self.items.insert(key, item);
    }

    /// Grab an object from the pool. If there are none left, this will allocate a new one.
    fn get(&mut self, key: &P::Key) -> Result<P> {
        match self.items.entry(key.clone()) {
            Entry::Occupied(mut entry) => {
                let vec = entry.get_vec_mut();
                match vec.pop() {
                    None => (self.create_fn)(key),
                    Some(item) => Ok(item),
                }
            }
            Entry::Vacant(_) => (self.create_fn)(key),
        }
    }
}

impl<P: Poolable> Pool<P> {
    /// Get mutable access to the inner pool
    fn with<F: FnOnce(&mut PoolInner<P>) -> R, R>(&self, f: F) -> R {
        let mut inner = self.inner.lock().unwrap();
        f(&mut inner)
    }

    /// Create a new pool. This must be supplied with a callback to be called
    /// when the pool needs to allocate a new object.
    /// Optionally also takes in a count of objects to preallocate using this callback.
    pub fn new(create_fn: impl FnMut(&P::Key) -> Result<P> + 'static) -> Result<Self> {
        let inner = PoolInner {
            items: MultiMap::new(),
            create_fn: Box::new(create_fn),
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }
}

impl<A: Allocator + 'static> ResourcePool<A> {
    pub fn new(info: ResourcePoolCreateInfo<A>) -> Result<Self> {
        let pipelines = PipelineCache::new(info.device.clone(), info.allocator.clone())?;
        let descriptors = DescriptorCache::new(info.device.clone())?;
        let device = info.device.clone();
        let mut alloc = info.allocator.clone();
        let allocators = Pool::new(move |key: &ScratchAllocatorCreateInfo| {
            ScratchAllocator::new(device.clone(), &mut alloc, info.scratch_size, key.usage)
        })?;
        let device = info.device.clone();
        let fences = Pool::new(move |_| Ok(Fence::new(device.clone(), false)?))?;

        Ok(Self {
            pipelines,
            descriptors,
            allocators,
            fences,
        })
    }
}

impl<A: Allocator> ResourcePool<A> {
    pub fn get_scratch_allocator(
        &self,
        usage: vk::BufferUsageFlags,
    ) -> Result<Pooled<ScratchAllocator<A>>> {
        ScratchAllocator::new_in_pool(
            &self.allocators,
            &ScratchAllocatorCreateInfo {
                usage,
            },
        )
    }

    pub fn next_frame(&self) {
        self.pipelines.next_frame();
        self.descriptors.next_frame();
    }
}

impl<A: Allocator> LocalPool<A> {
    pub fn new(pool: ResourcePool<A>) -> Result<Self> {
        let vertex_alloc = pool.get_scratch_allocator(
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let index_alloc = pool.get_scratch_allocator(
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        let uniform_alloc = pool.get_scratch_allocator(
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;
        let storage_alloc = pool.get_scratch_allocator(
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        Ok(Self {
            pool,
            vertex_allocator: vertex_alloc,
            index_allocator: index_alloc,
            uniform_allocator: uniform_alloc,
            storage_allocator: storage_alloc,
        })
    }

    /// Allocate a scratch vertex buffer, which is only valid for the scope of this local pool.
    /// See also: [`ScratchAllocator`](crate::ScratchAllocator)
    pub fn allocate_scratch_vbo(&mut self, size: vk::DeviceSize) -> Result<BufferView> {
        self.vertex_allocator.allocate(size)
    }
    /// Allocate a scratch index buffer, which is only valid for the scope of this local pool.
    /// See also: [`ScratchAllocator`](crate::ScratchAllocator)
    pub fn allocate_scratch_ibo(&mut self, size: vk::DeviceSize) -> Result<BufferView> {
        self.index_allocator.allocate(size)
    }
    /// Allocate a scratch uniform buffer, which is only valid for the scope of this local pool.
    /// See also: [`ScratchAllocator`](crate::ScratchAllocator)
    pub fn allocate_scratch_ubo(&mut self, size: vk::DeviceSize) -> Result<BufferView> {
        self.uniform_allocator.allocate(size)
    }
    /// Allocate a scratch shader storage buffer, which is only valid for the scope of this local pool.
    /// See also: [`ScratchAllocator`](crate::ScratchAllocator)
    pub fn allocate_scratch_ssbo(&mut self, size: vk::DeviceSize) -> Result<BufferView> {
        self.storage_allocator.allocate(size)
    }
}
