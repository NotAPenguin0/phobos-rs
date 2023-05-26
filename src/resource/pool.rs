use std::hash::Hash;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use multimap::{Entry, MultiMap};

use crate::{Allocator, DescriptorCache, Device, Fence, PipelineCache, ScratchAllocator};

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
}

/// Represents a pooled object. When this is dropped, it's released back to the pool where it can
/// be reused immediately.
pub struct Pooled<P: Poolable> {
    item: Option<P>,
    pool: Pool<P>,
    key: Option<P::Key>,
}

type BoxedCreateFunc<P> = Box<dyn Fn(&<P as Poolable>::Key) -> Result<P>>;

struct PoolInner<P: Poolable> {
    items: MultiMap<P::Key, P>,
    create_fn: BoxedCreateFunc<P>,
}

/// Represents an object pool that can be allocated from
pub struct Pool<P: Poolable> {
    inner: Arc<Mutex<PoolInner<P>>>,
}

/// Acts as a global resource pool that can safely be shared everywhere.
#[derive(Clone)]
pub struct ResourcePool<A: Allocator> {
    /// Pipeline cache used to create pipelines on demand
    pub pipelines: PipelineCache<A>,
    /// Descriptor cache used to create descriptor sets on demand
    pub descriptors: DescriptorCache,
    /// Scratch allocator pool used to easily create scratch buffers anywhere
    pub allocators: Pool<ScratchAllocator<A>>,
    /// Fence pool to reuse fences where possible
    pub fences: Pool<Fence>,
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
    pub fn new(create_fn: impl Fn(&P::Key) -> Result<P> + 'static) -> Result<Self> {
        let inner = PoolInner {
            items: MultiMap::new(),
            create_fn: Box::new(create_fn),
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }
}

impl<A: Allocator> ResourcePool<A> {
    pub fn new(device: Device, allocator: A) -> Result<Self> {
        todo!()
    }
}
