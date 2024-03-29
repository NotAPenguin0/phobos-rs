//! Generic implementation of a resource cache

use std::collections::{hash_map, HashMap};
use std::hash::Hash;

use anyhow::Result;

use crate::Device;

/// Trait representing a resource key in a cache. This key must be hashable and cloneable.
pub trait ResourceKey: Hash + Eq + Clone {
    /// Whether this resource is persistent. This means it will
    /// never be cleaned up by the cache, even if its lifetime expires.
    /// Use with caution.
    fn persistent(&self) -> bool;
}

/// Trait that needs to be implemented by types managed by a [`Cache`]
pub trait Resource {
    /// Key type used for looking up and possibly creating new resources. Must be hashable and cloneable.
    type Key: ResourceKey;
    /// Additional parameter passed through from [`Cache::get_or_create`] to [`Resource::create`].
    type ExtraParams<'a>;
    /// Amount of calls to [`Cache::next_frame`] have to happen without accessing this resource for it to be deallocated.
    const MAX_TIME_TO_LIVE: u32;

    /// Allocates a new resource to be stored in the cache. This function may error, but this error will propagate through the cache's access function.
    fn create(device: Device, key: &Self::Key, params: Self::ExtraParams<'_>) -> Result<Self>
    where
        Self: Sized;
}

struct Entry<R> {
    value: R,
    ttl: u32,
    persistent: bool,
}

/// Implements a smart resource cache that deallocates resources that have not been accessed in a while.
/// to use this for a type `R`, it's enough for `R` to implement the [`Resource`] trait.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Cache<R: Resource + Sized> {
    #[derivative(Debug = "ignore")]
    device: Device,
    #[derivative(Debug = "ignore")]
    store: HashMap<R::Key, Entry<R>>,
}

impl<R: Resource + Sized> Cache<R> {
    /// Create a new resource cache from a Vulkan device.
    pub fn new(device: Device) -> Self {
        Self {
            device,
            store: Default::default(),
        }
    }

    /// Access a resource in the cache by its key. The extra parameters needed are defined by the resource type,
    /// and can be used to allocate a resource if it doesn't exist.
    /// # Lifetime
    /// This function returns a reference to a resource stored in the cache. This reference is valid as long as
    /// the cache is exists.
    /// # Errors
    /// This function can only error if the requested resource did not exist, and allocation of it failed.
    pub fn get_or_create<'a, 'b, 's: 'b>(
        &'s mut self,
        key: &R::Key,
        params: R::ExtraParams<'a>,
    ) -> Result<&'b R> {
        let entry = self.store.entry(key.clone());
        let entry = match entry {
            hash_map::Entry::Occupied(entry) => entry.into_mut(),
            hash_map::Entry::Vacant(entry) => entry.insert(Entry {
                value: R::create(self.device.clone(), key, params)?,
                ttl: R::MAX_TIME_TO_LIVE,
                persistent: key.persistent(),
            }),
        };
        entry.ttl = R::MAX_TIME_TO_LIVE;
        Ok(&entry.value)
    }

    /// Updates the cache to deallocate resources that have not been accessed for too long.
    pub(crate) fn next_frame(&mut self) {
        self.store.iter_mut().for_each(|(_, entry)| {
            if !entry.persistent {
                entry.ttl -= 1
            }
        });
        self.store
            .retain(|_, entry| entry.persistent || entry.ttl != 0);
    }
}
