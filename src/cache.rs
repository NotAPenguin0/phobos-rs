use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use crate::{Device, Error};

pub trait Resource {
    type Key: Hash + Eq + Clone;
    type ExtraParams<'a>;
    const MAX_TIME_TO_LIVE: u32;

    fn create(device: Arc<Device>, key: &Self::Key, params: Self::ExtraParams<'_>) -> Result<Self, Error> where Self: Sized;
}

struct Entry<R> {
    pub value: R,
    pub ttl: u32
}

pub struct Cache<R> where R: Resource {
    device: Arc<Device>,
    store: HashMap<R::Key, Entry<R>>
}

impl<R> Cache<R> where R: Resource + Sized {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            store: Default::default(),
        }
    }

    pub fn get_or_create<'a>(&mut self, key: &R::Key, params: R::ExtraParams<'a>) -> Result<&R, Error> {
        Ok(&self.store.entry(key.clone()).or_insert_with(|| Entry {
            value: R::create(self.device.clone(), &key, params).unwrap(), // TODO: bad unwrap
            ttl: R::MAX_TIME_TO_LIVE,
        }).value)
    }

    /// Advance time-to-live values in the cache, deleting objects values that have not been accessed for a while.
    pub(crate) fn next_frame(&mut self) {
        self.store.iter_mut().for_each(| (_, entry)| entry.ttl = entry.ttl - 1 );
        self.store.retain(|_, entry| entry.ttl != 0);
    }
}