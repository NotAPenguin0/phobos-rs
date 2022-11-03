use ash::{vk, Entry};

pub struct Context {
    vk_entry: Entry,
}

impl Context {
    pub fn new() -> Context {
        Context {
            vk_entry: unsafe { Entry::load().unwrap() }
        }
    }
}