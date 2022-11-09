use std::sync::Arc;
use ash::vk;
use crate::{Device, Error};

use gpu_allocator::vulkan as vk_alloc;

#[derive(Debug)]
struct Buffer {
    device: Arc<Device>,
    memory: vk_alloc::Allocation,
    pub handle: vk::Buffer,
    pub size: vk::DeviceSize,
}

impl Buffer {
    pub fn new(allocator: &mut vk_alloc::Allocator) -> Result<Self, Error> {
        todo!()
    }
}