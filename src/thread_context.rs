use std::sync::{Arc, Mutex};
use ash::vk;
use crate::{Device, InFlightContext, ScratchAllocator};
use anyhow::Result;
use gpu_allocator::vulkan::Allocator;

pub struct ThreadContext {
    vbo_allocator: ScratchAllocator,
    ibo_allocator: ScratchAllocator,
    ubo_allocator: ScratchAllocator,
    ssbo_allocator: ScratchAllocator,
}

impl ThreadContext {
    /// Spawn a new thread context with local scratch allocators.
    pub fn new(device: Arc<Device>, allocator: Arc<Mutex<Allocator>>, scratch_size: Option<vk::DeviceSize>) -> Result<Self> {
        let scratch_size = match scratch_size {
            None => 1 as vk::DeviceSize,
            Some(size) => size
        };

        Ok(Self {
            vbo_allocator: ScratchAllocator::new(device.clone(), allocator.clone(), scratch_size, vk::BufferUsageFlags::VERTEX_BUFFER)?,
            ibo_allocator: ScratchAllocator::new(device.clone(), allocator.clone(), scratch_size, vk::BufferUsageFlags::INDEX_BUFFER)?,
            ubo_allocator: ScratchAllocator::new(device.clone(), allocator.clone(), scratch_size, vk::BufferUsageFlags::UNIFORM_BUFFER)?,
            ssbo_allocator: ScratchAllocator::new(device.clone(), allocator.clone(), scratch_size, vk::BufferUsageFlags::STORAGE_BUFFER)?,
        })
    }

    /// Gets an [`InFlightContext`] for this current thread context. This can be useful for using the rendergraph API
    /// in a thread context.
    pub fn get_ifc(&mut self) -> InFlightContext {
        InFlightContext {
            swapchain_image: None,
            vertex_allocator: &mut self.vbo_allocator,
            index_allocator: &mut self.ibo_allocator,
            uniform_allocator: &mut self.ubo_allocator,
            storage_allocator: &mut self.ssbo_allocator,
        }
    }
}