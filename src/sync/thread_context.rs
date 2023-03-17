use std::sync::{Arc};
use ash::vk;
use crate::{Allocator, DefaultAllocator, Device, InFlightContext, ScratchAllocator};
use anyhow::Result;

/// Thread context with linear allocators that can be used as a substitute
/// [`InFlightContext`] outside of a frame.
pub struct ThreadContext<A: Allocator = DefaultAllocator> {
    vbo_allocator: ScratchAllocator<A>,
    ibo_allocator: ScratchAllocator<A>,
    ubo_allocator: ScratchAllocator<A>,
    ssbo_allocator: ScratchAllocator<A>,
}

impl<A: Allocator> ThreadContext<A> {
    /// Spawn a new thread context with local scratch allocators.
    pub fn new(device: Arc<Device>, mut allocator: A, scratch_size: Option<impl Into<vk::DeviceSize>>) -> Result<Self> {
        let scratch_size = match scratch_size {
            None => 1 as vk::DeviceSize,
            Some(size) => size.into()
        };

        Ok(Self {
            vbo_allocator: ScratchAllocator::<A>::new(device.clone(), &mut allocator, scratch_size, vk::BufferUsageFlags::VERTEX_BUFFER)?,
            ibo_allocator: ScratchAllocator::<A>::new(device.clone(), &mut allocator, scratch_size, vk::BufferUsageFlags::INDEX_BUFFER)?,
            ubo_allocator: ScratchAllocator::<A>::new(device.clone(), &mut allocator, scratch_size, vk::BufferUsageFlags::UNIFORM_BUFFER)?,
            ssbo_allocator: ScratchAllocator::<A>::new(device.clone(), &mut allocator, scratch_size, vk::BufferUsageFlags::STORAGE_BUFFER)?,
        })
    }

    /// Gets an [`InFlightContext`] for this current thread context. This can be useful for using the rendergraph API
    /// in a thread context.
    pub fn get_ifc(&mut self) -> InFlightContext<A> {
        InFlightContext {
            swapchain_image: None,
            swapchain_image_index: None,
            vertex_allocator: &mut self.vbo_allocator,
            index_allocator: &mut self.ibo_allocator,
            uniform_allocator: &mut self.ubo_allocator,
            storage_allocator: &mut self.ssbo_allocator,
        }
    }
}