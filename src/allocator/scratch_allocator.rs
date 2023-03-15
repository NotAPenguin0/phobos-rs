//! The [`ScratchAllocator`] is a very simple linear allocator that can be used for scratch resources.
//! It is exposed through the [`InFlightContext`] struct, but you can also create your own instances elsewhere.
//!
//! The allocator works by linearly incrementing an offset on every allocation. Deallocation is only possible by calling
//! [`ScratchAllocator::reset`], which will free all memory and reset the offset to zero.
//!
//! # Example
//!
//! ```
//! use ash::vk;
//! use phobos as ph;
//!
//! // Create a scratch allocator with at most 1 KiB of available memory for uniform buffers
//! let mut allocator = ph::ScratchAllocator::new(device.clone(), alloc.clone(), (1 * 1024) as vk::DeviceSize, vk::BufferUsageFlags::UNIFORM_BUFFER);
//!
//! // Allocate a 64 byte uniform buffer and use it
//! let buffer = allocator.allocate(64 as vk::DeviceSize)?;
//! // For buffer usage, check the buffer module documentation.
//!
//! // Once we're ready for the next batch of allocations, call reset(). This must happen
//! // after the GPU is done using the contents of all allocated buffers.
//! // For the allocators in the InFlightContext, this is done for you already.
//! allocator.reset();
//! ```

use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use ash::vk;
use gpu_allocator::AllocationError::OutOfMemory;
use crate::{Allocator, Buffer, BufferView, DefaultAllocator, Device, Error, MemoryType};
use crate::Error::AllocationError;
use anyhow::Result;

#[derive(Debug)]
pub struct ScratchAllocator<A: Allocator = DefaultAllocator> {
    buffer: Buffer<A>,
    offset: vk::DeviceSize,
    alignment: vk::DeviceSize,
}

impl<A: Allocator> ScratchAllocator<A> {
    pub fn new(device: Arc<Device>, allocator: &mut A, max_size: vk::DeviceSize, usage: vk::BufferUsageFlags) -> Result<Self> {
        let buffer = Buffer::new(device.clone(), allocator, max_size, usage, MemoryType::CpuToGpu)?;
        let alignment = if usage.intersects(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER) {
            16
        } else if usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
            device.properties.limits.min_uniform_buffer_offset_alignment
        } else if usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
            device.properties.limits.min_storage_buffer_offset_alignment
        } else {
            unimplemented!()
        };

        return if buffer.is_mapped() {
            Ok(Self {
                buffer,
                offset: 0,
                alignment
            })
        } else {
            Err(anyhow::Error::from(Error::UnmappableBuffer))
        }
    }

    pub fn allocate(&mut self, size: vk::DeviceSize) -> Result<BufferView> {
        // Part of the buffer that is over the min alignment
        let unaligned_part = size % self.alignment;
        // Amount of padding bytes to insert
        let padding = self.alignment - unaligned_part;
        let padded_size = size + padding;
        return if self.offset + padded_size > self.buffer.size {
            Err(anyhow::Error::from(AllocationError(OutOfMemory)))
        } else {
            let offset = self.offset;
            self.offset += padded_size;
            let Some(pointer) = self.buffer.pointer else { panic!() };
            Ok(BufferView {
                handle: self.buffer.handle,
                pointer: unsafe { NonNull::new(pointer.as_ptr().offset(offset as isize)) },
                offset,
                size,
            })
        }
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }
}