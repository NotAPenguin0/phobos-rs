use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use ash::vk;
use gpu_allocator::AllocationError::OutOfMemory;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;
use crate::{Buffer, BufferView, Device, Error};
use crate::Error::AllocationError;
use anyhow::Result;

#[derive(Debug)]
pub struct ScratchAllocator {
    buffer: Buffer,
    offset: vk::DeviceSize,
    alignment: vk::DeviceSize,
}

impl ScratchAllocator {
    pub fn new(device: Arc<Device>, allocator: Arc<Mutex<Allocator>>, max_size: vk::DeviceSize, usage: vk::BufferUsageFlags) -> Result<Self> {
        let buffer = Buffer::new(device.clone(), allocator, max_size, usage, MemoryLocation::CpuToGpu)?;
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