//! Similarly to the [`image`] module, this module exposes two types: [`Buffer`] and [`BufferView`]. The difference here is that a
//! [`BufferView`] does not own a vulkan resource, so it cane be freely copied around as long as the owning [`Buffer`] lives.
//!
//! It also exposes some utilities for writing to memory-mapped buffers. For this you can use [`BufferView::mapped_slice`]. This only succeeds
//! if the buffer was allocated from a mappable heap (one that has the `HOST_VISIBLE` bit set).
//!
//! # Example
//!
//! ```
//! use ash::vk;
//! use gpu_allocator::MemoryLocation;
//! use phobos as ph;
//! // Allocate a new buffer
//! let buf = Buffer::new(device.clone(),
//!                       alloc.clone(),
//!                       // 16 bytes large
//!                       16 as vk::DeviceSize,
//!                       // We will use this buffer as a uniform buffer only
//!                       vk::BufferUsageFlags::UNIFORM_BUFFER,
//!                       // CpuToGpu will always set HOST_VISIBLE and HOST_COHERENT, and try to set DEVICE_LOCAL.
//!                       // Usually this resides on the PCIe BAR.
//!                       MemoryLocation::CpuToGpu);
//! // Obtain a buffer view to the entire buffer.
//! let mut view = buf.view_full();
//! // Obtain a slice of floats
//! let slice = view.mapped_slice::<f32>()?;
//! // Write some arbitrary data
//! let data = [1.0, 0.0, 1.0, 1.0];
//! slice.copy_from_slice(&data);
//! ```

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc};
use ash::vk;
use crate::{Allocation, Allocator, DefaultAllocator, Device, Error, MemoryType};

use anyhow::Result;


#[derive(Derivative)]
#[derivative(Debug)]
pub struct Buffer<A: Allocator = DefaultAllocator> {
    #[derivative(Debug="ignore")]
    device: Arc<Device>,
    #[derivative(Debug="ignore")]
    allocator: A,
    #[derivative(Debug="ignore")]
    memory: A::Allocation,
    pub(crate) pointer: Option<NonNull<c_void>>,
    pub handle: vk::Buffer,
    pub size: vk::DeviceSize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferView {
    pub(crate) handle: vk::Buffer,
    pub(crate) pointer: Option<NonNull<c_void>>,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
}

impl<A: Allocator> Buffer<A> {
    pub fn new(device: Arc<Device>, allocator: &mut A, size: impl Into<vk::DeviceSize>, usage: vk::BufferUsageFlags, location: MemoryType) -> Result<Self> {
        let size = size.into();
        let handle = unsafe {
            device.create_buffer(&vk::BufferCreateInfo {
                s_type: vk::StructureType::BUFFER_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: vk::BufferCreateFlags::empty(),
                size,
                usage,
                sharing_mode: vk::SharingMode::CONCURRENT,
                queue_family_index_count: device.queue_families.len() as u32,
                p_queue_family_indices: device.queue_families.as_ptr(),
            }, None)?
        };

        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let memory = allocator.allocate("buffer", &requirements, location)?;

        unsafe { device.bind_buffer_memory(handle, memory.memory(), memory.offset())? };

        Ok(Self {
            device,
            allocator: allocator.clone(),
            pointer: memory.mapped_ptr(),
            memory,
            handle,
            size,
        })
    }

    pub fn new_device_local(device: Arc<Device>, allocator: &mut A, size: impl Into<vk::DeviceSize>, usage: vk::BufferUsageFlags) -> Result<Self> {
        Self::new(device, allocator, size, usage, MemoryType::GpuOnly)
    }

    pub fn view(&self, offset: impl Into<vk::DeviceSize>, size: impl Into<vk::DeviceSize>) -> Result<BufferView> {
        let offset = offset.into();
        let size = size.into();
        return if offset + size >= self.size {
            Err(anyhow::Error::from(Error::BufferViewOutOfRange))
        } else {
            Ok(BufferView {
                handle: self.handle,
                offset,
                pointer: unsafe { self.pointer.map(|p| NonNull::new(p.as_ptr().offset(offset as isize)).unwrap() ) },
                size,
            })
        }
    }

    pub fn view_full(&self) -> BufferView {
        BufferView {
            handle: self.handle,
            pointer: self.pointer,
            offset: 0,
            size: self.size,
        }
    }

    pub fn is_mapped(&self) -> bool {
        self.pointer.is_some()
    }
}

impl<A: Allocator> Drop for Buffer<A> {
    fn drop(&mut self) {
        let memory = std::mem::take(&mut self.memory);
        self.allocator.free(memory).unwrap();
        unsafe { self.device.destroy_buffer(self.handle, None); }
    }
}

impl BufferView {
    pub fn mapped_slice<T>(&mut self) -> Result<&mut [T]> {
        if let Some(pointer) = self.pointer {
            Ok(unsafe { std::slice::from_raw_parts_mut(pointer.cast::<T>().as_ptr(),  self.size as usize / std::mem::size_of::<T>()) })
        } else {
            Err(anyhow::Error::from(Error::UnmappableBuffer))
        }
    }
}