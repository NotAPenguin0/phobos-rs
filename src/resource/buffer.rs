//! Wrappers for `VkBuffer` objects.
//!
//! Similarly to the [`image`](crate::image) module, this module exposes two types: [`Buffer`] and [`BufferView`]. The difference here is that a
//! [`BufferView`] does not own a vulkan resource, so it cane be freely copied around as long as the owning [`Buffer`] lives.
//!
//! It also exposes some utilities for writing to memory-mapped buffers. For this you can use [`BufferView::mapped_slice`]. This only succeeds
//! if the buffer was allocated from a mappable heap (one that has the `HOST_VISIBLE` bit set).
//!
//! # Example
//!
//! ```
//! use phobos::prelude::*;
//!
//! // Allocate a new buffer
//! let buf = Buffer::new(device.clone(),
//!                       alloc.clone(),
//!                       // 16 bytes large
//!                       16 as vk::DeviceSize,
//!                       // We will use this buffer as a uniform buffer only
//!                       vk::BufferUsageFlags::UNIFORM_BUFFER,
//!                       // CpuToGpu will always set HOST_VISIBLE and HOST_COHERENT, and try to set DEVICE_LOCAL.
//!                       // Usually this resides on the PCIe BAR.
//!                       MemoryType::CpuToGpu);
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

use anyhow::Result;
use ash::vk;
use ash::vk::Handle;

use crate::{Allocation, Allocator, DefaultAllocator, Device, Error, MemoryType};
use crate::core::traits::{AsRaw, Nameable};
use crate::util::align::align;

/// Wrapper around a [`VkBuffer`](vk::Buffer).
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Buffer<A: Allocator = DefaultAllocator> {
    #[derivative(Debug = "ignore")]
    device: Device,
    #[derivative(Debug = "ignore")]
    #[allow(dead_code)]
    memory: A::Allocation,
    address: vk::DeviceAddress,
    pointer: Option<NonNull<c_void>>,
    handle: vk::Buffer,
    size: vk::DeviceSize,
}

// SAFETY: The unsafe part of this is the mapped pointer, but this is a pointer to GPU memory
// so its value is not dropped when sending this to a different thread.
unsafe impl<A: Allocator> Send for Buffer<A> {}

/// View into a specific offset and range of a [`Buffer`].
/// Care should be taken with the lifetime of this, as there is no checking that the buffer
/// is not dropped while using this.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferView {
    handle: vk::Buffer,
    pointer: Option<NonNull<c_void>>,
    address: vk::DeviceAddress,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

// SAFETY: The unsafe part of this is the mapped pointer, but this is a pointer to GPU memory
// so its value is not dropped when sending this to a different thread.
unsafe impl Send for BufferView {}

impl<A: Allocator> Buffer<A> {
    /// Allocate a new buffer with a specific size, at a specific memory location.
    /// All usage flags must be given.
    pub fn new(
        device: Device,
        allocator: &mut A,
        size: impl Into<vk::DeviceSize>,
        usage: vk::BufferUsageFlags,
        location: MemoryType,
    ) -> Result<Self> {
        let size = size.into();
        let sharing_mode = if device.is_single_queue() {
            vk::SharingMode::EXCLUSIVE
        } else {
            vk::SharingMode::CONCURRENT
        };
        let handle = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo {
                    s_type: vk::StructureType::BUFFER_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::BufferCreateFlags::empty(),
                    size,
                    usage: usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    sharing_mode,
                    queue_family_index_count: if sharing_mode == vk::SharingMode::CONCURRENT {
                        device.queue_families().len() as u32
                    } else {
                        0
                    },
                    p_queue_family_indices: if sharing_mode == vk::SharingMode::CONCURRENT {
                        device.queue_families().as_ptr()
                    } else {
                        std::ptr::null()
                    },
                },
                None,
            )?
        };
        #[cfg(feature = "log-objects")]
        trace!("Created new VkBuffer {handle:p} (size = {size} bytes)");

        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let memory = allocator.allocate("buffer", &requirements, location)?;

        unsafe { device.bind_buffer_memory(handle, memory.memory(), memory.offset())? };

        let address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                s_type: vk::StructureType::BUFFER_DEVICE_ADDRESS_INFO,
                p_next: std::ptr::null(),
                buffer: handle,
            })
        };

        Ok(Self {
            device,
            pointer: memory.mapped_ptr(),
            memory,
            handle,
            size,
            address,
        })
    }

    /// Allocate a new buffer with a specific alignment instead of the inferred alignment from the usage flags.
    pub fn new_aligned(
        device: Device,
        allocator: &mut A,
        size: impl Into<vk::DeviceSize>,
        alignment: impl Into<vk::DeviceSize>,
        usage: vk::BufferUsageFlags,
        location: MemoryType,
    ) -> Result<Self> {
        let alignment = alignment.into();
        let size = align(size.into(), alignment);
        let sharing_mode = if device.is_single_queue() {
            vk::SharingMode::EXCLUSIVE
        } else {
            vk::SharingMode::CONCURRENT
        };
        let handle = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo {
                    s_type: vk::StructureType::BUFFER_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::BufferCreateFlags::empty(),
                    size,
                    usage: usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    sharing_mode,
                    queue_family_index_count: if sharing_mode == vk::SharingMode::CONCURRENT {
                        device.queue_families().len() as u32
                    } else {
                        0
                    },
                    p_queue_family_indices: if sharing_mode == vk::SharingMode::CONCURRENT {
                        device.queue_families().as_ptr()
                    } else {
                        std::ptr::null()
                    },
                },
                None,
            )?
        };
        #[cfg(feature = "log-objects")]
        trace!("Created new VkBuffer {handle:p} (size = {size} bytes)");

        let mut requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        requirements.alignment = alignment;
        let memory = allocator.allocate("buffer", &requirements, location)?;

        unsafe { device.bind_buffer_memory(handle, memory.memory(), memory.offset())? };

        let address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                s_type: vk::StructureType::BUFFER_DEVICE_ADDRESS_INFO,
                p_next: std::ptr::null(),
                buffer: handle,
            })
        };

        Ok(Self {
            device,
            pointer: memory.mapped_ptr(),
            memory,
            handle,
            size,
            address,
        })
    }

    /// Allocate a new buffer with device local memory (VRAM). This is usually the correct memory location for most buffers.
    pub fn new_device_local(
        device: Device,
        allocator: &mut A,
        size: impl Into<vk::DeviceSize>,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        Self::new(device, allocator, size, usage, MemoryType::GpuOnly)
    }

    /// Creates a view into an offset and size of the buffer.
    /// # Lifetime
    /// This view is valid as long as the buffer is valid.
    /// # Errors
    /// Fails if `offset + size > self.size`.
    pub fn view(
        &self,
        offset: impl Into<vk::DeviceSize>,
        size: impl Into<vk::DeviceSize>,
    ) -> Result<BufferView> {
        let offset = offset.into();
        let size = size.into();
        if offset + size > self.size {
            Err(anyhow::Error::from(Error::BufferViewOutOfRange))
        } else {
            Ok(BufferView {
                handle: self.handle,
                offset,
                pointer: unsafe {
                    self.pointer
                        .map(|p| NonNull::new(p.as_ptr().offset(offset as isize)).unwrap())
                },
                address: self.address + offset,
                size,
            })
        }
    }

    /// Creates a view of the entire buffer.
    /// # Lifetime
    /// This view is valid as long as the buffer is valid.
    pub fn view_full(&self) -> BufferView {
        BufferView {
            handle: self.handle,
            pointer: self.pointer,
            offset: 0,
            address: self.address,
            size: self.size,
        }
    }

    /// True if this buffer has a mapped pointer and thus can directly be written to.
    pub fn is_mapped(&self) -> bool {
        self.pointer.is_some()
    }

    /// Obtain a handle to the raw vulkan buffer object.
    /// # Safety
    /// * The caller must make sure to not use this handle after `self` is dropped.
    /// * The caller must not call `vkDestroyBuffer` on this handle.
    pub unsafe fn handle(&self) -> vk::Buffer {
        self.handle
    }

    /// Get the size of this buffer
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// Get the device address of this buffer
    pub fn address(&self) -> vk::DeviceAddress {
        self.address
    }
}

unsafe impl AsRaw for Buffer {
    unsafe fn as_raw(&self) -> u64 {
        self.handle().as_raw()
    }
}

impl Nameable for Buffer {
    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::BUFFER;
}

impl<A: Allocator> Drop for Buffer<A> {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkBuffer {:p}", self.handle);
        unsafe {
            self.device.destroy_buffer(self.handle, None);
        }
    }
}

impl BufferView {
    /// Obtain a slice to the mapped memory of this buffer.
    /// # Errors
    /// Fails if this buffer is not mappable (not `HOST_VISIBLE`).
    pub fn mapped_slice<T>(&mut self) -> Result<&mut [T]> {
        if let Some(pointer) = self.pointer {
            Ok(unsafe {
                std::slice::from_raw_parts_mut(
                    pointer.cast::<T>().as_ptr(),
                    self.size as usize / std::mem::size_of::<T>(),
                )
            })
        } else {
            Err(anyhow::Error::from(Error::UnmappableBuffer))
        }
    }

    /// Obtain a handle to the raw vulkan buffer object.
    /// # Safety
    /// * The caller must make sure to not use this handle after `self` is dropped.
    /// * The caller must not call `vkDestroyBuffer` on this handle.
    pub unsafe fn handle(&self) -> vk::Buffer {
        self.handle
    }

    /// Get the offset of this buffer view into the owning buffer
    pub fn offset(&self) -> vk::DeviceSize {
        self.offset
    }

    /// Get the size of this buffer view.
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// Get the device address of the start of this buffer view.
    pub fn address(&self) -> vk::DeviceAddress {
        self.address
    }
}
