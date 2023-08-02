//! A linear allocator that can be used for scratch resources.
//!
//! It is exposed through the [`InFlightContext`](crate::InFlightContext) struct, but you can also create your own instances elsewhere.
//!
//! The allocator works by linearly incrementing an offset on every allocation. Deallocation is only possible by calling
//! [`ScratchAllocator::reset`], which will free all memory and reset the offset to zero.
//!
//! # Example
//! ```
//! # use phobos::prelude::*;
//! # use anyhow::Result;
//! // Function that uses the buffer in some way and returns a fence
//! // that is signaled when the work is done.
//! fn use_the_buffer(buffer: BufferView) -> Fence<()> {
//!     unimplemented!()
//! }
//!
//! fn use_scratch_allocator<A: Allocator>(device: Device, alloc: &mut A) -> Result<()> {
//!     let mut allocator = ScratchAllocator::new(device.clone(), alloc, 128 as u64, vk::BufferUsageFlags::UNIFORM_BUFFER)?;
//!     let buffer: BufferView = allocator.allocate(128 as u64)?;
//!     let mut fence = use_the_buffer(buffer);
//!     fence.wait()?;
//!     // SAFETY: We just waited for the fence, so all work using our allocator is done.
//!     unsafe { allocator.reset(); }
//!     // We are back at the beginning of the allocator, so there are 128 bytes free again.
//!     let buffer: BufferView = allocator.allocate(128 as u64)?;
//!     Ok(())
//! }
//! ```

use anyhow::Result;
use ash::vk;
use gpu_allocator::AllocationError::OutOfMemory;

use crate::{Allocator, Buffer, BufferView, DefaultAllocator, Device, Error, MemoryType};
use crate::Error::AllocationError;
use crate::pool::Poolable;

/// A linear allocator used for short-lived resources. A good example of such a resource is a buffer
/// that needs to be updated every frame, like a uniform buffer for transform data.
/// Because of this typical usage, the scratch allocator allocates memory with [`MemoryType::CpuToGpu`].
///
/// The best way to obtain a scratch allocator is through a [`LocalPool`](crate::pool::LocalPool). This gives
/// a scratch allocator that is reset and recycled at the end of the pool's lifetime.
///
/// See also: [`LocalPool`](crate::pool::LocalPool), [`MemoryType`]
///
/// # Example
/// ```
/// # use phobos::prelude::*;
/// # use anyhow::Result;
/// // Function that uses the buffer in some way and returns a fence
/// // that is signaled when the work is done.
/// fn use_the_buffer(buffer: BufferView) -> Fence<()> {
///     unimplemented!()
/// }
///
/// fn use_scratch_allocator<A: Allocator>(device: Device, alloc: &mut A) -> Result<()> {
///     let mut allocator = ScratchAllocator::new(device.clone(), alloc, 128 as u64, vk::BufferUsageFlags::UNIFORM_BUFFER)?;
///     let buffer: BufferView = allocator.allocate(128 as u64)?;
///     let mut fence = use_the_buffer(buffer);
///     fence.wait()?;
///     // SAFETY: We just waited for the fence, so all work using our allocator is done.
///     unsafe { allocator.reset(); }
///     // We are back at the beginning of the allocator, so there are 128 bytes free again.
///     let buffer: BufferView = allocator.allocate(128 as u64)?;
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct ScratchAllocator<A: Allocator = DefaultAllocator> {
    device: Device,
    allocator: A,
    buffers: Vec<Buffer<A>>,
    current_buffer: usize,
    local_offset: vk::DeviceSize,
    chunk_size: vk::DeviceSize,
    alignment: vk::DeviceSize,
}

impl<A: Allocator> ScratchAllocator<A> {
    /// Create a new scratch allocator with a minimum capacity for internally allocated chunks.
    /// The actual allocated size may be slightly larger to satisfy alignment requirements.
    /// Alignment requirement is the maximum alignment needed for any buffer type. For more granular control, use
    /// [`Self::new_with_alignment()`]
    /// # Errors
    /// * Fails if the chunk size is not a multiple of the alignment value
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use anyhow::Result;
    /// fn make_scratch_allocator<A: Allocator>(device: Device, alloc: &mut A) -> Result<ScratchAllocator<A>> {
    ///     ScratchAllocator::new(device, alloc, 1024 as usize)
    /// }
    /// ```
    pub fn new(
        device: Device,
        allocator: &mut A,
        chunk_size: u64,
    ) -> Result<Self> {
        Self::new_with_alignment(device, allocator, 256, chunk_size)
    }

    /// Create a new scratch allocator with given alignment. The alignment used must be large enough to satisfy the alignment requirements
    /// of all buffer usage flags buffers from this allocator will be used with.
    /// # Errors
    /// * Fails if the internal allocation fails. This is possible when VRAM runs out.
    /// * Fails if the memory heap used for the allocation is not mappable.
    /// * Fails if the chunk size is not a multiple of the alignment value
    pub fn new_with_alignment(
        device: Device,
        allocator: &mut A,
        alignment: u64,
        chunk_size: u64,
    ) -> Result<Self> {
        if chunk_size % alignment != 0 {
            anyhow::bail!("Chunk size must be a multiple of alignment");
        }

        let buffer = Buffer::new(device.clone(), allocator, chunk_size, MemoryType::CpuToGpu)?;
        if !buffer.is_mapped() {
            anyhow::bail!(Error::UnmappableBuffer);
        }

        Ok(Self {
            buffers: vec![buffer],
            local_offset: 0,
            chunk_size,
            current_buffer: 0,
            alignment,
            device,
            allocator: allocator.clone(),
        })
    }

    /// Allocate at least size bytes from the allocator. The actual amount allocated may be slightly more to satisfy alignment
    /// requirements.
    /// # Errors
    /// * Fails if the internal allocation fails. This is possible when VRAM runs out.
    /// * Fails if the memory heap used for the allocation is not mappable.
    /// # Example
    /// ```
    /// # use phobos::prelude::*;
    /// # use anyhow::Result;
    /// fn use_scratch_allocator<A: Allocator>(device: Device, alloc: &mut A) -> Result<()> {
    ///     let mut allocator = ScratchAllocator::new(device.clone(), alloc, 1 * 1024u64)?;
    ///     let buffer: BufferView = allocator.allocate(64 as u64)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn allocate(&mut self, size: impl Into<vk::DeviceSize>) -> Result<BufferView> {
        let size: u64 = size.into();

        // Round up to the preferred alignment value
        let padded_size = ((size as f32) / (self.alignment as f32)).ceil() as u64 * self.alignment;
        
        // Check if we can use the current (last) buffer
        let current_buffer = self.buffers.get(self.current_buffer);
        let use_current_buffer = current_buffer.map(|buffer| self.local_offset + padded_size < buffer.size()).unwrap_or_default();

        // Get a buffer view to return
        let view = if use_current_buffer {
            current_buffer.unwrap().view(self.local_offset, size)
        } else {
            // In case we want to allocate something larger than the chunk size
            let whole_buffer_size = size.max(self.chunk_size);
            let whole_buffer_size = ((whole_buffer_size as f32) / (self.alignment as f32)).ceil() as u64 * self.alignment;
            
            // Create a new chunked buffer with the chunk size 
            let buffer = Buffer::new(self.device.clone(), &mut self.allocator, whole_buffer_size, MemoryType::CpuToGpu)?;
            if !buffer.is_mapped() {
                anyhow::bail!(Error::UnmappableBuffer);
            }

            self.local_offset = 0;
            let view = buffer.view(0u64, size);
            self.buffers.push(buffer);
            self.current_buffer = self.buffers.len() - 1;
            view
        };

        self.local_offset += padded_size;
        view
    }

    /// Resets the current offset into the allocator back to the beginning. Proper external synchronization needs to be
    /// added to ensure old buffers are not overwritten. This is usually done by using allocators from a [`LocalPool`](crate::pool::LocalPool)
    /// and keeping the pool alive as long as GPU execution.
    /// # Safety
    /// This function is safe if the old allocations can be completely discarded by the next time [`Self::allocate()`] is called.
    /// # Example
    /// ```
    /// # use phobos::prelude::*;
    /// # use anyhow::Result;
    /// // Function that uses the buffer in some way and returns a fence
    /// // that is signaled when the work is done.
    /// fn use_the_buffer(buffer: BufferView) -> Fence<()> {
    ///     unimplemented!()
    /// }
    ///
    /// fn use_scratch_allocator<A: Allocator>(device: Device, alloc: &mut A) -> Result<()> {
    ///     let mut allocator = ScratchAllocator::new(device.clone(), alloc, 128 as u64)?;
    ///     let buffer: BufferView = allocator.allocate(128 as u64)?;
    ///     let mut fence = use_the_buffer(buffer);
    ///     fence.wait()?;
    ///     // SAFETY: We just waited for the fence, so all work using our allocator is done.
    ///     unsafe { allocator.reset(); }
    ///     // We are back at the beginning of the allocator, so there are 128 bytes free again.
    ///     let buffer: BufferView = allocator.allocate(128 as u64)?;
    ///     Ok(())
    /// }
    /// ```
    pub unsafe fn reset(&mut self) {
        self.current_buffer = 0;
        self.local_offset = 0;

        // Note: This doesn't drop the internally stored chunks for the sake of performance
        // allows the scratch allocator to not have to re-allocate new GPU memory every time
    }
}

impl<A: Allocator> Poolable for ScratchAllocator<A> {
    type Key = ();

    fn on_release(&mut self) {
        unsafe { self.reset() }
    }
}
