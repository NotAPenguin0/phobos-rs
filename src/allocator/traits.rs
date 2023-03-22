use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::Result;
use ash::vk;

use crate::allocator::memory_type::MemoryType;

/// To supply custom allocators to phobos, this trait must be implemented.
/// Note that all allocators must be `Clone`, `Send` and `Sync`. To do this, wrap internal state in
/// `Arc<Mutex<T>>` where applicable.
pub trait Allocator: Clone + Send + Sync {
    /// Allocation type for this allocator. Must implement [`Allocation`]
    type Allocation: Allocation;

    /// Allocates raw memory of a specific memory type. The given name is used for internal tracking.
    fn allocate(
        &mut self,
        name: &'static str,
        requirements: &vk::MemoryRequirements,
        ty: MemoryType,
    ) -> Result<Self::Allocation>;
    /// Free some memory allocated from this allocator.
    fn free(&mut self, allocation: Self::Allocation) -> Result<()>;
}

/// Represents an allocation. This trait exposes methods for accessing the underlying device memory, obtain a mapped pointer, etc.
pub trait Allocation: Default {
    /// Access the underlying [`VkDeviceMemory`]. Remember to always `Self::offset()` into this.
    unsafe fn memory(&self) -> vk::DeviceMemory;
    /// The offset of this allocation in the underlying memory block.
    fn offset(&self) -> vk::DeviceSize;
    /// Obtain a mapped pointer to the memory, or None if this is not possible.
    fn mapped_ptr(&self) -> Option<NonNull<c_void>>;
}
