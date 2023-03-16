use std::ffi::c_void;
use std::ptr::NonNull;
use ash::vk;
use anyhow::Result;
use crate::allocator::memory_type::MemoryType;

pub trait Allocator: Clone + Send + Sync {
    type Allocation: Allocation;

    fn allocate(&mut self, name: &'static str, requirements: &vk::MemoryRequirements, ty: MemoryType) -> Result<Self::Allocation>;
    fn free(&mut self, allocation: Self::Allocation) -> Result<()>;
}

pub trait Allocation: Default {
    unsafe fn memory(&self) -> vk::DeviceMemory;
    fn offset(&self) -> vk::DeviceSize;
    fn mapped_ptr(&self) -> Option<NonNull<c_void>>;
}