use anyhow::Result;
use ash::vk;
use ash::vk::Handle;

use phobos::{Allocation, Allocator, MemoryType, ScratchAllocator};

mod framework;

#[test]
pub fn basic_allocator_usage() -> Result<()> {
    let context = framework::make_context()?;
    let mut allocator = context.allocator.clone();
    let allocation = allocator.allocate(
        "allocation",
        &vk::MemoryRequirements {
            size: 1024,
            alignment: 1,
            // Assume all memory types are valid for this allocation
            memory_type_bits: 0xFFFFFFFF,
        },
        MemoryType::GpuOnly,
    )?;
    assert_ne!(
        unsafe { allocation.memory().as_raw() },
        0,
        "VkDeviceMemory used for allocation should not be null"
    );
    // Also try to explicitly freeing the allocation
    allocator.free(allocation)?;
    Ok(())
}

#[test]
pub fn cpu_to_gpu_is_mappable() -> Result<()> {
    let context = framework::make_context()?;
    let mut allocator = context.allocator.clone();
    let allocation = allocator.allocate(
        "allocation",
        &vk::MemoryRequirements {
            size: 1024,
            alignment: 1,
            // Assume all memory types are valid for this allocation
            memory_type_bits: 0xFFFFFFFF,
        },
        MemoryType::CpuToGpu,
    )?;
    assert!(
        allocation.mapped_ptr().is_some(),
        "Memory allocated with CpuToGpu should be mappable"
    );
    Ok(())
}

#[test]
pub fn make_scratch_allocator() -> Result<()> {
    let mut context = framework::make_context()?;

    let _scratch_allocator =
        ScratchAllocator::new(context.device.clone(), &mut context.allocator, 1024 as u64)?;
    Ok(())
}

#[test]
pub fn use_scratch_allocator() -> Result<()> {
    let mut context = framework::make_context()?;
    let mut scratch_allocator =
        ScratchAllocator::new(context.device.clone(), &mut context.allocator, 1024 as u64)?;
    // Try allocating a buffer that should fit in the scratch allocator's memory.
    let _buffer = scratch_allocator.allocate(128 as u64)?;
    Ok(())
}

#[test]
pub fn use_entire_scratch_allocator() -> Result<()> {
    // Try allocating the entire scratch allocator's memory for a single buffer
    let mut context = framework::make_context()?;
    let mut scratch_allocator =
        ScratchAllocator::new(context.device.clone(), &mut context.allocator, 1024 as u64)?;
    let _buffer = scratch_allocator.allocate(1024 as u64)?;
    Ok(())
}

#[test]
pub fn scratch_allocator_out_of_memory() -> Result<()> {
    let mut context = framework::make_context()?;
    let mut scratch_allocator =
        ScratchAllocator::new(context.device.clone(), &mut context.allocator, 1024 as u64)?;
    // First allocate a smaller buffer
    let _buffer = scratch_allocator.allocate(512 as u64)?;
    // This should definitely exceed the capacity of the allocator
    let result = scratch_allocator.allocate(2048 as u64);
    assert!(result.is_err(), "Allocating more memory than available from a scratch allocator should fail");
    Ok(())
}

#[test]
pub fn reset_scratch_allocator() -> Result<()> {
    let mut context = framework::make_context()?;
    let mut scratch_allocator =
        ScratchAllocator::new(context.device.clone(), &mut context.allocator, 1024 as u64)?;
    // Allocate a first buffer.
    let _buffer = scratch_allocator.allocate(800 as u64)?;
    // Now reset it, so we should be able to allocate again
    unsafe { scratch_allocator.reset(); }
    let _buffer = scratch_allocator.allocate(800 as u64)?;
    Ok(())
}