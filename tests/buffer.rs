use anyhow::Result;
use ash::vk;
use ash::vk::Handle;

use phobos::{Buffer, MemoryType};

mod framework;

#[test]
pub fn alloc_buffer() -> Result<()> {
    let mut context = framework::make_context().expect("Can initialize context.");

    const ALLOC_SIZE: u64 = 1024u64;

    let buffer = Buffer::new(
        context.device.clone(),
        &mut context.allocator,
        ALLOC_SIZE,
        MemoryType::GpuOnly,
    )?;

    assert_ne!(unsafe { buffer.handle().as_raw() }, 0, "Buffer handle should not be null.");
    assert!(
        buffer.size() >= ALLOC_SIZE,
        "Allocated buffer should at least fit requested size."
    );
    assert_ne!(buffer.address(), 0, "Buffer device address should not be null.");

    Ok(())
}

#[test]
pub fn alloc_aligned_buffer() -> Result<()> {
    let mut context = framework::make_context().expect("Can initialize context.");

    const ALLOC_SIZE: u64 = 1000u64;
    const ALIGN: u64 = 128;

    let buffer = Buffer::new_aligned(
        context.device.clone(),
        &mut context.allocator,
        ALLOC_SIZE,
        ALIGN,
        MemoryType::GpuOnly,
    )?;

    assert_ne!(unsafe { buffer.handle().as_raw() }, 0, "Buffer handle should not be null.");
    assert!(
        buffer.size() >= ALLOC_SIZE,
        "Allocated buffer should at least fit requested size."
    );
    assert_eq!(buffer.size() % ALIGN, 0, "Buffer size should be aligned correctly.");
    assert_eq!(buffer.address() % ALIGN, 0, "Buffer address should be aligned correctly.");

    Ok(())
}

#[test]
pub fn buffer_view_full() -> Result<()> {
    let mut context = framework::make_context().expect("Can initialize context.");

    const ALLOC_SIZE: u64 = 1024u64;

    let buffer = Buffer::new(
        context.device.clone(),
        &mut context.allocator,
        ALLOC_SIZE,
        MemoryType::GpuOnly,
    )?;

    let view = buffer.view_full();
    assert_eq!(view.address(), buffer.address(), "Full view should have same address as buffer");
    assert_eq!(view.offset(), 0, "Full view should have zero offset");
    assert_eq!(view.size(), buffer.size(), "Full view should have same size as buffer");
    assert_eq!(
        unsafe { view.handle() },
        unsafe { buffer.handle() },
        "View and buffer should refer to the same buffer"
    );

    Ok(())
}

#[test]
pub fn buffer_view() -> Result<()> {
    let mut context = framework::make_context().expect("Can initialize context.");

    const ALLOC_SIZE: u64 = 1024u64;

    let buffer = Buffer::new(
        context.device.clone(),
        &mut context.allocator,
        ALLOC_SIZE,
        MemoryType::GpuOnly,
    )?;

    let view = buffer
        .view(ALLOC_SIZE / 2, ALLOC_SIZE / 2)
        .expect("Can view second half of buffer");
    assert_eq!(
        view.address(),
        buffer.address() + ALLOC_SIZE / 2,
        "Buffer address respects pointer arithmetic"
    );
    assert_eq!(view.offset(), ALLOC_SIZE / 2, "View should have correct offset");
    assert_eq!(view.size(), ALLOC_SIZE / 2, "View should have correct size");
    assert_eq!(
        unsafe { view.handle() },
        unsafe { buffer.handle() },
        "View and buffer should refer to the same buffer"
    );

    Ok(())
}

#[test]
pub fn invalid_buffer_view() -> Result<()> {
    let mut context = framework::make_context().expect("Can initialize context.");

    const ALLOC_SIZE: u64 = 1024u64;

    let buffer = Buffer::new(
        context.device.clone(),
        &mut context.allocator,
        ALLOC_SIZE,
        MemoryType::GpuOnly,
    )?;

    buffer
        .view(0u64, ALLOC_SIZE * 2)
        .expect_err("Cannot create view outside of buffer range");

    Ok(())
}
