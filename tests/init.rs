use anyhow::Result;
use ash::vk::Handle;

use phobos::{QueueRequest, QueueType};
use phobos::domain::{Compute, Graphics, Transfer};

mod framework;

#[test]
pub fn can_initialize() -> Result<()> {
    let _context = framework::make_context()?;
    Ok(())
}

#[test]
pub fn multi_queue_initialize() -> Result<()> {
    let context = framework::make_context_with_queues([
        QueueRequest {
            dedicated: false,
            queue_type: QueueType::Graphics,
        },
        QueueRequest {
            dedicated: true,
            queue_type: QueueType::Transfer,
        },
        QueueRequest {
            dedicated: true,
            queue_type: QueueType::Compute,
        },
    ])?;

    // Now try to obtain an instance of all requested queues. Note that we cannot assert these do not refer to the same physical queue,
    // as they might not.
    {
        let _graphics = context.exec.get_queue::<Graphics>();
    }

    {
        let _transfer = context.exec.get_queue::<Transfer>();
    }

    {
        let _compute = context.exec.get_queue::<Compute>();
    }

    Ok(())
}

#[test]
pub fn requesting_raytracing_does_not_fail() -> Result<()> {
    // Even if raytracing is not available this call should not return an Err case
    let _context = framework::make_context_with_settings(|settings| settings.raytracing(true))?;
    Ok(())
}

#[test]
pub fn vulkan_loaded() -> Result<()> {
    let context = framework::make_context()?;
    let handle = context.instance.handle();
    assert_ne!(handle.as_raw(), 0, "VkInstance handle should not be zero");
    let loader = unsafe { context.instance.loader() };
    assert_ne!(loader.fp_v1_0().create_instance as *const std::ffi::c_void, std::ptr::null(), "Vulkan function pointers should not be null");
    Ok(())
}

#[test]
pub fn valid_device() -> Result<()> {
    let context = framework::make_context()?;
    let handle = unsafe { context.device.handle() };
    assert_ne!(handle.handle().as_raw(), 0, "VkDevice handle should not be zero");
    // Also try a vulkan function call on it to make sure it is loaded properly
    unsafe { handle.device_wait_idle()?; }
    Ok(())
}