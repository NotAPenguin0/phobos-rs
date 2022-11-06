use std::sync::Arc;
use ash::vk;
use crate::Device;

/// Wrapper around a [`VkFence`](vk::Fence) object.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Fence {
    #[derivative(Debug="ignore")]
    pub device: Arc<Device>,
    pub handle: vk::Fence,
}

/// Wrapper around a [`VkSemaphore`](vk::Semaphore) object.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Semaphore {
    #[derivative(Debug="ignore")]
    pub device: Arc<Device>,
    pub handle: vk::Semaphore,
}

impl Fence {
    /// Create a new fence, possibly in the singaled status.
    pub fn new(device: Arc<Device>, signaled: bool) -> Result<Self, vk::Result> {
        Ok(Fence {
            device: device.clone(),
            handle: unsafe {
                device.create_fence(&vk::FenceCreateInfo::builder()
                        .flags(if signaled { vk::FenceCreateFlags::SIGNALED } else { vk::FenceCreateFlags::empty() }),
    None)?
            }
        })
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe { self.device.destroy_fence(self.handle, None); }
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { self.device.destroy_semaphore(self.handle, None); }
    }
}