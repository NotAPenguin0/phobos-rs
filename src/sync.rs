use std::future::Future;
use std::pin::Pin;
use std::slice;
use std::sync::Arc;
use std::task::{Context, Poll};
use ash::prelude::VkResult;
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

impl Fence {
    /// Waits for the fence to be signaled with no timeout.
    pub fn wait(&self) -> VkResult<()> {
        unsafe { self.device.wait_for_fences(slice::from_ref(&self.handle), true, u64::MAX) }
    }

    pub fn reset(&self) -> VkResult<()> {
        unsafe { self.device.reset_fences(slice::from_ref(&self.handle)) }
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