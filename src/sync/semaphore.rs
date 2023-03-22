use std::sync::Arc;

use ash::vk;

use crate::Device;

/// Wrapper around a [`VkSemaphore`](vk::Semaphore) object. Semaphores are used for GPU-GPU sync.
#[derive(Debug)]
pub struct Semaphore {
    device: Arc<Device>,
    handle: vk::Semaphore,
}

impl Semaphore {
    /// Create a new `VkSemaphore` object.
    pub fn new(device: Arc<Device>) -> Result<Self, vk::Result> {
        let info = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
        };
        Ok(Semaphore {
            handle: unsafe {
                device.create_semaphore(&info, None)?
            },
            device,
        })
    }

    pub unsafe fn handle(&self) -> vk::Semaphore {
        self.handle
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { self.device.destroy_semaphore(self.handle, None); }
    }
}