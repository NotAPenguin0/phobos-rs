use ash::vk;

use crate::Device;

/// Wrapper around a [`VkSemaphore`](vk::Semaphore) object. Semaphores are used for GPU-GPU sync.
#[derive(Debug)]
pub struct Semaphore {
    device: Device,
    handle: vk::Semaphore,
}

impl Semaphore {
    /// Create a new `VkSemaphore` object.
    pub fn new(device: Device) -> Result<Self, vk::Result> {
        let info = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
        };
        Ok(Semaphore {
            handle: unsafe { device.create_semaphore(&info, None)? },
            device,
        })
    }

    /// Get unsafe access to the underlying `VkSemaphore` object.
    /// # Safety
    /// Any vulkan calls that mutate the semaphore may put the system in an undefined state.
    pub unsafe fn handle(&self) -> vk::Semaphore {
        self.handle
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.handle, None);
        }
    }
}
