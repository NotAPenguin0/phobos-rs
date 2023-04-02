use anyhow::Result;
use ash::vk;

use crate::prelude::*;

/// The command pool is where command buffers are allocated from.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct CommandPool {
    #[derivative(Debug = "ignore")]
    device: Device,
    handle: vk::CommandPool,
}

impl CommandPool {
    /// Create a new command pool over a queue family with specified flags.
    pub fn new(device: Device, family: u32, flags: vk::CommandPoolCreateFlags) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags,
            queue_family_index: family,
        };
        let handle = unsafe { device.create_command_pool(&info, None)? };

        Ok(CommandPool {
            device,
            handle,
        })
    }

    /// Get unsafe access to the underlying `VkCommandPool` handle.
    /// # Safety
    /// - Access to the command pool **and** command buffers allocated from it must be externally synchronized.
    pub unsafe fn handle(&self) -> vk::CommandPool {
        self.handle
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.handle, None);
        }
    }
}
