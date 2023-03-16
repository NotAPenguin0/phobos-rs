use std::sync::Arc;
use ash::vk;
use crate::prelude::*;

use anyhow::Result;

#[derive(Derivative)]
#[derivative(Debug)]
pub struct CommandPool {
    #[derivative(Debug="ignore")]
    device: Arc<Device>,
    pub(crate) handle: vk::CommandPool
}

impl CommandPool {
    pub fn new(device: Arc<Device>, family: u32, flags: vk::CommandPoolCreateFlags) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags,
            queue_family_index: family,
        };
        let handle = unsafe { device.create_command_pool(&info, None)? };

        Ok(CommandPool {
            device,
            handle
        })
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe { self.device.destroy_command_pool(self.handle, None); }
    }
}