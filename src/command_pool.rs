use std::sync::Arc;
use ash::vk;
use crate::{Device, Error};

#[derive(Derivative)]
#[derivative(Debug)]
pub struct CommandPool {
    #[derivative(Debug="ignore")]
    device: Arc<Device>,
    pub(crate) handle: vk::CommandPool
}

impl CommandPool {
    pub fn new(device: Arc<Device>, family: u32, flags: vk::CommandPoolCreateFlags) -> Result<Self, Error> {
        let handle = unsafe { device.create_command_pool(&vk::CommandPoolCreateInfo::builder()
            .queue_family_index(family)
            .flags(flags),
        None)? };

        Ok(CommandPool {
            device: device.clone(),
            handle
        })
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe { self.device.destroy_command_pool(self.handle, None); }
    }
}