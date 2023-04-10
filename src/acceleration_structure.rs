use anyhow::Result;
use ash::vk;

use crate::Device;
use crate::core::device::ExtensionID;

pub struct AccelerationStructure {
    device: Device,
    handle: vk::AccelerationStructureKHR,
}

pub struct AccelerationStructureCreateInfo {}

impl AccelerationStructure {
    pub fn new(device: Device, info: AccelerationStructureCreateInfo) -> Result<Self> {
        device.require_extension(ExtensionID::AccelerationStructure)?;
        let fns = device.acceleration_structure().unwrap();

        todo!()
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.device
                .acceleration_structure()
                .unwrap()
                .destroy_acceleration_structure(self.handle, None);
        }
    }
}
