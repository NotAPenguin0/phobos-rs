use anyhow::Result;
use ash::vk;

use crate::{BufferView, Device};
use crate::core::device::ExtensionID;
use crate::util::to_vk::IntoVulkanType;

pub struct AccelerationStructure {
    device: Device,
    handle: vk::AccelerationStructureKHR,
}

#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone)]
pub enum AccelerationStructureType {
    TopLevel,
    BottomLevel,
    Generic,
}

impl IntoVulkanType for AccelerationStructureType {
    type Output = vk::AccelerationStructureTypeKHR;

    fn into_vulkan(self) -> Self::Output {
        match self {
            AccelerationStructureType::TopLevel => {
                vk::AccelerationStructureTypeKHR::TOP_LEVEL
            }
            AccelerationStructureType::BottomLevel => {
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL
            }
            AccelerationStructureType::Generic => {
                vk::AccelerationStructureTypeKHR::GENERIC
            }
        }
    }
}

impl AccelerationStructure {
    pub fn new(device: Device, ty: AccelerationStructureType, buffer: BufferView, flags: vk::AccelerationStructureCreateFlagsKHR) -> Result<Self> {
        device.require_extension(ExtensionID::AccelerationStructure)?;
        let fns = device.acceleration_structure().unwrap();
        if ty == AccelerationStructureType::Generic {
            warn!("Applications should avoid using Generic acceleration structures, this is intended for API translation layers. See https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureCreateInfoKHR.html");
        }
        let info = vk::AccelerationStructureCreateInfoKHR {
            s_type: vk::StructureType::ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            p_next: std::ptr::null(),
            create_flags: flags,
            buffer: unsafe { buffer.handle() },
            offset: buffer.offset(),
            size: buffer.size(),
            ty: ty.into_vulkan(),
            // should be left at zero
            device_address: 0,
        };

        let handle = unsafe { fns.create_acceleration_structure(&info, None)? };
        Ok(Self {
            device,
            handle,
        })
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
