use ash::vk;

use crate::util::to_vk::IntoVulkanType;

#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone)]
pub enum AccelerationStructureBuildType {
    Host,
    Device,
}

impl IntoVulkanType for AccelerationStructureBuildType {
    type Output = vk::AccelerationStructureBuildTypeKHR;

    fn into_vulkan(self) -> Self::Output {
        match self {
            AccelerationStructureBuildType::Host => vk::AccelerationStructureBuildTypeKHR::HOST,
            AccelerationStructureBuildType::Device => vk::AccelerationStructureBuildTypeKHR::DEVICE,
        }
    }
}