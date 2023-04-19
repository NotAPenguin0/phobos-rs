use ash::vk;

use crate::util::to_vk::IntoVulkanType;

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
            AccelerationStructureType::TopLevel => vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            AccelerationStructureType::BottomLevel => vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            AccelerationStructureType::Generic => vk::AccelerationStructureTypeKHR::GENERIC,
        }
    }
}
