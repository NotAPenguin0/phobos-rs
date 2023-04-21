//! Acceleration structure type

use ash::vk;

use crate::util::to_vk::IntoVulkanType;

/// Acceleration structure type
#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone)]
pub enum AccelerationStructureType {
    /// A top level acceleration structure holds instances of a BLAS
    TopLevel,
    /// A bottom level acceleration structure stores mesh geometry
    BottomLevel,
    /// Usage of this is discouraged, but it is included for completeness
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
