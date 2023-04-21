//! Parameter for acceleration structure build commands

use ash::vk;

use crate::util::to_vk::IntoVulkanType;

/// The acceleration structure build type indicates whether the AS will be built on the Host or on the Device.
/// Prefer using Device build operations.
#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone)]
pub enum AccelerationStructureBuildType {
    /// Build on the Host
    Host,
    /// Build on the Device
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