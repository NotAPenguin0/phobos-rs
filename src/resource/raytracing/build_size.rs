//! Utilities for querying acceleration structure build sizes

use anyhow::{bail, Result};
use ash::vk;

use crate::{AccelerationStructure, AccelerationStructureBuildInfo, AccelerationStructureBuildType, Device};
use crate::core::device::ExtensionID;
use crate::util::align::align;
use crate::util::to_vk::{AsVulkanType, IntoVulkanType};

/// Holds the required sizes of buffers for the acceleration structure
#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone)]
pub struct AccelerationStructureBuildSize {
    /// Required size of the acceleration structure
    pub size: vk::DeviceSize,
    /// Required size of the scratch buffer for update operations
    pub update_scratch_size: vk::DeviceSize,
    /// Required size of the scratch buffer for build operations
    pub build_scratch_size: vk::DeviceSize,
}

/// Get the build sizes for this acceleration structure build info
pub fn query_build_size(device: &Device, ty: AccelerationStructureBuildType, info: &AccelerationStructureBuildInfo, primitive_counts: &[u32]) -> Result<AccelerationStructureBuildSize> {
    device.require_extension(ExtensionID::AccelerationStructure)?;
    let fns = device.acceleration_structure().unwrap();

    if primitive_counts.len() != info.geometry.geometries.len() {
        bail!(
                "max primitive count length should match the number of geometries (expected: {}, actual: {})",
                info.geometry.geometries.len(),
                primitive_counts.len()
            );
    }

    let sizes = unsafe { fns.get_acceleration_structure_build_sizes(ty.into_vulkan(), &info.geometry.as_vulkan(), primitive_counts) };
    let scratch_align = device
        .acceleration_structure_properties()?
        .min_acceleration_structure_scratch_offset_alignment as vk::DeviceSize;
    Ok(AccelerationStructureBuildSize {
        size: align(sizes.acceleration_structure_size, AccelerationStructure::alignment()),
        update_scratch_size: align(sizes.update_scratch_size, scratch_align),
        build_scratch_size: align(sizes.build_scratch_size, scratch_align),
    })
}