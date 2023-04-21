//! Exposes different acceleration structure geometry types

use ash::vk;

pub use instances::*;
pub use triangles::*;

use crate::{AccelerationStructure, AccelerationStructureType};
use crate::util::address::DeviceOrHostAddress;
use crate::util::to_vk::{AsVulkanType, IntoVulkanType};

pub mod triangles;
pub mod instances;

/// All information required to build the geometry of an acceleration structure
pub struct AccelerationStructureBuildGeometryInfo<'a> {
    /// The acceleration structure type
    pub ty: AccelerationStructureType,
    /// Acceleration structure build flags
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
    /// The acceleration structure build mode
    pub mode: vk::BuildAccelerationStructureModeKHR,
    /// Source acceleration structure
    pub src: Option<&'a AccelerationStructure>,
    /// Destination acceleration structure
    pub dst: Option<&'a AccelerationStructure>,
    /// Geometry data in this acceleration structure
    pub geometries: Vec<vk::AccelerationStructureGeometryKHR>,
    /// Scratch data used for building
    pub scratch_data: DeviceOrHostAddress,
}

impl<'a> AsVulkanType for AccelerationStructureBuildGeometryInfo<'a> {
    type Output = vk::AccelerationStructureBuildGeometryInfoKHR;

    fn as_vulkan(&self) -> Self::Output {
        vk::AccelerationStructureBuildGeometryInfoKHR {
            s_type: vk::StructureType::ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            p_next: std::ptr::null(),
            ty: self.ty.into_vulkan(),
            flags: self.flags,
            mode: self.mode,
            src_acceleration_structure: self.src.map(|a| unsafe { a.handle() }).unwrap_or_default(),
            dst_acceleration_structure: self.dst.map(|a| unsafe { a.handle() }).unwrap_or_default(),
            geometry_count: self.geometries.len() as u32,
            p_geometries: self.geometries.as_ptr(),
            pp_geometries: std::ptr::null(),
            scratch_data: self.scratch_data.as_vulkan(),
        }
    }
}