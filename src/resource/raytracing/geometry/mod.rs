use ash::vk;

pub use instances::*;
pub use triangles::*;

use crate::{AccelerationStructure, AccelerationStructureType};
use crate::util::address::DeviceOrHostAddress;
use crate::util::to_vk::{AsVulkanType, IntoVulkanType};

pub mod triangles;
pub mod instances;

pub struct AccelerationStructureBuildGeometryInfo<'a> {
    pub ty: AccelerationStructureType,
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
    pub mode: vk::BuildAccelerationStructureModeKHR,
    pub src: Option<&'a AccelerationStructure>,
    pub dst: Option<&'a AccelerationStructure>,
    pub geometries: Vec<vk::AccelerationStructureGeometryKHR>,
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