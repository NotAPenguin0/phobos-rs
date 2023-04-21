//! Exposes structures with build info for acceleration structures

use ash::vk;

use crate::{AccelerationStructure, AccelerationStructureBuildGeometryInfo, AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryTrianglesData, AccelerationStructureType};
use crate::util::address::DeviceOrHostAddress;
use crate::util::to_vk::{AsVulkanType, IntoVulkanType};

/// All information required to build an acceleration structure
pub struct AccelerationStructureBuildInfo<'a> {
    pub(crate) geometry: AccelerationStructureBuildGeometryInfo<'a>,
    pub(crate) build_range_infos: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
}

impl<'a> Default for AccelerationStructureBuildInfo<'a> {
    /// Create a default acceleration structure build info
    fn default() -> Self {
        Self {
            geometry: AccelerationStructureBuildGeometryInfo {
                ty: AccelerationStructureType::TopLevel,
                flags: Default::default(),
                mode: Default::default(),
                src: None,
                dst: None,
                geometries: vec![],
                scratch_data: DeviceOrHostAddress::null_host(),
            },
            build_range_infos: vec![],
        }
    }
}

impl<'a> AccelerationStructureBuildInfo<'a> {
    /// Create a build info struct for AS build operations
    pub fn new_build() -> Self {
        Self::default()
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
    }

    /// Create a build info struct for AS update operations
    pub fn new_update() -> Self {
        Self::default()
            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
    }

    /// Set the acceleration structure type
    pub fn set_type(mut self, ty: AccelerationStructureType) -> Self {
        self.geometry.ty = ty;
        self
    }

    /// Set the acceleration structure build flags
    pub fn flags(mut self, flags: vk::BuildAccelerationStructureFlagsKHR) -> Self {
        self.geometry.flags = flags;
        self
    }

    /// Set the acceleration structure build mode
    pub fn mode(mut self, mode: vk::BuildAccelerationStructureModeKHR) -> Self {
        self.geometry.mode = mode;
        self
    }

    /// Set the source acceleration structure
    pub fn src(mut self, src: &'a AccelerationStructure) -> Self {
        self.geometry.src = Some(src);
        self
    }

    /// Set the destination acceleration structure
    pub fn dst(mut self, dst: &'a AccelerationStructure) -> Self {
        self.geometry.dst = Some(dst);
        self
    }

    /// Add a triangle geometry to this acceleration structure
    pub fn push_triangles(mut self, triangles: AccelerationStructureGeometryTrianglesData) -> Self {
        self = self.push_geometry(vk::AccelerationStructureGeometryKHR {
            s_type: vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            p_next: std::ptr::null(),
            flags: triangles.flags,
            geometry_type: vk::GeometryTypeKHR::TRIANGLES,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                triangles: triangles.into_vulkan(),
            },
        });
        self
    }

    /// Add an AABB geometry to this acceleration structure
    pub fn push_aabbs(mut self, aabbs: vk::AccelerationStructureGeometryAabbsDataKHR, flags: vk::GeometryFlagsKHR) -> Self {
        self = self.push_geometry(vk::AccelerationStructureGeometryKHR {
            s_type: vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            p_next: std::ptr::null(),
            geometry_type: vk::GeometryTypeKHR::AABBS,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                aabbs,
            },
            flags,
        });
        self
    }

    /// Add instance geometry to this acceleration structure
    pub fn push_instances(mut self, instances: AccelerationStructureGeometryInstancesData) -> Self {
        self = self.push_geometry(vk::AccelerationStructureGeometryKHR {
            s_type: vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            p_next: std::ptr::null(),
            geometry_type: vk::GeometryTypeKHR::INSTANCES,
            flags: instances.flags,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                instances: instances.into_vulkan(),
            },
        });
        self
    }

    /// Add any geoemtry to this acceleration structure
    pub fn push_geometry(mut self, geometry: vk::AccelerationStructureGeometryKHR) -> Self {
        self.geometry.geometries.push(geometry);
        self
    }

    /// Set the scratch buffer device address used for the build operation
    pub fn scratch_data(mut self, data: impl Into<DeviceOrHostAddress>) -> Self {
        self.geometry.scratch_data = data.into();
        self
    }

    /// Add a primitive range to this acceleration structure
    pub fn push_range(mut self, primitive_count: u32, primitive_offset: u32, first_vertex: u32, transform_offset: u32) -> Self {
        self.build_range_infos.push(vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count,
            primitive_offset,
            first_vertex,
            transform_offset,
        });
        self
    }

    /// Get the acceleration structure type
    pub fn ty(&self) -> AccelerationStructureType {
        self.geometry.ty
    }

    /// Get the Vulkan structs describing this build info
    pub fn as_vulkan(
        &'a self,
    ) -> (
        vk::AccelerationStructureBuildGeometryInfoKHR,
        &'a [vk::AccelerationStructureBuildRangeInfoKHR],
    ) {
        (self.geometry.as_vulkan(), self.build_range_infos.as_slice())
    }
}
