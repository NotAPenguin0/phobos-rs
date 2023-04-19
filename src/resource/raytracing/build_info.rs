use ash::vk;

use crate::{AccelerationStructure, AccelerationStructureBuildGeometryInfo, AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryTrianglesData, AccelerationStructureType};
use crate::util::address::DeviceOrHostAddress;
use crate::util::to_vk::{AsVulkanType, IntoVulkanType};

pub struct AccelerationStructureBuildInfo<'a> {
    pub(crate) geometry: AccelerationStructureBuildGeometryInfo<'a>,
    pub(crate) build_range_infos: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
}

impl<'a> Default for AccelerationStructureBuildInfo<'a> {
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
    pub fn new_build() -> Self {
        Self::default()
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
    }

    pub fn new_update() -> Self {
        Self::default()
            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
    }

    pub fn set_type(mut self, ty: AccelerationStructureType) -> Self {
        self.geometry.ty = ty;
        self
    }

    pub fn flags(mut self, flags: vk::BuildAccelerationStructureFlagsKHR) -> Self {
        self.geometry.flags = flags;
        self
    }

    pub fn mode(mut self, mode: vk::BuildAccelerationStructureModeKHR) -> Self {
        self.geometry.mode = mode;
        self
    }

    pub fn src(mut self, src: &'a AccelerationStructure) -> Self {
        self.geometry.src = Some(src);
        self
    }

    pub fn dst(mut self, dst: &'a AccelerationStructure) -> Self {
        self.geometry.dst = Some(dst);
        self
    }

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

    pub fn push_geometry(mut self, geometry: vk::AccelerationStructureGeometryKHR) -> Self {
        self.geometry.geometries.push(geometry);
        self
    }

    pub fn scratch_data(mut self, data: impl Into<DeviceOrHostAddress>) -> Self {
        self.geometry.scratch_data = data.into();
        self
    }

    pub fn push_range(mut self, primitive_count: u32, primitive_offset: u32, first_vertex: u32, transform_offset: u32) -> Self {
        self.build_range_infos.push(vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count,
            primitive_offset,
            first_vertex,
            transform_offset,
        });
        self
    }

    pub fn ty(&self) -> AccelerationStructureType {
        self.geometry.ty
    }

    pub fn as_vulkan(
        &'a self,
    ) -> (
        vk::AccelerationStructureBuildGeometryInfoKHR,
        &'a [vk::AccelerationStructureBuildRangeInfoKHR],
    ) {
        (self.geometry.as_vulkan(), self.build_range_infos.as_slice())
    }
}
