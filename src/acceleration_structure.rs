use anyhow::{bail, Result};
use ash::vk;

use crate::{BufferView, Device};
use crate::core::device::ExtensionID;
use crate::util::address::{DeviceOrHostAddress, DeviceOrHostAddressConst};
use crate::util::align::align;
use crate::util::to_vk::{AsVulkanType, IntoVulkanType};

pub struct AccelerationStructure {
    device: Device,
    handle: vk::AccelerationStructureKHR,
    ty: AccelerationStructureType,
}

#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone)]
pub enum AccelerationStructureType {
    TopLevel,
    BottomLevel,
    Generic,
}

#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone)]
pub enum AccelerationStructureBuildType {
    Host,
    Device,
}

#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone)]
pub struct AccelerationStructureBuildSize {
    pub size: vk::DeviceSize,
    pub update_scratch_size: vk::DeviceSize,
    pub build_scratch_size: vk::DeviceSize,
}

pub struct AccelerationStructureGeometryTrianglesData {
    pub format: vk::Format,
    pub vertex_data: DeviceOrHostAddressConst,
    pub stride: vk::DeviceSize,
    pub max_vertex: u32,
    pub index_type: vk::IndexType,
    pub index_data: DeviceOrHostAddressConst,
    pub transform_data: DeviceOrHostAddressConst,
    pub flags: vk::GeometryFlagsKHR,
}

pub struct AccelerationStructureBuildGeometryInfo<'a> {
    pub ty: AccelerationStructureType,
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
    pub mode: vk::BuildAccelerationStructureModeKHR,
    pub src: Option<&'a AccelerationStructure>,
    pub dst: Option<&'a AccelerationStructure>,
    pub geometries: Vec<vk::AccelerationStructureGeometryKHR>,
    pub scratch_data: DeviceOrHostAddress,
}

impl AccelerationStructureGeometryTrianglesData {
    pub fn default() -> Self {
        Self {
            format: vk::Format::default(),
            vertex_data: DeviceOrHostAddressConst::null_host(),
            stride: 0,
            max_vertex: 0,
            index_type: vk::IndexType::NONE_KHR,
            index_data: DeviceOrHostAddressConst::null_host(),
            transform_data: DeviceOrHostAddressConst::null_host(),
            flags: Default::default(),
        }
    }

    pub fn format(mut self, format: impl Into<vk::Format>) -> Self {
        self.format = format.into();
        self
    }

    pub fn vertex_data(mut self, data: impl Into<DeviceOrHostAddressConst>) -> Self {
        self.vertex_data = data.into();
        self
    }

    pub fn stride(mut self, stride: impl Into<vk::DeviceSize>) -> Self {
        self.stride = stride.into();
        self
    }

    pub fn max_vertex(mut self, max_vertex: u32) -> Self {
        self.max_vertex = max_vertex;
        self
    }

    pub fn index_data(mut self, ty: vk::IndexType, data: impl Into<DeviceOrHostAddressConst>) -> Self {
        self.index_type = ty;
        self.index_data = data.into();
        self
    }

    pub fn transform_data(mut self, data: impl Into<DeviceOrHostAddressConst>) -> Self {
        self.transform_data = data.into();
        self
    }

    pub fn flags(mut self, flags: vk::GeometryFlagsKHR) -> Self {
        self.flags = flags;
        self
    }
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

impl IntoVulkanType for AccelerationStructureBuildType {
    type Output = vk::AccelerationStructureBuildTypeKHR;

    fn into_vulkan(self) -> Self::Output {
        match self {
            AccelerationStructureBuildType::Host => vk::AccelerationStructureBuildTypeKHR::HOST,
            AccelerationStructureBuildType::Device => vk::AccelerationStructureBuildTypeKHR::DEVICE,
        }
    }
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
            src_acceleration_structure: self.src.map(|a| a.handle).unwrap_or_default(),
            dst_acceleration_structure: self.dst.map(|a| a.handle).unwrap_or_default(),
            geometry_count: self.geometries.len() as u32,
            p_geometries: self.geometries.as_ptr(),
            pp_geometries: std::ptr::null(),
            scratch_data: self.scratch_data.as_vulkan(),
        }
    }
}

impl IntoVulkanType for AccelerationStructureGeometryTrianglesData {
    type Output = vk::AccelerationStructureGeometryTrianglesDataKHR;

    fn into_vulkan(self) -> Self::Output {
        vk::AccelerationStructureGeometryTrianglesDataKHR {
            s_type: vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            p_next: std::ptr::null(),
            vertex_format: self.format,
            vertex_data: self.vertex_data.as_vulkan(),
            vertex_stride: self.stride,
            max_vertex: self.max_vertex,
            index_type: self.index_type,
            index_data: self.index_data.as_vulkan(),
            transform_data: self.transform_data.as_vulkan(),
        }
    }
}

pub struct AccelerationStructureBuildInfo<'a> {
    geometry: AccelerationStructureBuildGeometryInfo<'a>,
    build_range_infos: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
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

    pub fn push_instances(mut self, instances: vk::AccelerationStructureGeometryInstancesDataKHR, flags: vk::GeometryFlagsKHR) -> Self {
        self = self.push_geometry(vk::AccelerationStructureGeometryKHR {
            s_type: vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            p_next: std::ptr::null(),
            geometry_type: vk::GeometryTypeKHR::INSTANCES,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                instances,
            },
            flags,
        });
        self
    }

    pub fn push_geometry(mut self, geometry: vk::AccelerationStructureGeometryKHR) -> Self {
        self.geometry.geometries.push(geometry);
        self
    }

    pub fn scratch_data(mut self, data: DeviceOrHostAddress) -> Self {
        self.geometry.scratch_data = data;
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

    pub fn as_vulkan(
        &'a self,
    ) -> (
        vk::AccelerationStructureBuildGeometryInfoKHR,
        &'a [vk::AccelerationStructureBuildRangeInfoKHR],
    ) {
        (self.geometry.as_vulkan(), self.build_range_infos.as_slice())
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

        #[cfg(feature = "log-objects")]
        trace!("Created VkAccelerationStructureKHR {:p}", handle);

        Ok(Self {
            device,
            handle,
            ty,
        })
    }

    pub fn build_sizes(
        device: &Device,
        ty: AccelerationStructureBuildType,
        info: &AccelerationStructureBuildInfo,
        primitive_counts: &[u32],
    ) -> Result<AccelerationStructureBuildSize> {
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
        // From the spec: 'offset is an offset in bytes from the base address of the buffer at which the acceleration structure will be stored, and must be a multiple of 256'
        // (https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkAccelerationStructureCreateInfoKHR.html)
        const AS_ALIGNMENT: vk::DeviceSize = 256;
        let scratch_align = device
            .acceleration_structure_properties()?
            .min_acceleration_structure_scratch_offset_alignment as vk::DeviceSize;
        Ok(AccelerationStructureBuildSize {
            size: align(sizes.acceleration_structure_size, AS_ALIGNMENT),
            update_scratch_size: align(sizes.update_scratch_size, scratch_align),
            build_scratch_size: align(sizes.update_scratch_size, scratch_align),
        })
    }

    pub unsafe fn handle(&self) -> vk::AccelerationStructureKHR {
        self.handle
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkAccelerationStructureKHR {:p}", self.handle);
        unsafe {
            self.device
                .acceleration_structure()
                .unwrap()
                .destroy_acceleration_structure(self.handle, None);
        }
    }
}
