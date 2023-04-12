use anyhow::{anyhow, bail, Result};
use ash::vk;
use ash::vk::Handle;

use crate::{BufferView, Device};
use crate::core::device::ExtensionID;
use crate::util::address::{DeviceOrHostAddress, DeviceOrHostAddressConst};
use crate::util::align::align;
use crate::util::to_vk::{AsVulkanType, IntoVulkanType};
use crate::util::transform::TransformMatrix;

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

#[derive(Default, Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct u24([u8; 3]);

impl From<u24> for u32 {
    fn from(value: u24) -> Self {
        let u24([a, b, c]) = value;
        #[cfg(target_endian = "little")]
        return u32::from_le_bytes([a, b, c, 0]);
        #[cfg(target_endian = "big")]
        return u32::from_be_bytes([0, a, b, c]);
    }
}

impl TryFrom<u32> for u24 {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self> {
        let [a, b, c, d] = value.to_ne_bytes();
        #[cfg(target_endian = "little")]
        return if d == 0 { Ok(u24([a, b, c])) } else { Err(anyhow!("u32 did not fit in u24")) };
        #[cfg(target_endian = "big")]
        return if a == 0 { Ok(u24([b, c, d])) } else { Err(anyhow!("u32 did not fit in u24")) };
    }
}

pub struct AccelerationStructureGeometryInstancesData {
    pub data: DeviceOrHostAddressConst,
    pub flags: vk::GeometryFlagsKHR,
}


#[derive(Default, Copy, Clone)]
#[repr(C, packed)]
pub struct AccelerationStructureInstance {
    pub transform: TransformMatrix,
    pub custom_index: u24,
    pub mask: u8,
    pub shader_binding_table_record_offset: u24,
    pub flags: u8,
    pub acceleration_structure: u64,
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

pub struct AccelerationStructureBuildInfo<'a> {
    geometry: AccelerationStructureBuildGeometryInfo<'a>,
    build_range_infos: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
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

impl IntoVulkanType for AccelerationStructureGeometryInstancesData {
    type Output = vk::AccelerationStructureGeometryInstancesDataKHR;

    fn into_vulkan(self) -> Self::Output {
        vk::AccelerationStructureGeometryInstancesDataKHR {
            s_type: vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
            p_next: std::ptr::null(),
            array_of_pointers: vk::FALSE,
            data: self.data.as_vulkan(),
        }
    }
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

impl AccelerationStructureInstance {
    pub fn custom_index(mut self, idx: u32) -> Result<Self> {
        self.custom_index = u24::try_from(idx)?;
        Ok(self)
    }

    pub fn mask(mut self, mask: u8) -> Self {
        self.mask = mask;
        self
    }

    pub fn sbt_record_offset(mut self, offset: u32) -> Result<Self> {
        self.shader_binding_table_record_offset = u24::try_from(offset)?;
        Ok(self)
    }

    pub fn flags(mut self, flags: vk::GeometryInstanceFlagsKHR) -> Self {
        self.flags = flags.as_raw() as u8;
        self
    }

    pub fn acceleration_structure(mut self, accel: &AccelerationStructure, mode: AccelerationStructureBuildType) -> Result<Self> {
        match mode {
            AccelerationStructureBuildType::Host => {
                self.acceleration_structure = accel.handle.as_raw();
            }
            AccelerationStructureBuildType::Device => {
                self.acceleration_structure = accel.address()?
            }
        };
        Ok(self)
    }

    pub fn transform(mut self, transform: TransformMatrix) -> Self {
        self.transform = transform;
        self
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
        trace!("Created new VkAccelerationStructureKHR {:p}", handle);

        Ok(Self {
            device,
            handle,
            ty,
        })
    }

    pub fn alignment() -> u64 {
        // From the spec: 'offset is an offset in bytes from the base address of the buffer at which the acceleration structure will be stored, and must be a multiple of 256'
        // (https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkAccelerationStructureCreateInfoKHR.html)
        256
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
        let scratch_align = device
            .acceleration_structure_properties()?
            .min_acceleration_structure_scratch_offset_alignment as vk::DeviceSize;
        Ok(AccelerationStructureBuildSize {
            size: align(sizes.acceleration_structure_size, Self::alignment()),
            update_scratch_size: align(sizes.update_scratch_size, scratch_align),
            build_scratch_size: align(sizes.update_scratch_size, scratch_align),
        })
    }

    pub unsafe fn handle(&self) -> vk::AccelerationStructureKHR {
        self.handle
    }

    pub fn address(&self) -> Result<vk::DeviceAddress> {
        self.device.require_extension(ExtensionID::AccelerationStructure)?;
        let fns = self.device.acceleration_structure().unwrap();
        unsafe {
            Ok(fns.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR {
                s_type: vk::StructureType::ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                p_next: std::ptr::null(),
                acceleration_structure: self.handle(),
            }))
        }
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
