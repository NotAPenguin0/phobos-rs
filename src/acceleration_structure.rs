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

pub struct AccelerationStructureBuildGeometryInfo<'a> {
    pub ty: AccelerationStructureType,
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
    pub mode: vk::BuildAccelerationStructureModeKHR,
    pub src: Option<&'a AccelerationStructure>,
    pub dst: Option<&'a AccelerationStructure>,
    pub geometries: &'a [vk::AccelerationStructureGeometryKHR],
    pub scratch_data: DeviceOrHostAddress,
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
        Ok(Self {
            device,
            handle,
        })
    }

    pub fn build_sizes(
        device: &Device,
        ty: AccelerationStructureBuildType,
        info: &AccelerationStructureBuildGeometryInfo,
        primitive_counts: &[u32],
    ) -> Result<AccelerationStructureBuildSize> {
        device.require_extension(ExtensionID::AccelerationStructure)?;
        let fns = device.acceleration_structure().unwrap();

        if primitive_counts.len() != info.geometries.len() {
            bail!(
                "max primitive count length should match the number of geometries (expected: {}, actual: {})",
                info.geometries.len(),
                primitive_counts.len()
            );
        }

        let sizes = unsafe { fns.get_acceleration_structure_build_sizes(ty.into_vulkan(), &info.as_vulkan(), primitive_counts) };
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
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.device
                .acceleration_structure()
                .unwrap()
                .destroy_acceleration_structure(self.handle, None);
        }
    }
}
