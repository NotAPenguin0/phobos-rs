use anyhow::Result;
use ash::vk;
use ash::vk::Packed24_8;

use crate::{AccelerationStructure, AccelerationStructureBuildType};
use crate::util::address::DeviceOrHostAddressConst;
use crate::util::to_vk::{AsVulkanType, IntoVulkanType};
use crate::util::transform::TransformMatrix;

pub struct AccelerationStructureGeometryInstancesData {
    pub data: DeviceOrHostAddressConst,
    pub flags: vk::GeometryFlagsKHR,
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct AccelerationStructureInstance(vk::AccelerationStructureInstanceKHR);

const_assert_eq!(std::mem::size_of::<AccelerationStructureInstance>(), 64);

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


impl Default for AccelerationStructureInstance {
    fn default() -> Self {
        Self(vk::AccelerationStructureInstanceKHR {
            transform: TransformMatrix::default().into_vulkan(),
            instance_custom_index_and_mask: Packed24_8::new(0, 0),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(0, 0),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                host_handle: vk::AccelerationStructureKHR::null()
            },
        })
    }
}

impl AccelerationStructureInstance {
    pub fn custom_index(mut self, idx: u32) -> Result<Self> {
        self.0.instance_custom_index_and_mask = Packed24_8::new(idx, self.0.instance_custom_index_and_mask.high_8());
        Ok(self)
    }

    pub fn mask(mut self, mask: u8) -> Self {
        self.0.instance_custom_index_and_mask = Packed24_8::new(self.0.instance_custom_index_and_mask.low_24(), mask);
        self
    }

    pub fn sbt_record_offset(mut self, offset: u32) -> Result<Self> {
        self.0.instance_shader_binding_table_record_offset_and_flags = Packed24_8::new(offset, self.0.instance_shader_binding_table_record_offset_and_flags.high_8());
        Ok(self)
    }

    pub fn flags(mut self, flags: vk::GeometryInstanceFlagsKHR) -> Self {
        self.0.instance_shader_binding_table_record_offset_and_flags = Packed24_8::new(self.0.instance_shader_binding_table_record_offset_and_flags.low_24(), flags.as_raw() as u8);
        self
    }

    pub fn acceleration_structure(mut self, accel: &AccelerationStructure, mode: AccelerationStructureBuildType) -> Result<Self> {
        match mode {
            AccelerationStructureBuildType::Host => {
                self.0.acceleration_structure_reference = vk::AccelerationStructureReferenceKHR {
                    host_handle: unsafe { accel.handle() }
                };
            }
            AccelerationStructureBuildType::Device => {
                self.0.acceleration_structure_reference = vk::AccelerationStructureReferenceKHR {
                    device_handle: accel.address()?
                };
            }
        };
        Ok(self)
    }

    pub fn transform(mut self, transform: TransformMatrix) -> Self {
        self.0.transform = transform.into_vulkan();
        self
    }
}