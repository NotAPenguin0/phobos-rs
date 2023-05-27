//! Wrappers for acceleration structure instance geometry data

use anyhow::Result;
use ash::vk;
use ash::vk::Packed24_8;

use crate::util::address::DeviceOrHostAddressConst;
use crate::util::to_vk::{AsVulkanType, IntoVulkanType};
use crate::util::transform::TransformMatrix;
use crate::{AccelerationStructure, AccelerationStructureBuildType};

/// Instance data in an acceleration structure
pub struct AccelerationStructureGeometryInstancesData {
    /// Data buffer filled with packed [`AccelerationStructureInstance`] structs.
    pub data: DeviceOrHostAddressConst,
    /// Geometry flags
    pub flags: vk::GeometryFlagsKHR,
}

/// An instance in the acceleration structure instance buffer
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
    /// Create a default instance in an acceleration structure
    fn default() -> Self {
        Self(vk::AccelerationStructureInstanceKHR {
            transform: TransformMatrix::default().into_vulkan(),
            instance_custom_index_and_mask: Packed24_8::new(0, 0),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(0, 0),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                host_handle: vk::AccelerationStructureKHR::null(),
            },
        })
    }
}

impl AccelerationStructureInstance {
    /// Set the custom index of this instance. The highest 8 bits of this value are ignored, only 24 bits of
    /// precision are supported
    pub fn custom_index(mut self, idx: u32) -> Result<Self> {
        self.0.instance_custom_index_and_mask =
            Packed24_8::new(idx, self.0.instance_custom_index_and_mask.high_8());
        Ok(self)
    }

    /// Set the mask of this instance, used to disable specific instances when tracing
    pub fn mask(mut self, mask: u8) -> Self {
        self.0.instance_custom_index_and_mask =
            Packed24_8::new(self.0.instance_custom_index_and_mask.low_24(), mask);
        self
    }

    /// Set the hit group offset into the shader binding table for this instance
    pub fn sbt_record_offset(mut self, offset: u32) -> Result<Self> {
        self.0.instance_shader_binding_table_record_offset_and_flags = Packed24_8::new(
            offset,
            self.0
                .instance_shader_binding_table_record_offset_and_flags
                .high_8(),
        );
        Ok(self)
    }

    /// Set the instance flags
    pub fn flags(mut self, flags: vk::GeometryInstanceFlagsKHR) -> Self {
        self.0.instance_shader_binding_table_record_offset_and_flags = Packed24_8::new(
            self.0
                .instance_shader_binding_table_record_offset_and_flags
                .low_24(),
            flags.as_raw() as u8,
        );
        self
    }

    /// Set the acceleration structure this instance refers to
    pub fn acceleration_structure(
        mut self,
        accel: &AccelerationStructure,
        mode: AccelerationStructureBuildType,
    ) -> Result<Self> {
        match mode {
            AccelerationStructureBuildType::Host => {
                self.0.acceleration_structure_reference = vk::AccelerationStructureReferenceKHR {
                    host_handle: unsafe { accel.handle() },
                };
            }
            AccelerationStructureBuildType::Device => {
                self.0.acceleration_structure_reference = vk::AccelerationStructureReferenceKHR {
                    device_handle: accel.address()?,
                };
            }
        };
        Ok(self)
    }

    /// Set this instance's transform matrix
    pub fn transform(mut self, transform: TransformMatrix) -> Self {
        self.0.transform = transform.into_vulkan();
        self
    }
}
