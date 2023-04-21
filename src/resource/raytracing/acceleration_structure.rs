//! Acceleration structure resource

use anyhow::Result;
use ash::vk;

use crate::{AccelerationStructureType, BufferView, Device};
use crate::core::device::ExtensionID;
use crate::util::to_vk::IntoVulkanType;

/// Wrapper around a [`VkAccelerationStructureKHR`](vk::AccelerationStructureKHR)
pub struct AccelerationStructure {
    device: Device,
    handle: vk::AccelerationStructureKHR,
    ty: AccelerationStructureType,
}

impl AccelerationStructure {
    /// Create a new acceleration structure.
    /// # Parameters
    /// * `device`  - The vulkan device.
    /// * `ty`      - The acceleration structure type. Use of [`AccelerationStructureType::Generic`] is discouraged.
    /// * `buffer`  - The backing memory buffer for this acceleration structure.
    /// * `flags`   - Acceleration structure create flags.
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

    /// Get the required alignment for the backing memory
    pub fn alignment() -> u64 {
        // From the spec: 'offset is an offset in bytes from the base address of the buffer at which the acceleration structure will be stored, and must be a multiple of 256'
        // (https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkAccelerationStructureCreateInfoKHR.html)
        256
    }

    /// Get unsafe access to the raw Vulkan handle of this acceleration structure
    /// # Safety
    /// Any mutation of the acceleration structure may put the system in an undefined state
    pub unsafe fn handle(&self) -> vk::AccelerationStructureKHR {
        self.handle
    }

    /// Get the device address of this acceleration structure
    pub fn address(&self) -> Result<vk::DeviceAddress> {
        self.device.require_extension(ExtensionID::AccelerationStructure)?;
        let fns = self.device.acceleration_structure().unwrap();
        unsafe {
            Ok(
                fns.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR {
                    s_type: vk::StructureType::ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                    p_next: std::ptr::null(),
                    acceleration_structure: self.handle(),
                }),
            )
        }
    }

    /// Get the acceleration structure type
    pub fn ty(&self) -> AccelerationStructureType {
        self.ty
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkAccelerationStructureKHR {:p}", self.handle);
        unsafe {
            self.device
                .acceleration_structure()
                // Since we created this object successfully surely the extension is supported
                .unwrap()
                .destroy_acceleration_structure(self.handle, None);
        }
    }
}
