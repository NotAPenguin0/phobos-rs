//! Wrappers around device addresses

use std::ffi::c_void;

use ash::vk;

use crate::util::to_vk::AsVulkanType;

/// Either a `VkDeviceAddress` or a mutable host pointer
#[derive(Copy, Clone)]
#[repr(C)]
pub enum DeviceOrHostAddress {
    /// Device pointer
    Device(vk::DeviceAddress),
    /// Mutable host pointer
    Host(*mut c_void),
}

/// Either a `VkDeviceAddress` or an immutable host pointer
#[derive(Copy, Clone)]
#[repr(C)]
pub enum DeviceOrHostAddressConst {
    /// Device pointer
    Device(vk::DeviceAddress),
    /// Immutable host pointer
    Host(*const c_void),
}

impl DeviceOrHostAddress {
    /// Create a null pointer on the device
    pub fn null_device() -> Self {
        Self::Device(vk::DeviceAddress::default())
    }

    /// Create a null pointer on the host
    pub fn null_host() -> Self {
        Self::Host(std::ptr::null_mut())
    }
}

impl DeviceOrHostAddressConst {
    /// Create a null pointer on the device
    pub fn null_device() -> Self {
        Self::Device(vk::DeviceAddress::default())
    }

    /// Create a null pointer on the host
    pub fn null_host() -> Self {
        Self::Host(std::ptr::null())
    }
}

impl AsVulkanType for DeviceOrHostAddress {
    type Output = vk::DeviceOrHostAddressKHR;

    fn as_vulkan(&self) -> Self::Output {
        match *self {
            DeviceOrHostAddress::Device(addr) => {
                vk::DeviceOrHostAddressKHR {
                    device_address: addr
                }
            }
            DeviceOrHostAddress::Host(ptr) => {
                vk::DeviceOrHostAddressKHR {
                    host_address: ptr
                }
            }
        }
    }
}

impl AsVulkanType for DeviceOrHostAddressConst {
    type Output = vk::DeviceOrHostAddressConstKHR;

    fn as_vulkan(&self) -> Self::Output {
        match *self {
            DeviceOrHostAddressConst::Device(addr) => {
                vk::DeviceOrHostAddressConstKHR {
                    device_address: addr
                }
            }
            DeviceOrHostAddressConst::Host(ptr) => {
                vk::DeviceOrHostAddressConstKHR {
                    host_address: ptr
                }
            }
        }
    }
}

impl From<vk::DeviceAddress> for DeviceOrHostAddress {
    fn from(value: vk::DeviceAddress) -> Self {
        Self::Device(value)
    }
}

impl From<*mut c_void> for DeviceOrHostAddress {
    fn from(value: *mut c_void) -> Self {
        Self::Host(value)
    }
}

impl From<vk::DeviceAddress> for DeviceOrHostAddressConst {
    fn from(value: vk::DeviceAddress) -> Self {
        Self::Device(value)
    }
}

impl From<*const c_void> for DeviceOrHostAddressConst {
    fn from(value: *const c_void) -> Self {
        Self::Host(value)
    }
}