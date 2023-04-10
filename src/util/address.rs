use std::ffi::c_void;

use ash::vk;

use crate::util::to_vk::AsVulkanType;

#[derive(Copy, Clone)]
#[repr(C)]
pub enum DeviceOrHostAddress {
    Device(vk::DeviceAddress),
    Host(*mut c_void),
}

#[derive(Copy, Clone)]
#[repr(C)]
pub enum DeviceOrHostAddressConst {
    Device(vk::DeviceAddress),
    Host(*const c_void),
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