use std::ffi::c_void;

use ash::vk;

pub enum PNext {
    WriteDescriptorSetAccelerationStructure(vk::WriteDescriptorSetAccelerationStructureKHR)
}

fn as_void_ptr<T>(value: &T) -> *const c_void {
    value as *const T as *const c_void
}

impl PNext {
    pub fn as_ptr(&self) -> *const c_void {
        match self {
            PNext::WriteDescriptorSetAccelerationStructure(value) => { as_void_ptr(value) }
        }
    }
}