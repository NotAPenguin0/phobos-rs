//! Utilities for dealing with generic pNext chains

use std::ffi::c_void;

use ash::vk;

/// A typed element in the pNext chain
pub enum PNext {
    WriteDescriptorSetAccelerationStructure(vk::WriteDescriptorSetAccelerationStructureKHR)
}

fn as_void_ptr<T>(value: &T) -> *const c_void {
    value as *const T as *const c_void
}

impl PNext {
    /// Get a raw pointer to insert into the Vulkan pNext chain
    pub fn as_ptr(&self) -> *const c_void {
        match self {
            PNext::WriteDescriptorSetAccelerationStructure(value) => { as_void_ptr(value) }
        }
    }
}