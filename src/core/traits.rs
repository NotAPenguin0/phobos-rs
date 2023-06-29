//! Defines traits for core

use ash::vk;

/// Direct access to the object's handle's as_raw representation
pub unsafe trait AsRaw {
    /// Get the as_raw u64 value of the handle underlying the object
    unsafe fn as_raw(&self) -> u64;
}

/// Represents an abstraction for naming objects in Vulkan for debuggers
pub trait Nameable: AsRaw {
    /// Change the name of the given object
    const OBJECT_TYPE: vk::ObjectType;
}
