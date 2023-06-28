//! Defines traits for core

use anyhow::Result;

pub unsafe trait AsRawHandle {
	unsafe fn handle() -> u64;
}

/// Represents an abstraction for naming objects in Vulkan for debuggers
pub trait Nameable {
	/// Change the name of the given object
	fn set_name(&self, device: &crate::core::device::Device, name: &str) -> Result<()>;
}