//! Defines traits for core

use anyhow::Result;
use crate::Device;

/// Direct access to the object's handle's as_raw representation
pub unsafe trait AsRaw {
	/// Get the as_raw u64 value of the handle underlying the object
	unsafe fn as_raw(&self) -> u64;
}

/// Represents an abstraction for naming objects in Vulkan for debuggers
pub trait Nameable: AsRaw {
	/// Change the name of the given object
	const OBJECT_TYPE: crate::vk::ObjectType;
}

impl Device {
	/// Set the name of any given compatible object for debugging purposes
	pub fn set_name<T: Nameable>(&self, object: &T, name: &str) -> Result<()> {
		let object_name = std::ffi::CString::new(name)?;
		let name_info = crate::vk::DebugUtilsObjectNameInfoEXT::builder()
			.object_type(<T as Nameable>::OBJECT_TYPE)
			.object_handle( unsafe {object.as_raw()} )
			.object_name(&object_name)
			.build();

		unsafe {
			Ok(self.debug_utils()?.set_debug_utils_object_name(
				self.handle().handle(),
				&name_info
			)?)
		}
	}
}