use std::ffi::{c_char, CStr, CString};
use std::mem::size_of;
use ash::vk;

/// Wraps a c string into a string, or an empty string if the provided c string was null.
/// Assumes the provided c string is null terminated.
pub(crate) unsafe fn wrap_c_str(s: *const c_char) -> String {
    return if s.is_null() {
        String::default()
    } else {
        CStr::from_ptr(s).to_string_lossy().to_owned().to_string()
    }
}

/// Safely unwraps a slice of strings into a vec of raw c strings.
pub(crate) fn unwrap_to_raw_strings(strings: &[CString]) -> Vec<*const c_char> {
    strings.iter().map(|string| string.as_ptr()).collect()
}

pub trait ByteSize {
    fn byte_size(&self) -> usize;
}

impl ByteSize for vk::Format {
    fn byte_size(&self) -> usize {
        match *self {
            vk::Format::R32G32B32_SFLOAT => 3 * size_of::<f32>(),
            vk::Format::R32G32B32A32_SFLOAT => 4 * size_of::<f32>(),
            _ => { todo!() }
        }
    }
}