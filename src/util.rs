use std::ffi::{c_char, CStr, CString};

/// Safely wraps a c string into a string, or an empty string if the provided c string was null.
/// Assumes the provided c string is null terminated.
pub fn wrap_c_str(s: *const c_char) -> String {
    return if s.is_null() {
        String::default()
    } else {
        unsafe { CStr::from_ptr(s).to_string_lossy().to_owned().to_string() }
    }
}

/// Safely unwraps a slice of strings into a vec of raw c strings.
pub fn unwrap_to_raw_strings(strings: &[CString]) -> Vec<*const c_char> {
    strings.iter().map(|string| string.as_ptr()).collect()
}