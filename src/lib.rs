extern crate core;

use std::borrow::Cow;
use std::ffi::{c_char, CStr, CString};
use ash::{vk, Entry};
use std::str::FromStr;

/// Application settings used to initialize the phobos context.
pub struct AppSettings {
    /// Application name. Possibly displayed in debugging tools, task manager, etc.
    pub name: String,
    /// Application version.
    pub version: (u32, u32, u32),
    /// Enable Vulkan validation layers for additional debug output. For developing this should almost always be on.
    pub enable_validation: bool
}

/// Main phobos context. This stores all global Vulkan state. Interaction with the device all happens through this
/// struct.
pub struct Context {
    /// Entry point for Vulkan functions.
    vk_entry: Entry,
    /// Stores the handle to the created VkInstance.
    instance: ash::Instance,
}

/// Safely wraps a c string into a string, or an empty string if the provided c string was null.
/// Assumes the provided c string is null terminated.
fn wrap_c_str(s: *const c_char) -> String {
    return if s.is_null() {
        String::default()
    } else {
        unsafe { CStr::from_ptr(s).to_string_lossy().to_owned().to_string() }
    }
}

fn unwrap_to_c_str_array(strings: &[&str]) -> *const *const c_char {

}

extern "system" fn vk_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void) -> vk::Bool32 {

    let callback_data = unsafe { *p_callback_data };
    let message_id_number = callback_data.message_id_number as i32;
    let message_id_name = wrap_c_str(callback_data.p_message_id_name);
    let message = wrap_c_str(callback_data.p_message);

    println!("[{:?}]:[{:?}]: {} ({}): {}\n",
             severity,
             msg_type,
             message_id_name,
             &message_id_number.to_string(),
             message
    );

    false as vk::Bool32
}

fn create_vk_instance(entry: &Entry, settings: &AppSettings) -> Option<ash::Instance> {
    let app_name = CString::new(settings.name.clone()).unwrap();
    let engine_name = CString::new("Phobos").unwrap();
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 2, 0),
        p_application_name: app_name.as_ptr(),
        p_engine_name: engine_name.as_ptr(),
        engine_version: vk::make_api_version(0,
         u32::from_str(env!("CARGO_PKG_VERSION_MAJOR")).unwrap(),
         u32::from_str(env!("CARGO_PKG_VERSION_MINOR")).unwrap(),
         u32::from_str(env!("CARGO_PKG_VERSION_PATCH")).unwrap()),
        ..Default::default()
    };

    let mut layers = Vec::<&str>::new();
    if settings.enable_validation {
        layers.push("VK_LAYER_KHRONOS_validation");
    }

    let instance_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        enabled_layer_count: layers.len() as u32,
        pp_enabled_layer_names: unwrap_to_c_str_array(layers.as_slice()),
        ..Default::default()
    };

    return unsafe { entry.create_instance(&instance_info, None).ok() };
}

impl Context {
    pub fn new(settings: AppSettings) -> Option<Context> {
        let entry = unsafe { Entry::load().unwrap() };
        let instance = create_vk_instance(&entry, &settings).unwrap();

        Some(Context {
            vk_entry: entry,
            instance
        })
    }
}