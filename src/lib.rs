use std::ffi::CString;
use ash::{vk, Entry};

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

unsafe extern "system" fn vk_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    user_data: *mut std::os::raw::c_void) -> vk::Bool32 {


    true as vk::Bool32
}

fn create_vk_instance(entry: &Entry, settings: &AppSettings) -> Option<ash::Instance> {
    let app_name = CString::new(settings.name.clone()).unwrap();
    let engine_name = CString::new("Phobos").unwrap();
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 2, 0),
        p_application_name: app_name.as_ptr(),
        p_engine_name: engine_name.as_ptr(),
        ..Default::default()
    };

    let instance_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
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