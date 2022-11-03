extern crate core;

use std::ffi::{CString};
use std::str::FromStr;
use ash::{vk, Entry, Instance};
use ash::extensions::ext::DebugUtils;
use ash_window;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle};

mod util;

/// Used as a dummy window interface in case of a headless context. Calling any of the `raw_xxx_handle()` functions on this will result in a panic.
pub struct HeadlessWindowInterface;

unsafe impl HasRawWindowHandle for HeadlessWindowInterface {
    fn raw_window_handle(&self) -> RawWindowHandle {
        panic!("Called raw_window_handle() on headless window context.");
    }
}

unsafe impl HasRawDisplayHandle for HeadlessWindowInterface {
    fn raw_display_handle(&self) -> RawDisplayHandle {
        panic!("Called raw_display_handle() on headless window context.");
    }
}

/// Parent trait combining all requirements for a window interface. To be a window interface, a type T must implement the following traits:
/// - [`HasRawWindowHandle`](raw_window_handle::HasRawWindowHandle)
/// - [`HasRawDisplayHandle`](raw_window_handle::HasRawDisplayHandle)
pub trait WindowInterface: HasRawWindowHandle + HasRawDisplayHandle {}
impl<T: HasRawWindowHandle + HasRawDisplayHandle> WindowInterface for T {}

/// Application settings used to initialize the phobos context.
pub struct AppSettings<'a, Window> where Window: WindowInterface {
    /// Application name. Possibly displayed in debugging tools, task manager, etc.
    pub name: String,
    /// Application version.
    pub version: (u32, u32, u32),
    /// Enable Vulkan validation layers for additional debug output. For developing this should almost always be on.
    pub enable_validation: bool,
    /// Optionally a reference to an object implementing a windowing system. If this is not None, it will be used to create a [`VkSurfaceKHR`](vk::SurfaceKHR) to present to.
    pub window: Option<&'a Window>,
}

/// Contains all information about a [`VkSurfaceKHR`](vk::SurfaceKHR)
pub struct Surface {
    /// Handle to the [`VkSurfaceKHR`](vk::SurfaceKHR)
    pub handle: vk::SurfaceKHR,
    /// [`VkSurfaceCapabilitiesKHR`](vk::SurfaceCapabilitiesKHR) structure storing information about surface capabilities.
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    /// List of [`VkSurfaceFormatKHR`](vk::SurfaceFormatKHR) with all formats this surface supports.
    pub formats: Vec<vk::SurfaceFormatKHR>,
    /// List of [`VkPresentModeKHR`](vk::PresentModeKHR) with all present modes this surface supports.
    pub present_modes: Vec<vk::PresentModeKHR>
}

/// Main phobos context. This stores all global Vulkan state. Interaction with the device all happens through this
/// struct.
pub struct Context {
    /// Entry point for Vulkan functions.
    vk_entry: Entry,
    /// Stores the handle to the created [`VkInstance`](ash::Instance).
    instance: Instance,
    /// handle to the [`VkDebugUtilsMessengerEXT`](vk::DebugUtilsMessengerEXT) object. None if `AppSettings::enable_validation` was `false` on initialization.
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    /// Surface information and handle. None if `AppSettings::create_headless` was `true` on initialization.
    surface: Option<Surface>,
}

extern "system" fn vk_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void) -> vk::Bool32 {

    let callback_data = unsafe { *p_callback_data };
    let message_id_number = callback_data.message_id_number as i32;
    let message_id_name = util::wrap_c_str(callback_data.p_message_id_name);
    let message = util::wrap_c_str(callback_data.p_message);

    println!("[{:?}]:[{:?}]: {} ({}): {}\n",
             severity,
             msg_type,
             message_id_name,
             &message_id_number.to_string(),
             message
    );

    false as vk::Bool32
}

fn create_vk_instance<Window>(entry: &Entry, settings: &AppSettings<Window>) -> Option<ash::Instance> where Window: WindowInterface {
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

    let mut layers = Vec::<CString>::new();
    let mut extensions = Vec::<CString>::new();

    if settings.enable_validation {
        layers.push(CString::new("VK_LAYER_KHRONOS_validation").unwrap());
        extensions.push(CString::from(DebugUtils::name()));
    }

    let layers_raw = util::unwrap_to_raw_strings(layers.as_slice());
    let extensions_raw = util::unwrap_to_raw_strings(extensions.as_slice());

    let instance_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(layers_raw.as_slice())
        .enabled_extension_names(extensions_raw.as_slice())
        .build();

    return unsafe { entry.create_instance(&instance_info, None).ok() };
}

fn create_debug_messenger(entry: &Entry, instance: &Instance) -> vk::DebugUtilsMessengerEXT {
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
        .pfn_user_callback(Some(vk_debug_callback))
        .build();

    let loader = DebugUtils::new(&entry, &instance);
    unsafe { loader.create_debug_utils_messenger(&create_info, None).unwrap() }
}

fn create_surface<Window>(settings: &AppSettings<Window>, entry: &Entry, instance: &Instance) -> Surface where Window: WindowInterface {
    let window = settings.window.unwrap();
    let surface_handle = unsafe { ash_window::create_surface(&entry, &instance, window.raw_display_handle(), window.raw_window_handle(), None).unwrap() };
    todo!()
}

impl Context {
    pub fn new<Window>(settings: AppSettings<Window>) -> Option<Context> where Window: WindowInterface {
        let entry = unsafe { Entry::load().unwrap() };
        let instance = create_vk_instance(&entry, &settings).unwrap();
        let debug_messenger = if settings.enable_validation {
            Some(create_debug_messenger(&entry, &instance))
        } else {
            None
        };
        let surface = if settings.window.is_some() {
            Some(create_surface(&settings, &entry, &instance))
        } else {
            None
        };

        Some(Context {
            vk_entry: entry,
            instance,
            debug_messenger,
            surface
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            if let Some(debug_messenger) = self.debug_messenger {
                let debug_utils_loader = DebugUtils::new(&self.vk_entry, &self.instance);
                debug_utils_loader.destroy_debug_utils_messenger(debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}