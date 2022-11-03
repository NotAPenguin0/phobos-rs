extern crate core;

use std::ffi::{CStr, CString};
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

/// Abstraction over vulkan queue capabilities. Note that in raw Vulkan, there is no 'Graphics queue'. Phobos will expose one, but behind the scenes the exposed
/// e.g. graphics queue and transfer could point to the same hardware queue.
#[derive(Copy, Clone, Default, Debug)]
pub enum QueueType {
    #[default]
    Graphics = vk::QueueFlags::GRAPHICS.as_raw() as isize,
    Compute = vk::QueueFlags::COMPUTE.as_raw() as isize,
    Transfer = vk::QueueFlags::TRANSFER.as_raw() as isize
}

/// Structure holding a queue with specific capabilities to request from the physical device.
#[derive(Debug)]
pub struct QueueRequest {
    /// Whether this queue should be dedicated if possible. For example, requesting a dedicated queue of type `QueueType::Transfer` will try to
    /// match this to a queue that does not have graphics or compute capabilities. On the other hand, requesting a dedicated graphics queue will not
    /// try to exclude transfer capabilities, as this is not possible per spec guarantees (a graphics queue must have transfer support)
    pub dedicated: bool,
    /// Capabilities that are requested from the queue.
    pub queue_type: QueueType
}

/// Minimum requirements for the GPU. This will be used to determine what physical device is selected.
#[derive(Default, Debug)]
pub struct GPURequirements {
    /// Whether a dedicated GPU is required. Setting this to true will discard integrated GPUs.
    pub dedicated: bool,
    /// Minimum amount of video memory required, in bytes. Note that this might count shared memory if RAM is shared.
    pub min_video_memory: usize,
    /// Minimum amount of dedicated video memory, in bytes. This only counts memory that is on the device.
    pub min_dedicated_video_memory: usize,
    /// Command queue types requested from the physical device.
    pub queues: Vec<QueueRequest>,
    /// Vulkan 1.0 features that are required from the physical device.
    pub features: vk::PhysicalDeviceFeatures,
    /// Vulkan 1.1 features that are required from the physical device.
    pub features_1_1: vk::PhysicalDeviceVulkan11Features,
    /// Vulkan 1.2 features that are required from the physical device.
    pub features_1_2: vk::PhysicalDeviceVulkan12Features,
}

/// Application settings used to initialize the phobos context.
#[derive(Default, Debug)]
pub struct AppSettings<'a, Window> where Window: WindowInterface {
    /// Application name. Possibly displayed in debugging tools, task manager, etc.
    pub name: String,
    /// Application version.
    pub version: (u32, u32, u32),
    /// Enable Vulkan validation layers for additional debug output. For developing this should almost always be on.
    pub enable_validation: bool,
    /// Optionally a reference to an object implementing a windowing system. If this is not None, it will be used to create a [`VkSurfaceKHR`](vk::SurfaceKHR) to present to.
    pub window: Option<&'a Window>,
    /// Minimum requirements the selected physical device should have.
    pub gpu_requirements: GPURequirements,
}

/// Contains all information about a [`VkSurfaceKHR`](vk::SurfaceKHR)
#[derive(Default, Debug)]
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

/// Stores all information of a queue that was found on the physical device.
#[derive(Default, Debug)]
pub struct QueueInfo {
    /// Functionality that this queue provides.
    pub queue_type: QueueType,
    /// Whether this is a dedicated queue or not.
    pub dedicated: bool,
    /// Whether this queue is capable of presenting to a surface.
    pub can_present: bool,
    /// The queue family index.
    family_index: u32,
}

/// A physical device abstracts away an actual device, like a graphics card or integrated graphics card.
#[derive(Default, Debug)]
pub struct PhysicalDevice {
    /// Handle to the [`VkPhysicalDevice`](vk::PhysicalDevice).
    pub handle: vk::PhysicalDevice,
    /// [`VkPhysicalDeviceProperties`](vk::PhysicalDeviceProperties) structure with properties of this physical device.
    pub properties: vk::PhysicalDeviceProperties,
    /// [`VkPhysicalDeviceMemoryProperties`](vk::PhysicalDeviceMemoryProperties) structure with memory properties of the physical device, such as
    /// available memory types and heaps.
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// List of [`VkQueueFamilyProperties`](vk::QueueFamilyProperties) with properties of each queue family on the device.
    pub queue_families: Vec<vk::QueueFamilyProperties>,
    /// List of [`QueueInfo`]  with requested queues abstracted away from the physical queues.
    pub queues: Vec<QueueInfo>
}

/// Stores function pointers for extension functions.
struct FuncPointers {
    /// Function pointers for the VK_DEBUG_UTILS_EXT extension
    pub debug_utils: Option<DebugUtils>,
    /// Function pointers for the VK_SURFACE_KHR extension
    pub surface: Option<ash::extensions::khr::Surface>
}

/// Main phobos context. This stores all global Vulkan state. Interaction with the device all happens through this
/// struct.
pub struct Context {
    /// Entry point for Vulkan functions.
    vk_entry: Entry,
    /// Stores the handle to the created [`VkInstance`](ash::Instance).
    instance: Instance,
    /// Extension function pointers.
    funcs: FuncPointers,
    /// handle to the [`VkDebugUtilsMessengerEXT`](vk::DebugUtilsMessengerEXT) object. None if `AppSettings::enable_validation` was `false` on initialization.
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    /// Surface information and handle. None if `AppSettings::create_headless` was `true` on initialization.
    surface: Option<Surface>,
    /// Physical device handle and properties.
    physical_device: PhysicalDevice,
}

extern "system" fn vk_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void) -> vk::Bool32 {

    let callback_data = unsafe { *p_callback_data };
    let message_id_number = callback_data.message_id_number as i32;
    let message_id_name = unsafe { util::wrap_c_str(callback_data.p_message_id_name) };
    let message = unsafe { util::wrap_c_str(callback_data.p_message) };

    // TODO: switch out logging with log crate: https://docs.rs/log

    println!("[{:?}]:[{:?}]: {} ({}): {}\n",
             severity,
             msg_type,
             message_id_name,
             &message_id_number.to_string(),
             message
    );

    false as vk::Bool32
}

fn create_vk_instance<Window>(entry: &Entry, settings: &AppSettings<Window>) -> Option<Instance> where Window: WindowInterface {
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

    if let Some(window) = settings.window {
        extensions.extend(
            ash_window::enumerate_required_extensions(window.raw_display_handle())
                .unwrap()
                .to_vec()
                .iter()
                .map(|&raw_str| unsafe { CString::from(CStr::from_ptr(raw_str)) })
        );
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

fn create_debug_messenger(funcs: &FuncPointers) -> vk::DebugUtilsMessengerEXT {
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
        .pfn_user_callback(Some(vk_debug_callback))
        .build();

    unsafe { funcs.debug_utils.as_ref().unwrap().create_debug_utils_messenger(&create_info, None).unwrap() }
}

fn create_surface<Window>(settings: &AppSettings<Window>, entry: &Entry, instance: &Instance) -> Surface where Window: WindowInterface {
    let window = settings.window.unwrap();
    let surface_handle = unsafe { ash_window::create_surface(&entry, &instance, window.raw_display_handle(), window.raw_window_handle(), None).unwrap() };
    return Surface {
        handle: surface_handle,
        // Surface capabilities will be queried by the VkPhysicalDevice.
        ..Default::default()
    };
}

fn total_video_memory(device: &PhysicalDevice) -> usize {
    device.memory_properties.memory_heaps.iter()
        .map(|heap| heap.size as usize)
        .sum()
}

fn total_device_memory(device: &PhysicalDevice) -> usize {
    device.memory_properties.memory_heaps.iter()
        .filter(|heap|
            heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        )
        .map(|heap| heap.size as usize)
        .sum()
}

fn get_queue_family_prefer_dedicated(families: &[vk::QueueFamilyProperties], queue_type: QueueType, avoid: vk::QueueFlags) -> Option<(usize, bool)> {
    let required = vk::QueueFlags::from_raw(queue_type as vk::Flags);
    families.iter().enumerate().fold(None, |current_best_match, (index, family) | -> Option<usize> {
        // Does not contain required flags, must skip
        if !family.queue_flags.contains(required) { return current_best_match; }
        // Contains required flags and does not contain *any* flags to avoid, this is an optimal match.
        // Note that to check of it doesn't contain any of the avoid flags, contains() will not work, we need to use intersects()
        if !family.queue_flags.intersects(avoid) { return Some(index) }

        // Only if we don't have a match yet, settle for a suboptimal match
        return if current_best_match.is_none() {
            Some(index)
        } else {
            current_best_match
        }

    })
    .map(|index| {
        return (index, !families[index].queue_flags.intersects(avoid));
    })
}

fn select_physical_device<Window>(settings: &AppSettings<Window>, surface: &Option<Surface>, funcs: &FuncPointers, instance: &Instance) -> PhysicalDevice where Window: WindowInterface {
    let devices = unsafe { instance.enumerate_physical_devices().expect("No physical devices found.") };

    devices.iter()
        .find_map(|device| -> Option<PhysicalDevice> {
            let mut physical_device = PhysicalDevice {
                handle: *device,
                properties: unsafe { instance.get_physical_device_properties(*device) },
                memory_properties: unsafe { instance.get_physical_device_memory_properties(*device) },
                queue_families: unsafe { instance.get_physical_device_queue_family_properties(*device) },
                ..Default::default()
            };

            if settings.gpu_requirements.dedicated && physical_device.properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU { return None; }
            if settings.gpu_requirements.min_video_memory > total_video_memory(&physical_device) { return None; }
            if settings.gpu_requirements.min_dedicated_video_memory > total_device_memory(&physical_device) { return None; }

            physical_device.queues = {
                settings.gpu_requirements.queues.iter()
                    .filter_map(|request| -> Option<QueueInfo> {
                        let avoid = if request.dedicated {
                            match request.queue_type {
                                QueueType::Graphics => vk::QueueFlags::COMPUTE,
                                QueueType::Compute => vk::QueueFlags::GRAPHICS,
                                QueueType::Transfer => vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS
                            }
                        } else { vk::QueueFlags::default() };

                        return if let Some((index, dedicated)) = get_queue_family_prefer_dedicated(physical_device.queue_families.as_slice(), request.queue_type, avoid) {
                            Some(QueueInfo {
                                queue_type: request.queue_type,
                                dedicated,
                                family_index: index as u32,
                                ..Default::default()
                            })
                        } else {
                            None
                        };
                    })
                    .collect()
            };

            // We now have a list of all the queues that were found matching our request. If this amount is smaller than the number of requested queues,
            // at least one is missing and could not be fulfilled. In this case we reject the device.
            if physical_device.queues.len() < settings.gpu_requirements.queues.len() { return None; }

            // Now check surface support (if we are not creating a headless context)
            if let Some(surface) = surface {
                // The surface is supported if one of the queues we found can present to it.
                let mut supported_queue = physical_device.queues.iter_mut().find(|queue| {
                    unsafe {
                        funcs.surface.as_ref().unwrap()
                            .get_physical_device_surface_support(
                                physical_device.handle,
                                queue.family_index,
                                surface.handle)
                            .unwrap()
                    }
                });
                if let Some(queue) = supported_queue {
                    // Flag that we can present to it
                    queue.can_present = true;
                } else { // No queue to present found, reject physical device
                    return None;
                }
            }

            return Some(physical_device);
        })
        .expect("No physical device matching requested capabilities found.")
}

fn fill_surface_details(surface: &mut Surface, physical_device: &PhysicalDevice, funcs: &FuncPointers) {
    let surface_funcs = funcs.surface.as_ref().unwrap();
    unsafe {
        surface.capabilities = surface_funcs.get_physical_device_surface_capabilities(physical_device.handle, surface.handle).unwrap();
        surface.formats = surface_funcs.get_physical_device_surface_formats(physical_device.handle, surface.handle).unwrap();
        surface.present_modes = surface_funcs.get_physical_device_surface_present_modes(physical_device.handle, surface.handle).unwrap();
    }
}

impl Context {
    pub fn new<Window>(settings: AppSettings<Window>) -> Option<Context> where Window: WindowInterface {
        let entry = unsafe { Entry::load().unwrap() };
        let instance = create_vk_instance(&entry, &settings).unwrap();
        let funcs = FuncPointers {
            debug_utils: Some(DebugUtils::new(&entry, &instance)),
            surface: settings.window.map(|_| ash::extensions::khr::Surface::new(&entry, &instance))
        };

        let debug_messenger = settings.enable_validation.then(|| create_debug_messenger(&funcs));
        let mut surface = settings.window.map(|_| create_surface(&settings, &entry, &instance));
        let physical_device = select_physical_device(&settings, &surface, &funcs, &instance);
        if let Some(surface) = surface.as_mut() {
            fill_surface_details(surface, &physical_device, &funcs);
        }

        Some(Context {
            vk_entry: entry,
            instance,
            funcs,
            debug_messenger,
            surface,
            physical_device,
        })

    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Some(surface) = &self.surface {
            unsafe { self.funcs.surface.as_ref().unwrap().destroy_surface(surface.handle, None); }
        }

        if let Some(debug_messenger) = self.debug_messenger {
            unsafe { self.funcs.debug_utils.as_ref().unwrap().destroy_debug_utils_messenger(debug_messenger, None); }
        }

        unsafe { self.instance.destroy_instance(None); }
    }
}