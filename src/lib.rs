extern crate core;

mod util;
mod init;

use ash::{vk, Entry, Instance, Device};
use ash::extensions::ext::DebugUtils;
use ash_window;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle};
use init::*;

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
pub struct FuncPointers {
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
    /// Logical device. This will be what is used for most Vulkan calls.
    device: Device,
}

impl Context {
    pub fn new<Window>(settings: AppSettings<Window>) -> Option<Context> where Window: WindowInterface {
        let entry = unsafe { Entry::load().unwrap() };
        let instance = init::create_vk_instance(&entry, &settings).unwrap();
        let funcs = FuncPointers {
            debug_utils: Some(DebugUtils::new(&entry, &instance)),
            surface: settings.window.map(|_| ash::extensions::khr::Surface::new(&entry, &instance))
        };

        let debug_messenger = settings.enable_validation.then(|| init::create_debug_messenger(&funcs));
        let mut surface = settings.window.map(|_| init::create_surface(&settings, &entry, &instance));
        let physical_device = init::select_physical_device(&settings, &surface, &funcs, &instance);
        if let Some(surface) = surface.as_mut() {
            init::fill_surface_details(surface, &physical_device, &funcs);
        }

        let device = init::create_device(&settings, &physical_device, &instance);

        Some(Context {
            vk_entry: entry,
            instance,
            funcs,
            debug_messenger,
            surface,
            physical_device,
            device
        })

    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { self.device.destroy_device(None); }

        if let Some(surface) = &self.surface {
            unsafe { self.funcs.surface.as_ref().unwrap().destroy_surface(surface.handle, None); }
        }

        if let Some(debug_messenger) = self.debug_messenger {
            unsafe { self.funcs.debug_utils.as_ref().unwrap().destroy_debug_utils_messenger(debug_messenger, None); }
        }

        unsafe { self.instance.destroy_instance(None); }
    }
}