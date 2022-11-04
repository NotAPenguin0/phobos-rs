extern crate core;

#[macro_use]
extern crate derivative;

mod util;
mod init;
mod window;
mod image;

use std::sync::Arc;
use ash::{vk, Entry, Instance, Device};
use ash::extensions::ext::DebugUtils;
use window::WindowInterface;

pub use image::*;

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
    /// Vulkan extensions that should be present and enabled.
    pub device_extensions: Vec<String>,
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
    /// Optionally a preferred surface format. This is ignored for a headless context. If set to None, a fallback surface format will be chosen.
    /// This format is `{BGRA8_SRGB, NONLINEAR_SRGB}` if it is available. Otherwise, the format is implementation-defined.
    pub surface_format: Option<vk::SurfaceFormatKHR>,
    /// Optionally a preferred present mode. This is ignored for a headless context. If set to None, this will fall back to
    /// `vk::PresentModeKHR::FIFO`, as this is guaranteed to always be supported.
    pub present_mode: Option<vk::PresentModeKHR>,
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
#[derive(Default, Debug, Copy, Clone)]
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

/// Stores queried properties of a Vulkan extension.
#[derive(Debug, Default)]
pub struct ExtensionProperties {
    /// Name of the extension.
    pub name: String,
    /// Specification version of the extension.
    pub spec_version: u32,
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
    /// Available Vulkan extensions.
    pub extension_properties: Vec<ExtensionProperties>,
    /// List of [`VkQueueFamilyProperties`](vk::QueueFamilyProperties) with properties of each queue family on the device.
    pub queue_families: Vec<vk::QueueFamilyProperties>,
    /// List of [`QueueInfo`]  with requested queues abstracted away from the physical queues.
    pub queues: Vec<QueueInfo>
}


/// A swapchain is an abstraction of a presentation system. It handles buffering, VSync, and acquiring images
/// to render and present frames to.
#[derive(Default, Debug)]
pub struct Swapchain {
    /// Handle to the [`VkSwapchainKHR`](vk::SwapchainKHR) object.
    pub handle: vk::SwapchainKHR,
    /// Swapchain image format.
    pub format: vk::SurfaceFormatKHR,
    /// Present mode. The only mode that is required by the spec to always be supported is `FIFO`.
    pub present_mode: vk::PresentModeKHR,
    /// Size of the swapchain images. This is effectively the window render area.
    pub extent: vk::Extent2D,
    /// Swapchain images to present to.
    pub images: Vec<ImageView>,
}

/// Exposes a logical command queue on the device.
#[derive(Debug, Default)]
pub struct Queue {
    /// Raw [`VkQueue`](vk::Queue) handle.
    handle: vk::Queue,
    /// Information about this queue, such as supported operations, family index, etc. See also [`QueueInfo`]
    info: QueueInfo,
}

/// Stores function pointers for extension functions.
#[derive(Default)]
pub struct FuncPointers {
    /// Function pointers for the VK_DEBUG_UTILS_EXT extension
    pub debug_utils: Option<DebugUtils>,
    /// Function pointers for the VK_SURFACE_KHR extension
    pub surface: Option<ash::extensions::khr::Surface>,
    /// Function pointers for the VK_SWAPCHAIN_KHR extension
    pub swapchain: Option<ash::extensions::khr::Swapchain>,
}

/// Main phobos context. This stores all global Vulkan state. Interaction with the device all happens through this
/// struct.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Context {
    /// Entry point for Vulkan functions.
    #[derivative(Debug="ignore")]
    vk_entry: Entry,
    /// Vulkan instance
    #[derivative(Debug="ignore")]
    instance: Instance,
    /// Extension function pointers.
    #[derivative(Debug="ignore")]
    funcs: FuncPointers,
    /// handle to the [`VkDebugUtilsMessengerEXT`](vk::DebugUtilsMessengerEXT) object. None if `AppSettings::enable_validation` was `false` on initialization.
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    /// Surface information and handle. None if `AppSettings::create_headless` was `true` on initialization.
    surface: Option<Surface>,
    /// Physical device handle and properties.
    physical_device: PhysicalDevice,
    /// Logical device. This will be what is used for most Vulkan calls.
    #[derivative(Debug="ignore")]
    device: Arc<Device>,
    /// Logical device command queues. Used for command buffer submission.
    queues: Vec<Queue>,
    /// Swapchain for presenting. None in headless mode.
    swapchain: Option<Swapchain>,
}

impl Context {
    pub fn new<Window>(settings: AppSettings<Window>) -> Option<Context> where Window: WindowInterface {
        let entry = unsafe { Entry::load().unwrap() };
        let instance = init::create_vk_instance(&entry, &settings).unwrap();
        let mut funcs = FuncPointers {
            debug_utils: Some(DebugUtils::new(&entry, &instance)),
            surface: settings.window.map(|_| ash::extensions::khr::Surface::new(&entry, &instance)),
            ..Default::default()
        };

        let debug_messenger = settings.enable_validation.then(|| init::create_debug_messenger(&funcs));
        let mut surface = settings.window.map(|_| init::create_surface(&settings, &entry, &instance));
        let physical_device = init::select_physical_device(&settings, &surface, &funcs, &instance);
        if let Some(surface) = surface.as_mut() {
            init::fill_surface_details(surface, &physical_device, &funcs);
        }

        let device = Arc::from(init::create_device(&settings, &physical_device, &instance));
        // After creating the device we can load the VK_SWAPCHAIN_KHR extension
        funcs.swapchain = settings.window.map(|_| ash::extensions::khr::Swapchain::new(&instance, device.as_ref()));
        let queues = init::get_queues(&physical_device, device.clone());

        let swapchain = {
            let surface = surface.as_ref().unwrap();
            let funcs = &funcs;
            let device = device.clone();
            settings.window.map(move |_| init::create_swapchain(device, &settings, surface, funcs))
        };

        Some(Context {
            vk_entry: entry,
            instance,
            funcs,
            debug_messenger,
            surface,
            physical_device,
            device,
            queues,
            swapchain
        })

    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Some(swapchain) = &self.swapchain {
            unsafe { self.funcs.swapchain.as_ref().unwrap().destroy_swapchain(swapchain.handle, None); }
        }
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