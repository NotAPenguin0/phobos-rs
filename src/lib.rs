//! Fast, powerful Vulkan abstraction library
//!
//! Phobos provides powerful Vulkan abstractions to automatically manage common issues like
//! synchronization and resource management. It hides away a lot of boilerplate that comes with
//! Vulkan, while still being highly flexible.
//! <br>
//! <br>
//! # Example
//!
//! For illustrative purposes, we will use winit here. Any windowing library can be supported by implementing a few trait objects
//! necessary to satisfy the [`window::WindowInterface`] trait.
//! ```
//! let event_loop = EventLoopBuilder::new().with_any_thread(true).build();
//! let window = WindowBuilder::new()
//!     .with_title("Phobos test app")
//!     .build(&event_loop)
//!     .unwrap();
//! ```
//! First, we will define an [`AppSettings`] structure that outlines requirements
//! and information about our application. Phobos will use this to
//! pick a suitable GPU to run your program on and initialize Vulkan for it.
//! ```
//! use phobos as ph;
//!
//! let settings = ph::AppSettings {
//!     version: (1, 0, 0),
//!     name: String::from("Phobos sample app"),
//!     enable_validation: true,
//!     window: Some(&window),
//!     surface_format: None, // Use default fallback format.
//!     present_mode: Some(PresentModeKHR::MAILBOX), // No vsync, double buffered
//!     gpu_requirements: ph::GPURequirements {
//!         dedicated: true,
//!         min_video_memory: 1 * 1024 * 1024 * 1024, // 1 GiB.
//!         min_dedicated_video_memory: 1 * 1024 * 1024 * 1024,
//!         queues: vec![
//!             // We request one of each queue, and prefer dedicated Transfer and Compute queues.
//!             ph::QueueRequest { dedicated: false, queue_type: ph::QueueType::Graphics },
//!             ph::QueueRequest { dedicated: true, queue_type: ph::QueueType::Transfer },
//!             ph::QueueRequest { dedicated: true, queue_type: ph::QueueType::Compute }
//!         ],
//!         // For this example we won't request any extensions or extra Vulkan features.
//!         ..Default::default()
//!     }
//! };
//! ```
//! Now we are ready to initialize the Phobos library.
//! ```
//! // Create Vulkan instance. This step is required.
//! let instance = ph::VkInstance::new(&settings).unwrap();
//! // Create a debug messenger object. This is not required, and only useful
//! // validation layers are enabled.
//! let debug_messenger = ph::DebugMessenger::new(&instance).unwrap();
//! let (surface, physical_device) = {
//!     // Create surface to render to. Not required for a compute-only context.
//!     let mut surface = Surface::new(&instance, &settings).unwrap();
//!     // Select a physical device based on gpu requirements we passed earlier.
//!     let physical_device = ph::PhysicalDevice::select(&settings, Some(&surface), &settings).unwrap();
//!     surface.query_details(&physical_device).unwrap();
//!     (surface, physical_device)
//! };
//! // Create Vulkan device. This is our main interface with the Vulkan API.
//! let device: Arc<ph::Device> = ph::Device::new(&instance, &physical_device, &settings).unwrap();
//! // Create execution manager, needed to execute commands.
//! let exec = ph::ExecutionManager::new(device.clone(), &physical_device).unwrap();
//! // Create swapchain and frame manager.
//! let mut frame = {
//!     let swapchain = ph::Swapchain::new(&instance, device.clone(), &settings, &surface).unwrap();
//!     ph::FrameManager::new(device.clone(), swapchain).unwrap()
//! };
//! ```

#![feature(specialization)]

extern crate core;

#[macro_use]
extern crate derivative;

mod util;
mod command_pool;
mod deferred_delete;
pub mod task_graph;
pub mod buffer;
pub mod window;
pub mod image;
pub mod frame;
pub mod sync;
pub mod queue;
pub mod error;
pub mod instance;
pub mod debug;
pub mod surface;
pub mod physical_device;
pub mod device;
pub mod execution_manager;
pub mod swapchain;
pub mod command_buffer;
pub mod pass;
pub mod pipeline;

use std::sync::Arc;
use ash::vk;
use window::WindowInterface;
use gpu_allocator::vulkan as vk_alloc;

pub use crate::image::*;
pub use crate::frame::*;
pub use crate::sync::*;
pub use crate::error::*;
pub use crate::instance::*;
pub use crate::debug::*;
pub use crate::surface::*;
pub use crate::physical_device::*;
pub use crate::queue::*;
pub use crate::device::*;
pub use crate::execution_manager::*;
pub use crate::swapchain::*;
pub use crate::window::*;
pub use crate::command_buffer::*;
pub use crate::buffer::*;
pub use crate::task_graph::*;

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
#[derive(Debug)]
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

impl<'a, Window> Default for AppSettings<'a, Window> where Window: WindowInterface {
    fn default() -> Self {
        AppSettings {
            name: String::from(""),
            version: (0, 0, 0),
            enable_validation: false,
            window: None,
            surface_format: None,
            present_mode: None,
            gpu_requirements: GPURequirements::default(),
        }
    }
}

pub struct AppBuilder<'a, Window> where Window: WindowInterface {
    inner: AppSettings<'a, Window>,
}

impl<'a, Window> AppBuilder<'a, Window> where Window: WindowInterface {   
    pub fn new() -> Self {
        AppBuilder { inner: AppSettings::default() }
    }

    pub fn name(mut self, name: String) -> Self {
        self.inner.name = name;
        self
    }

    pub fn version(mut self, ver: (u32, u32, u32)) -> Self {
        self.inner.version = ver;
        self
    }

    pub fn validation(mut self, val: bool) -> Self {
        self.inner.enable_validation = val;
        self
    }

    pub fn window(mut self, window: &'a Window) -> Self {
        self.inner.window = Some(window);
        self
    }

    pub fn surface_format(mut self, format: vk::SurfaceFormatKHR) -> Self {
        self.inner.surface_format = Some(format);
        self
    }

    pub fn present_mode(mut self, mode: vk::PresentModeKHR) -> Self {
        self.inner.present_mode = Some(mode);
        self
    }

    pub fn gpu(mut self, gpu: GPURequirements) -> Self {
        self.inner.gpu_requirements = gpu;
        self
    }

    pub fn build(self) -> AppSettings<'a, Window> {
        self.inner
    }
}

pub fn create_allocator(instance: &VkInstance, device: Arc<Device>, physical_device: &PhysicalDevice) -> Result<vk_alloc::Allocator, Error> {
    Ok(vk_alloc::Allocator::new(
        &vk_alloc::AllocatorCreateDesc {
            instance: instance.instance.clone(),
            device: device.handle.clone(),
            physical_device: physical_device.handle.clone(),
            debug_settings: Default::default(),
            buffer_device_address: false // We might change this if the bufferDeviceAddress feature gets enabled.
        }
    )?)
}