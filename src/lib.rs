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
//! let settings = ph::AppBuilder::new()
//!         .version((1, 0, 0))
//!         .name(String::from("Phobos demo app"))
//!         .validation(true)
//!         .window(&window)
//!         .present_mode(vk::PresentModeKHR::MAILBOX)
//!         .scratch_size(1 * 1024) // 1 KiB scratch memory per buffer type per frame
//!         .gpu(ph::GPURequirements {
//!             dedicated: true,
//!             min_video_memory: 1 * 1024 * 1024 * 1024, // 1 GiB.
//!             min_dedicated_video_memory: 1 * 1024 * 1024 * 1024,
//!             queues: vec![
//!                 ph::QueueRequest { dedicated: false, queue_type: ph::QueueType::Graphics },
//!                 ph::QueueRequest { dedicated: true, queue_type: ph::QueueType::Transfer },
//!                 ph::QueueRequest { dedicated: true, queue_type: ph::QueueType::Compute }
//!             ],
//!             ..Default::default()
//!         })
//!         .build();
//! ```
//! Now we are ready to initialize the Phobos library.
//! ```
//! // Create Vulkan instance. This step is required.
//! let instance = ph::VkInstance::new(&settings)?;
//! // Create a debug messenger object. This is not required, and only useful
//! // validation layers are enabled.
//! let debug_messenger = ph::DebugMessenger::new(&instance)?;
//! let (surface, physical_device) = {
//!     // Create surface to render to. Not required for a compute-only context.
//!     let mut surface = Surface::new(&instance, &settings)?;
//!     // Select a physical device based on gpu requirements we passed earlier.
//!     let physical_device = ph::PhysicalDevice::select(&settings, Some(&surface), &settings)?;
//!     surface.query_details(&physical_device)?;
//!     (surface, physical_device)
//! };
//! // Create Vulkan device. This is our main interface with the Vulkan API.
//! let device: Arc<ph::Device> = ph::Device::new(&instance, &physical_device, &settings)?;
//! // Create the GPU allocator
//! let mut alloc = ph::create_allocator(&instance, device.clone(), &physical_device)?;
//! // Create execution manager, needed to execute commands.
//! let exec = ph::ExecutionManager::new(device.clone(), &physical_device)?;
//! // Create swapchain and frame manager.
//! let mut frame = {
//!     let swapchain = ph::Swapchain::new(&instance, device.clone(), &settings, &surface)?;
//!     ph::FrameManager::new(device.clone(), alloc.clone(), &settings, swapchain)?
//! };
//! ```
//! For further example code, check out the following modules
//! - [`pipeline`] for pipeline creation and management.
//! - [`frame`] for managing your main loop and frame rendering logic.
//! - [`task_graph`] for creating a task graph to record commands.
//! - [`pass`] for defining passes inside a task graph.
//! - [`descriptor`] for descriptor set management.
//! - [`command_buffer`] for different Vulkan commands available.
//! - [`scratch_allocator`] is a powerful and simple allocator for quick scratch buffers within a frame or thread context.
//! - [`image`] for managing [`VkImage`](vk::Image) and [`VkImageView`](vk::ImageView) objects.
//! - [`buffer`] for managing [`VkBuffer`](vk::Buffer) objects.
//! - [`util`] for various utilities and common patterns like buffer uploads.

#![feature(never_type)]
#![feature(fn_traits)]
#![feature(stmt_expr_attributes)]

#[macro_use]
extern crate derivative;
#[macro_use] extern crate log;

pub mod prelude;
pub use crate::prelude::*;

pub mod buffer;
pub mod image;
pub mod command_buffer;
pub mod pipeline;
pub mod descriptor;
pub mod sampler;
pub mod util;
pub mod core;
pub mod sync;
pub mod domain;
pub mod graph;
pub mod allocator;
pub mod wsi;