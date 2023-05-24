//! Fast, powerful Vulkan abstraction library
//!
//! Phobos provides powerful Vulkan abstractions to automatically manage common issues like
//! synchronization and resource management. It hides away a lot of boilerplate that comes with
//! Vulkan, while still being highly flexible.
//!
//! To get started, the easiest way is to simply
//! ```
//! // Import all important traits
//! use phobos::prelude::traits;
//! // Import types under a namespace.
//! use phobos::prelude as ph;
//!
//! // Or, if you dont care about using the types under a namespace
//! use phobos::prelude::*;
//! ```
//!
//! # Example
//!
//! For illustrative purposes, we will use winit here. Any windowing library can be supported by implementing a few trait objects
//! necessary to satisfy the [`WindowInterface`](crate::WindowInterface) trait.
//! ```
//! use winit::window::WindowBuilder;
//! use winit::event_loop::EventLoopBuilder;
//! let event_loop = EventLoopBuilder::new().build();
//! let window = WindowBuilder::new()
//!     .with_title("Phobos test app")
//!     .build(&event_loop)
//!     .unwrap();
//! ```
//! First, we will define an [`AppSettings`](crate::AppSettings) structure that outlines requirements
//! and information about our application. Phobos will use this to
//! pick a suitable GPU to run your program on and initialize Vulkan for it.
//! ```
//! use phobos::prelude::*;
//!
//! let settings = AppBuilder::new()
//!         .version((1, 0, 0))
//!         .name("Phobos demo app")
//!         .validation(true)
//!         .window(&window)
//!         .present_mode(vk::PresentModeKHR::MAILBOX)
//!         .scratch_size(1 * 1024u64) // 1 KiB scratch memory per buffer type per frame
//!         .gpu(GPURequirements {
//!             dedicated: true,
//!             min_video_memory: 1 * 1024 * 1024 * 1024, // 1 GiB.
//!             min_dedicated_video_memory: 1 * 1024 * 1024 * 1024,
//!             queues: vec![
//!                 QueueRequest { dedicated: false, queue_type: QueueType::Graphics },
//!                 QueueRequest { dedicated: true, queue_type: QueueType::Transfer },
//!                 QueueRequest { dedicated: true, queue_type: QueueType::Compute }
//!             ],
//!             ..Default::default()
//!         })
//!         .build();
//! ```
//! Now we are ready to initialize the Phobos library.
//! ```
//! use phobos::prelude::*;
//! let (
//!     instance,
//!     physical_device,
//!     surface,
//!     device,
//!     allocator,
//!     exec,
//!     frame,
//!     Some(debug_messenger)
//! ) = WindowedContext::init(&settings)? else {
//!     panic!("Asked for debug messenger but didn't get one.")
//! };
//!
//! ```
//! For more initialization options, see [`initialize()`], [`initialize_with_allocator()`] and
//! [`WindowedContext::init_with_allocator()`](crate::core::init::WindowedContext::init_with_allocator).
//!
//! For further example code, check out the following modules
//! - [`pipeline`] for pipeline creation and management.
//! - [`wsi`] for managing your main loop and frame rendering logic.
//! - [`graph`] for creating a pass graph to record commands.
//! - [`sync`] for various synchronization primitives, threading utilities, gpu futures and queue synchronization.
//! - [`descriptor`] for descriptor set management.
//! - [`command_buffer`] for different Vulkan commands available.
//! - [`allocator`] For various allocators and related utilities.
//! - [`image`] for managing [`VkImage`](vk::Image) and [`VkImageView`](vk::ImageView) objects.
//! - [`buffer`] for managing [`VkBuffer`](vk::Buffer) objects.
//! - [`util`] for various utilities and common patterns like buffer uploads.

#![feature(min_specialization)]
#![cfg_attr(feature = "fsr2", feature(new_uninit))]

#![warn(missing_docs)]

#[macro_use]
extern crate derivative;
#[macro_use]
extern crate log;
#[macro_use]
extern crate static_assertions;

pub use crate::prelude::*;

pub mod prelude;

pub mod allocator;
pub mod command_buffer;
pub mod core;
pub mod descriptor;
pub mod graph;
pub mod pipeline;
pub mod sync;
pub mod util;
pub mod wsi;
pub mod resource;

#[cfg(feature = "fsr2")]
pub mod fsr2;
