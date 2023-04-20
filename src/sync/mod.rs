//! The sync module provides utilities dealing with Vulkan synchronization outside the scope of the
//! pass graph.
//!
//! - The [`fence`] module provides a wrapper around `VkFence` objects, used for CPU-GPU sync,
//! as well as an implementation for [`Future`](std::future::Future) for them.
//! - The [`semaphore`] module provides a simple wrapper around `VkSemaphore` objects, used for GPU-GPU sync.
//! - The [`execution_manager`] module abstracts away vulkan queues and synchronizes access to them by using the
//! [`domain`](crate::domain) system. Most of the time, submissions should go through here.
//! - [`thread_context`] provides an in-flight thread context that can act as a replacement [`InFlightContext`](crate::InFlightContext)
//! when one is expected.
//! - [`submit_batch`] provides a utility to chain [`Semaphore`](crate::Semaphore)s together and submit them all
//! as one batch.

pub mod execution_manager;
pub mod fence;
pub mod semaphore;
pub mod submit_batch;
pub mod thread_context;
pub mod domain;
