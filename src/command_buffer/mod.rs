//! Most functions in this module are a relatively thin wrapper over Vulkan commands.
//!
//! # Domains
//!
//! The most important feature is that of execution domains. Commands are divided into four domains:
//! - [Transfer](crate::domain::Transfer): All transfer and copy related commands.
//! - [Graphics](crate::domain::Graphics): All graphics and rendering related commands.
//! - [Compute](crate::domain::Compute): GPU compute commands, most notably `vkCmdDispatch`
//! - [All](crate::domain::All): All of the above.
//!
//! This concept abstracts over that of queue families. A command buffer over a domain is allocated from a queue that supports all operations
//! on its domain, and as few other domains (to try to catch dedicated transfer/async compute queues). For this reason, always try to
//! allocate from the most restrictive domain as you can.
//!
//! # Incomplete command buffers
//!
//! Vulkan command buffers need to call `vkEndCommandBuffer` before they can be submitted. After this call, no more commands should be
//! recorded to it. For this reason, we expose two command buffer types. The [`IncompleteCommandBuffer`] still accepts commands, and can only
//! be converted into a [`CommandBuffer`] by calling [`IncompleteCommandBuffer::finish`](crate::command_buffer::IncompleteCommandBuffer::finish). This turns it into a complete command buffer, which can
//! be submitted to the execution manager.
//!
//! # Commands
//! All commands are implemented through traits for each domain. These are all defined inside the [`traits`] module, and are most easily imported
//! through the [`prelude`](crate::prelude).
//!
//! There are also a bunch of methods that do not directly translate to Vulkan commands, for example for binding descriptor sets directly.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::MutexGuard;

use anyhow::Result;
use ash::vk;

use crate::{CmdBuffer, DescriptorCache, DescriptorSetBuilder, Device, Error, ExecutionManager, PipelineCache};
use crate::core::queue::Queue;
use crate::domain::ExecutionDomain;
use crate::pipeline::create_info::PipelineRenderingInfo;

pub mod compute;
pub mod graphics;
pub mod incomplete;
pub mod traits;
pub mod transfer;

pub(crate) mod command_pool;
pub(crate) mod state;

/// This struct represents a finished command buffer. This command buffer can't be recorded to anymore.
/// It can only be obtained by calling [`IncompleteCommandBuffer::finish()`].
/// # Example
/// ```
/// # use phobos::*;
/// # use phobos::domain::ExecutionDomain;
/// # use anyhow::Result;
///
/// fn finish_and_wait<D: ExecutionDomain + 'static>(exec: ExecutionManager, cmd: IncompleteCommandBuffer<D>) -> Result<()> {
///     let cmd: CommandBuffer<D> = cmd.finish()?;
///     let mut fence = exec.submit(cmd)?;
///     fence.wait()
/// }
/// ```
#[derive(Debug)]
pub struct CommandBuffer<D: ExecutionDomain> {
    handle: vk::CommandBuffer,
    _domain: PhantomData<D>,
}

/// This struct represents an incomplete command buffer.
/// This is a command buffer that has not been called [`IncompleteCommandBuffer::finish()`] on yet.
/// Calling this method will turn it into an immutable command buffer which can then be submitted
/// to the queue it was allocated from. See also [`ExecutionManager`].
///
/// # Example
/// ```
/// # use phobos::*;
/// # use phobos::domain::{ExecutionDomain, Graphics};
/// # use anyhow::Result;
///
/// fn submit_some_commands(exec: ExecutionManager) -> Result<()> {
///     let cmd: IncompleteCommandBuffer<Graphics> = exec.on_domain::<Graphics>(None, None)?;
///     // ... record some commands
///     let mut fence = exec.submit(cmd.finish()?)?;
///     fence.wait()?;
///     Ok(())
/// }
/// ```
/// # Descriptor sets
/// Descriptor sets can be bound simply by calling the correct `bind_xxx` functions on the command buffer.
/// (see for example [`IncompleteCommandBuffer::bind_uniform_buffer()`])
/// It should be noted that these are not actually bound yet on calling this function.
/// Instead, the next `draw()` or `dispatch()` call flushes these bind calls and does an actual `vkCmdBindDescriptorSets` call.
/// This also forgets the old binding state, so to update the bindings you need to re-bind all previously bound sets (this is something
/// that could change in the future, see <https://github.com/NotAPenguin0/phobos-rs/issues/23>)
#[derive(Derivative)]
#[derivative(Debug)]
pub struct IncompleteCommandBuffer<'q, D: ExecutionDomain> {
    #[derivative(Debug = "ignore")]
    device: Device,
    handle: vk::CommandBuffer,
    queue_lock: MutexGuard<'q, Queue>,
    current_pipeline_layout: vk::PipelineLayout,
    current_set_layouts: Vec<vk::DescriptorSetLayout>,
    current_bindpoint: vk::PipelineBindPoint,
    // TODO: Note: technically not correct
    current_rendering_state: Option<PipelineRenderingInfo>,
    current_render_area: vk::Rect2D,
    current_descriptor_sets: Option<HashMap<u32, DescriptorSetBuilder<'static>>>,
    descriptor_state_needs_update: bool,
    // TODO: Only update disturbed descriptor sets
    descriptor_cache: Option<DescriptorCache>,
    pipeline_cache: Option<PipelineCache>,
    _domain: PhantomData<D>,
}

impl<D: ExecutionDomain> CmdBuffer for CommandBuffer<D> {
    /// Immediately delete a command buffer. Generally you do not need to call this manually, since
    /// commands buffers submitted through the [`ExecutionManager`] already do this cleanup.
    /// # Safety
    /// * This command buffer must not currently be executing on the GPU.
    unsafe fn delete(&mut self, exec: ExecutionManager) -> Result<()> {
        let queue = exec.get_queue::<D>().ok_or(Error::NoCapableQueue)?;
        let handle = self.handle;
        self.handle = vk::CommandBuffer::null();
        queue.free_command_buffer::<Self>(handle)
    }
}

impl<D: ExecutionDomain> CommandBuffer<D> {
    /// Get unsafe access to the underlying command buffer
    /// # Safety
    /// Any vulkan calls that modify the command buffer state may lead to validation errors or put the
    /// system in an undefined state.
    pub unsafe fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }
}
