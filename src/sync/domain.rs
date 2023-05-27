//! Domains abstract over the concept of Vulkan queue families.
//!
//! Commands are divided into four domains:
//! - [Transfer](crate::domain::Transfer): All transfer and copy related commands.
//! - [Graphics](crate::domain::Graphics): All graphics and rendering related commands.
//! - [Compute](crate::domain::Compute): GPU compute commands, most notably `vkCmdDispatch`
//! - [All](crate::domain::All): All of the above.
//!
//! A command buffer over a domain is allocated from a queue that supports all operations
//! on its domain, and as few other domains (to try to catch dedicated transfer/async compute queues). For this reason, always try to
//! allocate from the most restrictive domain as you can.
//!

use ash::vk;

use crate::command_buffer::traits::IncompleteCmdBuffer;
use crate::command_buffer::IncompleteCommandBuffer;
use crate::core::queue::Queue;
use crate::{Allocator, QueueType};

/// This trait defines an execution domain. An execution domain must specify a command buffer type,
/// and expose a function that checks whether a queue is compatible with it or not.
pub trait ExecutionDomain {
    /// Returns true if the selected queue can be used to submit commands from this entire domain
    /// to.
    fn queue_is_compatible(queue: &Queue) -> bool;
    /// Type of the command buffer that will be submitted to this domain.
    /// This type must implement the [`IncompleteCmdBuffer`] trait.
    type CmdBuf<'q, A: Allocator>: IncompleteCmdBuffer<'q, A>;
}

/// Supports all operations (graphics, transfer and compute).
/// This may not always be available (although it usually is).
/// For your main rendering operations, this is typically the correct domain to
/// choose.
pub struct All;
/// Supports graphics operations. Additionally, any domain supporting graphics also supports
/// transfer operations as required by the Vulkan specification.
pub struct Graphics;
/// Supports transfer operations. You should only use this domain for dedicated transfer operations,
/// such as data uploads. When possible, a dedicated transfer queue will be used.
pub struct Transfer;
/// Supports compute operations. For main rendering, you typically want to use the [`All`] domain
/// instead of this, as switching between queues for every compute operation has too much overhead.
pub struct Compute;

impl ExecutionDomain for Graphics {
    /// Returns true if the selected queue can be used to submit commands from this entire domain
    /// to.
    fn queue_is_compatible(queue: &Queue) -> bool {
        queue.info().queue_type == QueueType::Graphics
    }

    /// Type of the command buffer that will be submitted to this domain.
    type CmdBuf<'q, A: Allocator> = IncompleteCommandBuffer<'q, Graphics, A>;
}

impl ExecutionDomain for Transfer {
    /// Returns true if the selected queue can be used to submit commands from this entire domain
    /// to.
    fn queue_is_compatible(queue: &Queue) -> bool {
        queue.info().queue_type == QueueType::Transfer
    }

    /// Type of the command buffer that will be submitted to this domain.
    type CmdBuf<'q, A: Allocator> = IncompleteCommandBuffer<'q, Transfer, A>;
}

impl ExecutionDomain for Compute {
    /// Returns true if the selected queue can be used to submit commands from this entire domain
    /// to.
    fn queue_is_compatible(queue: &Queue) -> bool {
        queue.info().queue_type == QueueType::Compute
    }

    /// Type of the command buffer that will be submitted to this domain.
    type CmdBuf<'q, A: Allocator> = IncompleteCommandBuffer<'q, Compute, A>;
}

impl ExecutionDomain for All {
    /// Returns true if the selected queue can be used to submit commands from this entire domain
    /// to.
    fn queue_is_compatible(queue: &Queue) -> bool {
        queue
            .info()
            .flags
            .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER)
    }

    /// Type of the command buffer that will be submitted to this domain.
    type CmdBuf<'q, A: Allocator> = IncompleteCommandBuffer<'q, All, A>;
}
