use ash::vk;
use crate::command_buffer::IncompleteCommandBuffer;
use crate::QueueType;
use crate::command_buffer::traits::IncompleteCmdBuffer;
use crate::core::queue::Queue;

/// This trait defines an execution domain. An execution domain must specify a command buffer type,
/// and expose a function that checks whether a queue is compatible with it or not.
pub trait ExecutionDomain {
    /// Returns true if the selected queue can be used to submit commands from this entire domain
    /// to.
    fn queue_is_compatible(queue: &Queue) -> bool;
    /// Type of the command buffer that will be submitted to this domain.
    /// This type must implement the [`IncompleteCmdBuffer`] trait.
    type CmdBuf<'q>: IncompleteCmdBuffer<'q>;
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
    fn queue_is_compatible(queue: &Queue) -> bool {
        queue.info.queue_type == QueueType::Graphics
    }

    type CmdBuf<'q> = IncompleteCommandBuffer<'q, Graphics>;
}

impl ExecutionDomain for Transfer {
    fn queue_is_compatible(queue: &Queue) -> bool {
        queue.info.queue_type == QueueType::Transfer
    }

    type CmdBuf<'q> = IncompleteCommandBuffer<'q, Transfer>;
}

impl ExecutionDomain for Compute {
    fn queue_is_compatible(queue: &Queue) -> bool {
        queue.info.queue_type == QueueType::Compute
    }

    type CmdBuf<'q> = IncompleteCommandBuffer<'q, Compute>;
}

impl ExecutionDomain for All {
    fn queue_is_compatible(queue: &Queue) -> bool {
        queue.info.flags.contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER)
    }

    type CmdBuf<'q> = IncompleteCommandBuffer<'q, All>;
}