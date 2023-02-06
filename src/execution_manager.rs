use std::collections::HashMap;
use std::sync::Arc;
use crate::{Device, Error, PhysicalDevice, Queue, QueueType};
use crate::command_buffer::*;
use anyhow::Result;

/// The execution manager is responsible for allocating command buffers on correct
/// queues. To obtain any command buffer, you must allocate it by calling
/// [`ExecutionManager::on_domain()`]. An execution domain is a type that implements
/// the [`domain::ExecutionDomain`] trait. Four domains are already defined, and these should cover
/// virtually every available use case.
///
/// - [`domain::All`] supports all operations and is essentially a combination of the other three domains.
/// - [`domain::Graphics`] supports only graphics operations.
/// - [`domain::Transfer`] supports only transfer operations.
/// - [`domain::Compute`] supports only compute operations.
///
/// Note that all domains also implement a couple commands that apply to all domains with no
/// restrictions on queue type support, such as pipeline barriers.
///
/// # Example
/// ```
/// use phobos::{domain, ExecutionManager};
/// // Create an execution manager first. You only want one of these.
/// let exec = ExecutionManager::new(device.clone(), &physical_device);
/// // Obtain a command buffer on the Transfer domain
/// let cmd = exec.on_domain::<domain::Transfer>()?
///               .copy_image(/*command parameters*/)
///               .finish();
/// // Submit the command buffer, either to this frame's command list,
/// // or to the execution manager for submitting commands outside of a
/// // frame context (such as on another thread).
/// ```
pub struct ExecutionManager {
    pub(crate) queues: Vec<Queue>,
}

pub mod domain {
    use ash::vk;
    use crate::{IncompleteCmdBuffer, Queue};
    use super::QueueType;
    use super::IncompleteCommandBuffer;

    /// This trait defines an execution domain. An execution domain must specify a command buffer type,
    /// and expose a function that checks whether a queue is compatible with it or not.
    pub trait ExecutionDomain {
        /// Returns true if the selected queue can be used to submit commands from this entire domain
        /// to.
        fn queue_is_compatible(queue: &Queue) -> bool;
        /// Type of the command buffer that will be submitted to this domain.
        /// This type must implement the [`IncompleteCmdBuffer`] trait.
        type CmdBuf: IncompleteCmdBuffer;
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

        type CmdBuf = IncompleteCommandBuffer<Graphics>;
    }

    impl ExecutionDomain for Transfer {
        fn queue_is_compatible(queue: &Queue) -> bool {
            queue.info.queue_type == QueueType::Transfer
        }

        type CmdBuf = IncompleteCommandBuffer<Transfer>;
    }

    impl ExecutionDomain for Compute {
        fn queue_is_compatible(queue: &Queue) -> bool {
            queue.info.queue_type == QueueType::Compute
        }

        type CmdBuf = IncompleteCommandBuffer<Compute>;
    }

    impl ExecutionDomain for All {
        fn queue_is_compatible(queue: &Queue) -> bool {
            queue.info.flags.contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER)
        }

        type CmdBuf = IncompleteCommandBuffer<All>;
    }
}

impl ExecutionManager {
    /// Create a new execution manager. You should only ever have on instance of this struct
    /// in your program.
    pub fn new(device: Arc<Device>, physical_device: &PhysicalDevice) -> Result<Self> {
        let mut counts = HashMap::new();
        let queues: Vec<Queue> = physical_device.queues.iter().map(|queue| -> Result<Queue> {
            let index = counts.entry(queue.family_index).or_insert(0 as u32);
            let handle = unsafe { device.get_device_queue(queue.family_index, *index) };
            // Note that we can unwrap() here, because if this does not return Some() then our algorithm is
            // bugged and this should panic.
            *counts.get_mut(&queue.family_index).unwrap() += 1;
            Queue::new(device.clone(), handle, *queue)
        }).collect::<Result<Vec<Queue>>>()?;

        Ok(ExecutionManager {
            queues
        })
    }

    /// Obtain a command buffer capable of operating on the specified domain.
    pub fn on_domain<D: domain::ExecutionDomain>(&self) -> Result<D::CmdBuf> {
        let queue = self.get_queue::<D>().ok_or(Error::NoCapableQueue)?;
        queue.allocate_command_buffer::<D::CmdBuf>()
    }

    /// Obtain a reference to a queue capable of presenting.
    pub(crate) fn get_present_queue(&self) -> Option<&Queue> {
        self.queues.iter().find(|&queue| queue.info.can_present)
    }

    /// Obtain a reference to a queue matching predicate.
    pub(crate) fn get_queue<D: domain::ExecutionDomain>(&self) -> Option<&Queue> {
        self.queues.iter().find(|&q| D::queue_is_compatible(&q))
    }
}