use std::collections::HashMap;
use std::sync::Arc;
use crate::{Device, Error, PhysicalDevice, Queue, QueueType};
use crate::command_buffer::*;

/// This struct is responsible for managing execution domains.
/// Command buffers will be created through this struct.
pub struct ExecutionManager {
    pub(crate) queues: Vec<Queue>,
}

pub mod domain {
    use ash::vk;
    use crate::{IncompleteCmdBuffer, Queue};
    use super::QueueType;
    use super::IncompleteCommandBuffer;

    pub trait ExecutionDomain {
        fn queue_is_compatible(queue: &Queue) -> bool;
        type CmdBuf: IncompleteCmdBuffer;
    }

    pub struct All;
    pub struct Graphics;
    pub struct Transfer;
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
    pub fn new(device: Arc<Device>, physical_device: &PhysicalDevice) -> Result<Self, Error> {
        let mut counts = HashMap::new();
        let queues: Vec<Queue> = physical_device.queues.iter().map(|queue| -> Result<Queue, Error> {
            let index = counts.entry(queue.family_index).or_insert(0 as u32);
            let handle = unsafe { device.get_device_queue(queue.family_index, *index) };
            // Note that we can unwrap() here, because if this does not return Some() then our algorithm is
            // bugged and this should panic.
            *counts.get_mut(&queue.family_index).unwrap() += 1;
            Queue::new(device.clone(), handle, *queue)
        }).collect::<Result<Vec<Queue>, Error>>()?;

        Ok(ExecutionManager {
            queues
        })
    }

    /// Obtain a command buffer capable of operating on the specified domain.
    pub fn on_domain<D: domain::ExecutionDomain>(&self) -> Result<D::CmdBuf, Error> {
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