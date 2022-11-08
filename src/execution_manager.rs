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
    use super::QueueType;
    use super::IncompleteCommandBuffer;

    pub trait ExecutionDomain {
        const QUEUE_TYPE: QueueType;
        type CmdBuf;
    }

    pub struct Graphics;
    pub struct Transfer;
    pub struct Compute;

    impl ExecutionDomain for Graphics {
        const QUEUE_TYPE: QueueType = QueueType::Graphics;
        type CmdBuf = IncompleteCommandBuffer<Graphics>;
    }

    impl ExecutionDomain for Transfer {
        const QUEUE_TYPE: QueueType = QueueType::Transfer;
        type CmdBuf = IncompleteCommandBuffer<Transfer>;
    }

    impl ExecutionDomain for Compute {
        const QUEUE_TYPE: QueueType = QueueType::Compute;
        type CmdBuf = IncompleteCommandBuffer<Compute>;
    }
}

impl ExecutionManager {
    /// Create a new execution manager. You should only ever have on instance of this struct
    /// in your program.
    pub fn new(device: Arc<Device>, physical_device: &PhysicalDevice) -> Result<Self, Error> {
        let mut counts = HashMap::new();
        let queues: Vec<Queue> = physical_device.queues.iter().map(|queue| -> Queue {
            let index = counts.entry(queue.family_index).or_insert(0 as u32);
            let handle = unsafe { device.get_device_queue(queue.family_index, *index) };
            // Note that we can unwrap() here, because if this does not return Some() then our algorithm is
            // bugged and this should panic.
            *counts.get_mut(&queue.family_index).unwrap() += 1;
            Queue::new(device.clone(), handle, *queue)
        }).collect();

        Ok(ExecutionManager {
            queues
        })
    }

    /// Obtain a command buffer capable of operating on the specified domain.
    pub fn on_domain<Domain: domain::ExecutionDomain>(&self) -> Result<Domain::CmdBuf, Error> {
        todo!()
    }

    /// Obtain a reference to a queue capable of presenting.
    pub(crate) fn get_present_queue(&self) -> Option<&Queue> {
        self.queues.iter().find(|&queue| queue.info.can_present)
    }
}