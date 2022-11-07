use std::collections::HashMap;
use std::sync::Arc;
use crate::{Device, Error, PhysicalDevice, Queue};

pub struct ExecutionManager {
    pub(crate) queues: Vec<Queue>,
}

impl ExecutionManager {
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
}