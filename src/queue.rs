use std::sync::Arc;
use ash::vk;
use crate::{sync, Device};

/// Abstraction over vulkan queue capabilities. Note that in raw Vulkan, there is no 'Graphics queue'. Phobos will expose one, but behind the scenes the exposed
/// e.g. graphics queue and transfer could point to the same hardware queue.
#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub enum QueueType {
    #[default]
    Graphics = vk::QueueFlags::GRAPHICS.as_raw() as isize,
    Compute = vk::QueueFlags::COMPUTE.as_raw() as isize,
    Transfer = vk::QueueFlags::TRANSFER.as_raw() as isize
}

/// Stores all information of a queue that was found on the physical device.
#[derive(Default, Debug, Copy, Clone)]
pub struct QueueInfo {
    /// Functionality that this queue provides.
    pub queue_type: QueueType,
    /// Whether this is a dedicated queue or not.
    pub dedicated: bool,
    /// Whether this queue is capable of presenting to a surface.
    pub can_present: bool,
    /// The queue family index.
    pub family_index: u32,
}

/// Exposes a logical command queue on the device.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Queue {
    #[derivative(Debug="ignore")]
    device: Arc<Device>,
    /// Raw [`VkQueue`](vk::Queue) handle.
    handle: vk::Queue,
    /// Information about this queue, such as supported operations, family index, etc. See also [`QueueInfo`]
    pub info: QueueInfo,
}

impl Queue {
    pub fn new(device: Arc<Device>, handle: vk::Queue, info: QueueInfo) -> Self {
        Queue {
            device,
            handle,
            info
        }
    }

    /// Submits a batch of submissions to the queue, and signals the given fence when the
    /// submission is done
    /// <br>
    /// <br>
    /// # Thread safety
    /// This function is **not yet** thread safe! This function is marked as unsafe for now to signal this.
    pub unsafe fn submit(&self, submits: &[vk::SubmitInfo], fence: &sync::Fence) -> Result<(), vk::Result> {
        self.device.queue_submit(self.handle, submits, fence.handle)
    }

    pub fn handle(&self) -> vk::Queue {
        self.handle
    }
}