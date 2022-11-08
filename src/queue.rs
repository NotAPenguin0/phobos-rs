use std::sync::Arc;
use ash::vk;
use crate::{sync, Device, Error, IncompleteCmdBuffer};
use crate::command_pool::*;

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
    /// All supported operations on this queue, instead of its primary type.
    pub flags: vk::QueueFlags,
}

/// Exposes a logical command queue on the device.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Queue {
    #[derivative(Debug="ignore")]
    device: Arc<Device>,
    /// Raw [`VkQueue`](vk::Queue) handle.
    handle: vk::Queue,
    /// Note that we are only creating one command pool.
    /// We will need to provide thread-safe access to this pool.
    /// TODO: measure lock contention on command pools and determine if we need a queue of pools to pull from instead.
    pool: CommandPool,
    /// Information about this queue, such as supported operations, family index, etc. See also [`QueueInfo`]
    pub info: QueueInfo,
}

impl Queue {
    pub fn new(device: Arc<Device>, handle: vk::Queue, info: QueueInfo) -> Result<Self, Error> {
        // We create a transient command pool because command buffers will be allocated and deallocated
        // frequently.
        let pool = CommandPool::new(device.clone(), info.family_index, vk::CommandPoolCreateFlags::TRANSIENT)?;
        Ok(Queue {
            device,
            handle,
            pool,
            info
        })
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

    pub(crate) fn allocate_command_buffer<CmdBuf: IncompleteCmdBuffer>(&self) -> Result<CmdBuf, Error> {
        let handle = unsafe { self.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.pool.handle)
                .command_buffer_count(1)
                .level(vk::CommandBufferLevel::PRIMARY))?
        }.into_iter().next().ok_or(Error::Uncategorized("Command buffer allocation failed."))?;

        CmdBuf::new(self.device.clone(), handle, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
    }
}