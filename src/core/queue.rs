use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::Result;
use ash::vk;

use crate::command_buffer::command_pool::CommandPool;
use crate::{CmdBuffer, DescriptorCache, Device, Error, Fence, IncompleteCmdBuffer, PipelineCache};

/// Abstraction over vulkan queue capabilities. Note that in raw Vulkan, there is no 'Graphics queue'. Phobos will expose one, but behind the scenes the exposed
/// e.g. graphics queue and transfer could point to the same hardware queue.
#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub enum QueueType {
    #[default]
    Graphics = vk::QueueFlags::GRAPHICS.as_raw() as isize,
    Compute = vk::QueueFlags::COMPUTE.as_raw() as isize,
    Transfer = vk::QueueFlags::TRANSFER.as_raw() as isize,
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

/// Physical VkQueue object.
#[derive(Debug)]
pub(crate) struct DeviceQueue {
    pub handle: vk::Queue,
}

/// Exposes a logical command queue on the device. Note that the physical `VkQueue` object could be multiplexed
/// between different logical queues (e.g. on devices with only one queue).
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Queue {
    #[derivative(Debug = "ignore")]
    device: Arc<Device>,
    queue: Arc<Mutex<DeviceQueue>>,
    /// Note that we are only creating one command pool.
    /// We will need to provide thread-safe access to this pool.
    /// TODO: measure lock contention on command pools and determine if we need a queue of pools to pull from instead.
    pool: CommandPool,
    /// Information about this queue, such as supported operations, family index, etc. See also [`QueueInfo`]
    info: QueueInfo,
}

impl Queue {
    pub(crate) fn new(device: Arc<Device>, queue: Arc<Mutex<DeviceQueue>>, info: QueueInfo) -> Result<Self> {
        // We create a transient command pool because command buffers will be allocated and deallocated
        // frequently.
        let pool = CommandPool::new(device.clone(), info.family_index, vk::CommandPoolCreateFlags::TRANSIENT)?;
        Ok(Queue {
            device,
            queue,
            pool,
            info,
        })
    }

    fn acquire_device_queue(&self) -> Result<MutexGuard<DeviceQueue>> {
        Ok(self.queue.lock().map_err(|_| Error::PoisonError)?)
    }

    /// Submits a batch of submissions to the queue, and signals the given fence when the
    /// submission is done
    /// <br>
    /// <br>
    /// # Thread safety
    /// This function is **not yet** thread safe! This function is marked as unsafe for now to signal this.
    pub unsafe fn submit(&self, submits: &[vk::SubmitInfo], fence: Option<&Fence>) -> Result<()> {
        let fence = match fence {
            None => vk::Fence::null(),
            Some(fence) => fence.handle(),
        };
        let queue = self.acquire_device_queue()?;
        Ok(self.device.queue_submit(queue.handle, submits, fence)?)
    }

    /// Submits a batch of submissions to the queue, and signals the given fence when the
    /// submission is done
    /// <br>
    /// <br>
    /// # Thread safety
    /// This function is **not yet** thread safe! This function is marked as unsafe for now to signal this.
    pub unsafe fn submit2(&self, submits: &[vk::SubmitInfo2], fence: Option<&Fence>) -> Result<()> {
        let fence = match fence {
            None => vk::Fence::null(),
            Some(fence) => fence.handle(),
        };
        let queue = self.acquire_device_queue()?;
        Ok(self.device.queue_submit2(queue.handle, submits, fence)?)
    }

    /// Obtain the raw vulkan handle of a queue.
    pub unsafe fn handle(&self) -> vk::Queue {
        let queue = self.acquire_device_queue().unwrap();
        queue.handle
    }

    pub(crate) fn allocate_command_buffer<'q, CmdBuf: IncompleteCmdBuffer<'q>>(
        device: Arc<Device>,
        queue_lock: MutexGuard<'q, Queue>,
        pipelines: Option<Arc<Mutex<PipelineCache>>>,
        descriptors: Option<Arc<Mutex<DescriptorCache>>>,
    ) -> Result<CmdBuf> {
        let info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            command_pool: unsafe { queue_lock.pool.handle() },
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
        };
        let handle = unsafe { device.allocate_command_buffers(&info)? }
            .into_iter()
            .next()
            .ok_or(Error::Uncategorized("Command buffer allocation failed."))?;

        CmdBuf::new(
            device,
            queue_lock,
            handle,
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            pipelines,
            descriptors,
        )
    }

    /// Instantly delete a command buffer, without taking synchronization into account.
    /// This function **must** be externally synchronized.
    pub(crate) unsafe fn free_command_buffer<CmdBuf: CmdBuffer>(&self, cmd: vk::CommandBuffer) -> Result<()> {
        Ok(self.device.free_command_buffers(self.pool.handle(), std::slice::from_ref(&cmd)))
    }

    pub fn info(&self) -> &QueueInfo {
        &self.info
    }
}
