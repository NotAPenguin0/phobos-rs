//! Exposes Vulkan queue objects, though these are always abstracted through the ExecutionManager.

use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::Result;
use ash::vk;

use crate::{
    Allocator, CmdBuffer, DescriptorCache, Device, Error, Fence, IncompleteCmdBuffer, PipelineCache,
};
use crate::command_buffer::command_pool::CommandPool;

/// Abstraction over vulkan queue capabilities. Note that in raw Vulkan, there is no 'Graphics queue'. Phobos will expose one, but behind the scenes the exposed
/// e.g. graphics and transfer queues could point to the same hardware queue. Synchronization for this is handled for you.
#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub enum QueueType {
    /// Queue that supports graphics operations. Per the vulkan spec, this queue also always supports
    /// transfer operations. Phobos will try to match this to a hardware queue that also
    /// supports compute operations. This is always guaranteed to be available if graphics operations
    /// are supported.
    #[default]
    Graphics = vk::QueueFlags::GRAPHICS.as_raw() as isize,
    /// Queue that supports compute operations. Per the vulkan spec, this queue also always supports
    /// transfer operations. Phobos will try to match this to a hardware queue that does not support
    /// graphics operations if possible, to make full use of async compute when available.
    Compute = vk::QueueFlags::COMPUTE.as_raw() as isize,
    /// Queue that supports transfer operations. Phobos will try to match this to a hardware queue that only supports
    /// transfer operations if possible.
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
    device: Device,
    queue: Arc<Mutex<DeviceQueue>>,
    /// Note that we are only creating one command pool.
    /// We will need to provide thread-safe access to this pool.
    /// TODO: measure lock contention on command pools and determine if we need a queue of pools to pull from instead.
    pool: CommandPool,
    /// Information about this queue, such as supported operations, family index, etc. See also [`QueueInfo`]
    info: QueueInfo,
    /// This queues queue family properties.
    family_properties: vk::QueueFamilyProperties,
}

impl Queue {
    pub(crate) fn new(
        device: Device,
        queue: Arc<Mutex<DeviceQueue>>,
        info: QueueInfo,
        family_properties: vk::QueueFamilyProperties,
    ) -> Result<Self> {
        // We create a transient command pool because command buffers will be allocated and deallocated
        // frequently.
        let pool = CommandPool::new(
            device.clone(),
            info.family_index,
            vk::CommandPoolCreateFlags::TRANSIENT,
        )?;
        Ok(Queue {
            device,
            queue,
            pool,
            info,
            family_properties,
        })
    }

    fn acquire_device_queue(&self) -> Result<MutexGuard<DeviceQueue>> {
        Ok(self.queue.lock().map_err(|_| Error::PoisonError)?)
    }

    /// Submits a batch of submissions to the queue, and signals the given fence when the
    /// submission is done. When possible, prefer submitting through the
    /// execution manager.
    pub fn submit(&self, submits: &[vk::SubmitInfo], fence: Option<&Fence>) -> Result<()> {
        let fence = match fence {
            None => vk::Fence::null(),
            // SAFETY: The user supplied a valid fence
            Some(fence) => unsafe { fence.handle() },
        };
        let queue = self.acquire_device_queue()?;
        // SAFETY:
        // * `fence` is null or a valid fence handle (see above).
        // * The user supplied a valid range of `VkSubmitInfo` structures.
        // * `queue` is a valid queue object.
        unsafe { Ok(self.device.queue_submit(queue.handle, submits, fence)?) }
    }

    /// Submits a batch of submissions to the queue, and signals the given fence when the
    /// submission is done. When possible, prefer submitting through the
    /// execution manager.
    ///
    /// This function is different from [`Queue::submit()`] only because it accepts `VkSubmitInfo2`[vk::SubmitInfo2] structures.
    /// This is a more modern version of the old API. The old API will be deprecated and removed eventually.
    pub fn submit2(&self, submits: &[vk::SubmitInfo2], fence: Option<&Fence>) -> Result<()> {
        let fence = match fence {
            None => vk::Fence::null(),
            // SAFETY: The user supplied a valid fence
            Some(fence) => unsafe { fence.handle() },
        };
        let queue = self.acquire_device_queue()?;
        // * `fence` is null or a valid fence handle (see above).
        // * The user supplied a valid range of `VkSubmitInfo2` structures.
        // * `queue` is a valid queue object.
        unsafe { Ok(self.device.queue_submit2(queue.handle, submits, fence)?) }
    }

    /// Obtain the raw vulkan handle of a queue.
    /// # Safety
    /// Any vulkan calls that mutate the `VkQueue` object may lead to race conditions or undefined behaviour.
    pub unsafe fn handle(&self) -> vk::Queue {
        let queue = self.acquire_device_queue().unwrap();
        queue.handle
    }

    pub(crate) fn allocate_command_buffer<'q, A: Allocator, CmdBuf: IncompleteCmdBuffer<'q, A>>(
        device: Device,
        queue_lock: MutexGuard<'q, Queue>,
        pipelines: PipelineCache<A>,
        descriptors: DescriptorCache,
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
            .ok_or_else(|| Error::Uncategorized("Command buffer allocation failed."))?;

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
    pub(crate) unsafe fn free_command_buffer<CmdBuf: CmdBuffer<A>, A: Allocator>(
        &self,
        cmd: vk::CommandBuffer,
    ) -> Result<()> {
        self.device
            .free_command_buffers(self.pool.handle(), std::slice::from_ref(&cmd));
        Ok(())
    }

    /// Get the properties of this queue, such as whether it is dedicated or not.
    pub fn info(&self) -> &QueueInfo {
        &self.info
    }

    /// Get the properties of this queue's family.
    pub fn family_properties(&self) -> &vk::QueueFamilyProperties {
        &self.family_properties
    }
}
