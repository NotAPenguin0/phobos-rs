//! Exposes the [`ExecutionManager`], used to allocate and submit command buffers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, TryLockError, TryLockResult};

use anyhow::Result;
use ash::vk;

use crate::{Allocator, CmdBuffer, DescriptorCache, Device, Error, Fence, PhysicalDevice, PipelineCache};
use crate::command_buffer::*;
use crate::core::queue::{DeviceQueue, Queue};
use crate::sync::domain::ExecutionDomain;
use crate::sync::submit_batch::SubmitBatch;

/// The execution manager is responsible for allocating command buffers on correct
/// queues. To obtain any command buffer, you must allocate it by calling
/// [`ExecutionManager::on_domain()`]. An execution domain is a type that implements
/// the [`domain::ExecutionDomain`](crate::domain::ExecutionDomain) trait. Four domains are already defined, and these should cover
/// virtually every available use case.
///
/// - [`domain::All`](crate::domain::All) supports all operations and is essentially a combination of the other three domains.
/// - [`domain::Graphics`](crate::domain::Graphics) supports only graphics operations.
/// - [`domain::Transfer`](crate::domain::Transfer) supports only transfer operations.
/// - [`domain::Compute`](crate::domain::Compute) supports only compute operations.
///
/// Note that all domains also implement a couple commands that apply to all domains with no
/// restrictions on queue type support, such as pipeline barriers.
///
/// # Example
/// ```
/// use phobos::prelude::*;
/// // Create an execution manager first. You only want one of these.
/// let exec = ExecutionManager::new(device.clone(), &physical_device);
/// // Obtain a command buffer on the Transfer domain
/// let cmd = exec.on_domain::<domain::Transfer>(None, None)?
///               .copy_image(/*command parameters*/)
///               .finish()?;
/// // Submit the command buffer, either to this frame's command list,
/// // or to the execution manager for submitting commands outside of a
/// // frame context (such as on another thread).
/// ```
#[derive(Debug, Clone)]
pub struct ExecutionManager {
    device: Device,
    queues: Arc<Vec<Mutex<Queue>>>,
}

fn max_queue_count(family: u32, families: &[vk::QueueFamilyProperties]) -> u32 {
    // TODO: missing queue family in the middle will panic
    families.get(family as usize).unwrap().queue_count
}

impl ExecutionManager {
    /// Create a new execution manager. You should only ever have on instance of this struct
    /// in your program.
    pub fn new(device: Device, physical_device: &PhysicalDevice) -> Result<Self> {
        let mut counts = HashMap::new();
        let mut device_queues = HashMap::new();

        let queues = physical_device
            .queues()
            .iter()
            .map(|queue| -> Result<Mutex<Queue>> {
                let index = counts.entry(queue.family_index).or_insert(0);
                // If we have exceeded the max count for this family, we need to reuse a device queue from earlier
                let device_queue = if *index >= max_queue_count(queue.family_index, physical_device.queue_families()) {
                    // Re-use a previously requested device queue. If this panics, the code is bugged (this is not a user error)
                    device_queues.get(&queue.family_index).cloned().unwrap()
                } else {
                    // Create a new DeviceQueue
                    let device_queue = Arc::new(Mutex::new(DeviceQueue {
                        handle: unsafe { device.get_device_queue(queue.family_index, *index) },
                    }));
                    // Note that we can unwrap() here, because if this does not return Some() then our algorithm is
                    // bugged and this should panic.
                    *counts.get_mut(&queue.family_index).unwrap() += 1;
                    // Store it
                    device_queues.insert(queue.family_index, device_queue.clone());
                    // Use this for our queue
                    device_queue
                };
                Ok(Mutex::new(Queue::new(
                    device.clone(),
                    device_queue,
                    *queue,
                    *physical_device.queue_families().get(queue.family_index as usize).unwrap(),
                )?))
            })
            .collect::<Result<Vec<Mutex<Queue>>>>()?;

        info!("Created device queues:");
        for queue in &queues {
            let lock = queue.lock().unwrap();
            let info = lock.info();
            info!(
                "Queue #{:?}({}) supports {:?} (dedicated: {}, can present: {})",
                info.queue_type, info.family_index, info.flags, info.dedicated, info.can_present
            )
        }

        Ok(ExecutionManager {
            device,
            queues: Arc::new(queues),
        })
    }

    /// Tries to obtain a command buffer over a domain, or returns an Err state if the lock is currently being held.
    /// If this command buffer needs access to pipelines or descriptor sets, pass in the relevant caches.
    pub fn try_on_domain<'q, D: ExecutionDomain, A: Allocator>(&'q self, pipelines: Option<PipelineCache<A>>, descriptors: Option<DescriptorCache>) -> Result<D::CmdBuf<'q, A>> {
        let queue = self.try_get_queue::<D>().map_err(|_| Error::QueueLocked)?;
        Queue::allocate_command_buffer::<'q, A, D::CmdBuf<'q, A>>(self.device.clone(), queue, pipelines, descriptors)
    }

    /// Obtain a command buffer capable of operating on the specified domain.
    /// If this command buffer needs access to pipelines or descriptor sets, pass in the relevant caches.
    pub fn on_domain<'q, D: ExecutionDomain, A: Allocator>(&'q self, pipelines: Option<PipelineCache<A>>, descriptors: Option<DescriptorCache>) -> Result<D::CmdBuf<'q, A>> {
        let queue = self.get_queue::<D>().ok_or_else(|| Error::NoCapableQueue)?;
        Queue::allocate_command_buffer::<'q, A, D::CmdBuf<'q, A>>(self.device.clone(), queue, pipelines, descriptors)
    }

    /// Begin a submit batch. Note that all submits in a batch are over a single domain (currently).
    /// # Example
    /// ```
    /// use phobos::prelude::*;
    /// let exec = ExecutionManager::new(device.clone(), &physical_device)?;
    /// let cmd1 = exec.on_domain::<domain::All>(None, None)?.finish()?;
    /// let cmd2 = exec.on_domain::<domain::All>(None, None)?.finish()?;
    /// let mut batch = exec.start_submit_batch()?;
    /// // Submit the first command buffer first
    /// batch.submit(cmd1)?
    ///      // The second command buffer waits at COLOR_ATTACHMENT_OUTPUT on the first command buffer's completion.
    ///      .then(PipelineStage::COLOR_ATTACHMENT_OUTPUT, cmd2, &mut batch)?;
    /// batch.finish()?.wait()?;
    /// ```
    pub fn start_submit_batch<D: ExecutionDomain + 'static>(&self) -> Result<SubmitBatch<D>> {
        SubmitBatch::new(self.device.clone(), self.clone())
    }

    /// Submit a command buffer to its queue.
    pub fn submit<D: ExecutionDomain + 'static>(&self, mut cmd: CommandBuffer<D>) -> Result<Fence> {
        let fence = Fence::new(self.device.clone(), false)?;

        let handle = unsafe { cmd.handle() };
        let info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &handle,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
        };

        let queue = self.get_queue::<D>().ok_or_else(|| Error::NoCapableQueue)?;
        queue.submit(std::slice::from_ref(&info), Some(&fence))?;
        let exec = self.clone();
        Ok(fence.with_cleanup(move || unsafe {
            cmd.delete(exec).unwrap();
        }))
    }

    /// Submit multiple SubmitInfo2 structures.
    pub(crate) fn submit_batch<D: ExecutionDomain>(&self, submits: &[vk::SubmitInfo2], fence: &Fence) -> Result<()> {
        let queue = self.get_queue::<D>().ok_or_else(|| Error::NoCapableQueue)?;
        queue.submit2(submits, Some(fence))?;
        Ok(())
    }

    /// Obtain a reference to a queue capable of presenting.
    pub(crate) fn get_present_queue(&self) -> Option<MutexGuard<Queue>> {
        self.queues
            .iter()
            .find(|&queue| queue.lock().unwrap().info().can_present)
            .map(|q| q.lock().unwrap())
    }

    /// Try to get a reference to a queue matching the domain, or return an error state if this would need to block
    /// to lock the queue.
    pub fn try_get_queue<D: ExecutionDomain>(&self) -> TryLockResult<MutexGuard<Queue>> {
        let q = self.queues.iter().find(|&q| {
            let q = q.try_lock();
            match q {
                Ok(queue) => D::queue_is_compatible(&queue),
                Err(_) => false,
            }
        });
        match q {
            None => Err(TryLockError::WouldBlock),
            Some(q) => Ok(q.lock()?),
        }
    }

    /// Obtain a reference to a queue matching the domain. Blocks if this queue is currently locked.
    pub fn get_queue<D: ExecutionDomain>(&self) -> Option<MutexGuard<Queue>> {
        self.queues
            .iter()
            .find(|&q| {
                let q = q.lock().unwrap();
                D::queue_is_compatible(&q)
            })
            .map(|q| q.lock().unwrap())
    }
}
