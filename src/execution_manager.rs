use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, TryLockError, TryLockResult};
use crate::{Device, Error, Fence, PhysicalDevice, Queue, QueueType};
use crate::command_buffer::*;
use anyhow::Result;
use ash::vk;
use crate::deferred_delete::DeletionQueue;
use crate::domain::ExecutionDomain;

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
#[derive(Debug)]
pub struct ExecutionManager {
    device: Arc<Device>,
    pub(crate) queues: Vec<Mutex<Queue>>,
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
}

impl ExecutionManager {
    /// Create a new execution manager. You should only ever have on instance of this struct
    /// in your program.
    pub fn new(device: Arc<Device>, physical_device: &PhysicalDevice) -> Result<Arc<Self>> {
        let mut counts = HashMap::new();
        let queues = physical_device.queues.iter().map(|queue| -> Result<Mutex<Queue>> {
            let index = counts.entry(queue.family_index).or_insert(0 as u32);
            let handle = unsafe { device.get_device_queue(queue.family_index, *index) };
            // Note that we can unwrap() here, because if this does not return Some() then our algorithm is
            // bugged and this should panic.
            *counts.get_mut(&queue.family_index).unwrap() += 1;
            Ok(Mutex::new(Queue::new(device.clone(), handle, *queue)?))
        }).collect::<Result<Vec<Mutex<Queue>>>>()?;

        info!("Created device queues:");
        for queue in &queues {
            let lock = queue.lock().unwrap();
            info!("Queue #{:?}({}) supports {:?} (dedicated: {}, can present: {})", lock.info.queue_type, lock.info.family_index, lock.info.flags, lock.info.dedicated, lock.info.can_present)
        }

        Ok(Arc::new(ExecutionManager {
            device: device.clone(),
            queues
        }))
    }

    /// Tries to obtain a command buffer over a domain, or returns an Err state if the lock is currently being held.
    pub fn try_on_domain<'q, D: domain::ExecutionDomain>(&'q self) -> Result<D::CmdBuf<'q>> {
        let queue = self.try_get_queue::<D>().map_err(|_| Error::QueueLocked)?;
        Queue::allocate_command_buffer::<'q, D::CmdBuf<'q>>(self.device.clone(), queue)
    }

    /// Obtain a command buffer capable of operating on the specified domain.
    pub fn on_domain<'q, D: domain::ExecutionDomain>(&'q self) -> Result<D::CmdBuf<'q>> {
        let queue = self.get_queue::<D>().ok_or(Error::NoCapableQueue)?;
        Queue::allocate_command_buffer::<'q, D::CmdBuf<'q>>(self.device.clone(), queue)
    }

    // Submit a command buffer to its queue. TODO: Add semaphores
    pub fn submit<'f, 'q, D: domain::ExecutionDomain + 'f>(exec: Arc<ExecutionManager>, mut cmd: CommandBuffer<D>) -> Result<Fence<'f>> {
        let fence = Fence::new(exec.device.clone(), false)?;

        let info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &cmd.handle,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
        };

        let queue = exec.get_queue::<D>().ok_or(Error::NoCapableQueue)?;
        unsafe { queue.submit(std::slice::from_ref(&info), Some(&fence))?; }

        let exec = exec.clone();
        Ok(fence.with_cleanup(move || {
            unsafe { cmd.delete(exec.clone()).unwrap(); }
        }))
    }

    /// Obtain a reference to a queue capable of presenting.
    pub(crate) fn get_present_queue(&self) -> Option<MutexGuard<Queue>> {
        self.queues.iter().find(|&queue| queue.lock().unwrap().info.can_present.clone()).map(|q| q.lock().unwrap())
    }

    pub fn try_get_queue<D: domain::ExecutionDomain>(&self) -> TryLockResult<MutexGuard<Queue>> {
        let q = self.queues.iter().find(|&q| {
            let q = q.try_lock();
            match q {
                Ok(queue) => { D::queue_is_compatible(&*queue) }
                Err(_) => { false }
            }
        });
        match q {
            None => { Err(TryLockError::WouldBlock) }
            Some(q) => { Ok(q.lock()?) }
        }
    }

    /// Obtain a reference to a queue matching predicate.
    pub fn get_queue<D: domain::ExecutionDomain>(&self) -> Option<MutexGuard<Queue>> {
        self.queues.iter().find(|&q| {
            let q = q.lock().unwrap();
            D::queue_is_compatible(&*q)
        }).map(|q| q.lock().unwrap())
    }
}