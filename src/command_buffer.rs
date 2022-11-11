use std::marker::PhantomData;
use std::sync::Arc;
use crate::execution_manager::domain::*;
use crate::execution_manager::domain;

use ash::vk;
use crate::{Device, Error, ExecutionManager, ImageView};

/// Trait representing a command buffer that supports graphics commands.
pub trait GraphicsCmdBuffer : TransferCmdBuffer {
    // doesn't do anything yet.
    fn draw(self) -> Self;
}

/// Trait representing a command buffer that supports transfer commands.
pub trait TransferCmdBuffer {

}

/// Trait representing a command buffer that supports compute commands.
pub trait ComputeCmdBuffer : TransferCmdBuffer {
    
}

/// This struct represents a finished command buffer. This command buffer can't be recorded to anymore.
/// It can only be obtained by calling finish() on an incomplete command buffer;
pub struct CommandBuffer<D: ExecutionDomain> {
    pub(crate) handle: vk::CommandBuffer,
    _domain: PhantomData<D>,
}

/// Completed command buffer
pub trait CmdBuffer {
    /// Delete the command buffer immediately.
    /// This is marked unsafe because there is no guarantee that the command buffer is not in use.
    unsafe fn delete(&mut self, exec: &ExecutionManager) -> Result<(), Error>;
}

/// Incomplete command buffer
pub trait IncompleteCmdBuffer {
    type Domain: ExecutionDomain;

    fn new(device: Arc<Device>, handle: vk::CommandBuffer, flags: vk::CommandBufferUsageFlags) -> Result<Self, Error> where Self: Sized;
    fn finish(self) -> Result<CommandBuffer<Self::Domain>, Error>;
}

/// This struct represents an incomplete command buffer.
/// This is a command buffer that has not been called [`IncompleteCommandBuffer::finish()`] on yet.
/// Calling this method will turn it into an immutable command buffer which can then be submitted
/// to the queue it was allocated from. See also [`ExecutionManager`].
///
/// # Example
/// ```
/// use phobos::{domain, ExecutionManager};
///
/// let exec = ExecutionManager::new(device.clone(), &physical_device);
/// let cmd = exec.on_domain::<domain::All>()?
///               // record commands to this command buffer
///               // ...
///               // convert into a complete command buffer by calling finish().
///               // This allows the command buffer to be submitted.
///               .finish();
/// ```
#[derive(Derivative)]
#[derivative(Debug)]
pub struct IncompleteCommandBuffer<D: ExecutionDomain> {
    #[derivative(Debug="ignore")]
    device: Arc<Device>,
    handle: vk::CommandBuffer,
    _domain: PhantomData<D>,
}

impl<D: ExecutionDomain> IncompleteCmdBuffer for IncompleteCommandBuffer<D> {
    type Domain = D;

    fn new(device: Arc<Device>, handle: vk::CommandBuffer, flags: vk::CommandBufferUsageFlags) -> Result<Self, Error> {
        unsafe { device.begin_command_buffer(
            handle,
            &vk::CommandBufferBeginInfo::builder().flags(flags))?
        };
        Ok(IncompleteCommandBuffer {
            device: device.clone(),
            handle,
            _domain: PhantomData
        })
    }

    /// Finish recording a command buffer and move its contents into a finished
    /// command buffer that can be submitted
    fn finish(self) -> Result<CommandBuffer<D>, Error> {
        unsafe { self.device.end_command_buffer(self.handle)? }
        Ok(CommandBuffer {
            handle: self.handle,
            _domain: PhantomData,
        })
    }
}

impl<D: ExecutionDomain> CmdBuffer for CommandBuffer<D> {
    unsafe fn delete(&mut self, exec: &ExecutionManager) -> Result<(), Error> {
        let queue = exec.get_queue::<D>().ok_or(Error::NoCapableQueue)?;
        queue.free_command_buffer::<Self>(self.handle)
    }
}

// Provides implementations for commands that work on all domains,
// and other helper functions
impl<D: ExecutionDomain> IncompleteCommandBuffer<D> {
    /// Transitions an image layout.
    /// Generally you will not need to call this function manually,
    /// using the render graph api you can do most transitions automatically.
    pub fn transition_image(self, image: &ImageView, src_stage: vk::PipelineStageFlags, dst_stage: vk::PipelineStageFlags,
                            from: vk::ImageLayout, to: vk::ImageLayout,
                            src_access: vk::AccessFlags, dst_access: vk::AccessFlags) -> Self {
        let barrier = vk::ImageMemoryBarrier::builder()
            .image(image.image.handle)
            .subresource_range(image.info.subresource_range())
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .old_layout(from)
            .new_layout(to)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .build();
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.handle,
                src_stage,
                dst_stage,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                std::slice::from_ref(&barrier)
            );
        }

        self
    }
}

trait GfxSupport : TransferSupport {}
trait TransferSupport {}
trait ComputeSupport : TransferSupport {}

impl GfxSupport for domain::Graphics {}
impl GfxSupport for domain::All {}
impl TransferSupport for domain::Graphics {}
impl TransferSupport for domain::Transfer {}
impl TransferSupport for domain::Compute {}
impl TransferSupport for domain::All {}
impl ComputeSupport for domain::Compute {}
impl ComputeSupport for domain::All {}

impl<D: GfxSupport + ExecutionDomain> GraphicsCmdBuffer for IncompleteCommandBuffer<D> {
    // Methods for graphics commands

    // nothing yet
    fn draw(self) -> Self {
        self
    }
}

impl<D: TransferSupport + ExecutionDomain> TransferCmdBuffer for IncompleteCommandBuffer<D> {
    // Methods for transfer commands
}

impl<D: ComputeSupport + ExecutionDomain> ComputeCmdBuffer for IncompleteCommandBuffer<D> {
    // Methods for compute commands
}