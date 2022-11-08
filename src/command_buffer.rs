use std::marker::PhantomData;
use crate::execution_manager::domain::*;
use crate::execution_manager::domain;

use ash::vk;

/// Trait representing a command buffer that supports graphics commands.
pub trait GraphicsCmdBuffer {

}

/// Trait representing a command buffer that supports transfer commands.
pub trait TransferCmdBuffer {

}

/// Trait representing a command buffer that supports transfer commands.
pub trait ComputeCmdBuffer {

}

/// This struct represents a finished command buffer. This command buffer can't be recorded to anymore.
/// It can only be obtained by calling finish() on an incomplete command buffer;
pub struct CommandBuffer {

}

/// This struct represents an incomplete command buffer.
/// This is a command buffer that has not been called [`IncompleteCommandBuffer::finish()`] on yet.
/// Calling this method will turn it into an immutable command buffer which can then be submitted
/// to the queue it was allocated from.
#[derive(Debug)]
pub struct IncompleteCommandBuffer<D: ExecutionDomain> {
    handle: vk::CommandBuffer,
    _domain: PhantomData<D>,
}

trait GfxSupport {}
trait TransferSupport {}
trait ComputeSupport {}

impl GfxSupport for domain::Graphics {}
impl TransferSupport for domain::Graphics {}
impl TransferSupport for domain::Transfer {}
impl TransferSupport for domain::Compute {}
impl ComputeSupport for domain::Compute {}

impl<D: ExecutionDomain> IncompleteCommandBuffer<D> {
    pub fn finish() -> CommandBuffer {
        todo!()
    }
}

impl<D: GfxSupport + ExecutionDomain> GraphicsCmdBuffer for IncompleteCommandBuffer<D> {
    // Methods for graphics commands
}

impl<D: TransferSupport + ExecutionDomain> TransferCmdBuffer for IncompleteCommandBuffer<D> {
    // Methods for transfer commands
}

impl<D: ComputeSupport + ExecutionDomain> ComputeCmdBuffer for IncompleteCommandBuffer<D> {
    // Methods for compute commands
}