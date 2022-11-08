use std::marker::PhantomData;
use crate::execution_manager::domain::*;
use crate::execution_manager::domain;

use ash::vk;

/// Trait representing a command buffer that supports graphics commands.
pub trait GraphicsCmdBuffer {
    // doesn't do anything yet.
    fn draw(self) -> Self;
}

/// Trait representing a command buffer that supports transfer commands.
pub trait TransferCmdBuffer {

}

/// Trait representing a command buffer that supports transfer commands.
pub trait ComputeCmdBuffer {

}

/// This struct represents a finished command buffer. This command buffer can't be recorded to anymore.
/// It can only be obtained by calling finish() on an incomplete command buffer;
pub struct CommandBuffer<D: ExecutionDomain> {
    _domain: PhantomData<D>,
}

// TODO: probably move this to a function inside the queue.
pub trait IncompleteCmdBuffer {
    type Domain: ExecutionDomain;

    fn new() -> Self;
    fn finish(self) -> CommandBuffer<Self::Domain>;
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

impl<D: ExecutionDomain> IncompleteCmdBuffer for IncompleteCommandBuffer<D> {
    type Domain = D;

    fn new() -> Self {
        IncompleteCommandBuffer {
            handle: vk::CommandBuffer::null(),
            _domain: PhantomData
        }
    }

    /// Finish recording a command buffer and move its contents into a finished
    /// command buffer that can be submitted
    fn finish(self) -> CommandBuffer<D> {
        // TODO
        CommandBuffer {
            _domain: PhantomData,
        }
    }
}

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