use crate::domain::ExecutionDomain;
use crate::{ComputeCmdBuffer, ComputeSupport};
use crate::command_buffer::IncompleteCommandBuffer;

impl<D: ComputeSupport + ExecutionDomain> ComputeCmdBuffer for IncompleteCommandBuffer<'_, D> {
    // Methods for compute commands
}