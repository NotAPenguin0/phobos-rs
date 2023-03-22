use crate::{ComputeCmdBuffer, ComputeSupport};
use crate::command_buffer::IncompleteCommandBuffer;
use crate::domain::ExecutionDomain;

impl<D: ComputeSupport + ExecutionDomain> ComputeCmdBuffer for IncompleteCommandBuffer<'_, D> {
    // Methods for compute commands
}
