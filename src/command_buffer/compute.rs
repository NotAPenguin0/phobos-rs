use crate::domain::ExecutionDomain;
use crate::{ComputeSupport, IncompleteCommandBuffer};
use crate::traits::ComputeCmdBuffer;

impl<D: ComputeSupport + ExecutionDomain> ComputeCmdBuffer for IncompleteCommandBuffer<'_, D> {
    // Methods for compute commands
}