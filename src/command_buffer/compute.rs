use anyhow::Result;
use ash::vk;

use crate::{ComputeCmdBuffer, ComputeSupport, Error};
use crate::command_buffer::IncompleteCommandBuffer;
use crate::domain::ExecutionDomain;

impl<D: ComputeSupport + ExecutionDomain> ComputeCmdBuffer for IncompleteCommandBuffer<'_, D> {
    fn bind_compute_pipeline(mut self, name: &str) -> Result<Self>
    where
        Self: Sized, {
        let Some(mut cache) = self.pipeline_cache.clone() else { return Err(Error::NoPipelineCache.into()); };
        {
            cache.with_compute_pipeline(name, |pipeline| {
                unsafe {
                    self.device
                        .cmd_bind_pipeline(self.handle, vk::PipelineBindPoint::COMPUTE, pipeline.handle);
                }
                self.current_bindpoint = vk::PipelineBindPoint::COMPUTE;
                self.current_pipeline_layout = pipeline.layout;
                self.current_set_layouts = pipeline.set_layouts.clone();
                Ok(())
            })?;
        }
        Ok(self)
    }

    fn dispatch(mut self, x: u32, y: u32, z: u32) -> Result<Self> {
        self = self.ensure_descriptor_state()?;
        unsafe {
            self.device.cmd_dispatch(self.handle, x, y, z);
        }
        Ok(self)
    }
    // Methods for compute commands
}
