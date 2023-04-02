use anyhow::Result;
use ash::vk;

use crate::{ComputeCmdBuffer, ComputeSupport, Error};
use crate::command_buffer::IncompleteCommandBuffer;
use crate::domain::ExecutionDomain;

impl<D: ComputeSupport + ExecutionDomain> ComputeCmdBuffer for IncompleteCommandBuffer<'_, D> {
    /// Sets the current compute pipeline by looking up the given name in the pipeline cache.
    /// # Errors
    /// - Fails with [`Error::NoPipelineCache`] if this command buffer was created without a pipeline cache.
    /// - Fails if the given name is not a valid compute pipeline name.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use phobos::domain::ExecutionDomain;
    /// # use anyhow::Result;
    ///
    /// // Assumes "my_pipeline" was previously added to the pipeline cache with `PipelineCache::create_named_compute_pipeline()`,
    /// // and that cmd was created with this cache.
    /// fn compute_pipeline<D: ExecutionDomain + ComputeSupport>(cmd: IncompleteCommandBuffer<D>) -> Result<IncompleteCommandBuffer<D>> {
    ///     cmd.bind_compute_pipeline("my_pipeline")
    /// }
    /// ```
    fn bind_compute_pipeline(mut self, name: &str) -> Result<Self>
        where
            Self: Sized, {
        let Some(mut cache) = self.pipeline_cache.clone() else { return Err(Error::NoPipelineCache.into()); };
        {
            cache.with_compute_pipeline(name, |pipeline| {
                self.bind_pipeline_impl(pipeline.handle, pipeline.layout, pipeline.set_layouts.clone(), vk::PipelineBindPoint::COMPUTE)
            })?;
        }
        Ok(self)
    }

    /// Dispatch compute invocations. `x`, `y` and `z` are the amount of workgroups in each dimension. The total amount of
    /// dispatches is then `(x * LocalSize.x, y * LocalSize.y, z * LocalSize.z)` as defined in the shader.
    ///
    /// This function also flushes the current descriptor set state. Any binding updates after this will completely overwrite
    /// the old binding state (even to different locations).
    ///
    /// See also: [`vkCmdDispatch`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdDispatch.html)
    ///
    /// # Errors
    /// * Fails if this required a descriptor set update and this command buffer was created without a
    ///   descriptor cache.
    /// * Fails if updating the descriptor state fails.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use phobos::domain::ExecutionDomain;
    /// # use anyhow::Result;
    ///
    /// // Assumes "my_pipeline" was previously added to the pipeline cache with `PipelineCache::create_named_compute_pipeline()`,
    /// // and that cmd was created with this cache.
    /// fn compute_pipeline<D: ExecutionDomain + ComputeSupport>(cmd: IncompleteCommandBuffer<D>) -> Result<IncompleteCommandBuffer<D>> {
    ///     // Bind the pipeline and dispatch 16 work groups in each dimension.
    ///     cmd.bind_compute_pipeline("my_pipeline")?
    ///        .dispatch(16, 16, 16)
    /// }
    /// ```
    fn dispatch(mut self, x: u32, y: u32, z: u32) -> Result<Self> {
        self = self.ensure_descriptor_state()?;
        unsafe {
            self.device.cmd_dispatch(self.handle, x, y, z);
        }
        Ok(self)
    }
    // Methods for compute commands
}
