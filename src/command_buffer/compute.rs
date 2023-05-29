//! Contains implementations of the compute domain for command buffers

use anyhow::Result;
use ash::vk;

use crate::{Allocator, ComputeCmdBuffer, ComputeSupport};
use crate::command_buffer::IncompleteCommandBuffer;
use crate::core::device::ExtensionID;
use crate::query_pool::{AccelerationStructurePropertyQuery, QueryPool};
use crate::raytracing::*;
use crate::sync::domain::ExecutionDomain;

impl<D: ComputeSupport + ExecutionDomain, A: Allocator> ComputeCmdBuffer
    for IncompleteCommandBuffer<'_, D, A>
{
    /// Sets the current compute pipeline by looking up the given name in the pipeline cache.
    /// # Errors
    /// - Fails if the given name is not a valid compute pipeline name.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use phobos::sync::domain::ExecutionDomain;
    /// # use anyhow::Result;
    /// // Assumes "my_pipeline" was previously added to the pipeline cache with `PipelineCache::create_named_compute_pipeline()`.
    /// fn compute_pipeline<D: ExecutionDomain + ComputeSupport>(cmd: IncompleteCommandBuffer<D>) -> Result<IncompleteCommandBuffer<D>> {
    ///     cmd.bind_compute_pipeline("my_pipeline")
    /// }
    /// ```
    fn bind_compute_pipeline(mut self, name: &str) -> Result<Self>
    where
        Self: Sized, {
        let cache = self.pipeline_cache.clone();
        cache.with_compute_pipeline(name, |pipeline| {
            self.bind_pipeline_impl(
                pipeline.handle,
                pipeline.layout,
                pipeline.set_layouts.clone(),
                vk::PipelineBindPoint::COMPUTE,
            )
        })?;

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
    /// * Fails if updating the descriptor state fails.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use phobos::sync::domain::ExecutionDomain;
    /// # use anyhow::Result;
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

    /// Build a single acceleration structure. This is a write operation to the acceleration structure, so
    /// it must be synchronized.
    fn build_acceleration_structure(self, info: &AccelerationStructureBuildInfo) -> Result<Self>
    where
        Self: Sized, {
        self.build_acceleration_structures(std::slice::from_ref(info))
    }

    /// Build multiple acceleration structures in a single Vulkan command.
    /// This is a write operation to the acceleration structures, so it must be
    /// synchronized
    fn build_acceleration_structures(
        self,
        info: &[AccelerationStructureBuildInfo],
    ) -> Result<Self>
    where
        Self: Sized, {
        self.device
            .require_extension(ExtensionID::AccelerationStructure)?;
        let as_vk = info.iter().map(|info| info.as_vulkan()).collect::<Vec<_>>();
        let geometries = as_vk
            .iter()
            .map(|(geometry, _)| *geometry)
            .collect::<Vec<_>>();
        let infos = as_vk.iter().map(|(_, ranges)| *ranges).collect::<Vec<_>>();
        unsafe {
            self.device
                .acceleration_structure()
                .unwrap()
                .cmd_build_acceleration_structures(
                    self.handle,
                    geometries.as_slice(),
                    infos.as_slice(),
                );
        }

        Ok(self)
    }

    /// Compact an acceleration structure. This is read operation on `src`, and a write operation on `dst`, which must both be
    /// externally synchronized.
    fn compact_acceleration_structure(
        self,
        src: &AccelerationStructure,
        dst: &AccelerationStructure,
    ) -> Result<Self> {
        self.device
            .require_extension(ExtensionID::AccelerationStructure)?;
        let fns = self.device.acceleration_structure().unwrap();
        let info = vk::CopyAccelerationStructureInfoKHR {
            s_type: vk::StructureType::COPY_ACCELERATION_STRUCTURE_INFO_KHR,
            p_next: std::ptr::null(),
            src: unsafe { src.handle() },
            dst: unsafe { dst.handle() },
            mode: vk::CopyAccelerationStructureModeKHR::COMPACT,
        };
        unsafe {
            fns.cmd_copy_acceleration_structure(self.handle, &info);
        };
        Ok(self)
    }

    /// Write acceleration structure properties to the query pool. The property written depends on
    /// the type of the query pool passed in and is automatically inferred to be
    /// [`Q::QUERY_TYPE`].
    ///
    /// This is a read operation on the acceleration structures, so it must be externally synchronized.
    fn write_acceleration_structures_properties<Q: AccelerationStructurePropertyQuery>(
        self,
        src: &[AccelerationStructure],
        query_pool: &mut QueryPool<Q>,
    ) -> Result<Self> {
        self.device
            .require_extension(ExtensionID::AccelerationStructure)?;
        let fns = self.device.acceleration_structure().unwrap();

        let handles = src
            .iter()
            .map(|a| unsafe { a.handle() })
            .collect::<Vec<_>>();
        let first = query_pool.current();
        unsafe {
            fns.cmd_write_acceleration_structures_properties(
                self.handle,
                handles.as_slice(),
                Q::QUERY_TYPE,
                query_pool.handle(),
                first,
            );
        }
        // Update the query pool to point to the next available query
        handles.iter().for_each(|_| {
            query_pool.next();
        });
        Ok(self)
    }

    /// Write acceleration structure properties to the query pool. The property written depends on
    /// the type of the query pool passed in and is automatically inferred to be
    /// [`Q::QUERY_TYPE`].
    ///
    /// This is a read operation on the acceleration structure, so it must be externally synchronized.
    fn write_acceleration_structure_properties<Q: AccelerationStructurePropertyQuery>(
        self,
        src: &AccelerationStructure,
        query_pool: &mut QueryPool<Q>,
    ) -> Result<Self> {
        self.write_acceleration_structures_properties(std::slice::from_ref(src), query_pool)
    }
}
