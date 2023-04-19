use std::sync::MutexGuard;

use anyhow::Result;
use ash::vk;

use crate::{BufferView, DescriptorCache, Device, domain, ExecutionManager, ImageView, PipelineCache};
use crate::command_buffer::CommandBuffer;
use crate::core::queue::Queue;
use crate::domain::ExecutionDomain;
use crate::query_pool::{AccelerationStructurePropertyQuery, QueryPool};
use crate::raytracing::acceleration_structure::{AccelerationStructure, AccelerationStructureBuildInfo};

/// Trait representing a command buffer that supports transfer commands.
pub trait TransferCmdBuffer {
    /// Copy one buffer view to another buffer view.
    /// Both views must have the same length.
    fn copy_buffer(self, src: &BufferView, dst: &BufferView) -> Result<Self>
        where
            Self: Sized;
    /// Copy a buffer to an image.
    fn copy_buffer_to_image(self, src: &BufferView, dst: &ImageView) -> Result<Self>
        where
            Self: Sized;
}

/// Trait representing a command buffer that supports graphics commands.
pub trait GraphicsCmdBuffer: TransferCmdBuffer {
    /// Automatically set viewport and scissor region to the entire render area
    fn full_viewport_scissor(self) -> Self;
    /// Sets the viewport. The equivalent of `vkCmdSetViewport`.
    fn viewport(self, viewport: vk::Viewport) -> Self;
    /// Sets the scissor region. Equivalent of `vkCmdSetScissor`.
    fn scissor(self, scissor: vk::Rect2D) -> Self;
    /// Record a single drawcall. Equivalent of `vkCmdDraw`.
    fn draw(self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> Result<Self>
    where
        Self: Sized;
    /// Record a single indexed drawcall. Equivalent of `vkCmdDrawIndexed`
    fn draw_indexed(self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) -> Result<Self>
    where
        Self: Sized;
    /// Bind a graphics pipeline with a given name.
    /// # Errors
    /// This function can report an error in case the pipeline name is not registered in the cache.
    fn bind_graphics_pipeline(self, name: &str) -> Result<Self>
    where
        Self: Sized;
    /// Bind a vertex buffer to the given vertex input binding.
    /// Equivalent of `vkCmdBindVertexBuffer`
    fn bind_vertex_buffer(self, binding: u32, buffer: &BufferView) -> Self
    where
        Self: Sized;
    /// Bind an index buffer. Equivalent of `vkCmdBindIndexBuffer`
    fn bind_index_buffer(self, buffer: &BufferView, ty: vk::IndexType) -> Self
    where
        Self: Sized;
    /// Blit an image. Equivalent to `vkCmdBlitImage`
    fn blit_image(self, src: &ImageView, dst: &ImageView, src_offsets: &[vk::Offset3D; 2], dst_offsets: &[vk::Offset3D; 2], filter: vk::Filter) -> Self
    where
        Self: Sized;

    /// Set the polygon mode. Only available if VK_EXT_extended_dynamic_state3 was enabled. Equivalent to `vkCmdSetPolygonMode`
    fn set_polygon_mode(self, mode: vk::PolygonMode) -> Result<Self>
    where
        Self: Sized;
}

/// Trait representing a command buffer that supports compute commands.
pub trait ComputeCmdBuffer: TransferCmdBuffer {
    /// Bind a compute pipeline with a given name.
    /// # Errors
    /// This function can report an error in case the pipeline name is not registered in the cache.
    fn bind_compute_pipeline(self, name: &str) -> Result<Self>
        where
            Self: Sized;

    /// Dispatch a compute invocation. See `vkCmdDispatch`
    fn dispatch(self, x: u32, y: u32, z: u32) -> Result<Self>
        where
            Self: Sized;

    fn build_acceleration_structure(self, info: &AccelerationStructureBuildInfo) -> Result<Self>
        where
            Self: Sized;

    fn build_acceleration_structures(self, info: &[AccelerationStructureBuildInfo]) -> Result<Self>
        where
            Self: Sized;

    fn compact_acceleration_structure(self, src: &AccelerationStructure, dst: &AccelerationStructure) -> Result<Self>
        where
            Self: Sized;

    fn write_acceleration_structures_properties<Q: AccelerationStructurePropertyQuery>(
        self,
        src: &[AccelerationStructure],
        query_pool: &mut QueryPool<Q>,
    ) -> Result<Self>
        where
            Self: Sized;

    fn write_acceleration_structure_properties<Q: AccelerationStructurePropertyQuery>(
        self,
        src: &AccelerationStructure,
        query_pool: &mut QueryPool<Q>,
    ) -> Result<Self>
        where
            Self: Sized;
}

/// Completed command buffer
pub trait CmdBuffer {
    /// Delete the command buffer immediately.
    /// This is marked unsafe because there is no guarantee that the command buffer is not in use.
    /// # Safety
    /// The caller must ensure this command buffer is not in use on the GPU.
    unsafe fn delete(&mut self, exec: ExecutionManager) -> Result<()>;
}

/// Incomplete command buffer
pub trait IncompleteCmdBuffer<'q> {
    type Domain: ExecutionDomain;

    fn new(
        device: Device,
        queue_lock: MutexGuard<'q, Queue>,
        handle: vk::CommandBuffer,
        flags: vk::CommandBufferUsageFlags,
        pipelines: Option<PipelineCache>,
        descriptors: Option<DescriptorCache>,
    ) -> Result<Self>
    where
        Self: Sized;
    fn finish(self) -> Result<CommandBuffer<Self::Domain>>;
}

/// Whether this domain supports graphics operations. Per the Vulkan spec, this
/// also implies it supports transfer operations.
pub trait GfxSupport: TransferSupport {}

/// Whether this domain supports transfer operations.
pub trait TransferSupport {}

/// Whether this domain supports compute operations. Per the Vulkan spec, this
/// also implies it supports transfer operations.
pub trait ComputeSupport: TransferSupport {}

impl GfxSupport for domain::Graphics {}
impl GfxSupport for domain::All {}
impl TransferSupport for domain::Graphics {}
impl TransferSupport for domain::Transfer {}
impl TransferSupport for domain::Compute {}
impl TransferSupport for domain::All {}
impl ComputeSupport for domain::Compute {}
impl ComputeSupport for domain::All {}
