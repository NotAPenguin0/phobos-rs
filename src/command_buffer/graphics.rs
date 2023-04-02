use anyhow::Result;
use ash::vk;

use crate::{BufferView, Error, GfxSupport, GraphicsCmdBuffer, ImageView};
use crate::command_buffer::IncompleteCommandBuffer;
use crate::core::device::ExtensionID;
use crate::domain::ExecutionDomain;

impl<D: GfxSupport + ExecutionDomain> GraphicsCmdBuffer for IncompleteCommandBuffer<'_, D> {
    /// Sets the viewport and scissor regions to the entire render area. Can only be called inside a renderpass.
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::*;
    /// # use phobos::domain::*;
    /// fn set_viewport<C: GraphicsCmdBuffer>(cmd: C) -> C {
    ///     // Now the current viewport and scissor cover the current attachment's entire area
    ///     cmd.full_viewport_scissor()
    /// }
    /// ```
    fn full_viewport_scissor(self) -> Self {
        let area = self.current_render_area;
        self.viewport(vk::Viewport {
            x: area.offset.x as f32,
            y: area.offset.y as f32,
            width: area.extent.width as f32,
            height: area.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })
        .scissor(area)
    }

    /// Sets the viewport. Directly translates to [`vkCmdSetViewport`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdSetViewport.html).
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::*;
    /// # use phobos::domain::*;
    /// fn set_viewport<C: GraphicsCmdBuffer>(cmd: C) -> C {
    ///     cmd.viewport(vk::Viewport {
    ///         x: 0.0,
    ///         y: 0.0,
    ///         width: 1920.0,
    ///         height: 1080.0,
    ///         min_depth: 0.0,
    ///         max_depth: 1.0,
    ///     })
    /// }
    /// ```
    fn viewport(self, viewport: vk::Viewport) -> Self {
        unsafe {
            self.device.cmd_set_viewport(self.handle, 0, std::slice::from_ref(&viewport));
        }
        self
    }

    /// Sets the scissor region. Directly translates to [`vkCmdSetScissor`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdSetScissor.html).
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::*;
    /// # use phobos::domain::*;
    /// fn set_scissor<C: GraphicsCmdBuffer>(cmd: C) -> C {
    ///     cmd.scissor(vk::Rect2D {
    ///         offset: Default::default(),
    ///         extent: vk::Extent2D {
    ///             width: 1920,
    ///             height: 1080
    ///         }
    ///     })
    /// }
    /// ```
    fn scissor(self, scissor: vk::Rect2D) -> Self {
        unsafe {
            self.device.cmd_set_scissor(self.handle, 0, std::slice::from_ref(&scissor));
        }
        self
    }

    /// Issue a drawcall. This will flush the current descriptor set state and actually bind the descriptor sets.
    /// Directly translates to [`vkCmdDraw`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdDraw.html).
    /// # Errors
    /// * Fails if flushing the descriptor state fails (this can happen if there is no descriptor cache).
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::*;
    /// # use phobos::domain::*;
    /// fn draw<C: GraphicsCmdBuffer>(cmd: C, vertex_buffer: &BufferView) -> Result<C> {
    ///     cmd.full_viewport_scissor()
    ///        .bind_graphics_pipeline("my_pipeline")?
    ///        .bind_vertex_buffer(0, vertex_buffer)
    ///        .draw(6, 1, 0, 0)
    /// }
    /// ```
    fn draw(mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> Result<Self> {
        self = self.ensure_descriptor_state()?;
        unsafe {
            self.device
                .cmd_draw(self.handle, vertex_count, instance_count, first_vertex, first_instance);
        }
        Ok(self)
    }

    /// Issue an indexed drawcall. This will flush the current descriptor state and actually bind the
    /// descriptor sets. Directly translates to [`vkCmdDrawIndexed`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdDrawIndexed.html).
    /// # Errors
    /// * Fails if flushing the descriptor state fails (this can happen if there is no descriptor cache).
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::*;
    /// # use phobos::domain::*;
    /// fn draw_indexed<C: GraphicsCmdBuffer>(cmd: C, vertex_buffer: &BufferView, index_buffer: &BufferView) -> Result<C> {
    ///     cmd.full_viewport_scissor()
    ///        .bind_graphics_pipeline("my_pipeline")?
    ///        .bind_vertex_buffer(0, vertex_buffer)
    ///        .bind_index_buffer(index_buffer, vk::IndexType::UINT32)
    ///        .draw_indexed(6, 1, 0, 0, 0)
    /// }
    /// ```
    fn draw_indexed(mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) -> Result<Self> {
        self = self.ensure_descriptor_state()?;
        unsafe {
            self.device.cmd_draw_indexed(
                self.handle,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        }
        Ok(self)
    }

    /// Bind a graphics pipeline by name.
    /// # Errors
    /// * Fails if the pipeline was not previously registered in the pipeline cache.
    /// * Fails if this command buffer has no pipeline cache.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use phobos::domain::ExecutionDomain;
    /// # use anyhow::Result;
    /// // Assumes "my_pipeline" was previously added to the pipeline cache with `PipelineCache::create_named_pipeline()`,
    /// // and that cmd was created with this cache.
    /// fn compute_pipeline<C: GraphicsCmdBuffer>(cmd: C) -> Result<C> {
    ///     cmd.bind_graphics_pipeline("my_pipeline")
    /// }
    /// ```
    fn bind_graphics_pipeline(mut self, name: &str) -> Result<Self> {
        let Some(mut cache) = self.pipeline_cache.clone() else { return Err(Error::NoPipelineCache.into()); };
        {
            let Some(rendering_state) = self.current_rendering_state.clone() else { return Err(Error::NoRenderpass.into()) };
            cache.with_pipeline(name, rendering_state, |pipeline|
                self.bind_pipeline_impl(pipeline.handle, pipeline.layout, pipeline.set_layouts.clone(), vk::PipelineBindPoint::GRAPHICS),
            )?;
        }
        Ok(self)
    }

    /// Binds a vertex buffer to the specified binding point. Note that currently there is no validation as to whether this
    /// binding actually exists for the given pipeline. Direct translation of [`vkCmdBindVertexBuffers`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBindVertexBuffers.html).
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::*;
    /// # use phobos::domain::*;
    /// fn draw<C: GraphicsCmdBuffer>(cmd: C, vertex_buffer: &BufferView) -> Result<C> {
    ///     cmd.bind_vertex_buffer(0, vertex_buffer)
    ///        .draw(6, 1, 0, 0)
    /// }
    /// ```
    fn bind_vertex_buffer(self, binding: u32, buffer: &BufferView) -> Self
    where
        Self: Sized, {
        let handle = unsafe { buffer.handle() };
        let offset = buffer.offset();
        unsafe {
            self.device.cmd_bind_vertex_buffers(
                self.handle,
                binding,
                std::slice::from_ref(&handle),
                std::slice::from_ref(&offset),
            )
        };
        self
    }

    /// Bind the an index buffer. The index type must match the actual type stored in the buffer.
    /// Direct translation of [`vkCmdBindIndexBuffer`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBindIndexBuffer.html)
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::*;
    /// # use phobos::domain::*;
    /// fn draw_indexed<C: GraphicsCmdBuffer>(cmd: C, vertex_buffer: &BufferView, index_buffer: &BufferView) -> Result<C> {
    ///     cmd.bind_vertex_buffer(0, vertex_buffer)
    ///        .bind_index_buffer(index_buffer, vk::IndexType::UINT32)
    ///        .draw_indexed(6, 1, 0, 0, 0)
    /// }
    /// ```
    fn bind_index_buffer(self, buffer: &BufferView, ty: vk::IndexType) -> Self
        where
            Self: Sized, {
        unsafe {
            self.device
                .cmd_bind_index_buffer(self.handle, buffer.handle(), buffer.offset(), ty);
        }
        self
    }

    /// Blit a source image to a destination image, using the specified offsets into the images and a filter. Direct and thin wrapper around
    /// [`vkCmdBlitImage`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBlitImage.html)
    fn blit_image(self, src: &ImageView, dst: &ImageView, src_offsets: &[vk::Offset3D; 2], dst_offsets: &[vk::Offset3D; 2], filter: vk::Filter) -> Self
    where
        Self: Sized, {
        let blit = vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: src.aspect(),
                mip_level: src.base_level(),
                base_array_layer: src.base_layer(),
                layer_count: src.layer_count(),
            },
            src_offsets: *src_offsets,
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: dst.aspect(),
                mip_level: dst.base_level(),
                base_array_layer: dst.base_layer(),
                layer_count: dst.layer_count(),
            },
            dst_offsets: *dst_offsets,
        };

        unsafe {
            self.device.cmd_blit_image(
                self.handle,
                src.image(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&blit),
                filter,
            );
        }
        self
    }

    /// Set the polygon mode. Only available if `VK_EXT_extended_dynamic_state3` was enabled on device creation.
    /// Equivalent to [`vkCmdSetPolygonModeEXT`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdSetPolygonModeEXT.html)
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::*;
    /// # use phobos::domain::*;
    /// fn set_polygon_mode<C: GraphicsCmdBuffer>(cmd: C) -> Result<C> {
    ///     // Subsequent drawcalls will get a wireframe view.
    ///     cmd.set_polygon_mode(vk::PolygonMode::LINE)
    /// }
    /// ```
    fn set_polygon_mode(self, mode: vk::PolygonMode) -> Result<Self> {
        let funcs = self
            .device
            .dynamic_state3()
            .ok_or::<anyhow::Error>(Error::ExtensionNotSupported(ExtensionID::ExtendedDynamicState3).into())?;
        // SAFETY: Vulkan API call. This function pointer is not null because we just verified its availability.
        unsafe {
            funcs.cmd_set_polygon_mode(self.handle, mode);
        }
        Ok(self)
    }
}
