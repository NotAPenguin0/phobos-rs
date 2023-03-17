use ash::vk;
use crate::domain::ExecutionDomain;
use crate::{BufferView, Error, GfxSupport, GraphicsCmdBuffer, ImageView};

use anyhow::Result;
use crate::command_buffer::IncompleteCommandBuffer;

impl<D: GfxSupport + ExecutionDomain> GraphicsCmdBuffer for IncompleteCommandBuffer<'_, D> {
    /// Sets the viewport and scissor regions to the entire render area. Can only be called inside a renderpass.
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
    fn viewport(self, viewport: vk::Viewport) -> Self {
        unsafe { self.device.cmd_set_viewport(self.handle, 0, std::slice::from_ref(&viewport)); }
        self
    }

    /// Sets the scissor region. Directly translates to [`vkCmdSetScissor`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdSetScissor.html).
    fn scissor(self, scissor: vk::Rect2D) -> Self {
        unsafe { self.device.cmd_set_scissor(self.handle, 0, std::slice::from_ref(&scissor)); }
        self
    }

    /// Issue a drawcall. This will flush the current descriptor set state and actually bind the descriptor sets.
    /// Directly translates to [`vkCmdDraw`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdDraw.html).
    fn draw(mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> Result<Self> {
        self = self.ensure_descriptor_state()?;
        unsafe { self.device.cmd_draw(self.handle, vertex_count, instance_count, first_vertex, first_instance); }
        Ok(self)
    }

    /// Issue an indexed drawcall. This will flush the current descriptor state and actually bind the
    /// descriptor sets. Directly translates to [`vkCmdDrawIndexed`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdDrawIndexed.html).
    fn draw_indexed(mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) -> Result<Self> {
        self = self.ensure_descriptor_state()?;
        unsafe { self.device.cmd_draw_indexed(self.handle, index_count, instance_count, first_index, vertex_offset, first_instance) }
        Ok(self)
    }

    /// Bind a graphics pipeline by name.
    /// # Errors
    /// Fails if the pipeline was not previously registered in the pipeline cache.
    fn bind_graphics_pipeline(mut self, name: &str) -> Result<Self> {
        let Some(cache) = &self.pipeline_cache else { return Err(Error::NoPipelineCache.into()); };
        {
            let mut cache = cache.lock().unwrap();
            let Some(rendering_state) = &self.current_rendering_state else { return Err(Error::NoRenderpass.into()) };
            let pipeline = cache.get_pipeline(name, &rendering_state)?;
            unsafe { self.device.cmd_bind_pipeline(self.handle, vk::PipelineBindPoint::GRAPHICS, pipeline.handle); }
            self.current_bindpoint = vk::PipelineBindPoint::GRAPHICS;
            self.current_pipeline_layout = pipeline.layout;
            self.current_set_layouts = pipeline.set_layouts.clone();
        }
        Ok(self)
    }

    /// Binds a vertex buffer to the specified binding point. Note that currently there is no validation as to whether this
    /// binding actually exists for the given pipeline. Direct translation of [`vkCmdBindVertexBuffers`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBindVertexBuffers.html).
    fn bind_vertex_buffer(self, binding: u32, buffer: &BufferView) -> Self where Self: Sized {
        unsafe { self.device.cmd_bind_vertex_buffers(self.handle, binding, std::slice::from_ref(&buffer.handle), std::slice::from_ref(&buffer.offset)) };
        self
    }

    /// Bind the an index buffer. The index type must match. Direct translation of [`vkCmdBindIndexBuffer`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBindIndexBuffer.html)
    fn bind_index_buffer(self, buffer: &BufferView, ty: vk::IndexType) -> Self where Self: Sized {
        unsafe { self.device.cmd_bind_index_buffer(self.handle, buffer.handle, buffer.offset, ty); }
        self
    }

    /// Blit a source image to a destination image, using the specified offsets into the images and a filter. Direct and thin wrapper around
    /// [`vkCmdBlitImage`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBlitImage.html)
    fn blit_image(self, src: &ImageView, dst: &ImageView, src_offsets: &[vk::Offset3D; 2], dst_offsets: &[vk::Offset3D; 2], filter: vk::Filter) -> Self where Self: Sized {
        let blit = vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: src.aspect,
                mip_level: src.base_level,
                base_array_layer: src.base_layer,
                layer_count: src.layer_count,
            },
            src_offsets: *src_offsets,
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: dst.aspect,
                mip_level: dst.base_level,
                base_array_layer: dst.base_layer,
                layer_count: dst.layer_count,
            },
            dst_offsets: *dst_offsets
        };

        unsafe {
            self.device.cmd_blit_image(
                self.handle,
                src.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&blit), filter);
        }
        self
    }
}