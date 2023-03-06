//! Most functions in this module are a relatively thin wrapper over Vulkan commands.
//!
//! # Domains
//!
//! The most important feature is that of execution domains. Commands are divided into four domains:
//! - Transfer: All transfer and copy related commands.
//! - Graphics: All graphics and rendering related commands.
//! - Compute: GPU compute commands, most notably `vkCmdDispatch`
//! - All: All of the above.
//!
//! This concept abstracts over that of queue families. A command buffer over a domain is allocated from a queue that supports all operations
//! on its domain, and as few other domains (to try to catch dedicated transfer/async compute queues). For this reason, always try to
//! allocate from the most restrictive domain as you can.
//!
//! # Incomplete command buffers
//!
//! Vulkan command buffers need to call `vkEndCommandBuffer` before they can be submitted. After this call, no more commands should be
//! recorded to it. For this reason, we expose two command buffer types. The [`IncompleteCommandBuffer`] still accepts commands, and can only
//! be converted into a [`CommandBuffer`] by calling [`IncompleteCommandBuffer::finish`]. This turns it into a complete commad buffer, which can
//! be submitted to the execution manager.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex, MutexGuard};
use crate::execution_manager::domain::*;
use crate::execution_manager::domain;

use ash::vk;
use crate::{BufferView, DebugMessenger, DescriptorCache, DescriptorSet, DescriptorSetBinding, Device, Error, ExecutionManager, ImageView, PipelineCache, PipelineRenderingInfo, Queue};

use anyhow::Result;
use ash::vk::{Filter, Offset3D};

pub(crate) struct RenderingAttachmentInfo {
    pub image_view: ImageView,
    pub image_layout: vk::ImageLayout,
    pub resolve_mode: Option<vk::ResolveModeFlags>,
    pub resolve_image_view: Option<ImageView>,
    pub resolve_image_layout: Option<vk::ImageLayout>,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub clear_value: vk::ClearValue,
}

pub(crate) struct RenderingInfo {
    pub flags: vk::RenderingFlags,
    pub render_area: vk::Rect2D,
    pub layer_count: u32,
    pub view_mask: u32,
    pub color_attachments: Vec<RenderingAttachmentInfo>,
    pub depth_attachment: Option<RenderingAttachmentInfo>,
    pub stencil_attachment: Option<RenderingAttachmentInfo>,
}

/// Trait representing a command buffer that supports graphics commands.
pub trait GraphicsCmdBuffer : TransferCmdBuffer {
    /// Sets the viewport. The equivalent of `vkCmdSetViewport`.
    fn viewport(self, viewport: vk::Viewport) -> Self;
    /// Sets the scissor region. Equivalent of `vkCmdSetScissor`.
    fn scissor(self, scissor: vk::Rect2D) -> Self;
    /// Record a single drawcall. Equivalent of `vkCmdDraw`.
    fn draw(self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> Self;
    /// Record a single indexed drawcall. Equivalent of `vkCmdDrawIndexed`
    fn draw_indexed(self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) -> Self;
    /// Bind a graphics pipeline with a given name. This is looked up from the given pipeline cache.
    /// # Errors
    /// This function can report an error in case the pipeline name is not registered in the cache.
    fn bind_graphics_pipeline(self, name: &str, cache: Arc<Mutex<PipelineCache>>) -> Result<Self> where Self: Sized;
    /// Bind a vertex buffer to the given vertex input binding.
    /// Equivalent of `vkCmdBindVertexBuffer`
    fn bind_vertex_buffer(self, binding: u32, buffer: BufferView) -> Self where Self: Sized;
    /// Bind an index buffer. Equivalent of `vkCmdBindIndexBuffer`
    fn bind_index_buffer(self, buffer: BufferView, ty: vk::IndexType) -> Self where Self: Sized;

    fn blit_image(self, src: &ImageView, dst: &ImageView, src_offsets: &[vk::Offset3D; 2], dst_offsets: &[vk::Offset3D; 2], filter: vk::Filter) -> Self where Self: Sized;
}

/// Trait representing a command buffer that supports transfer commands.
pub trait TransferCmdBuffer {
    fn copy_buffer(self, src: &BufferView, dst: &BufferView) -> Result<Self> where Self: Sized;
    fn copy_buffer_to_image(self, src: &BufferView, dst: &ImageView) -> Result<Self> where Self: Sized;
}

/// Trait representing a command buffer that supports compute commands.
pub trait ComputeCmdBuffer : TransferCmdBuffer {
    
}

/// This struct represents a finished command buffer. This command buffer can't be recorded to anymore.
/// It can only be obtained by calling finish() on an incomplete command buffer;
pub struct CommandBuffer<D: ExecutionDomain> {
    pub(crate) handle: vk::CommandBuffer,
    _domain: PhantomData<D>,
}

/// Completed command buffer
pub trait CmdBuffer {
    /// Delete the command buffer immediately.
    /// This is marked unsafe because there is no guarantee that the command buffer is not in use.
    unsafe fn delete(&mut self, exec: Arc<ExecutionManager>) -> Result<()>;
}

/// Incomplete command buffer
pub trait IncompleteCmdBuffer<'q> {
    type Domain: ExecutionDomain;

    fn new(device: Arc<Device>,
           queue_lock: MutexGuard<'q, Queue>,
           handle: vk::CommandBuffer,
           flags: vk::CommandBufferUsageFlags)
        -> Result<Self> where Self: Sized;
    fn finish(self) -> Result<CommandBuffer<Self::Domain>>;
}

/// This struct represents an incomplete command buffer.
/// This is a command buffer that has not been called [`IncompleteCommandBuffer::finish()`] on yet.
/// Calling this method will turn it into an immutable command buffer which can then be submitted
/// to the queue it was allocated from. See also [`ExecutionManager`].
///
/// # Example
/// ```
/// use phobos::{domain, ExecutionManager};
///
/// let exec = ExecutionManager::new(device.clone(), &physical_device);
/// let cmd = exec.on_domain::<domain::All>()?
///               // record commands to this command buffer
///               // ...
///               // convert into a complete command buffer by calling finish().
///               // This allows the command buffer to be submitted.
///               .finish();
/// ```
#[derive(Derivative)]
#[derivative(Debug)]
pub struct IncompleteCommandBuffer<'q, D: ExecutionDomain> {
    #[derivative(Debug="ignore")]
    device: Arc<Device>,
    handle: vk::CommandBuffer,
    queue_lock: MutexGuard<'q, Queue>,
    current_pipeline_layout: vk::PipelineLayout,
    current_set_layouts: Vec<vk::DescriptorSetLayout>,
    current_bindpoint: vk::PipelineBindPoint, // TODO: Note: technically not correct
    current_rendering_state: Option<PipelineRenderingInfo>,
    _domain: PhantomData<D>,
}

impl<'q, D: ExecutionDomain> IncompleteCmdBuffer<'q> for IncompleteCommandBuffer<'q, D> {
    type Domain = D;

    fn new(device: Arc<Device>, queue_lock: MutexGuard<'q, Queue>, handle: vk::CommandBuffer, flags: vk::CommandBufferUsageFlags) -> Result<Self> {
        unsafe { device.begin_command_buffer(
            handle,
            &vk::CommandBufferBeginInfo::builder().flags(flags))?
        };
        Ok(IncompleteCommandBuffer {
            device: device.clone(),
            handle,
            queue_lock,
            current_pipeline_layout: vk::PipelineLayout::null(),
            current_set_layouts: vec![],
            current_bindpoint: vk::PipelineBindPoint::default(),
            current_rendering_state: None,
            _domain: PhantomData
        })
    }

    /// Finish recording a command buffer and move its contents into a finished
    /// command buffer that can be submitted
    fn finish(self) -> Result<CommandBuffer<D>> {
        unsafe { self.device.end_command_buffer(self.handle)? }
        Ok(CommandBuffer {
            handle: self.handle,
            _domain: PhantomData,
        })
    }
}

impl<'q, D: ExecutionDomain> CmdBuffer for CommandBuffer<D> {
    unsafe fn delete(&mut self, exec: Arc<ExecutionManager>) -> Result<()> {
        let queue = exec.get_queue::<D>().ok_or(Error::NoCapableQueue)?;
        let handle = self.handle;
        self.handle = vk::CommandBuffer::null();
        queue.free_command_buffer::<Self>(handle)
    }
}

impl<D: ExecutionDomain> IncompleteCommandBuffer<'_, D> {
    /// Obtain a reference to a descriptor set inside the given cache that stores the requested bindings.
    /// This potentially allocates a new descriptor set and writes to it.
    /// # Lifetime
    /// The returned descriptor set lives as long as the cache allows it to live, so it should not be stored across multiple frames.
    /// # Errors
    /// - This function can potentially error if allocating the descriptor set fails. The descriptor cache has builtin ways to handle
    /// a full descriptor pool, but other errors are passed through.
    /// - This function errors if a requested set was not specified in the current pipeline's pipeline layout.
    /// - This function errors if no pipeline is bound.
    pub fn get_descriptor_set<'a>(&mut self, set: u32, mut bindings: DescriptorSetBinding, cache: &'a mut DescriptorCache) -> Result<&'a DescriptorSet> {
        let layout = self.current_set_layouts.get(set as usize).ok_or(Error::NoDescriptorSetLayout)?;
        bindings.layout = layout.clone();
        cache.get_descriptor_set(bindings)
    }

    /// Bind a descriptor set to the command buffer. This descriptor set can be obtained by calling
    /// [`Self::get_descriptor_set`]. Note that the index should be the same as the one used in get_descriptor_set.
    /// To safely do this, prefer using [`Self::bind_new_descriptor_set`] to combine these two functions into one call.
    pub fn bind_descriptor_set(self, index: u32, set: &DescriptorSet) -> Self {
        unsafe {
            self.device.cmd_bind_descriptor_sets(self.handle, self.current_bindpoint, self.current_pipeline_layout,
                                     index,
                                std::slice::from_ref(&set.handle),
                               &[]); }
        self
    }

    /// Obtain a descriptor set from the cache, and immediately bind it to the given index.
    /// # Errors
    /// - This function can error if locking the descriptor cache fails
    /// - This function can error if allocating the descriptor set fails.
    /// # Example
    /// ```
    /// use phobos::{DescriptorSetBuilder, domain, ExecutionManager};
    /// let exec = ExecutionManager::new(device.clone(), &physical_device);    ///
    /// let cmd = exec.on_domain::<domain::All>()?
    ///     .bind_graphics_pipeline("my_pipeline", pipeline_cache.clone())
    ///     .bind_new_descriptor_set(0, descriptor_cache.clone(), DescriptorSetBuilder::new()
    ///         .bind_sampled_image(0, image_view, &sampler)
    ///         .build())
    ///     // ...
    ///     .finish();
    ///
    /// ```
    pub fn bind_new_descriptor_set(mut self, index: u32, cache: Arc<Mutex<DescriptorCache>>, bindings: DescriptorSetBinding, ) -> Result<Self> {
        let mut cache = cache.lock().or_else(|_| Err(anyhow::Error::from(Error::PoisonError)))?;
        let set = self.get_descriptor_set(index, bindings, &mut cache)?;
        Ok(self.bind_descriptor_set(index, set))
    }

    /// Transitions an image layout.
    /// Generally you will not need to call this function manually,
    /// using the render graph api you can do most transitions automatically.
    pub fn transition_image(self, image: &ImageView, src_stage: vk::PipelineStageFlags, dst_stage: vk::PipelineStageFlags,
                            from: vk::ImageLayout, to: vk::ImageLayout,
                            src_access: vk::AccessFlags, dst_access: vk::AccessFlags) -> Self {
        let barrier = vk::ImageMemoryBarrier::builder()
            .image(image.image)
            .subresource_range(image.subresource_range())
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .old_layout(from)
            .new_layout(to)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .build();
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.handle,
                src_stage,
                dst_stage,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                std::slice::from_ref(&barrier)
            );
        }

        self
    }

    /// vkCmdPipelineBarrier2. Prefer using this over regular pipeline barriers if possible, to make
    /// full use of VK_KHR_SYNCHRONIZATION_2.
    pub fn pipeline_barrier_2(self, dependency: &vk::DependencyInfo) -> Self {
        unsafe {
            self.device.cmd_pipeline_barrier2(self.handle, &dependency);
        }

        self
    }

    pub(crate) fn begin_rendering(mut self, info: &RenderingInfo) -> Self {
        let map_attachment = |attachment: &RenderingAttachmentInfo| {
            vk::RenderingAttachmentInfo {
                s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
                p_next: std::ptr::null(),
                image_view: attachment.image_view.handle,
                image_layout: attachment.image_layout,
                resolve_mode: attachment.resolve_mode.unwrap_or(vk::ResolveModeFlagsKHR::NONE),
                resolve_image_view: match &attachment.resolve_image_view {
                    Some(view) => view.handle,
                    None => vk::ImageView::null()
                },
                resolve_image_layout: attachment.resolve_image_layout.unwrap_or(vk::ImageLayout::UNDEFINED),
                load_op: attachment.load_op,
                store_op: attachment.store_op,
                clear_value: attachment.clear_value,
            }
        };

        let color_attachments = info.color_attachments.iter().map(map_attachment)
            .collect::<Vec<_>>();
        let depth_attachment = info.depth_attachment.as_ref().map(map_attachment);
        let stencil_attachment = info.stencil_attachment.as_ref().map(map_attachment);
        let vk_info = vk::RenderingInfo {
            s_type: vk::StructureType::RENDERING_INFO,
            p_next: std::ptr::null(),
            flags: info.flags,
            render_area: info.render_area,
            layer_count: info.layer_count,
            view_mask: info.view_mask,
            color_attachment_count: color_attachments.len() as u32,
            p_color_attachments: color_attachments.as_ptr(),
            p_depth_attachment: match depth_attachment {
                Some(attachment) => &attachment,
                None => std::ptr::null()
            },
            p_stencil_attachment: match stencil_attachment {
                Some(attachment) => &attachment,
                None => std::ptr::null()
            },
        };

        unsafe {
            self.device.cmd_begin_rendering(self.handle, &vk_info);
        }

        self.current_rendering_state = Some(PipelineRenderingInfo {
            view_mask: info.view_mask,
            color_formats: info.color_attachments.iter().map(|attachment| attachment.image_view.format).collect(),
            depth_format: info.depth_attachment.as_ref().map(|attachment| attachment.image_view.format),
            stencil_format: info.stencil_attachment.as_ref().map(|attachment| attachment.image_view.format),
        });

        self
    }

    pub(crate) fn end_rendering(mut self) -> Self {
        unsafe {
            self.device.cmd_end_rendering(self.handle);
        }
        self.current_rendering_state = None;

        self
    }

    #[cfg(feature="debug-markers")]
    pub fn begin_label(self, label: vk::DebugUtilsLabelEXT, debug: &DebugMessenger) -> Self {
        unsafe {
            debug.functions.cmd_begin_debug_utils_label(self.handle, &label);
        }
        self
    }

    pub fn end_label(self, debug: &DebugMessenger) -> Self {
        unsafe {
            debug.functions.cmd_end_debug_utils_label(self.handle);
        }
        self
    }

    pub unsafe fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }
}

trait GfxSupport : TransferSupport {}
trait TransferSupport {}
trait ComputeSupport : TransferSupport {}

impl GfxSupport for domain::Graphics {}
impl GfxSupport for domain::All {}
impl TransferSupport for domain::Graphics {}
impl TransferSupport for domain::Transfer {}
impl TransferSupport for domain::Compute {}
impl TransferSupport for domain::All {}
impl ComputeSupport for domain::Compute {}
impl ComputeSupport for domain::All {}

impl<D: GfxSupport + ExecutionDomain> GraphicsCmdBuffer for IncompleteCommandBuffer<'_, D> {
    fn viewport(self, viewport: vk::Viewport) -> Self {
        unsafe { self.device.cmd_set_viewport(self.handle, 0, std::slice::from_ref(&viewport)); }
        self
    }

    fn scissor(self, scissor: vk::Rect2D) -> Self {
        unsafe { self.device.cmd_set_scissor(self.handle, 0, std::slice::from_ref(&scissor)); }
        self
    }

    fn draw(self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> Self {
        unsafe { self.device.cmd_draw(self.handle, vertex_count, instance_count, first_vertex, first_instance); }
        self
    }

    fn draw_indexed(self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) -> Self {
        unsafe { self.device.cmd_draw_indexed(self.handle, index_count, instance_count, first_index, vertex_offset, first_instance) }
        self
    }

    fn bind_graphics_pipeline(mut self, name: &str, cache: Arc<Mutex<PipelineCache>>) -> Result<Self> {
        let mut cache = cache.lock().unwrap();
        let Some(rendering_state) = &self.current_rendering_state else { return Err(Error::NoRenderpass.into()) };
        let pipeline = cache.get_pipeline(name, &rendering_state)?;
        unsafe { self.device.cmd_bind_pipeline(self.handle, vk::PipelineBindPoint::GRAPHICS, pipeline.handle); }
        self.current_bindpoint = vk::PipelineBindPoint::GRAPHICS;
        self.current_pipeline_layout = pipeline.layout;
        self.current_set_layouts = pipeline.set_layouts.clone();
        Ok(self)
    }

    fn bind_vertex_buffer(self, binding: u32, buffer: BufferView) -> Self where Self: Sized {
        unsafe { self.device.cmd_bind_vertex_buffers(self.handle, binding, std::slice::from_ref(&buffer.handle), std::slice::from_ref(&buffer.offset)) };
        self
    }

    fn bind_index_buffer(self, buffer: BufferView, ty: vk::IndexType) -> Self where Self: Sized {
        unsafe { self.device.cmd_bind_index_buffer(self.handle, buffer.handle, buffer.offset, ty); }
        self
    }

    fn blit_image(self, src: &ImageView, dst: &ImageView, src_offsets: &[Offset3D; 2], dst_offsets: &[Offset3D; 2], filter: Filter) -> Self where Self: Sized {
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

impl<D: TransferSupport + ExecutionDomain> TransferCmdBuffer for IncompleteCommandBuffer<'_, D> {
    fn copy_buffer(self, src: &BufferView, dst: &BufferView) -> Result<Self> {
        if src.size != dst.size {
            return Err(Error::InvalidBufferCopy.into());
        }

        let copy = vk::BufferCopy {
            src_offset: src.offset,
            dst_offset: dst.offset,
            size: src.size
        };

        unsafe { self.device.cmd_copy_buffer(self.handle, src.handle, dst.handle, std::slice::from_ref(&copy)); }

        Ok(self)
    }

    fn copy_buffer_to_image(self, src: &BufferView, dst: &ImageView) -> Result<Self> where Self: Sized {
        let copy = vk::BufferImageCopy {
            buffer_offset: src.offset,
            buffer_row_length: dst.size.width,
            buffer_image_height: dst.size.height,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: dst.aspect,
                mip_level: dst.base_level,
                base_array_layer: dst.base_layer,
                layer_count: dst.layer_count,
            },
            image_offset: Default::default(),
            image_extent: dst.size,
        };

        unsafe { self.device.cmd_copy_buffer_to_image(self.handle, src.handle, dst.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, std::slice::from_ref(&copy)); }

        Ok(self)
    }
}

impl<D: ComputeSupport + ExecutionDomain> ComputeCmdBuffer for IncompleteCommandBuffer<'_, D> {
    // Methods for compute commands
}