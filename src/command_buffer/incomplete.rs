use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::Result;
use ash::vk;

use crate::{BufferView, DebugMessenger, DescriptorCache, DescriptorSet, DescriptorSetBuilder, Device, Error, ImageView, IncompleteCmdBuffer, PhysicalResourceBindings, PipelineCache, PipelineStage, Sampler, VirtualResource};
use crate::command_buffer::{CommandBuffer, IncompleteCommandBuffer};
use crate::command_buffer::state::{RenderingAttachmentInfo, RenderingInfo};
use crate::core::queue::Queue;
use crate::descriptor::descriptor_set::DescriptorSetBinding;
use crate::domain::ExecutionDomain;
use crate::pipeline::create_info::PipelineRenderingInfo;

impl<'q, D: ExecutionDomain> IncompleteCmdBuffer<'q> for IncompleteCommandBuffer<'q, D> {
    type Domain = D;

    fn new(
        device: Arc<Device>,
        queue_lock: MutexGuard<'q, Queue>,
        handle: vk::CommandBuffer,
        flags: vk::CommandBufferUsageFlags,
        pipelines: Option<Arc<Mutex<PipelineCache>>>,
        descriptors: Option<Arc<Mutex<DescriptorCache>>>,
    ) -> Result<Self> {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: std::ptr::null(),
                flags,
                p_inheritance_info: std::ptr::null(),
            };
            device.begin_command_buffer(handle, &begin_info)?;
        };
        Ok(IncompleteCommandBuffer {
            device,
            handle,
            queue_lock,
            current_pipeline_layout: vk::PipelineLayout::null(),
            current_set_layouts: vec![],
            current_bindpoint: vk::PipelineBindPoint::default(),
            current_rendering_state: None,
            current_render_area: Default::default(),
            current_descriptor_sets: None,
            descriptor_state_needs_update: false,
            descriptor_cache: descriptors,
            pipeline_cache: pipelines,
            _domain: PhantomData,
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
    pub(super) fn get_descriptor_set<'a>(&mut self, set: u32, mut bindings: DescriptorSetBinding, cache: &'a mut DescriptorCache) -> Result<&'a DescriptorSet> {
        let layout = self.current_set_layouts.get(set as usize).ok_or(Error::NoDescriptorSetLayout)?;
        bindings.layout = layout.clone();
        cache.get_descriptor_set(bindings)
    }

    /// Bind a descriptor set to the command buffer. This descriptor set can be obtained by calling
    /// [`Self::get_descriptor_set`]. Note that the index should be the same as the one used in get_descriptor_set.
    /// To safely do this, prefer using [`Self::bind_new_descriptor_set`] to combine these two functions into one call.
    pub(super) fn bind_descriptor_set(&self, index: u32, set: &DescriptorSet) {
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.handle,
                self.current_bindpoint,
                self.current_pipeline_layout,
                index,
                std::slice::from_ref(&set.handle),
                &[],
            );
        }
    }

    pub(super) fn modify_descriptor_set(&mut self, set: u32, f: impl FnOnce(&mut DescriptorSetBuilder) -> Result<()>) -> Result<()> {
        if self.current_descriptor_sets.is_none() {
            self.current_descriptor_sets = Some(HashMap::new());
        }

        match self.current_descriptor_sets.as_mut().unwrap().entry(set) {
            Entry::Occupied(mut entry) => {
                f(entry.get_mut())?;
            }
            Entry::Vacant(entry) => {
                let mut builder = DescriptorSetBuilder::new();
                f(&mut builder)?;
                entry.insert(builder);
            }
        };
        self.descriptor_state_needs_update = true;
        Ok(())
    }

    pub(super) fn ensure_descriptor_state(mut self) -> Result<Self> {
        // No need to do anything
        if !self.descriptor_state_needs_update {
            return Ok(self);
        }

        let Some(cache) = &self.descriptor_cache else { return Err(Error::NoDescriptorCache.into()); };
        {
            let mut cache = cache.lock().unwrap();
            for (index, builder) in self.current_descriptor_sets.take().unwrap() {
                let mut info = builder.build();
                info.layout = *self.current_set_layouts.get(index as usize).unwrap();
                let set = cache.get_descriptor_set(info)?;
                self.bind_descriptor_set(index, set);
            }
        }

        // We updated all our descriptor sets, were good now.
        self.descriptor_state_needs_update = false;
        Ok(self)
    }

    /// Obtain a descriptor set from the cache, and immediately bind it to the given index.
    /// # Errors
    /// - This function can error if locking the descriptor cache fails
    /// - This function can error if allocating the descriptor set fails.
    /// # Example
    /// ```
    /// use phobos::prelude::*;
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
    #[deprecated(since = "0.5.0", note = "Use the new bind_xxx functions of the command buffer.")]
    pub fn bind_new_descriptor_set(mut self, index: u32, cache: &Arc<Mutex<DescriptorCache>>, bindings: DescriptorSetBinding) -> Result<Self> {
        let mut cache = cache.lock().or_else(|_| Err(anyhow::Error::from(Error::PoisonError)))?;
        let set = self.get_descriptor_set(index, bindings, &mut cache)?;
        self.bind_descriptor_set(index, set);
        Ok(self)
    }

    /// Clears all currently bound descriptors.
    pub fn forget_descriptor_state(mut self) -> Self {
        self.current_descriptor_sets = None;
        self.descriptor_state_needs_update = true;
        self
    }

    /// Resolve a virtual resource from the given bindings, and bind it as a sampled image to the given slot.
    /// # Errors
    /// Fails if the virtual resource has no binding associated to it.
    pub fn resolve_and_bind_sampled_image(
        mut self,
        set: u32,
        binding: u32,
        resource: &VirtualResource,
        sampler: &Sampler,
        bindings: &PhysicalResourceBindings,
    ) -> Result<Self> {
        self.modify_descriptor_set(set, |builder| {
            builder.resolve_and_bind_sampled_image(binding, resource, sampler, bindings)
        })?;
        Ok(self)
    }

    /// Binds a combined image + sampler to the specified slot.
    pub fn bind_sampled_image(mut self, set: u32, binding: u32, image: &ImageView, sampler: &Sampler) -> Result<Self> {
        self.modify_descriptor_set(set, |builder| {
            builder.bind_sampled_image(binding, image, sampler);
            Ok(())
        })?;
        Ok(self)
    }

    /// Binds a uniform buffer to the specified slot.
    pub fn bind_uniform_buffer(mut self, set: u32, binding: u32, buffer: &BufferView) -> Result<Self> {
        self.modify_descriptor_set(set, |builder| {
            builder.bind_uniform_buffer(binding, buffer);
            Ok(())
        })?;
        Ok(self)
    }

    /// Binds a storage buffer buffer to the specified slot.
    pub fn bind_storage_buffer(mut self, set: u32, binding: u32, buffer: &BufferView) -> Result<Self> {
        self.modify_descriptor_set(set, |builder| {
            builder.bind_storage_buffer(binding, buffer);
            Ok(())
        })?;
        Ok(self)
    }

    /// Transitions an image layout.
    /// Generally you will not need to call this function manually,
    /// using the render graph api you can do most transitions automatically.
    pub fn transition_image(
        self,
        image: &ImageView,
        src_stage: PipelineStage,
        dst_stage: PipelineStage,
        from: vk::ImageLayout,
        to: vk::ImageLayout,
        src_access: vk::AccessFlags2,
        dst_access: vk::AccessFlags2,
    ) -> Self {
        let barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: src_stage,
            src_access_mask: src_access,
            dst_stage_mask: dst_stage,
            dst_access_mask: dst_access,
            old_layout: from,
            new_layout: to,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: unsafe { image.image() },
            subresource_range: image.subresource_range(),
        };
        let dependency = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::BY_REGION,
            memory_barrier_count: 0,
            p_memory_barriers: std::ptr::null(),
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: std::ptr::null(),
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &barrier,
        };
        self.pipeline_barrier_2(&dependency)
    }

    /// vkCmdPipelineBarrier2. Prefer using this over regular pipeline barriers if possible, to make
    /// full use of `VK_KHR_SYNCHRONIZATION_2`.
    pub fn pipeline_barrier_2(self, dependency: &vk::DependencyInfo) -> Self {
        unsafe {
            self.device.cmd_pipeline_barrier2(self.handle, &dependency);
        }
        self
    }

    /// Upload push constants. These are small packets of data stored inside the command buffer, so their state is tracked while executing.
    /// Direct translation of [`vkCmdPushConstants`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdPushConstants.html).
    /// Tends to crash on some drivers if the specified push constant range does not exist (possible due to unused variable optimization in the shader,
    /// or incorrect stage flags specified)
    pub fn push_constants<T: Copy>(self, stage: vk::ShaderStageFlags, offset: u32, data: &[T]) -> Self {
        // TODO: Validate push constant ranges with current pipeline layout to prevent crashes.
        unsafe {
            let (_, data, _) = data.align_to::<u8>();
            self.device
                .cmd_push_constants(self.handle, self.current_pipeline_layout, stage, offset, data);
        }
        self
    }

    pub(crate) fn begin_rendering(mut self, info: &RenderingInfo) -> Self {
        let map_attachment = |attachment: &RenderingAttachmentInfo| vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            p_next: std::ptr::null(),
            image_view: unsafe { attachment.image_view.handle() },
            image_layout: attachment.image_layout,
            resolve_mode: attachment.resolve_mode.unwrap_or(vk::ResolveModeFlagsKHR::NONE),
            resolve_image_view: match &attachment.resolve_image_view {
                Some(view) => unsafe { view.handle() },
                None => vk::ImageView::null(),
            },
            resolve_image_layout: attachment.resolve_image_layout.unwrap_or(vk::ImageLayout::UNDEFINED),
            load_op: attachment.load_op,
            store_op: attachment.store_op,
            clear_value: attachment.clear_value,
        };

        let color_attachments = info.color_attachments.iter().map(map_attachment).collect::<Vec<_>>();
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
            p_depth_attachment: match &depth_attachment {
                Some(attachment) => attachment,
                None => std::ptr::null(),
            },
            p_stencil_attachment: match &stencil_attachment {
                Some(attachment) => attachment,
                None => std::ptr::null(),
            },
        };

        unsafe {
            self.device.cmd_begin_rendering(self.handle, &vk_info);
        }

        self.current_rendering_state = Some(PipelineRenderingInfo {
            view_mask: info.view_mask,
            color_formats: info
                .color_attachments
                .iter()
                .map(|attachment| attachment.image_view.format())
                .collect(),
            depth_format: info.depth_attachment.as_ref().map(|attachment| attachment.image_view.format()),
            stencil_format: info
                .stencil_attachment
                .as_ref()
                .map(|attachment| attachment.image_view.format()),
        });
        self.current_render_area = info.render_area;

        self
    }

    pub(crate) fn end_rendering(mut self) -> Self {
        unsafe {
            self.device.cmd_end_rendering(self.handle);
        }
        self.current_rendering_state = None;
        self.current_render_area = vk::Rect2D::default();

        self
    }

    #[cfg(feature = "debug-markers")]
    pub fn begin_label(self, label: vk::DebugUtilsLabelEXT, debug: &DebugMessenger) -> Self {
        unsafe {
            debug.cmd_begin_debug_utils_label(self.handle, &label);
        }
        self
    }

    #[cfg(feature = "debug-markers")]
    pub fn end_label(self, debug: &DebugMessenger) -> Self {
        unsafe {
            debug.cmd_end_debug_utils_label(self.handle);
        }
        self
    }

    pub unsafe fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }
}
