//! Extra utilities for command buffers not tied to a domain

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, MutexGuard};

use anyhow::{anyhow, ensure, Result};
use ash::vk;

use crate::{
    Allocator, BufferView, DebugMessenger, DescriptorCache, DescriptorSet, DescriptorSetBuilder,
    Device, ImageView, IncompleteCmdBuffer, PhysicalResourceBindings, PipelineCache, PipelineStage,
    Sampler, VirtualResource,
};
use crate::command_buffer::{CommandBuffer, IncompleteCommandBuffer};
use crate::command_buffer::state::{RenderingAttachmentInfo, RenderingInfo};
use crate::core::queue::Queue;
use crate::pipeline::create_info::PipelineRenderingInfo;
use crate::query_pool::{QueryPool, ScopedQuery, TimestampQuery};
use crate::raytracing::acceleration_structure::AccelerationStructure;
use crate::sync::domain::ExecutionDomain;

impl<'q, D: ExecutionDomain, A: Allocator> IncompleteCmdBuffer<'q, A>
    for IncompleteCommandBuffer<'q, D, A>
{
    type Domain = D;

    /// Create a new command buffer ready for recording.
    /// This will hold the lock on the queue until [`IncompleteCommandBuffer::finish()`] is called.
    fn new(
        device: Device,
        queue_lock: MutexGuard<'q, Queue>,
        handle: vk::CommandBuffer,
        flags: vk::CommandBufferUsageFlags,
        pipelines: PipelineCache<A>,
        descriptors: DescriptorCache,
    ) -> Result<Self> {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: std::ptr::null(),
                flags,
                p_inheritance_info: std::ptr::null(),
            };
            // SAFETY:
            // * A valid VkDevice was passed in
            // * The command buffer passed in may not be NULL
            // * The begin_info structure is valid.
            device.begin_command_buffer(handle, &begin_info)?;
        };
        Ok(IncompleteCommandBuffer {
            device,
            handle,
            timestamp_valid_bits: queue_lock.family_properties().timestamp_valid_bits,
            queue_lock,
            current_pipeline_layout: vk::PipelineLayout::null(),
            current_set_layouts: vec![],
            current_bindpoint: vk::PipelineBindPoint::default(),
            current_rendering_state: None,
            current_render_area: Default::default(),
            current_descriptor_sets: None,
            descriptor_state_needs_update: false,
            current_sbt_regions: None,
            descriptor_cache: descriptors,
            pipeline_cache: pipelines,
            _domain: PhantomData,
        })
    }

    /// Finish recording this command buffer. After calling this, no more commands can be
    /// recorded to this object and it should be submitted. This also releases the lock on the queue, so
    /// call this as soon as you are done recording to minimize contention of that lock.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use anyhow::Result;
    /// # use phobos::sync::domain::{ExecutionDomain, Graphics};
    /// fn finish_command_buffer<D: ExecutionDomain>(cmd: IncompleteCommandBuffer<D>) -> Result<CommandBuffer<D>> {
    ///     // Releases the lock on a queue associated with the domain `D`, allowing other command
    ///     // buffers to start recording on this domain.
    ///     let cmd = cmd.finish()?;
    ///     Ok(cmd)
    /// }
    /// ```
    fn finish(self) -> Result<CommandBuffer<D>> {
        // SAFETY:
        // * `self` is valid, so `device` and `self.handle` are valid.
        // * `self` is valid, so this command buffer is in the recording state (see `new()`).
        unsafe { self.device.end_command_buffer(self.handle)? }
        Ok(CommandBuffer {
            handle: self.handle,
            _domain: PhantomData,
        })
    }
}

impl<D: ExecutionDomain, A: Allocator> IncompleteCommandBuffer<'_, D, A> {
    /// Bind a descriptor set to the command buffer.
    /// # Errors
    /// - Fails if no pipeline was bound.
    pub(super) fn bind_descriptor_set(&self, index: u32, set: &DescriptorSet) -> Result<()> {
        ensure!(
            self.current_pipeline_layout != vk::PipelineLayout::null(),
            "cannot bind descriptor set at index {index} without binding a pipeline first."
        );
        unsafe {
            // SAFETY:
            // * self is valid, so self.handle is valid.
            // * We just verified using the ensure statement above that a pipeline is bound.
            // * We assume index is a valid descriptor set index, otherwise we get a validation layer error
            // * Caller passed in a valid descriptor set object.
            self.device.cmd_bind_descriptor_sets(
                self.handle,
                self.current_bindpoint,
                self.current_pipeline_layout,
                index,
                std::slice::from_ref(&set.handle),
                &[],
            );
        }
        Ok(())
    }

    /// Modify the descriptor set state at a given set binding.
    /// # Errors
    /// * Fails if the supplied callback fails.
    pub(super) fn modify_descriptor_set(
        &mut self,
        set: u32,
        f: impl FnOnce(&mut DescriptorSetBuilder) -> Result<()>,
    ) -> Result<()> {
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

    /// If there are unwritten descriptor sets, update the entire descriptor set state by binding a new set.
    /// # Errors
    /// * Fails if the descriptor set cache lookup fails.
    /// * Fails if binding the descriptor set fails.
    pub(super) fn ensure_descriptor_state(mut self) -> Result<Self> {
        // No need to do anything
        if !self.descriptor_state_needs_update {
            return Ok(self);
        }

        let cache = self.descriptor_cache.clone();
        for (index, builder) in self.current_descriptor_sets.take().unwrap() {
            let mut info = builder.build();
            info.layout = *self.current_set_layouts.get(index as usize).unwrap();
            cache.with_descriptor_set(info, |set| {
                self.bind_descriptor_set(index, set)?;
                Ok(())
            })?;
        }

        // We updated all our descriptor sets, were good now.
        self.descriptor_state_needs_update = false;
        Ok(self)
    }

    /// Binds the given pipeline to the given bindpoint.
    /// # Errors
    /// None
    pub(super) fn bind_pipeline_impl(
        &mut self,
        handle: vk::Pipeline,
        layout: vk::PipelineLayout,
        set_layouts: Vec<vk::DescriptorSetLayout>,
        bind_point: vk::PipelineBindPoint,
    ) -> Result<()> {
        unsafe {
            // SAFETY:
            // * `self` is valid, so `self.device` and `self.handle` are valid vulkan objects.
            // * `pipeline.handle` is a valid entry from the pipeline cache, so it is a valid compute pipeline.
            self.device
                .cmd_bind_pipeline(self.handle, bind_point, handle);
        }
        self.current_bindpoint = bind_point;
        self.current_pipeline_layout = layout;
        self.current_set_layouts = set_layouts.clone();
        Ok(())
    }

    /// Clear descriptor set state. Calling this will reset the current descriptor state to nothing being bound.
    /// It does not explicitly unbind descriptor sets, but the next `draw()` or `dispatch()` call will
    /// reflect this change. This function is not extremely useful at the moment.
    /// # Example
    /// ```
    /// # use phobos::sync::domain::ExecutionDomain;
    /// # use phobos::{BufferView, IncompleteCommandBuffer};
    /// # use anyhow::Result;
    /// fn use_descriptor_forget<'q, D: ExecutionDomain>(cmd: IncompleteCommandBuffer<'q, D>, buffer: &BufferView, other_buffer: &BufferView) -> Result<IncompleteCommandBuffer<'q, D>> {
    ///     cmd.bind_uniform_buffer(0, 0, buffer)?
    ///         // Forget the previous binding
    ///        .forget_descriptor_state()
    ///         // And overwrite it with a new one.
    ///        .bind_uniform_buffer(0, 0, other_buffer)
    /// }
    /// ```
    pub fn forget_descriptor_state(mut self) -> Self {
        self.current_descriptor_sets = None;
        self.descriptor_state_needs_update = true;
        self
    }

    /// Binds a new descriptor with descriptor type [`vk::DescriptorType::COMBINED_IMAGE_SAMPLER`]. The image bound to this is
    /// the image obtained by resolving the input resource from the given resource bindings. The sampler bound to this
    /// is the one given. This binding is not actually flushed to the command buffer until the next draw or dispatch call.
    ///
    /// Expects the image to be in [`vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL`]
    /// # Errors
    /// * Fails if the virtual resource has no physical binding associated to it.
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::sync::domain::ExecutionDomain;
    /// # use phobos::*;
    /// fn use_resolve_and_bind<'q, D: ExecutionDomain>(cmd: IncompleteCommandBuffer<'q, D>, image: &ImageView, sampler: &Sampler) -> Result<IncompleteCommandBuffer<'q, D>> {
    ///     let resource = image!("image");
    ///     let mut bindings = PhysicalResourceBindings::new();
    ///     bindings.bind_image("image", image);
    ///
    ///     cmd.resolve_and_bind_sampled_image(0, 0, &resource, sampler, &bindings)
    /// }
    /// ```
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

    /// Binds a new descriptor with type [`vk::DescriptorType::COMBINED_IMAGE_SAMPLER`].
    /// This binding is not actually flushed to the command buffer until the next draw or dispatch call.
    ///
    /// Expects the image to be in [`vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL`]
    /// # Errors
    /// None
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::sync::domain::ExecutionDomain;
    /// # use phobos::*;
    /// fn use_bind_sampled_image<'q, D: ExecutionDomain + GfxSupport>(cmd: IncompleteCommandBuffer<'q, D>, image: &ImageView, sampler: &Sampler) -> Result<IncompleteCommandBuffer<'q, D>> {
    ///     cmd.bind_sampled_image(0, 0, image, sampler)?
    ///         // This drawcall will flush the descriptor state and bind proper descriptor sets.
    ///        .draw(6, 1, 0, 0)
    /// }
    /// ```
    pub fn bind_sampled_image(
        mut self,
        set: u32,
        binding: u32,
        image: &ImageView,
        sampler: &Sampler,
    ) -> Result<Self> {
        self.modify_descriptor_set(set, |builder| {
            builder.bind_sampled_image(binding, image, sampler);
            Ok(())
        })?;
        Ok(self)
    }

    /// Binds a new descriptor with type [`vk::DescriptorType::UNIFORM_BUFFER`].
    /// This binding is not actually flushed to the command buffer until the next draw or dispatch call.
    /// # Errors
    /// None
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::sync::domain::ExecutionDomain;
    /// # use phobos::*;
    /// fn use_bind_uniform_buffer<'q, D: ExecutionDomain + GfxSupport>(cmd: IncompleteCommandBuffer<'q, D>, buffer: &BufferView) -> Result<IncompleteCommandBuffer<'q, D>> {
    ///     cmd.bind_uniform_buffer(0, 0, buffer)?
    ///         // This drawcall will flush the descriptor state and bind proper descriptor sets.
    ///        .draw(6, 1, 0, 0)
    /// }
    /// ```
    pub fn bind_uniform_buffer(
        mut self,
        set: u32,
        binding: u32,
        buffer: &BufferView,
    ) -> Result<Self> {
        self.modify_descriptor_set(set, |builder| {
            builder.bind_uniform_buffer(binding, buffer);
            Ok(())
        })?;
        Ok(self)
    }

    /// Binds a new descriptor with type [`vk::DescriptorType::STORAGE_BUFFER`].
    /// This binding is not actually flushed to the command buffer until the next draw or dispatch call.
    /// # Errors
    /// None
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::sync::domain::ExecutionDomain;
    /// # use phobos::*;
    /// fn use_bind_storage_buffer<'q, D: ExecutionDomain + GfxSupport>(cmd: IncompleteCommandBuffer<'q, D>, buffer: &BufferView) -> Result<IncompleteCommandBuffer<'q, D>> {
    ///     cmd.bind_storage_buffer(0, 0, buffer)?
    ///         // This drawcall will flush the descriptor state and bind proper descriptor sets.
    ///        .draw(6, 1, 0, 0)
    /// }
    /// ```
    pub fn bind_storage_buffer(
        mut self,
        set: u32,
        binding: u32,
        buffer: &BufferView,
    ) -> Result<Self> {
        self.modify_descriptor_set(set, |builder| {
            builder.bind_storage_buffer(binding, buffer);
            Ok(())
        })?;
        Ok(self)
    }

    /// Binds a new descriptor with type [`vk::DescriptorType::STORAGE_IMAGE`].
    /// This binding is not actually flushed to the command buffer until the next draw or dispatch call.
    ///
    /// Expects the image to be in [`vk::ImageLayout::GENERAL`]
    /// # Errors
    /// None
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::sync::domain::ExecutionDomain;
    /// # use phobos::*;
    /// fn use_bind_storage_image<'q, D: ExecutionDomain + GfxSupport>(cmd: IncompleteCommandBuffer<'q, D>, image: &ImageView) -> Result<IncompleteCommandBuffer<'q, D>> {
    ///     cmd.bind_storage_image(0, 0, image)?
    ///         // This drawcall will flush the descriptor state and bind proper descriptor sets.
    ///        .draw(6, 1, 0, 0)
    /// }
    /// ```
    pub fn bind_storage_image(mut self, set: u32, binding: u32, image: &ImageView) -> Result<Self> {
        self.modify_descriptor_set(set, |builder| {
            builder.bind_storage_image(binding, image);
            Ok(())
        })?;
        Ok(self)
    }

    /// Binds a new descriptor with descriptor type [`vk::DescriptorType::STORAGE_IMAGE`]. The image bound to this is
    /// the image obtained by resolving the input resource from the given resource bindings.
    /// This binding is not actually flushed to the command buffer until the next draw or dispatch call.
    ///
    /// Expects the image to be in [`vk::ImageLayout::GENERAL`]
    /// # Errors
    /// * Fails if the virtual resource has no physical binding associated to it.
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::sync::domain::ExecutionDomain;
    /// # use phobos::*;
    /// fn use_resolve_and_bind<'q, D: ExecutionDomain>(cmd: IncompleteCommandBuffer<'q, D>, image: &ImageView) -> Result<IncompleteCommandBuffer<'q, D>> {
    ///     let resource = VirtualResource::image("image");
    ///     let mut bindings = PhysicalResourceBindings::new();
    ///     bindings.bind_image("image", image);
    ///
    ///     cmd.resolve_and_bind_storage_image(0, 0, &resource, &bindings)
    /// }
    /// ```
    pub fn resolve_and_bind_storage_image(
        mut self,
        set: u32,
        binding: u32,
        resource: &VirtualResource,
        bindings: &PhysicalResourceBindings,
    ) -> Result<Self> {
        self.modify_descriptor_set(set, |builder| {
            builder.resolve_and_bind_storage_image(binding, resource, bindings)
        })?;
        Ok(self)
    }

    /// Binds a new descriptor with descriptor type [`vk::DescriptorType::ACCELERATION_STRUCTURE_KHR`]. The
    /// `VK_KHR_acceleration_structure` extension must be enabled for this (use [`AppBuilder::raytracing()`](crate::AppBuilder::raytracing() to enable).
    /// # Example
    /// ```
    /// # use anyhow::Result;
    /// # use phobos::sync::domain::ExecutionDomain;
    /// # use phobos::*;
    /// fn use_bind_acceleration_structure<'q, D: ExecutionDomain + GfxSupport>(cmd: IncompleteCommandBuffer<'q, D>, accel: &AccelerationStructure) -> Result<IncompleteCommandBuffer<'q, D>> {
    ///     cmd.use_bind_acceleration_structure(0, 0, accel)?
    ///         // This call will flush the descriptor state and bind proper descriptor sets.
    ///        .trace_rays(1920, 1080, 1)
    /// }
    /// ```
    pub fn bind_acceleration_structure(
        mut self,
        set: u32,
        binding: u32,
        accel: &AccelerationStructure,
    ) -> Result<Self> {
        self.modify_descriptor_set(set, |builder| {
            builder.bind_acceleration_structure(binding, accel);
            Ok(())
        })?;
        Ok(self)
    }

    /// Transitions an image layout manually. For attachment layouts and other
    /// resources used in the pass graph, this can be done automatically.
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
            // SAFETY: A valid image view object has a valid `VkImage` handle.
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
        self.pipeline_barrier(&dependency)
    }

    /// Insert a global memory barrier. If you want to create a barrier for a buffer, prefer using this as every driver
    /// implements buffer barriers as global memory barriers anyway.
    /// Uses [`vkCmdPipelineBarrier2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdPipelineBarrier2KHR.html) directly.
    pub fn memory_barrier(
        self,
        src_stage: PipelineStage,
        src_access: vk::AccessFlags2,
        dst_stage: PipelineStage,
        dst_access: vk::AccessFlags2,
    ) -> Self {
        let barrier = vk::MemoryBarrier2 {
            s_type: vk::StructureType::MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: src_stage,
            src_access_mask: src_access,
            dst_stage_mask: dst_stage,
            dst_access_mask: dst_access,
        };

        let dependency = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: vk::DependencyFlags::BY_REGION,
            memory_barrier_count: 1,
            p_memory_barriers: &barrier,
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: std::ptr::null(),
            image_memory_barrier_count: 0,
            p_image_memory_barriers: std::ptr::null(),
        };

        self.pipeline_barrier(&dependency)
    }

    /// The direct equivalent of a raw [`vkCmdPipelineBarrier2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdPipelineBarrier2KHR.html) call.
    /// Before calling this, make sure there is not an automatic way to insert this barrier, for example
    /// using the pass graph or using [`IncompleteCommandBuffer::transition_image()`].
    pub fn pipeline_barrier(self, dependency: &vk::DependencyInfo) -> Self {
        unsafe {
            self.device.cmd_pipeline_barrier2(self.handle, dependency);
        }
        self
    }

    /// Upload a single value of push constants. These are small packets of data stored inside the command buffer, so their state is tracked while recording and executing.
    /// Direct translation of [`vkCmdPushConstants`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdPushConstants.html).
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use phobos::sync::domain::ExecutionDomain;
    /// fn use_push_constant<D: ExecutionDomain>(cmd: IncompleteCommandBuffer<D>) -> IncompleteCommandBuffer<D> {
    ///     // Assumes a pipeline is bound, and that this pipeline has a vertex shader with the specified push constant range.
    ///     let data: f32 = 1.0;
    ///     cmd.push_constant(vk::ShaderStageFlags::VERTEX, 0, &data)
    /// }
    pub fn push_constant<T: Copy + Sized>(
        self,
        stage: vk::ShaderStageFlags,
        offset: u32,
        data: &T,
    ) -> Self {
        self.push_constants(stage, offset, std::slice::from_ref(data))
    }

    /// Upload push constants. These are small packets of data stored inside the command buffer, so their state is tracked while recording and executing.
    /// Direct translation of [`vkCmdPushConstants`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdPushConstants.html).
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use phobos::sync::domain::ExecutionDomain;
    /// fn use_push_constants<D: ExecutionDomain>(cmd: IncompleteCommandBuffer<D>) -> IncompleteCommandBuffer<D> {
    ///     // Assumes a pipeline is bound, and that this pipeline has a vertex shader with the specified push constant range.
    ///     let data: [f32; 2] = [64.0, 32.0];
    ///     cmd.push_constants(vk::ShaderStageFlags::VERTEX, 0, &data)
    /// }
    /// ```
    pub fn push_constants<T: Copy + Sized>(
        self,
        stage: vk::ShaderStageFlags,
        offset: u32,
        data: &[T],
    ) -> Self {
        // TODO: Validate push constant ranges with current pipeline layout to prevent crashes.
        unsafe {
            // SAFETY: every data structure can be aligned to a byte slice.
            let (_, data, _) = data.align_to::<u8>();
            // SAFETY: self is valid, everything else is up to validation layers.
            self.device.cmd_push_constants(
                self.handle,
                self.current_pipeline_layout,
                stage,
                offset,
                data,
            );
        }
        self
    }

    /// Begin a scoped query. Not all query types are scoped, so the query type must implement
    /// [`ScopedQuery`].
    pub fn begin_query<Q: ScopedQuery>(self, query_pool: &QueryPool<Q>, index: u32) -> Self {
        unsafe {
            self.device.cmd_begin_query(
                self.handle,
                query_pool.handle(),
                index,
                vk::QueryControlFlags::default(),
            );
        }
        self
    }

    /// End a scoped query. This query must be started with [`Self::begin_query()`] first.
    pub fn end_query<Q: ScopedQuery>(self, query_pool: &QueryPool<Q>, index: u32) -> Self {
        unsafe {
            self.device
                .cmd_end_query(self.handle, query_pool.handle(), index);
        }
        self
    }

    /// Write a timestamp to the next entry in a query pool.
    /// # Errors
    /// * Fails if the query pool is out of entries.
    pub fn write_timestamp(
        self,
        query_pool: &mut QueryPool<TimestampQuery>,
        stage: PipelineStage,
    ) -> Result<Self> {
        let index = query_pool
            .next()
            .ok_or_else(|| anyhow!("Query pool capacity exceeded"))?;
        query_pool.write_timestamp(self.timestamp_valid_bits, self.handle, stage, index);
        Ok(self)
    }

    /// Begins a dynamic renderpass. This must be called before binding any pipelines.
    pub(crate) fn begin_rendering(mut self, info: &RenderingInfo) -> Self {
        let map_attachment = |attachment: &RenderingAttachmentInfo| vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            p_next: std::ptr::null(),
            // SAFETY: A valid RenderingAttachmentInfo always stores a valid image view
            image_view: unsafe { attachment.image_view.handle() },
            image_layout: attachment.image_layout,
            resolve_mode: attachment
                .resolve_mode
                .unwrap_or(vk::ResolveModeFlagsKHR::NONE),
            resolve_image_view: match &attachment.resolve_image_view {
                // SAFETY: A valid RenderingAttachmentInfo always stores a valid image view
                Some(view) => unsafe { view.handle() },
                None => vk::ImageView::null(),
            },
            resolve_image_layout: attachment
                .resolve_image_layout
                .unwrap_or(vk::ImageLayout::UNDEFINED),
            load_op: attachment.load_op,
            store_op: attachment.store_op,
            clear_value: attachment.clear_value,
        };

        let color_attachments = info
            .color_attachments
            .iter()
            .map(map_attachment)
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
            // SAFETY: self is valid, vk_info is valid.
            self.device.cmd_begin_rendering(self.handle, &vk_info);
        }

        self.current_rendering_state = Some(PipelineRenderingInfo {
            view_mask: info.view_mask,
            color_formats: info
                .color_attachments
                .iter()
                .map(|attachment| attachment.image_view.format())
                .collect(),
            depth_format: info
                .depth_attachment
                .as_ref()
                .map(|attachment| attachment.image_view.format()),
            stencil_format: info
                .stencil_attachment
                .as_ref()
                .map(|attachment| attachment.image_view.format()),
        });
        self.current_render_area = info.render_area;

        self
    }

    /// Ends a dynamic renderpass.
    pub(crate) fn end_rendering(mut self) -> Self {
        unsafe {
            // Safety: self is valid, the caller must ensure begin_rendering() was called first.
            self.device.cmd_end_rendering(self.handle);
        }
        self.current_rendering_state = None;
        self.current_render_area = vk::Rect2D::default();

        self
    }

    /// Start a label region.
    #[cfg(feature = "debug-markers")]
    pub(crate) fn begin_label(
        self,
        label: vk::DebugUtilsLabelEXT,
        debug: &Arc<DebugMessenger>,
    ) -> Self {
        unsafe {
            debug.cmd_begin_debug_utils_label(self.handle, &label);
        }
        self
    }

    /// End a label region.
    #[cfg(feature = "debug-markers")]
    pub(crate) fn end_label(self, debug: &Arc<DebugMessenger>) -> Self {
        unsafe {
            debug.cmd_end_debug_utils_label(self.handle);
        }
        self
    }

    /// Get unsafe access to the underlying `VkCommandBuffer` handle.
    /// # Safety
    /// Any vulkan calls that mutate the command buffer's state may put the system in an undefined state.
    pub unsafe fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }
}
