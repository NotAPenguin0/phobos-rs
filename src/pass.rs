//! This module mainly exposes the [`PassBuilder`] struct, used for correctly defining passes in a
//! [`GpuTaskGraph`]. For documentation on how to use the task graph, refer to the [`task_graph`] module.
//! There are a few different types of passes. Each pass must declare its inputs and outputs, and can optionally
//! specify a closure to be executed when the pass is recorded to a command buffer. Additionally, a color can be given to each pass
//! which will show up in debuggers like `RenderDoc` if the `debug-markers` feature is enabled.
//!
//! # Example
//!
//! In this example we will define two passes: One that writes to an offscreen texture, and one that samples from this
//! texture to render it to the screen. This is a very simple dependency, but a very common pattern.
//! The task graph system will ensure access is properly synchronized, and the offscreen image is properly transitioned from its
//! initial layout before execution, to `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL`, to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`.
//!
//! First, we define some virtual resources.
//! ```
//! use phobos as ph;
//!
//! let offscreen = ph::VirtualResource::new("offscreen".to_string());
//! let swapchain = ph::VirtualResource::new("swapchain".to_string());
//! ```
//! Now we create the offscreen pass. Note that we use [`PassBuilder::render()`] to create a render pass.
//! This is a pass that outputs to at least one color or depth attachment by using the graphics pipeline.
//! ```
//! use ash::vk;
//!
//! let offscreen_pass = ph::PassBuilder::render("offscreen".to_string())
//!     // All this does is make the pass show up red in graphics debuggers, if
//!     // the debug-markers feature is enabled.
//!     .color([1.0, 0.0, 0.0, 1.0])
//!     // Add a single color attachment that will be cleared to red.
//!     .color_attachment(offscreen.clone(),
//!                       vk::AttachmentLoadOp::CLEAR,
//!                       Some(vk::ClearColorValue{ float32: [1.0, 0.0, 0.0, 1.0] }))?
//!     .build();
//!
//! ```
//! Next we can create the pass that will sample from this pass. To do this, we have to declare that we will sample
//! using [`PassBuilder::sample_image`]. Note that this is only necessary for images that are used elsewhere in the frame.
//! Regular textures for rendering do not need to be declared like this.
//!
//! ```
//! use ash::vk;
//!
//! // This is important. The virtual resource that encodes the output of the offscreen pass is not the same as the one
//! // we gave it in `color_attachment`. We have to look up the correct version using `Pass::output()`.
//! let input_resource = offscreen_pass.output(&offscreen).unwrap();
//!
//! let sample_pass = ph::PassBuilder::render("sample".to_string())
//!     // Let's color this pass green
//!     .color([0.0, 1.0, 0.0, 1.0])
//!     // Clear the swapchain to black.
//!     .color_attachment(swapchain.clone(),
//!                       vk::AttachmentLoadOp::CLEAR,
//!                       Some(vk::ClearColorValue{ float32: [0.0, 0.0, 0.0, 1.0] }))?
//!     // We sample the input resource in the fragment shader.
//!     .sample_image(input_resource.clone(), ph::PipelineStage::FRAGMENT_SHADER)
//!     .execute(|mut cmd, ifc, bindings| {
//!         // Commands to sample from the input texture go here
//!         Ok(cmd)
//!     })
//!     .build();
//! ```
//!
//! Binding physical resources and recording is covered under the [`task_graph`] module documentation.

use ash::vk;
use crate::{AttachmentType, Error, GpuResource, IncompleteCommandBuffer, InFlightContext, PhysicalResourceBindings, ResourceUsage, VirtualResource};
use crate::domain::ExecutionDomain;
use crate::pipeline::PipelineStage;
use anyhow::Result;

/// Represents one pass in a GPU task graph. You can obtain one using a [`PassBuilder`].
pub struct Pass<'exec, 'q, D> where D: ExecutionDomain {
    pub(crate) name: String,
    pub(crate) color: Option<[f32; 4]>,
    pub(crate) inputs: Vec<GpuResource>,
    pub(crate) outputs: Vec<GpuResource>,
    pub(crate) execute: Box<dyn FnMut(IncompleteCommandBuffer<'q, D>, &mut InFlightContext, &PhysicalResourceBindings) -> Result<IncompleteCommandBuffer<'q, D>>  + 'exec>,
    pub(crate) is_renderpass: bool
}

/// Used to create [`Pass`] objects correctly.
pub struct PassBuilder<'exec, 'q, D> where D: ExecutionDomain {
    inner: Pass<'exec, 'q, D>,
}

impl<'exec, 'q, D> Pass<'exec, 'q, D> where D: ExecutionDomain {
    /// Returns the output virtual resource associated with the input resource.
    pub fn output(&self, resource: &VirtualResource) -> Option<VirtualResource> {
        self.outputs.iter().filter_map(|output| {
            if VirtualResource::are_associated(resource, &output.resource) { Some(output.resource.clone()) }
            else { None }
        }).next()
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<'exec, 'q, D> PassBuilder<'exec, 'q, D> where D: ExecutionDomain {

    /// Create a new pass for generic commands. Does not support commands that are located inside a renderpass.
    pub fn new(name: impl Into<String>) -> Self {
        PassBuilder {
            inner: Pass {
                name: name.into(),
                color: None,
                execute: Box::new(|c, _, _| Ok(c)),
                inputs: vec![],
                outputs: vec![],
                is_renderpass: false
            },
        }
    }


    /// Create a new renderpass.
    pub fn render(name: impl Into<String>) -> Self {
        PassBuilder {
            inner: Pass {
                name: name.into(),
                color: None,
                execute: Box::new(|c, _, _| Ok(c)),
                inputs: vec![],
                outputs: vec![],
                is_renderpass: true
            },
        }
    }

    /// Create a pass for presenting to the swapchain.
    /// Note that this doesn't actually do the presentation, it just adds the proper sync for it.
    /// If you are presenting to an output of the graph, this is required.
    pub fn present(name: impl Into<String>, swapchain: VirtualResource) -> Pass<'exec, 'q, D> {
        Pass {
            name: name.into(),
            color: None,
            inputs: vec![GpuResource{
                usage: ResourceUsage::Present,
                resource: swapchain,
                stage: PipelineStage::BOTTOM_OF_PIPE,
                layout: vk::ImageLayout::PRESENT_SRC_KHR,
                clear_value: None,
                load_op: None,
            }],
            outputs: vec![],
            execute: Box::new(|c, _, _| Ok(c)),
            is_renderpass: false
        }
    }

    #[cfg(feature="debug-markers")]
    pub fn color(mut self, color: [f32; 4]) -> Self {
        self.inner.color = Some(color);
        self
    }

    /// Adds a color attachment to this pass. If [`vk::AttachmentLoadOp::CLEAR`] was specified, `clear` must not be None.
    pub fn color_attachment(mut self, resource: VirtualResource, op: vk::AttachmentLoadOp, clear: Option<vk::ClearColorValue>) -> Result<Self> {
        if !self.inner.is_renderpass { return Err(Error::Uncategorized("Cannot attach color attachment to a pass that is not a renderpass").into()) }
        if op == vk::AttachmentLoadOp::CLEAR && clear.is_none() { return Err(anyhow::Error::from(Error::NoClearValue)); }
        self.inner.inputs.push(GpuResource {
            usage: ResourceUsage::Attachment(AttachmentType::Color),
            resource: resource.clone(),
            // from https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineStageFlagBits2.html.
            // 'Load' and 'Clear' operations happen in COLOR_ATTACHMENT_OUTPUT,
            // Note that VK_PIPELINE_STAGE_2_CLEAR_BIT exists, but this only applies to vkCmdClear*
            stage: match op {
                vk::AttachmentLoadOp::LOAD => PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                vk::AttachmentLoadOp::CLEAR => PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                _ => todo!()
            },
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            clear_value: None,
            load_op: None,
        });

        self.inner.outputs.push(GpuResource{
            usage: ResourceUsage::Attachment(AttachmentType::Color),
            resource: resource.upgrade(),
            stage: PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            clear_value: clear.map(|c| vk::ClearValue { color: c }),
            load_op: Some(op)
        });

        Ok(self)
    }

    /// Adds a depth attachment to this pass. If [`vk::AttachmentLoadOp::CLEAR`] was specified, `clear` must not be None.
    pub fn depth_attachment(mut self, resource: VirtualResource, op: vk::AttachmentLoadOp, clear: Option<vk::ClearDepthStencilValue>) -> Result<Self> {
        if !self.inner.is_renderpass { return Err(Error::Uncategorized("Cannot attach depth attachment to a pass that is not a renderpass").into()) }
        self.inner.inputs.push(GpuResource {
            usage: ResourceUsage::Attachment(AttachmentType::Depth),
            resource: resource.clone(),
            stage: match op {
                // from https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineStageFlagBits2.html.
                // 'Load' operations on depth/stencil attachments happen in EARLY_FRAGMENT_TESTS.
                vk::AttachmentLoadOp::LOAD => PipelineStage::EARLY_FRAGMENT_TESTS,
                vk::AttachmentLoadOp::CLEAR => PipelineStage::EARLY_FRAGMENT_TESTS,
                _ => todo!()
            },
            layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            clear_value: None,
            load_op: None,
        });

        self.inner.outputs.push(GpuResource {
            usage: ResourceUsage::Attachment(AttachmentType::Depth),
            resource: resource.upgrade(),
            // Depth/stencil writes happen in LATE_FRAGMENT_TESTS.
            // It's also legal to specify COLOR_ATTACHMENT_OUTPUT, but this is more precise.
            stage: PipelineStage::LATE_FRAGMENT_TESTS,
            layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            clear_value: clear.map(|c| vk::ClearValue { depth_stencil: c }),
            load_op: Some(op)
        });

        Ok(self)
    }

    /// Declare that a resource will be used as a sampled image in the given pipeline stages.
    pub fn sample_image(mut self, resource: VirtualResource, stage: PipelineStage) -> Self {
        self.inner.inputs.push(GpuResource {
            usage: ResourceUsage::ShaderRead,
            resource,
            stage,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            clear_value: None,
            load_op: None
        });
        self
    }

    /// Set the function to be called when recording this pass.
    pub fn execute(mut self, exec: impl FnMut(IncompleteCommandBuffer<'q, D>, &mut InFlightContext, &PhysicalResourceBindings) -> Result<IncompleteCommandBuffer<'q, D>>  + 'exec) -> Self {
        self.inner.execute = Box::new(exec);
        self
    }

    /// Obtain a built [`Pass`] object.
    pub fn build(self) -> Pass<'exec, 'q, D> {
        self.inner
    }
}