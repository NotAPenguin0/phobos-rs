//! This module mainly exposes the [`PassBuilder`] struct, used for correctly defining passes in a
//! [`PassGraph`](crate::PassGraph). For documentation on how to use the pass graph, refer to the [`graph`](crate::graph) module level documentation.
//! There are a few different types of passes. Each pass must declare its inputs and outputs, and can optionally
//! specify a closure to be executed when the pass is recorded to a command buffer. Additionally, a color can be given to each pass
//! which will show up in debuggers like [*RenderDoc*](https://renderdoc.org/) if the `debug-markers` feature is enabled.
//!
//! # Example
//!
//! In this example we will define two passes: One that writes to an offscreen texture, and one that samples from this
//! texture to render it to the screen. This is a very simple dependency, but a very common pattern.
//! The task graph system will ensure access is properly synchronized, and the offscreen image is properly transitioned from its
//! initial layout before execution, to `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL`, and then to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`
//! before the final pass.
//!
//! First, we define some virtual resources.
//! ```
//! use phobos::prelude::*;
//!
//! let offscreen = ph::VirtualResource::image("offscreen");
//! let swapchain = ph::VirtualResource::image("swapchain");
//! ```
//! Now we create the offscreen pass. Note that we use [`PassBuilder::render()`] to create a render pass.
//! This is a pass that outputs to at least one color or depth attachment by using the graphics pipeline.
//! A pass that was not created through this method cannot define any attachments, this is useful for
//! e.g. compute-only passes.
//! ```
//! use phobos::prelude::*;
//!
//! let offscreen_pass = PassBuilder::render("offscreen")
//!     // All this does is make the pass show up red in graphics debuggers, if
//!     // the debug-markers feature is enabled.
//!     .color([1.0, 0.0, 0.0, 1.0])
//!     // Add a single color attachment that will be cleared to red.
//!     .color_attachment(&offscreen,
//!                       vk::AttachmentLoadOp::CLEAR,
//!                       Some(vk::ClearColorValue{ float32: [1.0, 0.0, 0.0, 1.0] }))?
//!     .build();
//!
//! ```
//! Next we can create the pass that will sample from this pass. To do this, we have to declare that we will sample
//! the virtual resource using [`PassBuilder::sample_image`].
//! Note that this is only necessary for images that are used elsewhere in the frame.
//! Regular textures for rendering do not need to be declared like this.
//!
//! ```
//! use phobos::prelude::*;
//!
//! // This is important. The virtual resource that encodes the output of the offscreen pass is not the same as the one
//! // we gave it in `color_attachment()`. We have to look up the correct version using `Pass::output()`.
//! let input_resource = offscreen_pass.output(&offscreen).unwrap();
//!
//! let sample_pass = PassBuilder::render("sample")
//!     // Let's color this pass green
//!     .color([0.0, 1.0, 0.0, 1.0])
//!     // Clear the swapchain to black.
//!     .color_attachment(swapchain.clone(),
//!                       vk::AttachmentLoadOp::CLEAR,
//!                       Some(vk::ClearColorValue{ float32: [0.0, 0.0, 0.0, 1.0] }))?
//!     // We sample the input resource in the fragment shader.
//!     .sample_image(&input_resource, PipelineStage::FRAGMENT_SHADER)
//!     .executor(|mut cmd, ifc, bindings| {
//!         // Draw a fullscreen quad using our sample pipeline and a descriptor set pointing to the input resource.
//!         // This assumes we created a pipeline before and a sampler before, and that we bind the proper resources
//!         // before recording the graph.
//!         cmd = cmd.bind_graphics_pipeline("fullscreen_sample")?
//!                  .resolve_and_bind_sampled_image(0, 0, &input_resource, &sampler, &bindings)?
//!                  .draw(6, 1, 0, 0)?;
//!         Ok(cmd)
//!     })
//!     .build();
//! ```
//!
//! Binding physical resources and recording is covered under the [`graph`](crate::graph) module documentation.

use anyhow::Result;
use ash::vk;

use crate::{Allocator, DefaultAllocator, Error, InFlightContext, PhysicalResourceBindings, VirtualResource};
use crate::command_buffer::IncompleteCommandBuffer;
use crate::domain::ExecutionDomain;
use crate::graph::pass_graph::PassResource;
use crate::graph::resource::{AttachmentType, ResourceUsage};
use crate::pipeline::PipelineStage;

/// The returned value from a pass callback function.
pub type PassFnResult<'q, D> = Result<IncompleteCommandBuffer<'q, D>>;

pub trait PassExecutor<D: ExecutionDomain, U, A: Allocator> {
    fn execute<'q>(
        &mut self,
        cmd: IncompleteCommandBuffer<'q, D>,
        ifc: &mut InFlightContext<A>,
        bindings: &PhysicalResourceBindings,
        user_data: &mut U,
    ) -> PassFnResult<'q, D>;
}

impl<D, U, A, F> PassExecutor<D, U, A> for F
    where
        D: ExecutionDomain,
        A: Allocator,
        F: for<'q> FnMut(IncompleteCommandBuffer<'q, D>, &mut InFlightContext<A>, &PhysicalResourceBindings, &mut U) -> PassFnResult<'q, D>,
{
    fn execute<'q>(
        &mut self,
        cmd: IncompleteCommandBuffer<'q, D>,
        ifc: &mut InFlightContext<A>,
        bindings: &PhysicalResourceBindings,
        user_data: &mut U,
    ) -> PassFnResult<'q, D> {
        self.call_mut((cmd, ifc, bindings, user_data))
    }
}

pub(crate) type BoxedPassFn<'cb, D, U, A> = Box<dyn PassExecutor<D, U, A> + 'cb>;

pub struct EmptyPassExecutor;

impl EmptyPassExecutor {
    pub fn new() -> Self {
        Self {}
    }

    pub fn new_boxed() -> Box<Self> {
        Box::new(Self::new())
    }
}

impl<D: ExecutionDomain, U, A: Allocator> PassExecutor<D, U, A> for EmptyPassExecutor {
    fn execute<'q>(
        &mut self,
        cmd: IncompleteCommandBuffer<'q, D>,
        _ifc: &mut InFlightContext<A>,
        _bindings: &PhysicalResourceBindings,
        _user_data: &mut U,
    ) -> PassFnResult<'q, D> {
        Ok(cmd)
    }
}

/// Represents one pass in a GPU task graph. You can obtain one using a [`PassBuilder`].
pub struct Pass<'cb, D: ExecutionDomain, U = (), A: Allocator = DefaultAllocator> {
    pub(crate) name: String,
    pub(crate) color: Option<[f32; 4]>,
    pub(crate) inputs: Vec<PassResource>,
    pub(crate) outputs: Vec<PassResource>,
    pub(crate) execute: BoxedPassFn<'cb, D, U, A>,
    pub(crate) is_renderpass: bool,
}

/// Used to create [`Pass`] objects correctly.
/// # Example
/// See the [`pass`](crate::graph::pass) module level documentation.
pub struct PassBuilder<'cb, D: ExecutionDomain, U = (), A: Allocator = DefaultAllocator> {
    inner: Pass<'cb, D, U, A>,
}

impl<'cb, D: ExecutionDomain, U, A: Allocator> Pass<'cb, D, U, A> {
    /// Returns the output virtual resource associated with the input resource.
    pub fn output(&self, resource: &VirtualResource) -> Option<&VirtualResource> {
        self.outputs
            .iter()
            .filter_map(|output| {
                if resource.is_associated_with(&output.resource) {
                    Some(&output.resource)
                } else {
                    None
                }
            })
            .next()
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<'cb, D: ExecutionDomain, U, A: Allocator> PassBuilder<'cb, D, U, A> {
    /// Create a new pass for generic commands. Does not support commands that are located inside a renderpass.
    pub fn new(name: impl Into<String>) -> Self {
        PassBuilder {
            inner: Pass {
                name: name.into(),
                color: None,
                execute: EmptyPassExecutor::new_boxed(),
                inputs: vec![],
                outputs: vec![],
                is_renderpass: false,
            },
        }
    }

    /// Create a new renderpass. This constructor is required for passes that render to any attachments.
    pub fn render(name: impl Into<String>) -> Self {
        PassBuilder {
            inner: Pass {
                name: name.into(),
                color: None,
                execute: EmptyPassExecutor::new_boxed(),
                inputs: vec![],
                outputs: vec![],
                is_renderpass: true,
            },
        }
    }

    /// Create a pass for presenting to the swapchain.
    /// Note that this doesn't actually do the presentation, it just adds the proper synchronization for it.
    /// If you are presenting to an output of the graph, this is required.
    pub fn present(name: impl Into<String>, swapchain: &VirtualResource) -> Pass<'cb, D, U, A> {
        Pass {
            name: name.into(),
            color: None,
            inputs: vec![PassResource {
                usage: ResourceUsage::Present,
                resource: swapchain.clone(),
                stage: PipelineStage::BOTTOM_OF_PIPE,
                layout: vk::ImageLayout::PRESENT_SRC_KHR,
                clear_value: None,
                load_op: None,
            }],
            outputs: vec![],
            execute: EmptyPassExecutor::new_boxed(),
            is_renderpass: false,
        }
    }

    /// Set the color of this pass. This can show up in graphics debuggers like RenderDoc.
    #[cfg(feature = "debug-markers")]
    pub fn color(mut self, color: [f32; 4]) -> Self {
        self.inner.color = Some(color);
        self
    }

    /// Adds a color attachment to this pass. If [`vk::AttachmentLoadOp::CLEAR`] was specified, `clear` must not be None.
    /// # Errors
    /// * Fails if this pass was not created using [`PassBuilder::render()`]
    /// * Fails if `op` was [`vk::AttachmentLoadOp::CLEAR`], but `clear` was [`None`].
    pub fn color_attachment(mut self, resource: &VirtualResource, op: vk::AttachmentLoadOp, clear: Option<vk::ClearColorValue>) -> Result<Self> {
        if !self.inner.is_renderpass {
            return Err(Error::Uncategorized("Cannot attach color attachment to a pass that is not a renderpass").into());
        }
        if op == vk::AttachmentLoadOp::CLEAR && clear.is_none() {
            return Err(anyhow::Error::from(Error::NoClearValue));
        }
        self.inner.inputs.push(PassResource {
            usage: ResourceUsage::Attachment(AttachmentType::Color),
            resource: resource.clone(),
            // from https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineStageFlagBits2.html.
            // 'Load' and 'Clear' operations happen in COLOR_ATTACHMENT_OUTPUT,
            // Note that VK_PIPELINE_STAGE_2_CLEAR_BIT exists, but this only applies to vkCmdClear*
            stage: match op {
                vk::AttachmentLoadOp::LOAD => PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                vk::AttachmentLoadOp::CLEAR => PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                _ => todo!(),
            },
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            clear_value: None,
            load_op: None,
        });

        self.inner.outputs.push(PassResource {
            usage: ResourceUsage::Attachment(AttachmentType::Color),
            resource: resource.upgrade(),
            stage: PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            clear_value: clear.map(|c| vk::ClearValue {
                color: c,
            }),
            load_op: Some(op),
        });

        Ok(self)
    }

    /// Adds a depth attachment to this pass. If [`vk::AttachmentLoadOp::CLEAR`] was specified, `clear` must not be None.
    /// # Errors
    /// * Fails if this pass was not created using [`PassBuilder::render()`]
    /// * Fails if `op` was [`vk::AttachmentLoadOp::CLEAR`], but `clear` was [`None`].
    pub fn depth_attachment(mut self, resource: &VirtualResource, op: vk::AttachmentLoadOp, clear: Option<vk::ClearDepthStencilValue>) -> Result<Self> {
        if !self.inner.is_renderpass {
            return Err(Error::Uncategorized("Cannot attach depth attachment to a pass that is not a renderpass").into());
        }
        self.inner.inputs.push(PassResource {
            usage: ResourceUsage::Attachment(AttachmentType::Depth),
            resource: resource.clone(),
            stage: match op {
                // from https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineStageFlagBits2.html.
                // 'Load' operations on depth/stencil attachments happen in EARLY_FRAGMENT_TESTS.
                vk::AttachmentLoadOp::LOAD => PipelineStage::EARLY_FRAGMENT_TESTS,
                vk::AttachmentLoadOp::CLEAR => PipelineStage::EARLY_FRAGMENT_TESTS,
                _ => todo!(),
            },
            layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            clear_value: None,
            load_op: None,
        });

        self.inner.outputs.push(PassResource {
            usage: ResourceUsage::Attachment(AttachmentType::Depth),
            resource: resource.upgrade(),
            // Depth/stencil writes happen in LATE_FRAGMENT_TESTS.
            // It's also legal to specify COLOR_ATTACHMENT_OUTPUT, but this is more precise.
            stage: PipelineStage::LATE_FRAGMENT_TESTS,
            layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            clear_value: clear.map(|c| vk::ClearValue {
                depth_stencil: c,
            }),
            load_op: Some(op),
        });

        Ok(self)
    }

    /// Does a hardware MSAA resolve from `src` into `dst`.
    pub fn resolve(mut self, src: &VirtualResource, dst: &VirtualResource) -> Self {
        self.inner.inputs.push(PassResource {
            usage: ResourceUsage::Attachment(AttachmentType::Resolve(src.clone())),
            resource: dst.clone(),
            stage: PipelineStage::COLOR_ATTACHMENT_OUTPUT, // RESOLVE is only for vkCmdResolve
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            clear_value: None,
            load_op: None,
        });

        self.inner.outputs.push(PassResource {
            usage: ResourceUsage::Attachment(AttachmentType::Resolve(src.clone())),
            resource: dst.upgrade(),
            stage: PipelineStage::COLOR_ATTACHMENT_OUTPUT, // RESOLVE is only for vkCmdResolve
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            clear_value: None,
            load_op: Some(vk::AttachmentLoadOp::DONT_CARE),
        });

        self
    }

    /// Declare that a resource will be used as a sampled image in the given pipeline stages.
    pub fn sample_image(mut self, resource: &VirtualResource, stage: PipelineStage) -> Self {
        self.inner.inputs.push(PassResource {
            usage: ResourceUsage::ShaderRead,
            resource: resource.clone(),
            stage,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            clear_value: None,
            load_op: None,
        });
        self
    }

    pub fn write_storage_image(mut self, resource: &VirtualResource, stage: PipelineStage) -> Self {
        self.inner.inputs.push(PassResource {
            usage: ResourceUsage::ShaderWrite,
            resource: resource.clone(),
            stage,
            layout: vk::ImageLayout::GENERAL,
            clear_value: None,
            load_op: None,
        });
        self.inner.outputs.push(PassResource {
            usage: ResourceUsage::ShaderWrite,
            resource: resource.upgrade(),
            stage,
            layout: vk::ImageLayout::GENERAL,
            clear_value: None,
            load_op: None,
        });
        self
    }

    pub fn read_storage_image(mut self, resource: &VirtualResource, stage: PipelineStage) -> Self {
        self.inner.inputs.push(PassResource {
            usage: ResourceUsage::ShaderRead,
            resource: resource.clone(),
            stage,
            layout: vk::ImageLayout::GENERAL,
            clear_value: None,
            load_op: None,
        });
        self
    }

    /// Set the executor to be called when recording this pass.
    pub fn executor(mut self, exec: impl PassExecutor<D, U, A> + 'cb) -> Self {
        self.inner.execute = Box::new(exec);
        self
    }

    pub fn execute_fn<F>(mut self, exec: F) -> Self
        where
            F: for<'q> FnMut(IncompleteCommandBuffer<'q, D>, &mut InFlightContext<A>, &PhysicalResourceBindings, &mut U) -> PassFnResult<'q, D> + 'cb, {
        self.inner.execute = Box::new(exec);
        self
    }

    /// Obtain a built [`Pass`] object.
    pub fn build(self) -> Pass<'cb, D, U, A> {
        self.inner
    }
}
