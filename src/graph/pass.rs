//! This module mainly exposes the [`PassBuilder`] struct, used for correctly defining passes in a
//! [`PassGraph`](crate::PassGraph).
//!
//! For documentation on how to use the pass graph, refer to the [`graph`](crate::graph) module level documentation.
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
//!     .color_attachment(&swapchain,
//!                       vk::AttachmentLoadOp::CLEAR,
//!                       Some(vk::ClearColorValue{ float32: [0.0, 0.0, 0.0, 1.0] }))?
//!     // We sample the input resource in the fragment shader.
//!     .sample_image(&input_resource, PipelineStage::FRAGMENT_SHADER)
//!     .execute_fn(|mut cmd, local_pool, bindings, _| {
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

#[cfg(feature = "fsr2")]
use anyhow::{anyhow, bail};
use anyhow::Result;
use ash::vk;

use crate::{Allocator, DefaultAllocator, Error, PhysicalResourceBindings, VirtualResource};
#[cfg(feature = "fsr2")]
use crate::{ComputeSupport, Device, ImageView};
use crate::command_buffer::IncompleteCommandBuffer;
#[cfg(feature = "fsr2")]
use crate::fsr2::{Fsr2DispatchDescription, Fsr2DispatchResources};
use crate::graph::pass_graph::PassResource;
#[cfg(feature = "fsr2")]
use crate::graph::physical_resource::PhysicalResource;
use crate::graph::resource::{AttachmentType, ResourceUsage};
use crate::pipeline::PipelineStage;
use crate::pool::LocalPool;
use crate::sync::domain::ExecutionDomain;
use crate::util::to_vk::IntoVulkanType;

/// The returned value from a pass callback function.
pub type PassFnResult<'q, D, A> = Result<IncompleteCommandBuffer<'q, D, A>>;

/// Defines a pass executor that can be called when the pass is recorded.
pub trait PassExecutor<D: ExecutionDomain, U, A: Allocator> {
    /// Record this pass to a command buffer.
    fn execute<'q>(
        &mut self,
        cmd: IncompleteCommandBuffer<'q, D, A>,
        local_pool: &mut LocalPool<A>,
        bindings: &PhysicalResourceBindings,
        user_data: &mut U,
    ) -> PassFnResult<'q, D, A>;
}

impl<D, U, A, F> PassExecutor<D, U, A> for F
where
    D: ExecutionDomain,
    A: Allocator,
    F: for<'q> FnMut(
        IncompleteCommandBuffer<'q, D, A>,
        &mut LocalPool<A>,
        &PhysicalResourceBindings,
        &mut U,
    ) -> PassFnResult<'q, D, A>,
{
    /// Record this pass to a command buffer by calling the given function.
    fn execute<'q>(
        &mut self,
        cmd: IncompleteCommandBuffer<'q, D, A>,
        local_pool: &mut LocalPool<A>,
        bindings: &PhysicalResourceBindings,
        user_data: &mut U,
    ) -> PassFnResult<'q, D, A> {
        self(cmd, local_pool, bindings, user_data)
    }
}

pub(crate) type BoxedPassFn<'cb, D, U, A> = Box<dyn PassExecutor<D, U, A> + 'cb>;

/// An empty pass executor that does nothing
pub struct EmptyPassExecutor;

impl EmptyPassExecutor {
    /// Creates an empty pass executor
    pub fn new() -> Self {
        Self {}
    }

    /// Create a new empty pass executor in a [`Box`]
    pub fn new_boxed() -> Box<Self> {
        Box::new(Self::new())
    }
}

impl<D: ExecutionDomain, U, A: Allocator> PassExecutor<D, U, A> for EmptyPassExecutor {
    /// Execute the empty pass executor by just returning the command buffer.
    fn execute<'q>(
        &mut self,
        cmd: IncompleteCommandBuffer<'q, D, A>,
        _local_pool: &mut LocalPool<A>,
        _bindings: &PhysicalResourceBindings,
        _user_data: &mut U,
    ) -> PassFnResult<'q, D, A> {
        Ok(cmd)
    }
}

/// Represents one pass in a GPU task graph. You can obtain one using a [`PassBuilder`].
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Pass<'cb, D: ExecutionDomain, U = (), A: Allocator = DefaultAllocator> {
    pub(crate) name: String,
    pub(crate) color: Option<[f32; 4]>,
    pub(crate) inputs: Vec<PassResource>,
    pub(crate) outputs: Vec<PassResource>,
    #[derivative(Debug = "ignore")]
    pub(crate) execute: BoxedPassFn<'cb, D, U, A>,
    pub(crate) is_renderpass: bool,
}

#[derive(Copy, Clone, Debug)]
pub enum ClearColor {
    Float([f32; 4]),
    Int([i32; 4]),
    Uint([u32; 4]),
}

#[derive(Copy, Clone, Default, Debug)]
pub struct ClearDepthStencil {
    pub depth: f32,
    pub stencil: u32,
}

impl IntoVulkanType for ClearColor {
    type Output = vk::ClearColorValue;

    fn into_vulkan(self) -> Self::Output {
        match self {
            ClearColor::Float(values) => vk::ClearColorValue {
                float32: values,
            },
            ClearColor::Int(values) => vk::ClearColorValue {
                int32: values,
            },
            ClearColor::Uint(values) => vk::ClearColorValue {
                uint32: values,
            },
        }
    }
}

impl IntoVulkanType for ClearDepthStencil {
    type Output = vk::ClearDepthStencilValue;

    fn into_vulkan(self) -> Self::Output {
        vk::ClearDepthStencilValue {
            depth: self.depth,
            stencil: self.stencil,
        }
    }
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

    /// Get the pass name
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
    pub fn color_attachment(
        mut self,
        resource: &VirtualResource,
        op: vk::AttachmentLoadOp,
        clear: Option<vk::ClearColorValue>,
    ) -> Result<Self> {
        if !self.inner.is_renderpass {
            return Err(Error::Uncategorized(
                "Cannot attach color attachment to a pass that is not a renderpass",
            )
            .into());
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

    /// Clear a color attachment with the specified clear color
    /// # Errors
    /// * Fails if this pass was not created using [`PassBuilder::render()`]
    pub fn clear_color_attachment(
        self,
        resource: &VirtualResource,
        color: ClearColor,
    ) -> Result<Self> {
        self.color_attachment(resource, vk::AttachmentLoadOp::CLEAR, Some(color.into_vulkan()))
    }

    /// Load a color attachment
    /// # Errors
    /// * Fails if this pass was not created using [`PassBuilder::render()`]
    pub fn load_color_attachment(self, resource: &VirtualResource) -> Result<Self> {
        self.color_attachment(resource, vk::AttachmentLoadOp::LOAD, None)
    }

    /// Clear the depth attachment with the specified clear values
    /// # Errors
    /// * Fails if this pass was not created using [`PassBuilder::render()`]
    pub fn clear_depth_attachment(
        self,
        resource: &VirtualResource,
        clear: ClearDepthStencil,
    ) -> Result<Self> {
        self.depth_attachment(resource, vk::AttachmentLoadOp::CLEAR, Some(clear.into_vulkan()))
    }

    /// Load a depth attachment
    /// # Errors
    /// * Fails if this pass was not created using [`PassBuilder::render()`]
    pub fn load_depth_attachment(self, resource: &VirtualResource) -> Result<Self> {
        self.depth_attachment(resource, vk::AttachmentLoadOp::LOAD, None)
    }

    /// Adds a depth attachment to this pass. If [`vk::AttachmentLoadOp::CLEAR`] was specified, `clear` must not be None.
    /// # Errors
    /// * Fails if this pass was not created using [`PassBuilder::render()`]
    /// * Fails if `op` was [`vk::AttachmentLoadOp::CLEAR`], but `clear` was [`None`].
    pub fn depth_attachment(
        mut self,
        resource: &VirtualResource,
        op: vk::AttachmentLoadOp,
        clear: Option<vk::ClearDepthStencilValue>,
    ) -> Result<Self> {
        if !self.inner.is_renderpass {
            return Err(Error::Uncategorized(
                "Cannot attach depth attachment to a pass that is not a renderpass",
            )
            .into());
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

    /// Does a hardware MSAA resolve from `src` into `dst`, but for depth images
    pub fn resolve_depth(mut self, src: &VirtualResource, dst: &VirtualResource) -> Self {
        self.inner.inputs.push(PassResource {
            usage: ResourceUsage::Attachment(AttachmentType::Resolve(src.clone())),
            resource: dst.clone(),
            stage: PipelineStage::COLOR_ATTACHMENT_OUTPUT, // RESOLVE is only for vkCmdResolve
            layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            clear_value: None,
            load_op: None,
        });

        self.inner.outputs.push(PassResource {
            usage: ResourceUsage::Attachment(AttachmentType::Resolve(src.clone())),
            resource: dst.upgrade(),
            stage: PipelineStage::COLOR_ATTACHMENT_OUTPUT, // RESOLVE is only for vkCmdResolve
            layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
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

    /// Declare that a resource will be used as a storage image that is written to in the given pipeline stages.
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

    /// Declare that a resource will be used as a storage image that will be read from in the given pipeline stages.
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

    #[allow(dead_code)]
    fn sample_optional_image(
        self,
        resource: &Option<VirtualResource>,
        stage: PipelineStage,
    ) -> Self {
        match resource {
            None => self,
            Some(resource) => self.sample_image(resource, stage),
        }
    }

    /// Set the executor to be called when recording this pass.
    pub fn executor(mut self, exec: impl PassExecutor<D, U, A> + 'cb) -> Self {
        self.inner.execute = Box::new(exec);
        self
    }

    /// Set the executor to be called when recording this pass. This method can be used to deduce types
    /// when a function is used as a pass executor.
    pub fn execute_fn<F>(mut self, exec: F) -> Self
    where
        F: for<'q> FnMut(
                IncompleteCommandBuffer<'q, D, A>,
                &mut LocalPool<A>,
                &PhysicalResourceBindings,
                &mut U,
            ) -> PassFnResult<'q, D, A>
            + 'cb, {
        self.inner.execute = Box::new(exec);
        self
    }

    /// Obtain a built [`Pass`] object.
    pub fn build(self) -> Pass<'cb, D, U, A> {
        self.inner
    }
}

/// Holds virtual resources needed to declare a FSR2 dispatch.
#[cfg(feature = "fsr2")]
#[derive(Debug, Clone)]
pub struct Fsr2DispatchVirtualResources {
    /// Color buffer for the current frame, at render resolution.
    pub color: VirtualResource,
    /// Depth buffer for the current frame, at render resolution
    pub depth: VirtualResource,
    /// Motion vectors for the current frame, at render resolution
    pub motion_vectors: VirtualResource,
    /// Optional 1x1 texture with the exposure value
    pub exposure: Option<VirtualResource>,
    /// Optional resource with the alpha value of reactive objects in the scene
    pub reactive: Option<VirtualResource>,
    /// Optional resource with the alpha value of special objects in the scene
    pub transparency_and_composition: Option<VirtualResource>,
    /// Output color buffer for the current frame at presentation resolution
    pub output: VirtualResource,
}

#[cfg(feature = "fsr2")]
impl Fsr2DispatchVirtualResources {
    fn resolve_image_resource(
        resource: &VirtualResource,
        bindings: &PhysicalResourceBindings,
    ) -> Result<ImageView> {
        let resolved = bindings.resolve(resource).ok_or_else(|| {
            anyhow!("Missing physical binding for required FSR2 resource: {}", resource.name())
        })?;
        let PhysicalResource::Image(image) = &resolved else {
            bail!("FSR2 resource {} should be an image", resource.name());
        };

        Ok(image.clone())
    }

    fn resolve_optional_image_resource(
        resource: &Option<VirtualResource>,
        bindings: &PhysicalResourceBindings,
    ) -> Result<Option<ImageView>> {
        match resource {
            None => Ok(None),
            Some(resource) => Ok(Some(Self::resolve_image_resource(resource, bindings)?)),
        }
    }

    /// Resolve all resources to their respective physical resources.
    /// Fails if any resource resolve fails.
    pub fn resolve(&self, bindings: &PhysicalResourceBindings) -> Result<Fsr2DispatchResources> {
        Ok(Fsr2DispatchResources {
            color: Self::resolve_image_resource(&self.color, bindings)?,
            depth: Self::resolve_image_resource(&self.depth, bindings)?,
            motion_vectors: Self::resolve_image_resource(&self.motion_vectors, bindings)?,
            exposure: Self::resolve_optional_image_resource(&self.exposure, bindings)?,
            reactive: Self::resolve_optional_image_resource(&self.reactive, bindings)?,
            transparency_and_composition: Self::resolve_optional_image_resource(
                &self.transparency_and_composition,
                bindings,
            )?,
            output: Self::resolve_image_resource(&self.output, bindings)?,
        })
    }
}

#[cfg(feature = "fsr2")]
impl<'cb, D: ExecutionDomain + ComputeSupport, U, A: Allocator> PassBuilder<'cb, D, U, A> {
    /// Create a pass for FSR2.
    pub fn fsr2(
        device: Device,
        descr: Fsr2DispatchDescription,
        resources: Fsr2DispatchVirtualResources,
    ) -> Pass<'cb, D, U, A> {
        let pass = PassBuilder::<'cb, D, U, A>::new("fsr2")
            .sample_image(&resources.color, PipelineStage::COMPUTE_SHADER)
            .sample_image(&resources.motion_vectors, PipelineStage::COMPUTE_SHADER)
            .sample_image(&resources.depth, PipelineStage::COMPUTE_SHADER)
            .sample_optional_image(&resources.exposure, PipelineStage::COMPUTE_SHADER)
            .sample_optional_image(&resources.reactive, PipelineStage::COMPUTE_SHADER)
            .sample_optional_image(
                &resources.transparency_and_composition,
                PipelineStage::COMPUTE_SHADER,
            )
            .write_storage_image(&resources.output, PipelineStage::COMPUTE_SHADER)
            .execute_fn(move |cmd, _, bindings, _| {
                let resolved_resources = resources.resolve(bindings)?;
                let mut fsr2 = device.fsr2_context();
                fsr2.dispatch(&descr, &resolved_resources, &cmd)?;
                Ok(cmd)
            })
            .build();
        pass
    }
}
