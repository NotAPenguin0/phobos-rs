use ash::vk;
use crate::{Error, GpuResource, IncompleteCommandBuffer, PhysicalResourceBindings, ResourceUsage, VirtualResource};
use crate::domain::ExecutionDomain;
use crate::pipeline::PipelineStage;

/// Represents one pass in a GPU task graph. You can obtain one using a [`PassBuilder`].
pub struct Pass<'exec, D> where D: ExecutionDomain {
    pub(crate) name: String,
    pub(crate) inputs: Vec<GpuResource>,
    pub(crate) outputs: Vec<GpuResource>,
    pub(crate) execute: Box<dyn FnMut(IncompleteCommandBuffer<D>, &PhysicalResourceBindings) -> IncompleteCommandBuffer<D> + 'exec>,
    pub(crate) is_renderpass: bool
}

/// Used to create [`Pass`] objects correctly.
pub struct PassBuilder<'exec, D> where D: ExecutionDomain {
    inner: Pass<'exec, D>,
}

impl<'exec, D> Pass<'exec, D> where D: ExecutionDomain {
    /// Returns the output virtual resource associated with the input resource.
    pub fn output(&self, resource: &VirtualResource) -> Option<VirtualResource> {
        self.outputs.iter().filter_map(|output| {
            if VirtualResource::are_associated(resource, &output.resource) { Some(output.resource.clone()) }
            else { None }
        }).next()
    }
}

impl<'exec, D> PassBuilder<'exec, D> where D: ExecutionDomain {
    /// Create a new renderpass.
    pub fn render(name: String) -> Self {
        PassBuilder {
            inner: Pass {
                name,
                execute: Box::new(|c, _| c),
                inputs: vec![],
                outputs: vec![],
                is_renderpass: true
            },
        }
    }

    /// Create a pass for presenting to the swapchain.
    /// Note that this doesn't actually do the presentation, it just adds the proper sync for it.
    /// If you are presenting to an output of the graph, this is required.
    pub fn present(name: String, swapchain: VirtualResource) -> Pass<'exec, D> {
        Pass {
            name,
            inputs: vec![GpuResource{
                usage: ResourceUsage::Present,
                resource: swapchain,
                stage: PipelineStage::BOTTOM_OF_PIPE,
                layout: vk::ImageLayout::PRESENT_SRC_KHR,
                clear_value: None,
                load_op: None,
            }],
            outputs: vec![],
            execute: Box::new(|c, _| c),
            is_renderpass: false
        }
    }

    /// Adds a color attachment to this pass. If [`vk::AttachmentLoadOp::CLEAR`] was specified, `clear` must not be None.
    pub fn color_attachment(mut self, resource: VirtualResource, op: vk::AttachmentLoadOp, clear: Option<vk::ClearColorValue>) -> Result<Self, Error> {
        if op == vk::AttachmentLoadOp::CLEAR && clear.is_none() { return Err(Error::NoClearValue); }
        self.inner.inputs.push(GpuResource {
            usage: ResourceUsage::Attachment,
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
            usage: ResourceUsage::Attachment,
            resource: resource.upgrade(),
            stage: PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            clear_value: clear.map(|c| vk::ClearValue { color: c }),
            load_op: Some(op)
        });

        Ok(self)
    }

    /// Adds a depth attachment to this pass. If [`vk::AttachmentLoadOp::CLEAR`] was specified, `clear` must not be None.
    pub fn depth_attachment(mut self, resource: VirtualResource, op: vk::AttachmentLoadOp, clear: Option<vk::ClearDepthStencilValue>) -> Self {
        self.inner.inputs.push(GpuResource {
            usage: ResourceUsage::Attachment,
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
            usage: ResourceUsage::Attachment,
            resource: resource.upgrade(),
            // Depth/stencil writes happen in LATE_FRAGMENT_TESTS.
            // It's also legal to specify COLOR_ATTACHMENT_OUTPUT, but this is more precise.
            stage: PipelineStage::LATE_FRAGMENT_TESTS,
            layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            clear_value: clear.map(|c| vk::ClearValue { depth_stencil: c }),
            load_op: Some(op)
        });

        self
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
    pub fn execute(mut self, exec: impl FnMut(IncompleteCommandBuffer<D>, &PhysicalResourceBindings) -> IncompleteCommandBuffer<D> + 'exec) -> Self {
        self.inner.execute = Box::new(exec);
        self
    }

    /// Obtain a built [`Pass`] object.
    pub fn build(self) -> Pass<'exec, D> {
        self.inner
    }
}