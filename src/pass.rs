use std::sync::Arc;
use ash::vk;
use crate::{GpuResource, IncompleteCommandBuffer, ResourceUsage, VirtualResource};
use crate::domain::ExecutionDomain;
use crate::pipeline::PipelineStage;

pub struct Pass<'exec, D> where D: ExecutionDomain {
    pub name: String,
    pub inputs: Vec<GpuResource>,
    pub outputs: Vec<GpuResource>,
    pub execute: Box<dyn FnMut(IncompleteCommandBuffer<D>) -> IncompleteCommandBuffer<D> + 'exec>,
    pub is_renderpass: bool
}

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
    pub fn render(name: String) -> Self {
        PassBuilder {
            inner: Pass {
                name,
                execute: Box::new(|c| c),
                inputs: vec![],
                outputs: vec![],
                is_renderpass: true
            },
        }
    }

    /// Create a pass for presenting to the swapchain.
    /// Note that this doesn't actually do the presentation, it just adds the proper sync for it.
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
            execute: Box::new(|c| c),
            is_renderpass: false
        }
    }

    pub fn color_attachment(mut self, resource: VirtualResource, op: vk::AttachmentLoadOp, clear: Option<vk::ClearColorValue>) -> Self {
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
        self
    }

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

    pub fn execute(mut self, exec: impl FnMut(IncompleteCommandBuffer<D>) -> IncompleteCommandBuffer<D> + 'exec) -> Self {
        self.inner.execute = Box::new(exec);
        self
    }

    pub fn get(self) -> Pass<'exec, D> {
        self.inner
    }
}