use ash::vk;
use ash::vk::AttachmentLoadOp;
use crate::{GpuResource, IncompleteCommandBuffer, ResourceUsage, VirtualResource};
use crate::domain::ExecutionDomain;
use crate::pipeline::PipelineStage;

pub struct Pass<D> where D: ExecutionDomain {
    pub name: String,
    pub inputs: Vec<GpuResource>,
    pub outputs: Vec<GpuResource>,
    pub execute: fn(&IncompleteCommandBuffer<D>) -> ()
}

pub struct PassBuilder<D> where D: ExecutionDomain {
    inner: Pass<D>,
}

impl<D> Pass<D> where D: ExecutionDomain {
    /// Returns the output virtual resource associated with the input resource.
    pub fn output(&self, resource: &VirtualResource) -> Option<VirtualResource> {
        self.outputs.iter().filter_map(|output| {
            if VirtualResource::are_associated(resource, &output.resource) { Some(output.resource.clone()) }
            else { None }
        }).next()
    }
}

impl<D> PassBuilder<D> where D: ExecutionDomain {
    pub fn new(name: String) -> Self {
        PassBuilder {
            inner: Pass {
                name,
                execute: |_| {},
                inputs: vec![],
                outputs: vec![]
            },
        }
    }

    /// Create a pass for presenting to the swapchain.
    /// Note that this doesn't actually do the presentation, it just adds the proper sync for it.
    pub fn present(name: String, swapchain: VirtualResource) -> Pass<D> {
        Pass {
            name,
            inputs: vec![GpuResource{
                usage: ResourceUsage::Present,
                resource: swapchain,
                stage: PipelineStage::BOTTOM_OF_PIPE,
            }],
            outputs: vec![],
            execute: |_| {}
        }
    }

    pub fn color_attachment(mut self, resource: VirtualResource, op: vk::AttachmentLoadOp) -> Self {
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
            }
        });

        self.inner.outputs.push(GpuResource{
            usage: ResourceUsage::Attachment,
            resource: resource.upgrade(),
            stage: PipelineStage::COLOR_ATTACHMENT_OUTPUT,
        });
        self
    }

    pub fn depth_attachment(mut self, resource: VirtualResource, op: vk::AttachmentLoadOp) -> Self {
        self.inner.inputs.push(GpuResource {
            usage: ResourceUsage::Attachment,
            resource: resource.clone(),
            stage: match op {
                // from https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineStageFlagBits2.html.
                // 'Load' operations on depth/stencil attachments happen in EARLY_FRAGMENT_TESTS.
                vk::AttachmentLoadOp::LOAD => PipelineStage::EARLY_FRAGMENT_TESTS,
                vk::AttachmentLoadOp::CLEAR => PipelineStage::EARLY_FRAGMENT_TESTS,
                _ => todo!()
            }
        });

        self.inner.outputs.push(GpuResource {
            usage: ResourceUsage::Attachment,
            resource: resource.upgrade(),
            // Depth/stencil writes happen in LATE_FRAGMENT_TESTS.
            // It's also legal to specify COLOR_ATTACHMENT_OUTPUT, but this is more precise.
            stage: PipelineStage::LATE_FRAGMENT_TESTS
        });

        self
    }

    pub fn sample_image(mut self, resource: VirtualResource, stage: PipelineStage) -> Self {
        self.inner.inputs.push(GpuResource{
            usage: ResourceUsage::ShaderRead,
            resource,
            stage,
        });
        self
    }

    pub fn get(self) -> Pass<D> {
        self.inner
    }
}