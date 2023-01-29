use ash::vk;
use ash::vk::AttachmentLoadOp;
use crate::{GpuResource, ResourceUsage, VirtualResource};
use crate::pipeline::PipelineStage;

#[derive(Default)]
pub struct Pass {
    pub name: String,
    pub inputs: Vec<GpuResource>,
    pub outputs: Vec<GpuResource>,
}

pub struct PassBuilder {
    inner: Pass,
}

impl Pass {
    /// Returns the output virtual resource associated with the input resource.
    pub fn output(&self, resource: &VirtualResource) -> Option<VirtualResource> {
        self.outputs.iter().filter_map(|output| {
            if VirtualResource::are_associated(resource, &output.resource) { Some(output.resource.clone()) }
            else { None }
        }).next()
    }
}

impl PassBuilder {
    pub fn new(name: String) -> Self {
        PassBuilder {
            inner: Pass {
                name,
                ..Default::default()
            },
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

    pub fn get(self) -> Pass {
        self.inner
    }
}