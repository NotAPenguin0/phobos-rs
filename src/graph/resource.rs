use ash::vk;
use crate::graph::virtual_resource::VirtualResource;

#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub enum ResourceType {
    #[default]
    Image,
    Buffer,
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub(crate) enum AttachmentType {
    #[default]
    Color,
    Depth,
    Resolve(VirtualResource)
}

/// Resource usage in a task graph.
#[derive(Debug, Default, PartialEq, Eq, Clone)]
#[allow(dead_code)]
pub(crate) enum ResourceUsage {
    #[default]
    Nothing,
    Present,
    Attachment(AttachmentType),
    ShaderRead,
    ShaderWrite,
}

impl ResourceUsage {
    pub fn access(&self) -> vk::AccessFlags2 {
        match self {
            ResourceUsage::Nothing => { vk::AccessFlags2::NONE }
            ResourceUsage::Present => { vk::AccessFlags2::NONE }
            ResourceUsage::Attachment(AttachmentType::Color) => { vk::AccessFlags2::COLOR_ATTACHMENT_WRITE }
            ResourceUsage::Attachment(AttachmentType::Depth) => { vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE }
            ResourceUsage::Attachment(AttachmentType::Resolve(_)) => { vk::AccessFlags2::COLOR_ATTACHMENT_WRITE }
            ResourceUsage::ShaderRead => { vk::AccessFlags2::SHADER_READ }
            ResourceUsage::ShaderWrite => { vk::AccessFlags2::SHADER_WRITE }
        }
    }

    pub fn is_read(&self) -> bool {
        match self {
            ResourceUsage::Nothing => { true }
            ResourceUsage::Present => { false }
            ResourceUsage::Attachment(_) => { false }
            ResourceUsage::ShaderRead => { true }
            ResourceUsage::ShaderWrite => { false }
        }
    }
}