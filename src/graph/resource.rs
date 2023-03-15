use crate::graph::virtual_resource::VirtualResource;

#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub enum ResourceType {
    #[default]
    Image,
    Buffer,
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub enum AttachmentType {
    #[default]
    Color,
    Depth,
    Resolve(VirtualResource)
}

/// Resource usage in a task graph.
#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub enum ResourceUsage {
    #[default]
    Nothing,
    Present,
    Attachment(AttachmentType),
    ShaderRead,
    ShaderWrite,
}