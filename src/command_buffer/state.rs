use ash::vk;
use crate::ImageView;

pub(crate) struct RenderingAttachmentInfo {
    pub image_view: ImageView,
    pub image_layout: vk::ImageLayout,
    pub resolve_mode: Option<vk::ResolveModeFlags>,
    pub resolve_image_view: Option<ImageView>,
    pub resolve_image_layout: Option<vk::ImageLayout>,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub clear_value: vk::ClearValue,
}

pub(crate) struct RenderingInfo {
    pub flags: vk::RenderingFlags,
    pub render_area: vk::Rect2D,
    pub layer_count: u32,
    pub view_mask: u32,
    pub color_attachments: Vec<RenderingAttachmentInfo>,
    pub depth_attachment: Option<RenderingAttachmentInfo>,
    pub stencil_attachment: Option<RenderingAttachmentInfo>,
}