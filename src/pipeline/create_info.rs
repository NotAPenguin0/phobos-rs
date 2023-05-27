//! Wrapper structs for pipeline create info objects.

use ash::vk;

use crate::pipeline::pipeline_layout::PipelineLayoutCreateInfo;
use crate::ShaderCreateInfo;

#[derive(Debug, Copy, Clone)]
pub(crate) struct VertexInputBindingDescription(pub(super) vk::VertexInputBindingDescription);

#[derive(Debug, Copy, Clone)]
pub(crate) struct VertexInputAttributeDescription(pub(super) vk::VertexInputAttributeDescription);

#[derive(Debug, Copy, Clone)]
pub(crate) struct PipelineInputAssemblyStateCreateInfo(
    pub(super) vk::PipelineInputAssemblyStateCreateInfo,
);

#[derive(Debug, Copy, Clone)]
pub(crate) struct PipelineDepthStencilStateCreateInfo(
    pub(super) vk::PipelineDepthStencilStateCreateInfo,
);

#[derive(Debug, Copy, Clone)]
pub(crate) struct PipelineRasterizationStateCreateInfo(
    pub(super) vk::PipelineRasterizationStateCreateInfo,
);

#[derive(Debug, Copy, Clone)]
pub(crate) struct PipelineMultisampleStateCreateInfo(
    pub(super) vk::PipelineMultisampleStateCreateInfo,
);

#[derive(Debug, Copy, Clone)]
pub(crate) struct PipelineColorBlendAttachmentState(
    pub(super) vk::PipelineColorBlendAttachmentState,
);

#[derive(Default, Debug, Copy, Clone)]
pub(crate) struct PipelineTessellationStateCreateInfo(
    pub(super) vk::PipelineTessellationStateCreateInfo,
);

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct PipelineRenderingInfo {
    pub view_mask: u32,
    pub color_formats: Vec<vk::Format>,
    pub depth_format: Option<vk::Format>,
    pub stencil_format: Option<vk::Format>,
}

/// Newtype wrapper for a Vulkan viewport. Implements `Hash` and `Eq`.
#[derive(Debug, Copy, Clone)]
pub struct Viewport(pub(super) vk::Viewport);

/// Newtype wrapper for a Vulkan Rect2D. Implements `Hash` and `Eq`.
#[derive(Debug, Copy, Clone)]
pub struct Rect2D(pub(super) vk::Rect2D);

/// Defines a full graphics pipeline. Use the pipeline builder to construct this properly.
#[derive(Debug, Clone, Derivative)]
#[derivative(PartialEq, Eq, Hash)]
pub struct PipelineCreateInfo {
    /// The shaders used in this pipeline
    pub shaders: Vec<ShaderCreateInfo>,
    pub(crate) name: String,
    pub(crate) layout: PipelineLayoutCreateInfo,
    pub(crate) vertex_input_bindings: Vec<VertexInputBindingDescription>,
    pub(crate) vertex_attributes: Vec<VertexInputAttributeDescription>,
    pub(crate) input_assembly: PipelineInputAssemblyStateCreateInfo,
    pub(crate) depth_stencil: PipelineDepthStencilStateCreateInfo,
    pub(crate) dynamic_states: Vec<vk::DynamicState>,
    pub(crate) rasterizer: PipelineRasterizationStateCreateInfo,
    pub(crate) multisample: PipelineMultisampleStateCreateInfo,
    pub(crate) blend_attachments: Vec<PipelineColorBlendAttachmentState>,
    pub(crate) viewports: Vec<Viewport>,
    pub(crate) scissors: Vec<Rect2D>,
    pub(crate) blend_enable_logic_op: bool,
    pub(crate) rendering_info: PipelineRenderingInfo,
    pub(crate) tesselation_info: Option<PipelineTessellationStateCreateInfo>,

    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub(super) vk_vertex_inputs: Vec<vk::VertexInputBindingDescription>,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub(super) vk_attributes: Vec<vk::VertexInputAttributeDescription>,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub(super) vertex_input_state: vk::PipelineVertexInputStateCreateInfo,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub(super) vk_viewports: Vec<vk::Viewport>,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub(super) vk_scissors: Vec<vk::Rect2D>,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub(super) viewport_state: vk::PipelineViewportStateCreateInfo,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub(super) vk_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState>,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub(super) blend_state: vk::PipelineColorBlendStateCreateInfo,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub(super) vk_dynamic_state: vk::PipelineDynamicStateCreateInfo,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub(super) vk_rendering_state: vk::PipelineRenderingCreateInfo,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub(super) vk_tessellation_state: Option<vk::PipelineTessellationStateCreateInfo>,
}

impl PipelineCreateInfo {
    pub(crate) fn build_rendering_state(&mut self) {
        self.vk_rendering_state = vk::PipelineRenderingCreateInfo::builder()
            .view_mask(self.rendering_info.view_mask)
            .color_attachment_formats(self.rendering_info.color_formats.as_slice())
            .depth_attachment_format(
                self.rendering_info
                    .depth_format
                    .unwrap_or(vk::Format::UNDEFINED),
            )
            .stencil_attachment_format(
                self.rendering_info
                    .stencil_format
                    .unwrap_or(vk::Format::UNDEFINED),
            )
            .build();
    }

    /// Build the inner state of the pipeline. This must be called at least once before actually creating the pipeline.
    /// When normally registering to the cache, this all happens automatically.
    pub fn build_inner(&mut self) {
        self.vk_attributes = self.vertex_attributes.iter().map(|v| v.0).collect();
        self.vk_vertex_inputs = self.vertex_input_bindings.iter().map(|v| v.0).collect();
        self.vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(self.vk_vertex_inputs.as_slice())
            .vertex_attribute_descriptions(self.vk_attributes.as_slice())
            .build();
        self.vk_viewports = self.viewports.iter().map(|v| v.0).collect();
        self.vk_scissors = self.scissors.iter().map(|v| v.0).collect();
        self.viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(self.vk_viewports.as_slice())
            .scissors(self.vk_scissors.as_slice())
            .build();
        self.vk_blend_attachments = self.blend_attachments.iter().map(|v| v.0).collect();
        self.blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(self.blend_enable_logic_op)
            .attachments(self.vk_blend_attachments.as_slice())
            .build();
        self.vk_dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(self.dynamic_states.as_slice())
            .build();
        self.vk_tessellation_state = self.tesselation_info.map(|info| {
            vk::PipelineTessellationStateCreateInfo::builder()
                .patch_control_points(info.0.patch_control_points)
                .flags(info.0.flags)
                .build()
        });
        self.build_rendering_state();
    }

    // Shader stage not yet filled out
    pub(crate) fn to_vk(&self, layout: vk::PipelineLayout) -> vk::GraphicsPipelineCreateInfo {
        vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: (&self.vk_rendering_state as *const _) as *const std::ffi::c_void,
            flags: Default::default(),
            stage_count: 0,
            p_stages: std::ptr::null(),
            p_vertex_input_state: &self.vertex_input_state,
            p_input_assembly_state: &self.input_assembly.0,
            p_tessellation_state: match &self.vk_tessellation_state {
                None => std::ptr::null(),
                Some(info) => info,
            },
            p_viewport_state: &self.viewport_state,
            p_rasterization_state: &self.rasterizer.0,
            p_multisample_state: &self.multisample.0,
            p_depth_stencil_state: &self.depth_stencil.0,
            p_color_blend_state: &self.blend_state,
            p_dynamic_state: &self.vk_dynamic_state,
            layout,
            render_pass: Default::default(),
            subpass: 0,
            base_pipeline_handle: Default::default(),
            base_pipeline_index: 0,
        }
    }
}
