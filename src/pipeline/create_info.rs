use ash::vk;
use crate::pipeline_layout::PipelineLayoutCreateInfo;
use crate::shader::ShaderCreateInfo;

#[derive(Debug, Copy, Clone)]
pub struct VertexInputBindingDescription(pub(super) vk::VertexInputBindingDescription);

#[derive(Debug, Copy, Clone)]
pub struct VertexInputAttributeDescription(pub(super) vk::VertexInputAttributeDescription);

#[derive(Debug, Copy, Clone)]
pub struct PipelineInputAssemblyStateCreateInfo(pub(super) vk::PipelineInputAssemblyStateCreateInfo);

#[derive(Debug, Copy, Clone)]
pub struct PipelineDepthStencilStateCreateInfo(pub(super) vk::PipelineDepthStencilStateCreateInfo);

#[derive(Debug, Copy, Clone)]
pub struct PipelineRasterizationStateCreateInfo(pub(super) vk::PipelineRasterizationStateCreateInfo);

#[derive(Debug, Copy, Clone)]
pub struct PipelineMultisampleStateCreateInfo(pub(super) vk::PipelineMultisampleStateCreateInfo);

#[derive(Debug, Copy, Clone)]
pub struct PipelineColorBlendAttachmentState(pub(super) vk::PipelineColorBlendAttachmentState);

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct PipelineRenderingInfo {
    pub view_mask: u32,
    pub color_formats: Vec<vk::Format>,
    pub depth_format: Option<vk::Format>,
    pub stencil_format: Option<vk::Format>,
}

#[derive(Debug, Copy, Clone)]
pub struct Viewport(pub(super) vk::Viewport);

#[derive(Debug, Copy, Clone)]
pub struct Rect2D(pub(super) vk::Rect2D);

/// Defines a full graphics pipeline. You can modify this manually, but all
/// information is also exposed through the pipeline builder,
/// with additional quality of life and presets.
#[derive(Debug, Clone, Derivative)]
#[derivative(PartialEq, Eq, Hash)]
pub struct PipelineCreateInfo {
    // TODO: Blend presets
    pub name: String,
    pub layout: PipelineLayoutCreateInfo,
    pub vertex_input_bindings: Vec<VertexInputBindingDescription>,
    pub vertex_attributes: Vec<VertexInputAttributeDescription>,
    pub shaders: Vec<ShaderCreateInfo>,
    pub input_assembly: PipelineInputAssemblyStateCreateInfo,
    pub depth_stencil: PipelineDepthStencilStateCreateInfo,
    pub dynamic_states: Vec<vk::DynamicState>,
    pub rasterizer: PipelineRasterizationStateCreateInfo,
    pub multisample: PipelineMultisampleStateCreateInfo,
    pub blend_attachments: Vec<PipelineColorBlendAttachmentState>,
    pub viewports: Vec<Viewport>,
    pub scissors: Vec<Rect2D>,
    pub blend_enable_logic_op: bool,
    pub(crate) rendering_info: PipelineRenderingInfo,

    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    pub(super) vk_vertex_inputs: Vec<vk::VertexInputBindingDescription>,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    pub(super) vk_attributes: Vec<vk::VertexInputAttributeDescription>,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    pub(super) vertex_input_state: vk::PipelineVertexInputStateCreateInfo,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    pub(super) vk_viewports: Vec<vk::Viewport>,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    pub(super) vk_scissors: Vec<vk::Rect2D>,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    pub(super) viewport_state: vk::PipelineViewportStateCreateInfo,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    pub(super) vk_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState>,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    pub(super) blend_state: vk::PipelineColorBlendStateCreateInfo,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    pub(super) vk_dynamic_state: vk::PipelineDynamicStateCreateInfo,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    pub(super) vk_rendering_state: vk::PipelineRenderingCreateInfo,
}


impl PipelineCreateInfo {
    pub fn build_rendering_state(&mut self) -> () {
        self.vk_rendering_state = vk::PipelineRenderingCreateInfo::builder()
            .view_mask(self.rendering_info.view_mask)
            .color_attachment_formats(self.rendering_info.color_formats.as_slice())
            .depth_attachment_format(self.rendering_info.depth_format.unwrap_or(vk::Format::UNDEFINED))
            .stencil_attachment_format(self.rendering_info.stencil_format.unwrap_or(vk::Format::UNDEFINED))
            .build();
    }

    pub fn build_inner(&mut self) -> () {
        self.vk_attributes = self.vertex_attributes.iter().map(|v| v.0.clone()).collect();
        self.vk_vertex_inputs = self.vertex_input_bindings.iter().map(|v| v.0.clone()).collect();
        self.vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(self.vk_vertex_inputs.as_slice())
            .vertex_attribute_descriptions(self.vk_attributes.as_slice())
            .build();
        self.vk_viewports = self.viewports.iter().map(|v| v.0.clone()).collect();
        self.vk_scissors = self.scissors.iter().map(|v| v.0.clone()).collect();
        self.viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(self.vk_viewports.as_slice())
            .scissors(self.vk_scissors.as_slice())
            .build();
        self.vk_blend_attachments = self.blend_attachments.iter().map(|v| v.0.clone()).collect();
        self.blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(self.blend_enable_logic_op)
            .attachments(self.vk_blend_attachments.as_slice())
            .build();
        self.vk_dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(self.dynamic_states.as_slice())
            .build();
        self.build_rendering_state();
    }

    // Shader stage not yet filled out
    pub(crate) fn to_vk(&self, layout: vk::PipelineLayout) -> vk::GraphicsPipelineCreateInfo {
        vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next:  (&self.vk_rendering_state as *const _) as *const std::ffi::c_void,
            flags: Default::default(),
            stage_count: 0,
            p_stages: std::ptr::null(),
            p_vertex_input_state: &self.vertex_input_state,
            p_input_assembly_state: &self.input_assembly.0,
            p_tessellation_state: std::ptr::null(),
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
