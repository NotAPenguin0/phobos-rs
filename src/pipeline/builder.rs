use std::collections::HashMap;
use ash::vk;

use crate::{ByteSize, Error};
use crate::create_info::*;

use anyhow::Result;
use crate::shader::ShaderCreateInfo;

/// Used to facilitate creating a graphics pipeline.
pub struct PipelineBuilder {
    inner: PipelineCreateInfo,
    vertex_binding_offsets: HashMap<u32, u32>,
}


impl PipelineBuilder {
    /// Create a new empty pipeline with default settings for everything.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            inner: PipelineCreateInfo {
                name: name.into(),
                layout: Default::default(),
                vertex_input_bindings: vec![],
                vertex_attributes: vec![],
                shaders: vec![],
                input_assembly: PipelineInputAssemblyStateCreateInfo {
                    0: vk::PipelineInputAssemblyStateCreateInfo {
                        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                        primitive_restart_enable: vk::FALSE,
                    }
                },
                depth_stencil: PipelineDepthStencilStateCreateInfo {
                    0: vk::PipelineDepthStencilStateCreateInfo {
                        s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        depth_test_enable: vk::FALSE,
                        depth_write_enable: vk::FALSE,
                        depth_compare_op: Default::default(),
                        depth_bounds_test_enable: vk::FALSE,
                        stencil_test_enable: vk::FALSE,
                        front: vk::StencilOpState {
                            fail_op: Default::default(),
                            pass_op: Default::default(),
                            depth_fail_op: Default::default(),
                            compare_op: Default::default(),
                            compare_mask: 0,
                            write_mask: 0,
                            reference: 0,
                        },
                        back: vk::StencilOpState {
                            fail_op: Default::default(),
                            pass_op: Default::default(),
                            depth_fail_op: Default::default(),
                            compare_op: Default::default(),
                            compare_mask: 0,
                            write_mask: 0,
                            reference: 0,
                        },
                        min_depth_bounds: 0.0,
                        max_depth_bounds: 0.0,
                    }
                },
                dynamic_states: vec![],
                rasterizer: PipelineRasterizationStateCreateInfo {
                    0: vk::PipelineRasterizationStateCreateInfo {
                        s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        depth_clamp_enable: vk::FALSE,
                        rasterizer_discard_enable: vk::FALSE,
                        polygon_mode: vk::PolygonMode::FILL,
                        cull_mode: vk::CullModeFlags::NONE,
                        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                        depth_bias_enable: vk::FALSE,
                        depth_bias_constant_factor: 0.0,
                        depth_bias_clamp: 0.0,
                        depth_bias_slope_factor: 0.0,
                        line_width: 1.0,
                    }
                },
                multisample: PipelineMultisampleStateCreateInfo {
                    0: vk::PipelineMultisampleStateCreateInfo {
                        s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        rasterization_samples: vk::SampleCountFlags::TYPE_1,
                        sample_shading_enable: vk::FALSE,
                        min_sample_shading: 0.0,
                        p_sample_mask: std::ptr::null(),
                        alpha_to_coverage_enable: vk::FALSE,
                        alpha_to_one_enable: vk::FALSE
                    }
                },
                blend_attachments: vec![],
                viewports: vec![],
                scissors: vec![],
                blend_enable_logic_op: false,
                rendering_info: PipelineRenderingInfo {
                    view_mask: 0,
                    color_formats: vec![],
                    depth_format: None,
                    stencil_format: None,
                },
                vk_vertex_inputs: vec![],
                vk_attributes: vec![],
                vertex_input_state: vk::PipelineVertexInputStateCreateInfo {
                    s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: Default::default(),
                    vertex_binding_description_count: 0,
                    p_vertex_binding_descriptions: std::ptr::null(),
                    vertex_attribute_description_count: 0,
                    p_vertex_attribute_descriptions: std::ptr::null(),
                },
                vk_viewports: vec![],
                vk_scissors: vec![],
                viewport_state: Default::default(),
                vk_blend_attachments: vec![],
                blend_state: Default::default(),
                vk_dynamic_state: Default::default(),
                vk_rendering_state: Default::default(),
            },
            vertex_binding_offsets: Default::default(),
        }
    }

    pub fn vertex_input(mut self, binding: u32, rate: vk::VertexInputRate) -> Self {
        self.vertex_binding_offsets.insert(0, 0);
        self.inner.vertex_input_bindings.push(VertexInputBindingDescription{
            0: vk::VertexInputBindingDescription {
                binding,
                stride: 0,
                input_rate: rate,
            }});
        self
    }

    pub fn vertex_attribute(mut self, binding: u32, location: u32, format: vk::Format) -> Result<Self> {
        let offset = self.vertex_binding_offsets.get_mut(&binding).ok_or(Error::NoVertexBinding)?;
        self.inner.vertex_attributes.push(VertexInputAttributeDescription{
            0: vk::VertexInputAttributeDescription {
                location,
                binding,
                format,
                offset: *offset,
            }});
        *offset += format.byte_size() as u32;
        for binding in &mut self.inner.vertex_input_bindings {
            binding.0.stride += format.byte_size() as u32;
        }

        Ok(self)
    }

    pub fn attach_shader(mut self, info: ShaderCreateInfo) -> Self {
        self.inner.shaders.push(info);
        self
    }

    pub fn depth_test(mut self, enable: bool) -> Self {
        self.inner.depth_stencil.0.depth_test_enable = vk::Bool32::from(enable);
        self
    }

    pub fn depth_write(mut self, enable: bool) -> Self {
        self.inner.depth_stencil.0.depth_write_enable = vk::Bool32::from(enable);
        self
    }

    pub fn depth_op(mut self, op: vk::CompareOp) -> Self {
        self.inner.depth_stencil.0.depth_compare_op = op;
        self
    }

    pub fn depth_clamp(mut self, enable: bool) -> Self {
        self.inner.rasterizer.0.depth_clamp_enable = vk::Bool32::from(enable);
        self
    }

    pub fn depth(self, test: bool, write: bool, clamp: bool, op: vk::CompareOp) -> Self {
        self.depth_test(test)
            .depth_write(write)
            .depth_clamp(clamp)
            .depth_op(op)
    }

    pub fn dynamic_state(mut self, state: vk::DynamicState) -> Self {
        self.inner.dynamic_states.push(state);
        // When setting a viewport dynamic state, we still need a dummy viewport to make validation shut up
        if state == vk::DynamicState::VIEWPORT {
            self.inner.viewports.push(Viewport{
                0: vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: 0.0,
                    height: 0.0,
                    min_depth: 0.0,
                    max_depth: 0.0,
                }
            })
        }
        // Same with scissor state
        if state == vk::DynamicState::SCISSOR {
            self.inner.scissors.push(Rect2D{
                0: vk::Rect2D {
                    offset: Default::default(),
                    extent: Default::default(),
                }})
        }

        self
    }

    pub fn dynamic_states(mut self, states: &[vk::DynamicState]) -> Self {
        for state in states {
            self = self.dynamic_state(*state);
        }
        self
    }

    pub fn polygon_mode(mut self, mode: vk::PolygonMode) -> Self {
        self.inner.rasterizer.0.polygon_mode = mode;
        self
    }

    pub fn cull_mask(mut self, cull: vk::CullModeFlags) -> Self {
        self.inner.rasterizer.0.cull_mode = cull;
        self
    }

    pub fn front_face(mut self, face: vk::FrontFace) -> Self {
        self.inner.rasterizer.0.front_face = face;
        self
    }

    pub fn samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.inner.multisample.0.rasterization_samples = samples;
        self
    }

    pub fn sample_shading(mut self, value: f32) -> Self {
        self.inner.multisample.0.sample_shading_enable = vk::TRUE;
        self.inner.multisample.0.min_sample_shading = value;
        self
    }

    pub fn blend_attachment_none(mut self) -> Self {
        self.inner.blend_attachments.push(PipelineColorBlendAttachmentState{
            0: vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ONE,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::RGBA,
            }});
        self
    }

    pub fn blend_additive_unmasked(mut self, src: vk::BlendFactor, dst: vk::BlendFactor, src_alpha: vk::BlendFactor, dst_alpha: vk::BlendFactor) -> Self {
        self.inner.blend_attachments.push(PipelineColorBlendAttachmentState{
            0: vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                src_color_blend_factor: src,
                dst_color_blend_factor: dst,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: src_alpha,
                dst_alpha_blend_factor: dst_alpha,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::RGBA,
            }});
        self
    }

    /// Build the pipeline create info structure.
    pub fn build(self) -> PipelineCreateInfo {
        self.inner
    }

    pub fn get_name(&self) -> &str {
        &self.inner.name
    }
}