//! The pipeline builder is used to easily create graphics pipelines correctly.

use std::collections::HashMap;

use anyhow::Result;
use ash::vk;

use crate::{ByteSize, Error, PipelineCreateInfo, ShaderCreateInfo};
use crate::pipeline::create_info::*;

/// Used to facilitate creating a graphics pipeline. For an example, please check the
/// [`pipeline`](crate::pipeline) module level documentation.
///
/// For information on each method, please check the Vulkan spec for
/// [`VkGraphicsPipelineCreateInfo`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html).
/// All builder methods correspond to entries directly, with minimal utilities added on top to enforce some invariants.
#[derive(Debug)]
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
                input_assembly: PipelineInputAssemblyStateCreateInfo(
                    vk::PipelineInputAssemblyStateCreateInfo {
                        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                        primitive_restart_enable: vk::FALSE,
                    },
                ),
                depth_stencil: PipelineDepthStencilStateCreateInfo(
                    vk::PipelineDepthStencilStateCreateInfo {
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
                    },
                ),
                dynamic_states: vec![],
                rasterizer: PipelineRasterizationStateCreateInfo(
                    vk::PipelineRasterizationStateCreateInfo {
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
                    },
                ),
                multisample: PipelineMultisampleStateCreateInfo(
                    vk::PipelineMultisampleStateCreateInfo {
                        s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: Default::default(),
                        rasterization_samples: vk::SampleCountFlags::TYPE_1,
                        sample_shading_enable: vk::FALSE,
                        min_sample_shading: 0.0,
                        p_sample_mask: std::ptr::null(),
                        alpha_to_coverage_enable: vk::FALSE,
                        alpha_to_one_enable: vk::FALSE,
                    },
                ),
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
                tesselation_info: None,
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
                vk_tessellation_state: None,
            },
            vertex_binding_offsets: Default::default(),
        }
    }

    /// Add a vertex input binding. These are the binding indices for `vkCmdBindVertexBuffers`
    pub fn vertex_input(mut self, binding: u32, rate: vk::VertexInputRate) -> Self {
        self.vertex_binding_offsets.insert(binding, 0);
        self.inner
            .vertex_input_bindings
            .push(VertexInputBindingDescription(vk::VertexInputBindingDescription {
                binding,
                stride: 0,
                input_rate: rate,
            }));
        self
    }

    /// Add a vertex attribute to the specified binding.
    /// Doing this will automatically calculate offsets and sizes, so make sure to add these in order of declaration in
    /// the shader.
    pub fn vertex_attribute(
        mut self,
        binding: u32,
        location: u32,
        format: vk::Format,
    ) -> Result<Self> {
        let offset = self
            .vertex_binding_offsets
            .get_mut(&binding)
            .ok_or_else(|| Error::NoVertexBinding)?;
        self.inner
            .vertex_attributes
            .push(VertexInputAttributeDescription(vk::VertexInputAttributeDescription {
                location,
                binding,
                format,
                offset: *offset,
            }));
        *offset += format.byte_size() as u32;
        for descr in &mut self.inner.vertex_input_bindings {
            if descr.0.binding == binding {
                descr.0.stride += format.byte_size() as u32;
            }
        }

        Ok(self)
    }

    /// Add a shader to the pipeline.
    pub fn attach_shader(mut self, info: ShaderCreateInfo) -> Self {
        self.inner.shaders.push(info);
        self
    }

    /// Set depth testing mode.
    pub fn depth_test(mut self, enable: bool) -> Self {
        self.inner.depth_stencil.0.depth_test_enable = vk::Bool32::from(enable);
        self
    }

    /// Set depth write mode.
    pub fn depth_write(mut self, enable: bool) -> Self {
        self.inner.depth_stencil.0.depth_write_enable = vk::Bool32::from(enable);
        self
    }

    /// Set the depth compare operation.
    pub fn depth_op(mut self, op: vk::CompareOp) -> Self {
        self.inner.depth_stencil.0.depth_compare_op = op;
        self
    }

    /// Toggle depth clamping.
    pub fn depth_clamp(mut self, enable: bool) -> Self {
        self.inner.rasterizer.0.depth_clamp_enable = vk::Bool32::from(enable);
        self
    }

    /// Configure all depth state in one call.
    pub fn depth(self, test: bool, write: bool, clamp: bool, op: vk::CompareOp) -> Self {
        self.depth_test(test)
            .depth_write(write)
            .depth_clamp(clamp)
            .depth_op(op)
    }

    /// Add a dynamic state to the pipeline.
    pub fn dynamic_state(mut self, state: vk::DynamicState) -> Self {
        self.inner.dynamic_states.push(state);
        // When setting a viewport dynamic state, we still need a dummy viewport to make validation shut up
        if state == vk::DynamicState::VIEWPORT {
            self.inner.viewports.push(Viewport(vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
                min_depth: 0.0,
                max_depth: 0.0,
            }))
        }
        // Same with scissor state
        if state == vk::DynamicState::SCISSOR {
            self.inner.scissors.push(Rect2D(vk::Rect2D {
                offset: Default::default(),
                extent: Default::default(),
            }))
        }

        self
    }

    /// Add dynamic states to the pipeline.
    pub fn dynamic_states(mut self, states: &[vk::DynamicState]) -> Self {
        for state in states {
            self = self.dynamic_state(*state);
        }
        self
    }

    /// Set the polygon mode.
    pub fn polygon_mode(mut self, mode: vk::PolygonMode) -> Self {
        self.inner.rasterizer.0.polygon_mode = mode;
        self
    }

    /// Set the face culling mask.
    pub fn cull_mask(mut self, cull: vk::CullModeFlags) -> Self {
        self.inner.rasterizer.0.cull_mode = cull;
        self
    }

    /// Set the front face.
    pub fn front_face(mut self, face: vk::FrontFace) -> Self {
        self.inner.rasterizer.0.front_face = face;
        self
    }

    /// Set the amount of MSAA samples.
    pub fn samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.inner.multisample.0.rasterization_samples = samples;
        self
    }

    /// Enable sample shading and set the sample shading rate.
    pub fn sample_shading(mut self, value: f32) -> Self {
        self.inner.multisample.0.sample_shading_enable = vk::TRUE;
        self.inner.multisample.0.min_sample_shading = value;
        self
    }

    /// Enable tessellation and set tessellation state.
    pub fn tessellation(
        mut self,
        patch_control_points: u32,
        flags: vk::PipelineTessellationStateCreateFlags,
    ) -> Self {
        let mut info = PipelineTessellationStateCreateInfo::default();
        info.0.patch_control_points = patch_control_points;
        info.0.flags = flags;
        self.inner.tesselation_info = Some(info);
        // When enabling tessellation, we need to set the primitive topology properly, see VUID-VkGraphicsPipelineCreateInfo-pStages-00736
        self.inner.input_assembly.0.topology = vk::PrimitiveTopology::PATCH_LIST;
        self
    }

    /// Add a blend attachment, but with no blending enabled.
    pub fn blend_attachment_none(mut self) -> Self {
        self.inner
            .blend_attachments
            .push(PipelineColorBlendAttachmentState(vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ONE,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::RGBA,
            }));
        self
    }

    /// Add a blend attachment writing to each color component
    pub fn blend_attachment(mut self, src_color: vk::BlendFactor, dst_color: vk::BlendFactor, color_op: vk::BlendOp, src_alpha: vk::BlendFactor, dst_alpha: vk::BlendFactor, alpha_op: vk::BlendOp) -> Self {
        self.inner.blend_attachments.push(PipelineColorBlendAttachmentState(vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            src_color_blend_factor: src_color,
            dst_color_blend_factor: dst_color,
            color_blend_op: color_op,
            src_alpha_blend_factor: src_alpha,
            dst_alpha_blend_factor: dst_alpha,
            alpha_blend_op: alpha_op,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        }));
        self
    }

    /// Add an additive blend attachment, writing to each color component.
    pub fn blend_additive_unmasked(
        mut self,
        src: vk::BlendFactor,
        dst: vk::BlendFactor,
        src_alpha: vk::BlendFactor,
        dst_alpha: vk::BlendFactor,
    ) -> Self {
        self.inner
            .blend_attachments
            .push(PipelineColorBlendAttachmentState(vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                src_color_blend_factor: src,
                dst_color_blend_factor: dst,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: src_alpha,
                dst_alpha_blend_factor: dst_alpha,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::RGBA,
            }));
        self
    }

    /// Build the pipeline create info structure.
    pub fn build(self) -> PipelineCreateInfo {
        self.inner
    }

    /// Obtain the pipeline name.
    pub fn name(&self) -> &str {
        &self.inner.name
    }
}
