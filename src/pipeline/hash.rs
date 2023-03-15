use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use crate::create_info::{PipelineColorBlendAttachmentState, PipelineDepthStencilStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineMultisampleStateCreateInfo, PipelineRasterizationStateCreateInfo, Rect2D, VertexInputAttributeDescription, VertexInputBindingDescription, Viewport};
use crate::pipeline_layout::PipelineLayoutCreateInfo;
use crate::set_layout::DescriptorSetLayoutCreateInfo;
use crate::shader::ShaderCreateInfo;

impl Hash for ShaderCreateInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.code_hash)
    }
}

impl Hash for VertexInputBindingDescription {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.binding.hash(state);
        self.0.stride.hash(state);
        self.0.input_rate.hash(state);
    }
}

impl Hash for VertexInputAttributeDescription {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.location.hash(state);
        self.0.binding.hash(state);
        self.0.format.hash(state);
        self.0.offset.hash(state);
    }
}

impl Hash for PipelineInputAssemblyStateCreateInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.flags.hash(state);
        self.0.topology.hash(state);
        self.0.primitive_restart_enable.hash(state);
    }
}

impl Hash for PipelineDepthStencilStateCreateInfo {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.0.flags.hash(hasher);
        self.0.depth_test_enable.hash(hasher);
        self.0.depth_write_enable.hash(hasher);
        self.0.depth_compare_op.hash(hasher);
        self.0.depth_bounds_test_enable.hash(hasher);
        self.0.stencil_test_enable.hash(hasher);
        self.0.front.compare_mask.hash(hasher);
        self.0.front.compare_op.hash(hasher);
        self.0.front.depth_fail_op.hash(hasher);
        self.0.front.fail_op.hash(hasher);
        self.0.front.pass_op.hash(hasher);
        self.0.front.reference.hash(hasher);
        self.0.front.write_mask.hash(hasher);
        self.0.back.compare_mask.hash(hasher);
        self.0.back.compare_op.hash(hasher);
        self.0.back.depth_fail_op.hash(hasher);
        self.0.back.fail_op.hash(hasher);
        self.0.back.pass_op.hash(hasher);
        self.0.back.reference.hash(hasher);
        self.0.back.write_mask.hash(hasher);
        self.0.min_depth_bounds.to_bits().hash(hasher);
        self.0.max_depth_bounds.to_bits().hash(hasher);
    }
}

impl Hash for PipelineRasterizationStateCreateInfo {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.0.flags.hash(hasher);
        self.0.depth_clamp_enable.hash(hasher);
        self.0.rasterizer_discard_enable .hash(hasher);
        self.0.polygon_mode.hash(hasher);
        self.0.cull_mode.hash(hasher);
        self.0.front_face.hash(hasher);
        self.0.depth_bias_enable.hash(hasher);
        self.0.depth_bias_constant_factor.to_bits().hash(hasher);
        self.0.depth_bias_clamp.to_bits().hash(hasher);
        self.0.depth_bias_slope_factor.to_bits().hash(hasher);
        self.0.line_width.to_bits().hash(hasher);
    }
}

impl Hash for PipelineMultisampleStateCreateInfo {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.0.flags.hash(hasher);
        self.0.rasterization_samples.hash(hasher);
        self.0.sample_shading_enable.hash(hasher);
        self.0.min_sample_shading.to_bits().hash(hasher);
        self.0.p_sample_mask.hash(hasher);
        self.0.alpha_to_coverage_enable.hash(hasher);
        self.0.alpha_to_one_enable.hash(hasher);
    }
}

impl Hash for PipelineColorBlendAttachmentState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.src_color_blend_factor.hash(state);
        self.0.dst_color_blend_factor.hash(state);
        self.0.color_blend_op.hash(state);
        self.0.src_alpha_blend_factor.hash(state);
        self.0.dst_alpha_blend_factor.hash(state);
        self.0.alpha_blend_op.hash(state);
        self.0.color_write_mask.hash(state);
    }
}

impl Hash for Viewport {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.0.x.to_bits().hash(hasher);
        self.0.y.to_bits().hash(hasher);
        self.0.width.to_bits().hash(hasher);
        self.0.height.to_bits().hash(hasher);
        self.0.min_depth.to_bits().hash(hasher);
        self.0.max_depth.to_bits().hash(hasher);
    }
}

impl Hash for Rect2D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.offset.x.hash(state);
        self.0.offset.y.hash(state);
        self.0.extent.width.hash(state);
        self.0.extent.height.hash(state);
    }
}

impl Hash for DescriptorSetLayoutCreateInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for binding in &self.bindings {
            binding.binding.hash(state);
            binding.descriptor_count.hash(state);
            binding.descriptor_type.hash(state);
            binding.stage_flags.hash(state);
            binding.p_immutable_samplers.hash(state);
        }
    }
}

impl Hash for PipelineLayoutCreateInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.flags.hash(state);
        self.set_layouts.hash(state);
        self.push_constants.hash(state);
    }
}

impl PartialEq<Self> for DescriptorSetLayoutCreateInfo {
    fn eq(&self, other: &Self) -> bool {
        // Given the low possibility for collisions, this is probably fine.
        // If it is not, I will eat the code.

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        self.hash(&mut h1);
        other.hash(&mut h2);

        h1.finish() == h2.finish()
    }
}

impl PartialEq<Self> for PipelineLayoutCreateInfo {
    fn eq(&self, other: &Self) -> bool {
        self.flags == other.flags &&
            self.set_layouts == other.set_layouts &&
            self.push_constants == other.push_constants
    }
}

impl PartialEq<Self> for ShaderCreateInfo {
    fn eq(&self, other: &Self) -> bool {
        self.code_hash == other.code_hash
    }
}

impl PartialEq<Self> for VertexInputBindingDescription {
    fn eq(&self, other: &Self) -> bool {
        self.0.binding == other.0.binding
            && self.0.stride == other.0.stride
            && self.0.input_rate == other.0.input_rate
    }
}

impl PartialEq<Self> for VertexInputAttributeDescription {
    fn eq(&self, other: &Self) -> bool {
        self.0.location == other.0.location
            && self.0.binding == other.0.binding
            && self.0.format == other.0.format
            && self.0.offset == other.0.offset
    }
}

impl PartialEq<Self> for PipelineInputAssemblyStateCreateInfo {
    fn eq(&self, other: &Self) -> bool {
        self.0.flags == other.0.flags
            && self.0.topology == other.0.topology
            && self.0.primitive_restart_enable == other.0.primitive_restart_enable
    }
}

impl PartialEq<Self> for PipelineDepthStencilStateCreateInfo {
    fn eq(&self, other: &Self) -> bool {
        self.0.flags == other.0.flags
            && self.0.depth_test_enable == other.0.depth_test_enable
            && self.0.depth_write_enable == other.0.depth_write_enable
            && self.0.depth_compare_op == other.0.depth_compare_op
            && self.0.depth_bounds_test_enable == other.0.depth_bounds_test_enable
            && self.0.stencil_test_enable == other.0.stencil_test_enable
            && self.0.min_depth_bounds == other.0.min_depth_bounds
            && self.0.max_depth_bounds == other.0.max_depth_bounds
    }
}

impl PartialEq<Self> for PipelineRasterizationStateCreateInfo {
    fn eq(&self, other: &Self) -> bool {
        self.0.flags == other.0.flags
            && self.0.depth_clamp_enable == other.0.depth_clamp_enable
            && self.0.rasterizer_discard_enable == other.0.rasterizer_discard_enable
            && self.0.polygon_mode == other.0.polygon_mode
            && self.0.cull_mode == other.0.cull_mode
            && self.0.front_face == other.0.front_face
            && self.0.depth_bias_enable == other.0.depth_bias_enable
            && self.0.depth_bias_constant_factor == other.0.depth_bias_constant_factor
            && self.0.depth_bias_clamp == other.0.depth_bias_clamp
            && self.0.depth_bias_slope_factor == other.0.depth_bias_slope_factor
            && self.0.line_width == other.0.line_width
    }
}

impl PartialEq<Self> for PipelineMultisampleStateCreateInfo {
    fn eq(&self, other: &Self) -> bool {
        self.0.flags == other.0.flags
            && self.0.rasterization_samples == other.0.rasterization_samples
            && self.0.sample_shading_enable == other.0.sample_shading_enable
            && self.0.min_sample_shading == other.0.min_sample_shading
            && self.0.p_sample_mask == other.0.p_sample_mask
            && self.0.alpha_to_coverage_enable == other.0.alpha_to_coverage_enable
            && self.0.alpha_to_one_enable == other.0.alpha_to_one_enable
    }
}

impl PartialEq<Self> for PipelineColorBlendAttachmentState {
    fn eq(&self, other: &Self) -> bool {
        self.0.blend_enable == other.0.blend_enable
            && self.0.src_color_blend_factor == other.0.src_color_blend_factor
            && self.0.dst_color_blend_factor == other.0.dst_color_blend_factor
            && self.0.color_blend_op == other.0.color_blend_op
            && self.0.src_alpha_blend_factor == other.0.src_alpha_blend_factor
            && self.0.dst_alpha_blend_factor == other.0.dst_alpha_blend_factor
            && self.0.alpha_blend_op == other.0.alpha_blend_op
            && self.0.color_write_mask == other.0.color_write_mask
    }
}

impl PartialEq for Viewport {
    fn eq(&self, other: &Self) -> bool {
        self.0.x == other.0.x
            && self.0.y == other.0.y
            && self.0.width == other.0.width
            && self.0.height == other.0.height
            && self.0.min_depth == other.0.min_depth
            && self.0.max_depth == other.0.max_depth
    }
}

impl PartialEq for Rect2D {
    fn eq(&self, other: &Self) -> bool {
        self.0.offset == other.0.offset
            && self.0.extent == other.0.extent
    }
}

impl Eq for DescriptorSetLayoutCreateInfo {}
impl Eq for PipelineLayoutCreateInfo {}
impl Eq for ShaderCreateInfo {}
impl Eq for VertexInputBindingDescription {}
impl Eq for VertexInputAttributeDescription {}
impl Eq for PipelineInputAssemblyStateCreateInfo {}
impl Eq for PipelineDepthStencilStateCreateInfo {}
impl Eq for PipelineRasterizationStateCreateInfo {}
impl Eq for PipelineMultisampleStateCreateInfo {}
impl Eq for PipelineColorBlendAttachmentState {}
impl Eq for Viewport {}
impl Eq for Rect2D {}

