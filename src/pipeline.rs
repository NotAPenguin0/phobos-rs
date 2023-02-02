use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use ash::vk;
use crate::{Device, Error};
use crate::cache::*;
use crate::util::ByteSize;

pub type PipelineStage = ash::vk::PipelineStageFlags2;

pub struct Shader {
    pub device: Arc<Device>,
    pub handle: vk::ShaderModule
}

#[derive(Debug, Clone)]
pub struct ShaderCreateInfo {
    pub stage: vk::ShaderStageFlags,
    pub code: Vec<u32>,
    pub code_hash: u64
}

// Note: Pipeline layout, descriptor set layout and pipeline are tied together tightly,
// When accessing a pipeline, we also have to register an access for the layouts to prevent them from being
// deleted.

pub struct DescriptorSetLayout {
    pub device: Arc<Device>,
    pub handle: vk::DescriptorSetLayout
}

#[derive(Debug, Clone, Default)]
pub struct DescriptorSetLayoutCreateInfo {
    pub bindings: Vec<vk::DescriptorSetLayoutBinding>
}

pub struct PipelineLayout {
    pub device: Arc<Device>,
    pub handle: vk::PipelineLayout
}

#[derive(Debug, Clone, Default, Copy, PartialEq, Eq, Hash)]
pub struct PushConstantRange {
    pub stage_flags: vk::ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug, Clone, Default)]
pub struct PipelineLayoutCreateInfo {
    pub flags: vk::PipelineLayoutCreateFlags,
    pub set_layouts: Vec<DescriptorSetLayoutCreateInfo>,
    pub push_constants: Vec<PushConstantRange>,
}

pub struct Pipeline {
    device: Arc<Device>,
    pub(crate) handle: vk::Pipeline
}

#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub struct VertexInputBindingDescription {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: vk::VertexInputRate
}

#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub struct VertexInputAttributeDescription {
    pub location: u32,
    pub binding: u32,
    pub format: vk::Format,
    pub offset: u32,
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct PipelineInputAssemblyStateCreateInfo {
    pub flags: vk::PipelineInputAssemblyStateCreateFlags,
    pub topology: vk::PrimitiveTopology,
    pub primitive_restart_enable: bool
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct StencilOpState {
    pub fail_op: vk::StencilOp,
    pub pass_op: vk::StencilOp,
    pub depth_fail_op: vk::StencilOp,
    pub compare_op: vk::CompareOp,
    pub compare_mask: u32,
    pub write_mask: u32,
    pub reference: u32,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PipelineDepthStencilStateCreateInfo {
    pub flags: vk::PipelineDepthStencilStateCreateFlags,
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: vk::CompareOp,
    pub depth_bounds_test_enable: bool,
    pub stencil_test_enable: bool,
    pub front: StencilOpState,
    pub back: StencilOpState,
    pub min_depth_bounds: f32,
    pub max_depth_bounds: f32,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PipelineRasterizationStateCreateInfo {
    pub flags: vk::PipelineRasterizationStateCreateFlags,
    pub depth_clamp_enable: bool,
    pub rasterizer_discard_enable: bool,
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub depth_bias_enable: bool,
    pub depth_bias_constant_factor: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope_factor: f32,
    pub line_width: f32,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PipelineMultisampleStateCreateInfo {
    pub flags: vk::PipelineMultisampleStateCreateFlags,
    pub rasterization_samples: vk::SampleCountFlags,
    pub sample_shading_enable: bool,
    pub min_sample_shading: f32,
    pub sample_mask:vk::SampleMask,
    pub alpha_to_coverage_enable: bool,
    pub alpha_to_one_enable: bool,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct PipelineColorBlendAttachmentState {
    pub blend_enable: bool,
    pub src_color_blend_factor: vk::BlendFactor,
    pub dst_color_blend_factor: vk::BlendFactor,
    pub color_blend_op: vk::BlendOp,
    pub src_alpha_blend_factor: vk::BlendFactor,
    pub dst_alpha_blend_factor: vk::BlendFactor,
    pub alpha_blend_op: vk::BlendOp,
    pub color_write_mask: vk::ColorComponentFlags,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Rect2D {
    pub offset_x: i32,
    pub offset_y: i32,
    pub width: u32,
    pub height: u32
}

#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub struct PipelineCreateInfo {
    // TODO: Shader reflection info
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
    pub blend_enable_logic_op: bool
}

pub struct PipelineBuilder {
    inner: PipelineCreateInfo,
    vertex_binding_offsets: HashMap<u32, u32>,
}

pub struct PipelineCache {
    shaders: Cache<Shader>,
    set_layouts: Cache<DescriptorSetLayout>,
    pipeline_layouts: Cache<PipelineLayout>,
    pipelines: Cache<Pipeline>,
    named_pipelines: HashMap<String, PipelineCreateInfo>,
}

impl Resource for Shader {
    type Key = ShaderCreateInfo;
    type ExtraParams<'a> = ();
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Arc<Device>, key: &Self::Key, _: Self::ExtraParams<'_>) -> Result<Self, Error> {
        let info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            code_size: key.code.len(),
            p_code: key.code.as_ptr(),
        };

        Ok(unsafe {
            Self {
                device: device.clone(),
                handle: unsafe { device.create_shader_module(&info, None)? },
            }
        })
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe { self.device.destroy_shader_module(self.handle, None); }
    }
}

impl ShaderCreateInfo {
    pub fn from_spirv(stage: vk::ShaderStageFlags, code: Vec<u32>) -> Self {
        let mut hasher = DefaultHasher::new();
        code.hash(&mut hasher);
        Self {
            stage,
            code,
            code_hash: hasher.finish(),
        }
    }
}

impl Resource for DescriptorSetLayout {
    type Key = DescriptorSetLayoutCreateInfo;
    type ExtraParams<'a> = ();
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Arc<Device>, key: &Self::Key, _: Self::ExtraParams<'_>) -> Result<Self, Error> {
        let info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(key.bindings.as_slice())
            .build();
        Ok(Self {
            device: device.clone(),
            handle: unsafe { device.create_descriptor_set_layout(&info, None)? }
        })
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_set_layout(self.handle, None);
        }
    }
}

impl Resource for PipelineLayout {
    type Key = PipelineLayoutCreateInfo;
    type ExtraParams<'a> = &'a mut Cache<DescriptorSetLayout>;
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Arc<Device>, key: &Self::Key, set_layout_cache: Self::ExtraParams<'_>) -> Result<Self, Error> {
        let info = vk::PipelineLayoutCreateInfo::builder()
            .flags(key.flags)
            .push_constant_ranges(key.push_constants.iter().map(|pc| pc.to_vk()).collect::<Vec<_>>().as_slice())
            .set_layouts(key.set_layouts.iter().map(|info|
                    set_layout_cache.get_or_create(&info, ()).unwrap().handle
                ).collect::<Vec<_>>().as_slice()
            )
            .build();
        Ok(Self {
            device: device.clone(),
            handle: unsafe { device.create_pipeline_layout(&info, None)? },
        })
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_layout(self.handle, None);
        }
    }
}

impl Resource for Pipeline {
    type Key = PipelineCreateInfo;
    type ExtraParams<'a> = (&'a mut Cache<Shader>);
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Arc<Device>, key: &Self::Key, params: Self::ExtraParams<'_>) -> Result<Self, Error> {
        todo!()
    }
}

impl PipelineBuilder {
    pub fn new(name: String) -> Self {
        Self {
            inner: PipelineCreateInfo {
                name,
                layout: Default::default(),
                vertex_input_bindings: vec![],
                vertex_attributes: vec![],
                shaders: vec![],
                input_assembly: PipelineInputAssemblyStateCreateInfo {
                    flags: Default::default(),
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    primitive_restart_enable: false,
                },
                depth_stencil: PipelineDepthStencilStateCreateInfo {
                    flags: Default::default(),
                    depth_test_enable: false,
                    depth_write_enable: false,
                    depth_compare_op: Default::default(),
                    depth_bounds_test_enable: false,
                    stencil_test_enable: false,
                    front: StencilOpState {
                        fail_op: Default::default(),
                        pass_op: Default::default(),
                        depth_fail_op: Default::default(),
                        compare_op: Default::default(),
                        compare_mask: 0,
                        write_mask: 0,
                        reference: 0,
                    },
                    back: StencilOpState {
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
                dynamic_states: vec![],
                rasterizer: PipelineRasterizationStateCreateInfo {
                    flags: Default::default(),
                    depth_clamp_enable: false,
                    rasterizer_discard_enable: false,
                    polygon_mode: vk::PolygonMode::FILL,
                    cull_mode: vk::CullModeFlags::NONE,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    depth_bias_enable: false,
                    depth_bias_constant_factor: 0.0,
                    depth_bias_clamp: 0.0,
                    depth_bias_slope_factor: 0.0,
                    line_width: 1.0,
                },
                multisample: PipelineMultisampleStateCreateInfo {
                    flags: Default::default(),
                    rasterization_samples: vk::SampleCountFlags::TYPE_1,
                    sample_shading_enable: false,
                    min_sample_shading: 0.0,
                    sample_mask: 0,
                    alpha_to_coverage_enable: false,
                    alpha_to_one_enable: false,
                },
                blend_attachments: vec![],
                viewports: vec![],
                scissors: vec![],
                blend_enable_logic_op: false,
            },
            vertex_binding_offsets: Default::default(),
        }
    }

    pub fn vertex_input(mut self, binding: u32, rate: vk::VertexInputRate) -> Self {
        self.vertex_binding_offsets.insert(0, 0);
        self.inner.vertex_input_bindings.push(VertexInputBindingDescription{
            binding,
            stride: 0,
            input_rate: rate,
        });
        self
    }

    pub fn vertex_attribute(mut self, binding: u32, location: u32, format: vk::Format) -> Result<Self, Error> {
        let offset = self.vertex_binding_offsets.get_mut(&binding).ok_or(Error::NoVertexBinding)?;
        self.inner.vertex_attributes.push(VertexInputAttributeDescription{
            location,
            binding,
            format,
            offset: *offset,
        });
        *offset += format.byte_size() as u32;
        for binding in &mut self.inner.vertex_input_bindings {
            binding.stride += format.byte_size() as u32;
        }

        Ok(self)
    }

    pub fn attach_shader(mut self, info: ShaderCreateInfo) -> Self {
        self.inner.shaders.push(info);
        self
    }

    pub fn depth_test(mut self, enable: bool) -> Self {
        self.inner.depth_stencil.depth_test_enable = enable;
        self
    }

    pub fn depth_write(mut self, enable: bool) -> Self {
        self.inner.depth_stencil.depth_write_enable = enable;
        self
    }

    pub fn depth_op(mut self, op: vk::CompareOp) -> Self {
        self.inner.depth_stencil.depth_compare_op = op;
        self
    }

    pub fn depth_clamp(mut self, enable: bool) -> Self {
        self.inner.rasterizer.depth_clamp_enable = enable;
        self
    }

    pub fn depth(mut self, test: bool, write: bool, clamp: bool, op: vk::CompareOp) -> Self {
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
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
                min_depth: 0.0,
                max_depth: 0.0,
            })
        }
        // Same with scissor state
        if state == vk::DynamicState::SCISSOR {
            self.inner.scissors.push(Rect2D{
                offset_x: 0,
                offset_y: 0,
                width: 0,
                height: 0,
            })
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
        self.inner.rasterizer.polygon_mode = mode;
        self
    }

    pub fn cull_mask(mut self, cull: vk::CullModeFlags) -> Self {
        self.inner.rasterizer.cull_mode = cull;
        self
    }

    pub fn front_face(mut self, face: vk::FrontFace) -> Self {
        self.inner.rasterizer.front_face = face;
        self
    }

    pub fn samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.inner.multisample.rasterization_samples = samples;
        self
    }

    pub fn sample_shading(mut self, value: f32) -> Self {
        self.inner.multisample.sample_shading_enable = true;
        self.inner.multisample.min_sample_shading = value;
        self
    }

    pub fn blend_attachment_none(mut self) -> Self {
        self.inner.blend_attachments.push(PipelineColorBlendAttachmentState{
            blend_enable: false,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ONE,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ONE,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        });
        self
    }

    // todo: shader reflection

    pub fn build(self) -> PipelineCreateInfo {
        self.inner
    }
}

impl PipelineCache {
    pub fn new(device: Arc<Device>) -> Result<Self, Error> {
        Ok(Self {
            shaders: Cache::new(device.clone()),
            set_layouts: Cache::new(device.clone()),
            pipeline_layouts: Cache::new(device.clone()),
            pipelines: Cache::new(device),
            named_pipelines: Default::default(),
        })
    }

    pub fn create_named_pipeline(&mut self, info: PipelineCreateInfo) -> Result<(), Error> {
        self.named_pipelines.insert(info.name.clone(), info);
        Ok(())
    }

    pub(crate) fn get_pipeline(&mut self, name: &str) -> Result<&Pipeline, Error> {
        let info = self.named_pipelines.get(name);
        let Some(info) = info else { return Err(Error::PipelineNotFound(name.to_string())); };
        // Also put in queries for descriptor set layouts and pipeline layout to make sure they are not destroyed.
        for layout in &info.layout.set_layouts {
            self.set_layouts.get_or_create(layout, ())?;
        }
        self.pipeline_layouts.get_or_create(&info.layout, &mut self.set_layouts)?;
        self.pipelines.get_or_create(info, &mut self.shaders)
    }

    /// Advance cache resource time to live so resources that have not been used in a while can be cleaned up
    pub fn next_frame(&mut self) {
        self.pipelines.next_frame();
        self.pipeline_layouts.next_frame();
        self.shaders.next_frame();
        self.set_layouts.next_frame();
    }
}

impl PushConstantRange {
    pub fn to_vk(&self) -> vk::PushConstantRange {
        vk::PushConstantRange {
            stage_flags: self.stage_flags,
            offset: self.offset,
            size: self.size,
        }
    }
}

impl Hash for ShaderCreateInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.code_hash)
    }
}

impl Hash for PipelineDepthStencilStateCreateInfo {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.flags.hash(hasher);
        self.depth_test_enable.hash(hasher);
        self.depth_write_enable.hash(hasher);
        self.depth_compare_op.hash(hasher);
        self.depth_bounds_test_enable.hash(hasher);
        self.stencil_test_enable.hash(hasher);
        self.front.hash(hasher);
        self.back.hash(hasher);
        self.min_depth_bounds.to_bits().hash(hasher);
        self.max_depth_bounds.to_bits().hash(hasher);
    }
}

impl Hash for PipelineRasterizationStateCreateInfo {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.flags.hash(hasher);
        self.depth_clamp_enable.hash(hasher);
        self.rasterizer_discard_enable .hash(hasher);
        self.polygon_mode.hash(hasher);
        self.cull_mode.hash(hasher);
        self.front_face.hash(hasher);
        self.depth_bias_enable.hash(hasher);
        self.depth_bias_constant_factor.to_bits().hash(hasher);
        self.depth_bias_clamp.to_bits().hash(hasher);
        self.depth_bias_slope_factor.to_bits().hash(hasher);
        self.line_width.to_bits().hash(hasher);
    }
}

impl Hash for PipelineMultisampleStateCreateInfo {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.flags.hash(hasher);
        self.rasterization_samples.hash(hasher);
        self.sample_shading_enable.hash(hasher);
        self.min_sample_shading.to_bits().hash(hasher);
        self.sample_mask.hash(hasher);
        self.alpha_to_coverage_enable.hash(hasher);
        self.alpha_to_one_enable.hash(hasher);
    }
}

impl Hash for Viewport {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.x.to_bits().hash(hasher);
        self.y.to_bits().hash(hasher);
        self.width.to_bits().hash(hasher);
        self.height.to_bits().hash(hasher);
        self.min_depth.to_bits().hash(hasher);
        self.max_depth.to_bits().hash(hasher);
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

impl Eq for DescriptorSetLayoutCreateInfo {

}

impl Eq for PipelineLayoutCreateInfo {

}

impl Eq for ShaderCreateInfo {

}

impl Eq for PipelineDepthStencilStateCreateInfo {

}

impl Eq for PipelineRasterizationStateCreateInfo {

}

impl Eq for PipelineMultisampleStateCreateInfo {

}

impl Eq for Viewport {

}