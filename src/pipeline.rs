use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::ffi::CString;
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

#[derive(Debug, Copy, Clone)]
pub struct VertexInputBindingDescription(vk::VertexInputBindingDescription);

#[derive(Debug, Copy, Clone)]
pub struct VertexInputAttributeDescription(vk::VertexInputAttributeDescription);

#[derive(Debug, Copy, Clone)]
pub struct PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateInfo);

#[derive(Debug, Copy, Clone)]
pub struct PipelineDepthStencilStateCreateInfo(vk::PipelineDepthStencilStateCreateInfo);

#[derive(Debug, Copy, Clone)]
pub struct PipelineRasterizationStateCreateInfo(vk::PipelineRasterizationStateCreateInfo);

#[derive(Debug, Copy, Clone)]
pub struct PipelineMultisampleStateCreateInfo(vk::PipelineMultisampleStateCreateInfo);

#[derive(Debug, Copy, Clone)]
pub struct PipelineColorBlendAttachmentState(vk::PipelineColorBlendAttachmentState);

#[derive(Debug, Copy, Clone)]
pub struct Viewport(vk::Viewport);

#[derive(Debug, Copy, Clone)]
pub struct Rect2D(vk::Rect2D);

#[derive(Debug, Clone, Derivative)]
#[derivative(PartialEq, Eq, Hash)]
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
    pub blend_enable_logic_op: bool,

    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    vk_vertex_inputs: Vec<vk::VertexInputBindingDescription>,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    vk_attributes: Vec<vk::VertexInputAttributeDescription>,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    vertex_input_state: vk::PipelineVertexInputStateCreateInfo,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    vk_viewports: Vec<vk::Viewport>,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    vk_scissors: Vec<vk::Rect2D>,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    viewport_state: vk::PipelineViewportStateCreateInfo,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    vk_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState>,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    blend_state: vk::PipelineColorBlendStateCreateInfo,
    #[derivative(PartialEq="ignore")]
    #[derivative(Hash="ignore")]
    vk_dynamic_state: vk::PipelineDynamicStateCreateInfo,
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
            code_size: key.code.len() * 4,
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

impl PipelineCreateInfo {
    // Shader stage not yet filled out
    pub(crate) fn to_vk(&self, layout: vk::PipelineLayout) -> vk::GraphicsPipelineCreateInfo {
        vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
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

impl Resource for Pipeline {
    type Key = PipelineCreateInfo;
    type ExtraParams<'a> = (&'a mut Cache<Shader>, &'a mut Cache<PipelineLayout>, &'a mut Cache<DescriptorSetLayout>);
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Arc<Device>, info: &Self::Key, params: Self::ExtraParams<'_>) -> Result<Self, Error> {
        let (shaders, pipeline_layouts, set_layouts) = params;
        let layout = pipeline_layouts.get_or_create(&info.layout, set_layouts)?;
        let mut pci = info.to_vk(layout.handle);
        // Set shader create info
        let entry = CString::new("main")?;
        let shader_info: Vec<_> = info.shaders.iter().map(|shader| -> vk::PipelineShaderStageCreateInfo {
            vk::PipelineShaderStageCreateInfo::builder()
                .name(&entry)
                .stage(shader.stage)
                .module(shaders.get_or_create(shader, ()).unwrap().handle)
                .build()
        }).collect();
        pci.stage_count = shader_info.len() as u32;
        pci.p_stages = shader_info.as_ptr();

        unsafe {
            Ok(Self {
                device: device.clone(),
                handle: device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pci), None)?.first().cloned().unwrap(),
            })
        }
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

    pub fn vertex_attribute(mut self, binding: u32, location: u32, format: vk::Format) -> Result<Self, Error> {
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

    // todo: shader reflection

    pub fn build(mut self) -> PipelineCreateInfo {
        self.inner.vk_attributes = self.inner.vertex_attributes.iter().map(|v| v.0.clone()).collect();
        self.inner.vk_vertex_inputs = self.inner.vertex_input_bindings.iter().map(|v| v.0.clone()).collect();
        self.inner.vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(self.inner.vk_vertex_inputs.as_slice())
            .vertex_attribute_descriptions(self.inner.vk_attributes.as_slice())
            .build();
        self.inner.vk_viewports = self.inner.viewports.iter().map(|v| v.0.clone()).collect();
        self.inner.vk_scissors = self.inner.scissors.iter().map(|v| v.0.clone()).collect();
        self.inner.viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(self.inner.vk_viewports.as_slice())
            .scissors(self.inner.vk_scissors.as_slice())
            .build();
        self.inner.vk_blend_attachments = self.inner.blend_attachments.iter().map(|v| v.0.clone()).collect();
        self.inner.blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(self.inner.blend_enable_logic_op)
            .attachments(self.inner.vk_blend_attachments.as_slice())
            .build();
        self.inner.vk_dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(self.inner.dynamic_states.as_slice())
            .build();
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
        self.pipelines.get_or_create(info, (&mut self.shaders, &mut self.pipeline_layouts, &mut self.set_layouts))
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

