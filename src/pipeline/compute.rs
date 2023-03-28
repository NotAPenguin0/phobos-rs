use ash::vk;

use crate::pipeline::pipeline_layout::PipelineLayoutCreateInfo;
use crate::ShaderCreateInfo;

/// Create info for a compute pipeline. Use the [`ComputePipelineBuilder`](crate::ComputePipelineBuilder)
/// struct to construct this.
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct ComputePipelineCreateInfo {
    pub shader: Option<ShaderCreateInfo>,
    pub(crate) name: String,
    pub(crate) layout: PipelineLayoutCreateInfo,
}

impl ComputePipelineCreateInfo {
    // create compute pipeline create info, but without the shader filled out
    pub(crate) fn to_vk(&self, layout: vk::PipelineLayout) -> vk::ComputePipelineCreateInfo {
        vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            stage: Default::default(),
            layout,
            base_pipeline_handle: Default::default(),
            base_pipeline_index: 0,
        }
    }
}

/// Builder struct similar to [`PipelineBuilder`](crate::PipelineBuilder), but for compute pipelines. Since these are much simpler,
/// it is also much easier to construct these.
#[derive(Debug)]
pub struct ComputePipelineBuilder {
    inner: ComputePipelineCreateInfo,
}

impl ComputePipelineBuilder {
    /// Create a new compute pipeline with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            inner: ComputePipelineCreateInfo {
                shader: None,
                name: name.into(),
                layout: Default::default(),
            },
        }
    }

    /// Set the compute shader that will be used for this pipeline. Note that compute pipelines
    /// can only have one shader.
    pub fn set_shader(mut self, shader: ShaderCreateInfo) -> Self {
        self.inner.shader = Some(shader);
        self
    }

    /// Build the compute pipeline create info.
    pub fn build(self) -> ComputePipelineCreateInfo {
        self.inner
    }

    /// Obtain the pipeline name.
    pub fn get_name(&self) -> &str {
        &self.inner.name
    }
}
