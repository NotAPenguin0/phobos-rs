//! Wrapper types for compute pipelines

use ash::vk;

use crate::pipeline::pipeline_layout::PipelineLayoutCreateInfo;
use crate::ShaderCreateInfo;

/// Create info for a compute pipeline. Use the [`ComputePipelineBuilder`](crate::ComputePipelineBuilder)
/// struct to construct this.
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct ComputePipelineCreateInfo {
    /// The shader used in this compute pipeline.
    pub shader: Option<ShaderCreateInfo>,
    pub(crate) name: String,
    pub(crate) layout: PipelineLayoutCreateInfo,
    pub(crate) persistent: bool,
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

/// Builder struct similar to [`PipelineBuilder`](crate::PipelineBuilder), but for compute pipelines. Since compute pipelines are much simpler,
/// there is much less work to do when building one.
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
                persistent: false,
            },
        }
    }

    /// Set the compute shader that will be used for this pipeline. Note that compute pipelines
    /// can only have one shader.
    pub fn set_shader(mut self, shader: ShaderCreateInfo) -> Self {
        self.inner.shader = Some(shader);
        self
    }

    /// Make this compute pipeline persistent, meaning it will never get cleaned up by the cache.
    /// Use this with caution, frequently recreating persistent pipelines will cause a pileup of memory.
    /// This is intentionally not available for graphics pipelines to avoid this issue, since those
    /// need to be recreated much more frequently.
    pub fn persistent(mut self) -> Self {
        self.inner.persistent = true;
        self
    }

    /// Build the compute pipeline create info.
    pub fn build(self) -> ComputePipelineCreateInfo {
        self.inner
    }

    /// Obtain the pipeline name.
    pub fn name(&self) -> &str {
        &self.inner.name
    }
}
