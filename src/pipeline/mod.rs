//! Deals with wrappers for creating and managing Vulkan pipeline objects and their related objects.
//!
//! This is a helper that manages creating pipelines, obtaining reflection information from them (if the `shader-reflection` feature is enabled).
//! You probably only want one of these in the entire application. Since it's used everywhere, to ensure safe access
//! is possible, the inner state of a [`PipelineCache`](crate::PipelineCache) is wrapped in an `Arc<RwLock<PipelineCacheInner>>`,
//! so this is `Send`, `Sync` and `Clone`.
//!
//! # Example
//! The following example uses the [`PipelineBuilder`](crate::PipelineBuilder) utility to make a graphics pipeline and add it to the pipeline cache.
//!
//! ```
//! use std::path::Path;
//! use phobos::prelude::*;
//!
//! let mut cache = PipelineCache::new(device.clone(), allocator.clone())?;
//!
//! // Load in some shader code for our pipelines.
//! // Note that `load_spirv_binary()` does not ship with phobos.
//! let vtx_code = load_spirv_binary(Path::new("path/to/vertex.glsl"));
//! let frag_code = load_spirv_binary(Path::new("path/to/fragment.glsl"));
//!
//! // Create shaders, these can safely be discarded after building the pipeline
//! // as they are just create info structs that are also stored internally
//! let vertex = ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::VERTEX, vtx_code);
//! let fragment = ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, frag_code);
//!
//! // Now we can build the actual pipeline.
//! let pci = PipelineBuilder::new("sample")
//!     // One vertex binding at binding 0. We have to specify this before adding attributes
//!     .vertex_input(0, vk::VertexInputRate::VERTEX)
//!     // Equivalent of `layout (location = 0) in vec2 Attr1;`
//!     // Note that this function can fail if the binding from before does not exist.
//!     .vertex_attribute(0, 0, vk::Format::R32G32_SFLOAT)?
//!     // Equivalent of `layout (location = 1) in vec2 Attr2;`
//!     .vertex_attribute(0, 1, vk::Format::R32G32_SFLOAT)?
//!     // To avoid having to recreate the pipeline every time the viewport changes, this is recommended.
//!     // Generally this does not decrease performance.
//!     .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
//!     // We don't want any blending, but we still need to specify what happens to our color output.
//!     .blend_attachment_none()
//!     // Disable face culling
//!     .cull_mask(vk::CullModeFlags::NONE)
//!     // Add our shaders
//!     .attach_shader(vertex)
//!     .attach_shader(fragment)
//!     .build();
//!
//! cache.create_named_pipeline(pci)?;
//! ```
//! # Correct usage
//! The pipeline cache internally frees up resources by destroying pipelines that have not been accessed in a long time.
//! To ensure this happens periodically, call [`PipelineCache::next_frame()`](crate::PipelineCache::next_frame) at the end of each iteration of your render loop.

use ash::vk;

use crate::pipeline::raytracing::ShaderBindingTable;
use crate::{Allocator, Device};

pub mod builder;
pub mod cache;
pub mod compute;
pub mod create_info;
pub mod hash;
pub mod pipeline_layout;
pub mod raytracing;
pub mod set_layout;
pub mod shader;

pub(crate) mod shader_reflection;

/// Pipeline stage in the GPU pipeline.
pub type PipelineStage = vk::PipelineStageFlags2;

/// A fully built Vulkan pipeline. This is a managed resource, so it cannot be manually
/// cloned or dropped.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Pipeline {
    #[derivative(Debug = "ignore")]
    device: Device,
    pub(crate) handle: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) set_layouts: Vec<vk::DescriptorSetLayout>,
}

/// A fully built Vulkan compute pipeline. This is a managed resource, so it cannot be manually
/// cloned or dropped.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct ComputePipeline {
    #[derivative(Debug = "ignore")]
    device: Device,
    pub(crate) handle: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) set_layouts: Vec<vk::DescriptorSetLayout>,
}

/// A fully built Vulkan ray tracing pipeline. This is a managed resource, so it cannot be manually
/// cloned or dropped.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct RayTracingPipeline<A: Allocator> {
    device: Device,
    pub(crate) handle: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) set_layouts: Vec<vk::DescriptorSetLayout>,
    pub(crate) shader_binding_table: ShaderBindingTable<A>,
}

/// Pipeline type.
#[derive(Debug)]
pub enum PipelineType {
    /// Graphics pipeline
    Graphics,
    /// Compute pipeline
    Compute,
    /// Raytracing pipeline
    RayTracing,
}
