//! Re-exports most commonly used types in the library

pub use ash::vk;

pub use traits::*;

pub use crate::allocator::default_allocator;
pub use crate::allocator::default_allocator::DefaultAllocator;
pub use crate::allocator::memory_type::MemoryType;
pub use crate::allocator::scratch_allocator::ScratchAllocator;
pub use crate::command_buffer::{CommandBuffer, IncompleteCommandBuffer};
pub use crate::core::app_info::*;
pub use crate::core::debug::DebugMessenger;
pub use crate::core::device::Device;
pub use crate::core::error::Error;
pub use crate::core::init::*;
pub use crate::core::instance::Instance;
pub use crate::core::physical_device::*;
pub use crate::core::queue::QueueType;
pub use crate::descriptor::cache::DescriptorCache;
pub use crate::descriptor::descriptor_set::DescriptorSet;
pub use crate::graph::pass::{ClearColor, ClearDepthStencil, Pass, PassBuilder};
pub use crate::graph::pass_graph::PassGraph;
pub use crate::graph::physical_resource::PhysicalResourceBindings;
pub use crate::graph::virtual_resource::VirtualResource;
pub use crate::pipeline::{PipelineStage, PipelineType};
pub use crate::pipeline::builder::PipelineBuilder;
pub use crate::pipeline::cache::PipelineCache;
pub use crate::pipeline::compute::{ComputePipelineBuilder, ComputePipelineCreateInfo};
pub use crate::pipeline::create_info::PipelineCreateInfo;
pub use crate::pipeline::hash::*;
pub use crate::pipeline::raytracing::RayTracingPipelineBuilder;
pub use crate::pipeline::shader::ShaderCreateInfo;
pub use crate::resource::*;
pub use crate::resource::buffer::{Buffer, BufferView};
pub use crate::resource::image::{Image, ImageView};
pub use crate::resource::query_pool::*;
pub use crate::resource::raytracing::*;
pub use crate::sampler::Sampler;
pub use crate::sync::domain;
pub use crate::sync::execution_manager::ExecutionManager;
pub use crate::sync::fence::*;
pub use crate::sync::semaphore::*;
pub use crate::util::address::*;
pub use crate::util::byte_size::ByteSize;
pub use crate::util::deferred_delete::DeletionQueue;
pub use crate::util::transform::TransformMatrix;
pub use crate::wsi::frame::{FrameManager, InFlightContext};
pub use crate::wsi::surface::Surface;
pub use crate::wsi::swapchain::Swapchain;

/// Re-exports all important traits of the library
pub mod traits {
    pub use crate::allocator::traits::*;
    pub use crate::command_buffer::traits::*;
    pub use crate::graph::pass_graph::GraphViz;
    pub use crate::graph::record::RecordGraphToCommandBuffer;
    pub use crate::wsi::window::{Window, WindowSize};
}
