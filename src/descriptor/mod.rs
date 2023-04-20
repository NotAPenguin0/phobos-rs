//! This module handles everything related to descriptor sets.
//!
//! Similarly to the [`pipeline`](crate::pipeline) module, this module exposes a [`DescriptorCache`](crate::DescriptorCache) struct.
//! This struct handles allocation of descriptor sets, writing to them and manages a descriptor pool.
//!
//! The descriptor pool automatically grows as more descriptors are allocated, removing the need to declare its size upfront.
//!
//! Binding descriptor sets is handled directly through the command buffer. This functionality is only available if the command
//! buffer was created using a descriptor and pipeline cache.
//!
//! # Example
//! ```
//! # use phobos::prelude::*;
//! # use anyhow::Result;
//! fn descriptor_sets_example<A: Allocator>(device: Device, alloc: A, exec: ExecutionManager, image: &ImageView, sampler: &Sampler) -> Result<()> {
//!     let descriptors = DescriptorCache::new(device.clone())?;
//!     let pipelines = PipelineCache::new(device.clone(), alloc)?;
//!     let cmd = exec.on_domain::<domain::All, A>(Some(pipelines), Some(cache))?
//!         .bind_graphics_pipeline("my_pipeline")?
//!         // In GLSL: layout(set = 0, binding = 0) uniform sampler2D tex;
//!         .bind_sampled_image(0, 0, image, &sampler)?
//!         .draw(6, 1, 0, 0)?;
//!     Ok(())
//! }
//! ```

pub mod builder;
pub mod cache;
pub mod descriptor_set;

mod descriptor_pool;
