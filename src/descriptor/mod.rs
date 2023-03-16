//! This module handles everything related to descriptor sets.
//! Similarly to the [`pipeline`] module, this module exposes a [`DescriptorCache`] struct.
//! This struct handles allocation of descriptor sets, writing to them and manages a descriptor pool.
//!
//! The descriptor pool automatically grows as more descriptors are allocated, removing the need to declare its size upfront.
//!
//! To allocate descriptor sets, use the provided [`DescriptorSetBuilder`] structure to specify bindings for
//! descriptor sets.
//!
//! # Example
//!
//! ```
//! use phobos as ph;
//!
//! let mut cache = ph::DescriptorCache::new(device.clone())?;
//! let set = ph::DescriptorSetBuilder::new()
//!           // In GLSL this would be a descriptor
//!           // layout(set = X, binding = 0) uniform sampler2D tex;
//!           .bind_sampled_image(0, my_image_view, &my_sampler)
//!           .build();
//! ```
//!
//! # Shader reflection
//!
//! Specifying bindings manually can be tedious, but it's fast. Using shader reflection allows you to omit this, at the cost
//! of one string hashmap lookup per binding. This is normally not expensive, so it's recommended to use this by enabling the
//! [`shader-reflection`] feature. Doing so exposes a new constructor [`DescriptorSetBuilder::with_reflection`] that can be used to attach
//! reflection info. It also gives access to new `bind_named_xxx` versions of all previous `bind` calls in the builder that use the provided
//! reflection information.
//!
//! ```
//! use phobos as ph;
//!
//! let set = {
//!     let cache: ph::PipelineCache = pipeline_cache.lock().unwrap();
//!     let reflection = cache.reflection_info("my_pipeline")?;
//!     ph::DescriptorSetBuilder::with_reflection(reflection)
//!         // In GLSL: layout(set = X, binding = Y) uniform sampler2D tex;
//!         .bind_named_sampled_image("tex", my_image_view, &my_sampler)?
//!         .build()
//! };
//! ```
//!
//! To bind a descriptor set, prefer using [`IncompleteCommandBuffer::bind_new_descriptor_set`] over the two separate calls.
//!

pub mod builder;
pub mod cache;
pub mod descriptor_set;

mod descriptor_pool;
