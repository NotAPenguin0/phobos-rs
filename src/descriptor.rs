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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use ash::vk;
use crate::cache::*;
use crate::{BufferView, Device, Error, ImageView, IncompleteCommandBuffer, PhysicalResource, PhysicalResourceBindings, Sampler, VirtualResource};
use crate::deferred_delete::DeletionQueue;
use anyhow::Result;
#[cfg(feature="shader-reflection")]
use crate::shader_reflection::ReflectionInfo;
use std::marker::PhantomData;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct DescriptorImageInfo {
    pub sampler: vk::Sampler,
    pub view: ImageView,
    pub layout: vk::ImageLayout
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct DescriptorBufferInfo {
    pub buffer: BufferView,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) enum DescriptorContents {
    Image(DescriptorImageInfo),
    Buffer(DescriptorBufferInfo),
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct DescriptorBinding {
    pub binding: u32,
    pub ty: vk::DescriptorType,
    pub descriptors: Vec<DescriptorContents>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DescriptorSetBinding {
    pub(crate) pool: vk::DescriptorPool,
    pub(crate) bindings: Vec<DescriptorBinding>,
    pub(crate) layout: vk::DescriptorSetLayout,
}

#[derive(Derivative)]
#[derivative(Debug, PartialEq, Eq)]
pub struct DescriptorSet {
    #[derivative(Debug="ignore")]
    #[derivative(PartialEq="ignore")]
    pub(crate) device: Arc<Device>,
    pub(crate) pool: vk::DescriptorPool,
    pub(crate) handle: vk::DescriptorSet
}

/// Defines how many descriptors a descriptor pool should be able to hold.
#[derive(Debug, Clone)]
pub(crate) struct DescriptorPoolSize(HashMap<vk::DescriptorType, u32>);

#[derive(Derivative)]
#[derivative(Debug)]
pub(crate) struct DescriptorPool {
    #[derivative(Debug="ignore")]
    pub device: Arc<Device>,
    pub handle: vk::DescriptorPool,
    pub size: DescriptorPoolSize
}

/// This structure uses a [`Cache`] over a [`DescriptorSet`] to automatically manage everything related to descriptor sets.
/// It can intelligently allocate and deallocate descriptor sets, and grow its internal descriptor pool when necessary.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct DescriptorCache {
    #[derivative(Debug="ignore")]
    device: Arc<Device>,
    cache: Cache<DescriptorSet>,
    pool: DescriptorPool,
    deferred_pool_delete: DeletionQueue<DescriptorPool>,
}

/// This structure is used to build up `DescriptorSetBinding` objects for requesting descriptor sets.
/// # Example usage
/// ```
/// use phobos::DescriptorSetBuilder;
/// // Create a descriptor set with a single binding, and bind `my_image_view` together with
/// // `my_sampler` as a combined image sampler.
/// let set = DescriptorSetBuilder::new()
///             .bind_sampled_image(0, my_image_view.clone(), &my_sampler)
///             .build();
/// ```
#[cfg(feature="shader-reflection")]
pub struct DescriptorSetBuilder<'a> {
    inner: DescriptorSetBinding,
    reflection: Option<&'a ReflectionInfo>
}


/// This structure is used to build up `DescriptorSetBinding` objects for requesting descriptor sets.
/// # Example usage
/// ```
/// use phobos::DescriptorSetBuilder;
/// // Create a descriptor set with a single binding, and bind `my_image_view` together with
/// // `my_sampler` as a combined image sampler.
/// let set = DescriptorSetBuilder::new()
///             .bind_sampled_image(0, my_image_view.clone(), &my_sampler)
///             .build();
/// ```
#[cfg(not(feature="shader-reflection"))]
pub struct DescriptorSetBuilder<'a> {
    inner: DescriptorSetBinding,
    _phantom: PhantomData<&'a i32>
}

fn binding_image_info(binding: &DescriptorBinding) -> Vec<vk::DescriptorImageInfo> {
    binding.descriptors.iter().map(|descriptor| {
        let DescriptorContents::Image(image) = descriptor else { panic!("Missing descriptor type case?") };
        vk::DescriptorImageInfo {
            sampler: image.sampler,
            image_view: image.view.handle,
            image_layout: image.layout,
        }
    })
    .collect()
}

fn binding_buffer_info(binding: &DescriptorBinding) -> Vec<vk::DescriptorBufferInfo> {
    binding.descriptors.iter().map(|descriptor| {
        let DescriptorContents::Buffer(buffer) = descriptor else { panic!("Missing descriptor type case?") };
        vk::DescriptorBufferInfo {
            buffer: buffer.buffer.handle,
            offset: buffer.buffer.offset,
            range: buffer.buffer.size,
        }
    })
    .collect()
}

impl Resource for DescriptorSet {
    type Key = DescriptorSetBinding;
    type ExtraParams<'a> = ();
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Arc<Device>, key: &Self::Key, _: Self::ExtraParams<'_>) -> Result<Self> where Self: Sized {
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(key.pool)
            .set_layouts(std::slice::from_ref(&key.layout))
            .build();
        let set = unsafe { device.allocate_descriptor_sets(&info) }?.first().cloned().unwrap();
        let writes = key.bindings.iter().map(|binding| {
            let mut write = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: set,
                dst_binding: binding.binding,
                dst_array_element: 0,
                descriptor_count: binding.descriptors.len() as u32,
                descriptor_type: binding.ty,
                p_image_info: std::ptr::null(),
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
            };
            // Now fill in the actual write info, based on the correct type
            let mut image_info = Vec::new();
            let mut buffer_info = Vec::new();
            match binding.ty {
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER => { image_info = binding_image_info(&binding); write.p_image_info = image_info.as_ptr(); },
                vk::DescriptorType::SAMPLED_IMAGE => { image_info = binding_image_info(&binding); write.p_image_info = image_info.as_ptr(); },
                vk::DescriptorType::UNIFORM_BUFFER => { buffer_info = binding_buffer_info(&binding); write.p_buffer_info = buffer_info.as_ptr(); },
                _ => { todo!(); }
            }
            write
        })
        .collect::<Vec<vk::WriteDescriptorSet>>();

        unsafe { device.update_descriptor_sets(writes.as_slice(), &[]); }

        Ok(DescriptorSet {
            device,
            pool: key.pool,
            handle: set,
        })
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        unsafe { self.device.free_descriptor_sets(self.pool, std::slice::from_ref(&self.handle)).unwrap(); }
    }
}

impl DescriptorPoolSize {
    pub fn new(min_capacity: u32) -> Self {
        let mut sizes = HashMap::new();
        sizes.insert(vk::DescriptorType::SAMPLER, min_capacity);
        sizes.insert(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, min_capacity);
        sizes.insert(vk::DescriptorType::SAMPLED_IMAGE, min_capacity);
        sizes.insert(vk::DescriptorType::STORAGE_IMAGE, min_capacity);
        sizes.insert(vk::DescriptorType::UNIFORM_TEXEL_BUFFER, min_capacity);
        sizes.insert(vk::DescriptorType::STORAGE_TEXEL_BUFFER, min_capacity);
        sizes.insert(vk::DescriptorType::UNIFORM_BUFFER, min_capacity);
        sizes.insert(vk::DescriptorType::STORAGE_BUFFER, min_capacity);
        sizes.insert(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC, min_capacity);
        sizes.insert(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC, min_capacity);
        sizes.insert(vk::DescriptorType::INPUT_ATTACHMENT, min_capacity);
        Self {
            0: sizes,
        }
    }
}

impl DescriptorPool {
    fn new(device: Arc<Device>, size: DescriptorPoolSize) -> Result<Self> {
        // Fold over all values to compute the sum of all sizes, this is the total amount of
        // descriptor sets that can be allocated from this pool.
        let max_sets = size.0.values().fold(0, |a, x| x + a);
        let flags = vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET;
        let pool_sizes = size.0.iter().map(|(descriptor_type, count)| {
            vk::DescriptorPoolSize {
                ty: *descriptor_type,
                descriptor_count: *count,
            }
        }).collect::<Vec<vk::DescriptorPoolSize>>();

        let info = vk::DescriptorPoolCreateInfo::builder()
            .flags(flags)
            .max_sets(max_sets)
            .pool_sizes(pool_sizes.as_slice())
            .build();

        Ok(Self{
            device: device.clone(),
            handle: unsafe { device.create_descriptor_pool(&info, None)? },
            size,
        })
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe { self.device.destroy_descriptor_pool(self.handle, None); }
    }
}

impl DescriptorCache {
    /// Create a new descriptor cache object.
    /// # Errors
    /// - This can fail if creating the initial descriptor pool fails.
    pub fn new(device: Arc<Device>) -> Result<Arc<Mutex<Self>>> {
        Ok(Arc::new(Mutex::new(Self {
            device: device.clone(),
            cache: Cache::new(device.clone()),
            pool: DescriptorPool::new(device.clone(), DescriptorPoolSize::new(1))?,
            deferred_pool_delete: DeletionQueue::new(16),
        })))
    }

    fn grow_pool_size(mut old_size: DescriptorPoolSize, request: &DescriptorSetBinding) -> DescriptorPoolSize {
        for (ty, count) in old_size.0.iter_mut() {
            if request.bindings.iter().find(|&binding| binding.ty == *ty).is_some() {
                *count = *count * 2;
            }
        }
        old_size
    }

    /// Get a new descriptor set with the given descriptor set binding.
    /// If the internal descriptor pool runs out of space, a new one will be created.
    /// # Errors
    /// - This function fails if no descriptor set layout was specified in `bindings`
    /// - This function fails the the requested descriptor set has no descriptors
    /// - This function fails if allocating a descriptor set failed due to an internal error.
    pub fn get_descriptor_set(&mut self, mut bindings: DescriptorSetBinding) -> Result<&DescriptorSet> {
        if bindings.bindings.is_empty() { return Err(anyhow::Error::from(Error::EmptyDescriptorBinding)); }
        if bindings.layout == vk::DescriptorSetLayout::null() { return Err(anyhow::Error::from(Error::NoDescriptorSetLayout)); }

        loop {
            bindings.pool = self.pool.handle;
            let set = self.cache.get_or_create(&bindings, ());
            match set {
                // Need to query again to fix lifetime compiler error
                Ok(_) => { return Ok(self.cache.get_or_create(&bindings, ()).unwrap()); }
                Err(_) => {
                    let new_size = Self::grow_pool_size(self.pool.size.clone(), &bindings);
                    // Create new pool, swap it out with the old one and then push the old one onto the deletion queue
                    let mut new_pool = DescriptorPool::new(self.device.clone(), new_size)?;
                    std::mem::swap(&mut new_pool, &mut self.pool);
                    self.deferred_pool_delete.push(new_pool);
                }
            }
        }
    }

    /// Advance the descriptor cache to the next frame. This allows resources to be reclaimed safely where possible.
    pub fn next_frame(&mut self) {
        self.cache.next_frame();
        self.deferred_pool_delete.next_frame();
    }
}

impl<'r> DescriptorSetBuilder<'r> {
    pub fn new() -> Self {
        Self {
            inner: DescriptorSetBinding {
                pool: vk::DescriptorPool::null(),
                bindings: vec![],
                layout: vk::DescriptorSetLayout::null(),
            },
            #[cfg(feature="shader-reflection")]
            reflection: None,
            #[cfg(not(feature="shader-reflection"))]
            _phantom: PhantomData::default(),
        }
    }

    #[cfg(feature="shader-reflection")]
    pub fn with_reflection(info: &'r ReflectionInfo) -> Self {
        Self {
            inner: DescriptorSetBinding {
                pool: vk::DescriptorPool::null(),
                bindings: vec![],
                layout: vk::DescriptorSetLayout::null(),
            },
            reflection: Some(info),
        }
    }

    pub fn resolve_and_bind_sampled_image(mut self, binding: u32, resource: VirtualResource, sampler: &Sampler, bindings: &PhysicalResourceBindings) -> Result<Self> {
        if let Some(PhysicalResource::Image(image)) = bindings.resolve(&resource) {
            Ok(self.bind_sampled_image(binding, image.clone(), sampler))
        } else {
            Err(Error::NoResourceBound(resource.uid.clone()).into())
        }
    }

    /// Bind an image view to the given binding as a [`vk::DescriptorType::COMBINED_IMAGE_SAMPLER`]
    pub fn bind_sampled_image(mut self, binding: u32, image: ImageView, sampler: &Sampler) -> Self {
        self.inner.bindings.push(DescriptorBinding {
            binding,
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptors: vec![ DescriptorContents::Image(DescriptorImageInfo {
                sampler: sampler.handle,
                view: image,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }) ],
        });
        self
    }

    #[cfg(feature="shader-reflection")]
    pub fn bind_named_sampled_image(mut self, name: &str, image: ImageView, sampler: &Sampler) -> Result<Self> {
        let Some(info) = self.reflection else { return Err(Error::NoReflectionInformation.into()); };
        let binding = info.bindings.get(name).ok_or(Error::NoBinding(name.to_string()))?;
        Ok(self.bind_sampled_image(binding.binding, image, sampler))
    }

    pub fn bind_uniform_buffer(mut self, binding: u32, buffer: BufferView) -> Self {
        self.inner.bindings.push(DescriptorBinding {
            binding,
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptors: vec![ DescriptorContents::Buffer(DescriptorBufferInfo {
                buffer,
            }) ],
        });

        self
    }

    #[cfg(feature="shader-reflection")]
    pub fn bind_named_uniform_buffer(mut self, name: &str, buffer: BufferView) -> Result<Self> {
        let Some(info) = self.reflection else { return Err(Error::NoReflectionInformation.into()); };
        let binding = info.bindings.get(name).ok_or(Error::NoBinding(name.to_string()))?;
        Ok(self.bind_uniform_buffer(binding.binding, buffer))
    }

    pub fn build(self) -> DescriptorSetBinding {
        self.inner
    }
}