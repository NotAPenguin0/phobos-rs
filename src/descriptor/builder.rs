//! The descriptor set builder is useful for building descriptor sets, but its public usage is now deprecated.
//! Instead, use the `bind_xxx` functions of [`IncompleteCommandBuffer`](crate::command_buffer::IncompleteCommandBuffer)

use anyhow::Result;
use ash::vk;

use crate::{BufferView, Error, ImageView, PhysicalResourceBindings, Sampler, VirtualResource};
use crate::descriptor::descriptor_set::{DescriptorBinding, DescriptorBufferInfo, DescriptorContents, DescriptorImageInfo, DescriptorSetBinding};
use crate::graph::physical_resource::PhysicalResource;
#[cfg(feature = "shader-reflection")]
use crate::pipeline::shader_reflection::ReflectionInfo;
use crate::raytracing::acceleration_structure::AccelerationStructure;

/// This structure is used to build up [`DescriptorSetBinding`](crate::descriptor::descriptor_set::DescriptorSetBinding) objects for requesting descriptor sets.
/// Public usage of this API is deprecated, use the provided methods inside [`IncompleteCommandBuffer`](crate::IncompleteCommandBuffer).
/// # Example usage
/// ```
/// # use phobos::*;
/// # use phobos::descriptor::descriptor_set::DescriptorSetBinding;
/// fn make_descriptor_set(image: &ImageView, sampler: &Sampler) -> DescriptorSetBinding {
///     DescriptorSetBuilder::new()
///         .bind_sampled_image(0, image, sampler)
///         .build()
/// }
///
/// ```
#[cfg(feature = "shader-reflection")]
#[derive(Debug)]
pub struct DescriptorSetBuilder<'a> {
    inner: DescriptorSetBinding,
    reflection: Option<&'a ReflectionInfo>,
}

/// This structure is used to build up [`DescriptorSetBinding`](crate::descriptor::descriptor_set::DescriptorSetBinding) objects for requesting descriptor sets.
/// Public usage of this API is deprecated, use the provided methods inside [`IncompleteCommandBuffer`](crate::IncompleteCommandBuffer).
/// # Example usage
/// ```
/// # use phobos::*;
/// # use phobos::descriptor::descriptor_set::DescriptorSetBinding;
/// fn make_descriptor_set(image: &ImageView, sampler: &Sampler) -> DescriptorSetBinding {
///     DescriptorSetBuilder::new()
///         .bind_sampled_image(0, image, sampler)
///         .build()
/// }
///
/// ```
#[cfg(not(feature = "shader-reflection"))]
pub struct DescriptorSetBuilder<'a> {
    inner: DescriptorSetBinding,
    _phantom: PhantomData<&'a ()>,
}

impl<'r> Default for DescriptorSetBuilder<'r> {
    /// Create a default descriptor set builder with no bindings.
    fn default() -> Self {
        Self {
            inner: DescriptorSetBinding {
                pool: vk::DescriptorPool::null(),
                bindings: vec![],
                layout: vk::DescriptorSetLayout::null(),
            },
            #[cfg(feature = "shader-reflection")]
            reflection: None,
            #[cfg(not(feature = "shader-reflection"))]
            _phantom: PhantomData::default(),
        }
    }
}

impl<'r> DescriptorSetBuilder<'r> {
    /// Create a new empty descriptor set builder with no reflection information.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new empty descriptor set builder with associated reflection information.
    /// This enables the usage of the `bind_named_xxx` set of functions.
    #[cfg(feature = "shader-reflection")]
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

    /// Resolve the virtual resource through the given bindings, and bind it to a specific slot as a combined image sampler.
    /// # Errors
    /// Fails if the binding did not exist, or did not contain an image.
    pub fn resolve_and_bind_sampled_image(
        &mut self,
        binding: u32,
        resource: &VirtualResource,
        sampler: &Sampler,
        bindings: &PhysicalResourceBindings,
    ) -> Result<()> {
        if let Some(PhysicalResource::Image(image)) = bindings.resolve(resource) {
            self.bind_sampled_image(binding, image, sampler);
            Ok(())
        } else {
            Err(Error::NoResourceBound(resource.uid().clone()).into())
        }
    }

    /// Bind an image view to the given binding as a [`vk::DescriptorType::COMBINED_IMAGE_SAMPLER`]
    pub fn bind_sampled_image(&mut self, binding: u32, image: &ImageView, sampler: &Sampler) {
        self.inner.bindings.push(DescriptorBinding {
            binding,
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptors: vec![DescriptorContents::Image(DescriptorImageInfo {
                sampler: unsafe { sampler.handle() },
                view: image.clone(),
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            })],
        });
    }

    /// Bind an image view to the given binding as a [`vk::DescriptorType::COMBINED_IMAGE_SAMPLER`].
    /// Uses the reflection information provided at construction to look up the correct binding slot by its name
    /// defined in the shader.
    /// # Errors
    /// Fails if `self` was not constructed with [`DescriptorSetBuilder::with_reflection()`].
    #[cfg(feature = "shader-reflection")]
    pub fn bind_named_sampled_image(&mut self, name: &str, image: &ImageView, sampler: &Sampler) -> Result<()> {
        let Some(info) = self.reflection else { return Err(Error::NoReflectionInformation.into()); };
        let binding = info.bindings.get(name).ok_or(Error::NoBinding(name.to_string()))?;
        self.bind_sampled_image(binding.binding, image, sampler);
        Ok(())
    }

    /// Bind a uniform buffer to the specified slot.
    pub fn bind_uniform_buffer(&mut self, binding: u32, buffer: &BufferView) {
        self.inner.bindings.push(DescriptorBinding {
            binding,
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptors: vec![DescriptorContents::Buffer(DescriptorBufferInfo {
                buffer: *buffer,
            })],
        });
    }

    /// Bind a buffer to the given binding as a [`vk::DescriptorType::UNIFORM_BUFFER`].
    /// Uses the reflection information provided at construction to look up the correct binding slot by its name
    /// defined in the shader.
    /// # Errors
    /// Fails if `self` was not constructed with [`DescriptorSetBuilder::with_reflection()`].
    #[cfg(feature = "shader-reflection")]
    pub fn bind_named_uniform_buffer(&mut self, name: &str, buffer: &BufferView) -> Result<()> {
        let Some(info) = self.reflection else { return Err(Error::NoReflectionInformation.into()); };
        let binding = info.bindings.get(name).ok_or(Error::NoBinding(name.to_string()))?;
        self.bind_uniform_buffer(binding.binding, buffer);
        Ok(())
    }

    /// Bind a storage buffer to the specified slot
    pub fn bind_storage_buffer(&mut self, binding: u32, buffer: &BufferView) {
        self.inner.bindings.push(DescriptorBinding {
            binding,
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptors: vec![DescriptorContents::Buffer(DescriptorBufferInfo {
                buffer: *buffer,
            })],
        })
    }

    /// Bind a storage image to the specified slot
    pub fn bind_storage_image(&mut self, binding: u32, image: &ImageView) {
        self.inner.bindings.push(DescriptorBinding {
            binding,
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptors: vec![DescriptorContents::Image(DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                view: image.clone(),
                layout: vk::ImageLayout::GENERAL,
            })],
        })
    }

    /// Resolve and bind a storage image to a specified slot.
    pub fn resolve_and_bind_storage_image(
        &mut self,
        binding: u32,
        resource: &VirtualResource,
        bindings: &PhysicalResourceBindings,
    ) -> Result<()> {
        if let Some(PhysicalResource::Image(image)) = bindings.resolve(resource) {
            self.bind_storage_image(binding, image);
            Ok(())
        } else {
            Err(Error::NoResourceBound(resource.uid().clone()).into())
        }
    }

    /// Bind an acceleration structure to the specified slot.
    pub fn bind_acceleration_structure(&mut self, binding: u32, accel: &AccelerationStructure) {
        self.inner.bindings.push(DescriptorBinding {
            binding,
            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptors: vec![
                DescriptorContents::AccelerationStructure(unsafe { accel.handle() })
            ],
        })
    }

    /// Build the descriptor set creation info to pass into the cache.
    pub fn build(self) -> DescriptorSetBinding {
        self.inner
    }
}
