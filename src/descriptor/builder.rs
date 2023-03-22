//! The descriptor set builder is useful for building descriptor sets, but its public usage is now deprecated.
//! Instead, use the `bind_xxx` functions of [`IncompleteCommandBuffer`](crate::command_buffer::IncompleteCommandBuffer)

use anyhow::Result;
use ash::vk;

use crate::{BufferView, Error, ImageView, PhysicalResourceBindings, Sampler, VirtualResource};
use crate::descriptor::descriptor_set::{
    DescriptorBinding, DescriptorBufferInfo, DescriptorContents, DescriptorImageInfo,
    DescriptorSetBinding,
};
use crate::graph::physical_resource::PhysicalResource;
#[cfg(feature = "shader-reflection")]
use crate::pipeline::shader_reflection::ReflectionInfo;

/// This structure is used to build up `DescriptorSetBinding` objects for requesting descriptor sets.
/// # Example usage
/// ```
/// use phobos::DescriptorSetBuilder;
/// // Create a descriptor set with a single binding, and bind `my_image_view` together with
/// // `my_sampler` as a combined image sampler.
/// let set = DescriptorSetBuilder::new()
///             .bind_sampled_image(0, &my_image_view, &my_sampler)
///             .build();
/// ```
#[cfg(feature = "shader-reflection")]
#[derive(Debug)]
pub struct DescriptorSetBuilder<'a> {
    inner: DescriptorSetBinding,
    reflection: Option<&'a ReflectionInfo>,
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
#[cfg(not(feature = "shader-reflection"))]
pub struct DescriptorSetBuilder<'a> {
    inner: DescriptorSetBinding,
    _phantom: PhantomData<&'a i32>,
}

impl<'r> DescriptorSetBuilder<'r> {
    /// Create a new empty descriptor set builder with no reflection information.
    pub fn new() -> Self {
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
    pub fn bind_sampled_image(&mut self, binding: u32, image: &ImageView, sampler: &Sampler) -> () {
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
    pub fn bind_named_sampled_image(
        &mut self,
        name: &str,
        image: &ImageView,
        sampler: &Sampler,
    ) -> Result<()> {
        let Some(info) = self.reflection else { return Err(Error::NoReflectionInformation.into()); };
        let binding = info
            .bindings
            .get(name)
            .ok_or(Error::NoBinding(name.to_string()))?;
        self.bind_sampled_image(binding.binding, image, sampler);
        Ok(())
    }

    /// Bind a uniform buffer to the specified slot.
    pub fn bind_uniform_buffer(&mut self, binding: u32, buffer: &BufferView) -> () {
        self.inner.bindings.push(DescriptorBinding {
            binding,
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptors: vec![DescriptorContents::Buffer(DescriptorBufferInfo {
                buffer: buffer.clone(),
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
        let binding = info
            .bindings
            .get(name)
            .ok_or(Error::NoBinding(name.to_string()))?;
        self.bind_uniform_buffer(binding.binding, buffer);
        Ok(())
    }

    /// Build the descriptor set creation info to pass into the cache.
    pub fn build(self) -> DescriptorSetBinding {
        self.inner
    }
}