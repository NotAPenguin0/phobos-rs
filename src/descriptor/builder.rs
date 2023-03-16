use crate::descriptor_set::{DescriptorBinding, DescriptorBufferInfo, DescriptorContents, DescriptorImageInfo, DescriptorSetBinding};
use crate::{BufferView, Error, ImageView, PhysicalResourceBindings, Sampler, VirtualResource};

use anyhow::Result;
use ash::vk;
use crate::graph::physical_resource::PhysicalResource;

#[cfg(feature="shader-reflection")]
use crate::pipeline::shader_reflection::ReflectionInfo;

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
#[derive(Debug)]
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

    pub fn resolve_and_bind_sampled_image(&mut self, binding: u32, resource: VirtualResource, sampler: &Sampler, bindings: &PhysicalResourceBindings) -> Result<()> {
        if let Some(PhysicalResource::Image(image)) = bindings.resolve(&resource) {
            self.bind_sampled_image(binding, image.clone(), sampler);
            Ok(())
        } else {
            Err(Error::NoResourceBound(resource.uid.clone()).into())
        }
    }

    /// Bind an image view to the given binding as a [`vk::DescriptorType::COMBINED_IMAGE_SAMPLER`]
    pub fn bind_sampled_image(&mut self, binding: u32, image: ImageView, sampler: &Sampler) -> () {
        self.inner.bindings.push(DescriptorBinding {
            binding,
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptors: vec![ DescriptorContents::Image(DescriptorImageInfo {
                sampler: sampler.handle,
                view: image,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }) ],
        });
    }

    #[cfg(feature="shader-reflection")]
    pub fn bind_named_sampled_image(&mut self, name: &str, image: ImageView, sampler: &Sampler) -> Result<()> {
        let Some(info) = self.reflection else { return Err(Error::NoReflectionInformation.into()); };
        let binding = info.bindings.get(name).ok_or(Error::NoBinding(name.to_string()))?;
        self.bind_sampled_image(binding.binding, image, sampler);
        Ok(())
    }

    pub fn bind_uniform_buffer(&mut self, binding: u32, buffer: BufferView) -> () {
        self.inner.bindings.push(DescriptorBinding {
            binding,
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptors: vec![ DescriptorContents::Buffer(DescriptorBufferInfo {
                buffer,
            }) ],
        });
    }

    #[cfg(feature="shader-reflection")]
    pub fn bind_named_uniform_buffer(&mut self, name: &str, buffer: BufferView) -> Result<()> {
        let Some(info) = self.reflection else { return Err(Error::NoReflectionInformation.into()); };
        let binding = info.bindings.get(name).ok_or(Error::NoBinding(name.to_string()))?;
        self.bind_uniform_buffer(binding.binding, buffer);
        Ok(())
    }

    pub fn build(self) -> DescriptorSetBinding {
        self.inner
    }
}