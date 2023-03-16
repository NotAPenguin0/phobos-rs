use std::sync::Arc;
use ash::vk;
use crate::{BufferView, Device, ImageView};
use crate::cache::Resource;

use anyhow::Result;

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

struct WriteDescriptorSet {
    set: vk::DescriptorSet,
    binding: u32,
    array_element: u32,
    count: u32,
    ty: vk::DescriptorType,
    image_info: Option<Vec<vk::DescriptorImageInfo>>,
    buffer_info: Option<Vec<vk::DescriptorBufferInfo>>,
}

impl Resource for DescriptorSet {
    type Key = DescriptorSetBinding;
    type ExtraParams<'a> = ();
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Arc<Device>, key: &Self::Key, _: Self::ExtraParams<'_>) -> Result<Self> where Self: Sized {
        let info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            descriptor_pool: key.pool,
            descriptor_set_count: 1,
            p_set_layouts: &key.layout,
        };
        let set = unsafe { device.allocate_descriptor_sets(&info) }?.first().cloned().unwrap();
        let writes = key.bindings.iter().map(|binding| {
            let mut write = WriteDescriptorSet {
                set,
                binding: binding.binding,
                array_element: 0,
                count: binding.descriptors.len() as u32,
                ty: binding.ty,
                image_info: None,
                buffer_info: None,
            };

            match binding.ty {
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER => { write.image_info = Some(binding_image_info(&binding)); },
                vk::DescriptorType::SAMPLED_IMAGE => { write.image_info = Some(binding_image_info(&binding)); },
                vk::DescriptorType::UNIFORM_BUFFER => { write.buffer_info = Some(binding_buffer_info(&binding)); },
                _ => { todo!(); }
            }
            write
        })
            .collect::<Vec<WriteDescriptorSet>>();

        let vk_writes = writes.iter().map(|write| {
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: write.set,
                dst_binding: write.binding,
                dst_array_element: write.array_element,
                descriptor_count: write.count,
                descriptor_type: write.ty,
                p_image_info: match &write.image_info {
                    None => { std::ptr::null() }
                    Some(image) => { image.as_ptr() }
                },
                p_buffer_info: match &write.buffer_info {
                    None => { std::ptr::null() }
                    Some(buffer) => { buffer.as_ptr() }
                },
                p_texel_buffer_view: std::ptr::null(),
            }
        }).collect::<Vec<_>>();

        unsafe { device.update_descriptor_sets(vk_writes.as_slice(), &[]); }

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