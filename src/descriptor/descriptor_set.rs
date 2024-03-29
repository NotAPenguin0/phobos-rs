//! Wrappers for descriptor set binding info

use anyhow::Result;
use ash::vk;

use crate::util::cache::{Resource, ResourceKey};
use crate::util::pnext::PNext;
use crate::{BufferView, Device, ImageView};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct DescriptorImageInfo {
    pub sampler: vk::Sampler,
    pub view: ImageView,
    pub layout: vk::ImageLayout,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct DescriptorBufferInfo {
    pub buffer: BufferView,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) enum DescriptorContents {
    Image(DescriptorImageInfo),
    Buffer(DescriptorBufferInfo),
    AccelerationStructure(vk::AccelerationStructureKHR),
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct DescriptorBinding {
    pub binding: u32,
    pub ty: vk::DescriptorType,
    pub descriptors: Vec<DescriptorContents>,
}

/// Specifies a set of bindings in a descriptor set. Can be created by a [`DescriptorSetBuilder`](crate::DescriptorSetBuilder).
/// Public usage of this has been deprecated in favor of using the descriptor set methods in [`IncompleteCommandBuffer`](crate::IncompleteCommandBuffer)
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DescriptorSetBinding {
    pub(crate) pool: vk::DescriptorPool,
    pub(crate) bindings: Vec<DescriptorBinding>,
    pub(crate) layout: vk::DescriptorSetLayout,
}

/// Wrapper over a Vulkan `VkDescriptorSet`. You don't explicitly need to use this, as the command buffer and descriptor cache can manage these
/// fully for you. See for example [`IncompleteCommandBuffer::bind_sampled_image()`](crate::IncompleteCommandBuffer::bind_sampled_image)
#[derive(Derivative)]
#[derivative(Debug, PartialEq, Eq)]
pub struct DescriptorSet {
    #[derivative(Debug = "ignore")]
    #[derivative(PartialEq = "ignore")]
    pub(crate) device: Device,
    pub(crate) pool: vk::DescriptorPool,
    pub(crate) handle: vk::DescriptorSet,
}

fn binding_image_info(binding: &DescriptorBinding) -> Vec<vk::DescriptorImageInfo> {
    binding
        .descriptors
        .iter()
        .map(|descriptor| {
            let DescriptorContents::Image(image) = descriptor else { panic!("Missing descriptor type case?") };
            vk::DescriptorImageInfo {
                sampler: image.sampler,
                image_view: unsafe { image.view.handle() },
                image_layout: image.layout,
            }
        })
        .collect()
}

fn binding_buffer_info(binding: &DescriptorBinding) -> Vec<vk::DescriptorBufferInfo> {
    binding
        .descriptors
        .iter()
        .map(|descriptor| {
            let DescriptorContents::Buffer(buffer) = descriptor else { panic!("Missing descriptor type case?") };
            vk::DescriptorBufferInfo {
                buffer: unsafe { buffer.buffer.handle() },
                offset: buffer.buffer.offset(),
                range: buffer.buffer.size(),
            }
        })
        .collect()
}

fn binding_accel_structure_info(binding: &DescriptorBinding) -> Vec<vk::AccelerationStructureKHR> {
    binding
        .descriptors
        .iter()
        .map(|descriptor| {
            let DescriptorContents::AccelerationStructure(handle) = descriptor else { panic!("Missing descriptor type case?") };
            *handle
        })
        .collect()
}

struct WriteDescriptorSet {
    pub set: vk::DescriptorSet,
    pub binding: u32,
    pub array_element: u32,
    pub count: u32,
    pub ty: vk::DescriptorType,
    pub image_info: Option<Vec<vk::DescriptorImageInfo>>,
    pub buffer_info: Option<Vec<vk::DescriptorBufferInfo>>,
    pub acceleration_structure_info: Option<Vec<vk::AccelerationStructureKHR>>,
}

impl ResourceKey for DescriptorSetBinding {
    fn persistent(&self) -> bool {
        false
    }
}

impl Resource for DescriptorSet {
    type Key = DescriptorSetBinding;
    type ExtraParams<'a> = ();
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Device, key: &Self::Key, _: Self::ExtraParams<'_>) -> Result<Self>
    where
        Self: Sized, {
        let info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            descriptor_pool: key.pool,
            descriptor_set_count: 1,
            p_set_layouts: &key.layout,
        };
        let set = unsafe { device.allocate_descriptor_sets(&info) }?
            .first()
            .cloned()
            .unwrap();
        #[cfg(feature = "log-objects")]
        trace!("Created new VkDescriptorSet {set:p}");

        let writes = key
            .bindings
            .iter()
            .map(|binding| {
                let mut write = WriteDescriptorSet {
                    set,
                    binding: binding.binding,
                    array_element: 0,
                    count: binding.descriptors.len() as u32,
                    ty: binding.ty,
                    image_info: None,
                    buffer_info: None,
                    acceleration_structure_info: None,
                };

                match binding.ty {
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                        write.image_info = Some(binding_image_info(binding));
                    }
                    vk::DescriptorType::SAMPLED_IMAGE => {
                        write.image_info = Some(binding_image_info(binding));
                    }
                    vk::DescriptorType::STORAGE_IMAGE => {
                        write.image_info = Some(binding_image_info(binding));
                    }
                    vk::DescriptorType::UNIFORM_BUFFER => {
                        write.buffer_info = Some(binding_buffer_info(binding));
                    }
                    vk::DescriptorType::STORAGE_BUFFER => {
                        write.buffer_info = Some(binding_buffer_info(binding));
                    }
                    vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => {
                        write.acceleration_structure_info =
                            Some(binding_accel_structure_info(binding));
                    }
                    _ => {
                        todo!();
                    }
                }
                write
            })
            .collect::<Vec<WriteDescriptorSet>>();

        let pnext = writes
            .iter()
            .map(|write| {
                if let Some(info) = &write.acceleration_structure_info {
                    Some(PNext::WriteDescriptorSetAccelerationStructure(
                        vk::WriteDescriptorSetAccelerationStructureKHR {
                            s_type:
                                vk::StructureType::WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                            p_next: std::ptr::null(),
                            acceleration_structure_count: info.len() as u32,
                            p_acceleration_structures: info.as_ptr(),
                        },
                    ))
                } else {
                    None
                }
            })
            .collect::<Vec<Option<PNext>>>();

        let vk_writes = writes
            .iter()
            .zip(&pnext)
            .map(|(write, p_next)| vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: p_next
                    .as_ref()
                    .map(|p_next| p_next.as_ptr())
                    .unwrap_or(std::ptr::null()),
                dst_set: write.set,
                dst_binding: write.binding,
                dst_array_element: write.array_element,
                descriptor_count: write.count,
                descriptor_type: write.ty,
                p_image_info: match &write.image_info {
                    None => std::ptr::null(),
                    Some(image) => image.as_ptr(),
                },
                p_buffer_info: match &write.buffer_info {
                    None => std::ptr::null(),
                    Some(buffer) => buffer.as_ptr(),
                },
                p_texel_buffer_view: std::ptr::null(),
            })
            .collect::<Vec<_>>();

        unsafe {
            device.update_descriptor_sets(vk_writes.as_slice(), &[]);
        }

        Ok(DescriptorSet {
            device,
            pool: key.pool,
            handle: set,
        })
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkDescriptorSet {:p}", self.handle);
        unsafe {
            self.device
                .free_descriptor_sets(self.pool, std::slice::from_ref(&self.handle))
                .unwrap();
        }
    }
}
