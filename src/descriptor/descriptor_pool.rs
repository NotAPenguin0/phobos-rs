use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use ash::vk;
use crate::Device;

use anyhow::Result;

/// Defines how many descriptors a descriptor pool should be able to hold.
#[derive(Debug, Clone)]
pub(super) struct DescriptorPoolSize(pub(super) HashMap<vk::DescriptorType, u32>);

#[derive(Derivative)]
#[derivative(Debug)]
pub(super) struct DescriptorPool {
    #[derivative(Debug="ignore")]
    pub device: Arc<Device>,
    pub handle: vk::DescriptorPool,
    pub size: DescriptorPoolSize
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
    pub(super) fn new(device: Arc<Device>, size: DescriptorPoolSize) -> Result<Self> {
        // TODO: this max_sets value is overly pessimistic as it doesnt account for multiple
        // descriptors being held in the same descriptor set. Ideally this grows with the pool too.
        let max_sets = size.0.values().fold(0, |a, x| x + a);
        let flags = vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET;
        let pool_sizes = size.0.iter().map(|(descriptor_type, count)| {
            vk::DescriptorPoolSize {
                ty: *descriptor_type,
                descriptor_count: *count,
            }
        }).collect::<Vec<vk::DescriptorPoolSize>>();

        let info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags,
            max_sets,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
        };

        Ok(Self{
            handle: unsafe { device.create_descriptor_pool(&info, None)? },
            device,
            size,
        })
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe { self.device.destroy_descriptor_pool(self.handle, None); }
    }
}


impl Display for DescriptorPoolSize {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut result = writeln!(f, "DescriptorPoolSize (");
        for (ty, size) in &self.0 {
            result = result.and_then(|_| writeln!(f, "{:?} => {}", ty, size))
        }
        result
    }
}