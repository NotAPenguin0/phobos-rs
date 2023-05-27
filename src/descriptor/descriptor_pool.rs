//! A descriptor pool automatically grows to allocate descriptor sets from. It is completely managed for you so you dont need to create one manually.

use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use anyhow::Result;
use ash::vk;

use crate::core::device::ExtensionID;
use crate::Device;

/// Defines how many descriptors a descriptor pool should be able to hold.
#[derive(Debug, Clone)]
pub(super) struct DescriptorPoolSize(pub(super) HashMap<vk::DescriptorType, u32>);

/// Memory pool for descriptor sets
#[derive(Derivative)]
#[derivative(Debug)]
pub(super) struct DescriptorPool {
    #[derivative(Debug = "ignore")]
    device: Device,
    handle: vk::DescriptorPool,
    size: DescriptorPoolSize,
}

impl DescriptorPoolSize {
    /// Create a new descriptor pool size description
    pub fn new(device: &Device, min_capacity: u32) -> Self {
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
        if device.is_extension_enabled(ExtensionID::AccelerationStructure) {
            sizes.insert(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, min_capacity);
        }
        Self(sizes)
    }
}

impl DescriptorPool {
    /// Create a new descriptor pool
    pub(super) fn new(device: Device, size: DescriptorPoolSize) -> Result<Self> {
        // TODO: this max_sets value is overly pessimistic as it doesnt account for multiple
        // descriptors being held in the same descriptor set. Ideally this grows with the pool too.
        let max_sets = size.0.values().fold(0, |a, x| x + a);
        let flags = vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET;
        let pool_sizes = size
            .0
            .iter()
            .map(|(descriptor_type, count)| vk::DescriptorPoolSize {
                ty: *descriptor_type,
                descriptor_count: *count,
            })
            .collect::<Vec<vk::DescriptorPoolSize>>();

        let info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags,
            max_sets,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
        };

        let handle = unsafe { device.create_descriptor_pool(&info, None)? };
        #[cfg(feature = "log-objects")]
        trace!("Created new VkDescriptorPool {handle:p}");

        Ok(Self {
            handle,
            device,
            size,
        })
    }

    /// Get the raw Vulkan handle of this descriptor pool
    pub(super) unsafe fn handle(&self) -> vk::DescriptorPool {
        self.handle
    }

    /// Get the current size of the descriptor pool.
    pub fn size(&self) -> &DescriptorPoolSize {
        &self.size
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkDescriptorPool {:p}", self.handle);
        unsafe {
            self.device.destroy_descriptor_pool(self.handle, None);
        }
    }
}

impl Display for DescriptorPoolSize {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut result = writeln!(f, "DescriptorPoolSize (");
        for (ty, size) in &self.0 {
            result = result.and_then(|_| writeln!(f, "{ty:?} => {size}"))
        }
        result
    }
}
