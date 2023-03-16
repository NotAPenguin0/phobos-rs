use std::sync::Arc;
use ash::vk;

use crate::Device;

use anyhow::Result;
use crate::util::cache::Resource;

use super::hash::*;

/// A fully built Vulkan descriptor set layout. This is a managed resource, so it cannot be manually
/// cloned or dropped.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct DescriptorSetLayout {
    #[derivative(Debug="ignore")]
    pub(crate) device: Arc<Device>,
    pub(crate) handle: vk::DescriptorSetLayout
}

/// Describes a descriptor set layout.
/// Generally you don't need to construct this manually, as shader reflection can infer all
/// information necessary.
#[derive(Debug, Clone, Default)]
pub struct DescriptorSetLayoutCreateInfo {
    pub bindings: Vec<vk::DescriptorSetLayoutBinding>
}


impl Resource for DescriptorSetLayout {
    type Key = DescriptorSetLayoutCreateInfo;
    type ExtraParams<'a> = ();
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Arc<Device>, key: &Self::Key, _: Self::ExtraParams<'_>) -> Result<Self> {
        let info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(key.bindings.as_slice())
            .build();
        Ok(Self {
            device: device.clone(),
            handle: unsafe { device.create_descriptor_set_layout(&info, None)? }
        })
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_set_layout(self.handle, None);
        }
    }
}
