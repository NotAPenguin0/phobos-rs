use std::sync::Arc;

use anyhow::Result;
use ash::vk;

use crate::Device;
use crate::util::cache::{Resource, ResourceKey};

/// A fully built Vulkan descriptor set layout. This is a managed resource, so it cannot be manually
/// created or dropped.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct DescriptorSetLayout {
    #[derivative(Debug = "ignore")]
    device: Arc<Device>,
    handle: vk::DescriptorSetLayout,
}

impl DescriptorSetLayout {
    pub unsafe fn handle(&self) -> vk::DescriptorSetLayout {
        self.handle
    }
}

/// Describes a descriptor set layout.
/// Generally you don't need to construct this manually, as shader reflection can infer all
/// information necessary.
#[derive(Debug, Clone, Default)]
pub struct DescriptorSetLayoutCreateInfo {
    pub bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl ResourceKey for DescriptorSetLayoutCreateInfo {
    fn persistent(&self) -> bool {
        false
    }
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
            handle: unsafe { device.create_descriptor_set_layout(&info, None)? },
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
