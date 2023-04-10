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
    device: Device,
    handle: vk::DescriptorSetLayout,
}

impl DescriptorSetLayout {
    /// Get unsafe access to the underlying `VkDescriptorSetLayout` object.
    /// # Safety
    /// Any vulkan calls that mutate the descriptor set layout may put the system in an undefined state.
    pub unsafe fn handle(&self) -> vk::DescriptorSetLayout {
        self.handle
    }
}

/// Describes a descriptor set layout.
/// Generally you don't need to construct this manually, as shader reflection can infer all
/// information necessary.
#[derive(Debug, Clone, Default)]
pub struct DescriptorSetLayoutCreateInfo {
    /// Descriptor set bindings for this set layout
    pub bindings: Vec<vk::DescriptorSetLayoutBinding>,
    /// Whether this descriptor set layout is persistent. Should only be true if the pipeline layout
    /// this belongs to is also persistent.
    pub persistent: bool,
}

impl ResourceKey for DescriptorSetLayoutCreateInfo {
    /// Whether this descriptor set layout is persistent.
    fn persistent(&self) -> bool {
        self.persistent
    }
}

impl Resource for DescriptorSetLayout {
    type Key = DescriptorSetLayoutCreateInfo;
    type ExtraParams<'a> = ();
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Device, key: &Self::Key, _: Self::ExtraParams<'_>) -> Result<Self> {
        let info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(key.bindings.as_slice())
            .build();
        let handle = unsafe { device.create_descriptor_set_layout(&info, None)? };
        #[cfg(feature = "log-objects")]
        trace!("Created new VkDescriptorSetLayout {handle:p}");
        Ok(Self {
            device: device.clone(),
            handle,
        })
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkDescriptorSetLayout {:p}", self.handle);
        unsafe {
            self.device.destroy_descriptor_set_layout(self.handle, None);
        }
    }
}
