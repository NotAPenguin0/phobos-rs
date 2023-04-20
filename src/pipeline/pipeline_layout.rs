//! Wrapper structs around `VkPipelineLayout` objects.

use anyhow::Result;
use ash::vk;

use crate::Device;
use crate::pipeline::set_layout::{DescriptorSetLayout, DescriptorSetLayoutCreateInfo};
use crate::util::cache::{Cache, Resource, ResourceKey};

/// A fully built Vulkan pipeline layout. This is a managed resource, so it cannot be manually
/// created or dropped.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct PipelineLayout {
    #[derivative(Debug = "ignore")]
    device: Device,
    handle: vk::PipelineLayout,
    set_layouts: Vec<vk::DescriptorSetLayout>,
}

/// Defines a range of Vulkan push constants, for manually defining a pipeline layout if you cannot
/// use shader reflection for whatever reason.
#[derive(Debug, Clone, Default, Copy, PartialEq, Eq, Hash)]
pub struct PushConstantRange {
    /// Shader stages where this push constant range is used
    pub stage_flags: vk::ShaderStageFlags,
    /// Offset into the global push constant block of this range
    pub offset: u32,
    /// Size of this push constant range
    pub size: u32,
}

/// Define a pipeline layout, this includes all descriptor bindings and push constant ranges used by the pipeline.
/// # Shader reflection
/// Using the `shader-reflection` feature allows you to completely omit constructing this manually. In this case,
/// shader reflection will be used to derive them automatically.
#[derive(Debug, Clone, Default)]
pub struct PipelineLayoutCreateInfo {
    /// Pipeline layout flags
    pub flags: vk::PipelineLayoutCreateFlags,
    /// Descriptor set layouts for this pipeline layout.
    pub set_layouts: Vec<DescriptorSetLayoutCreateInfo>,
    /// Push constant ranges used in this pipeline
    pub push_constants: Vec<PushConstantRange>,
    /// Whether this pipeline layout is persistent, e.g. whether it should be kept alive forever
    /// by the cache. Use this with caution, as it can cause large memory spikes for frequently changing
    /// pipeline layouts.
    pub persistent: bool,
}

impl PipelineLayout {
    /// Get unsafe access to the internal `VkPipelineLayout`.
    /// # Safety
    /// Any vulkan calls that mutate this pipeline layout may put the system in an undefined state.
    pub unsafe fn handle(&self) -> vk::PipelineLayout {
        self.handle
    }

    /// Get the descriptor set layouts of this pipeline layout.
    pub fn set_layouts(&self) -> &[vk::DescriptorSetLayout] {
        self.set_layouts.as_slice()
    }
}

impl ResourceKey for PipelineLayoutCreateInfo {
    /// Whether this pipeline layout is persistent or not.
    fn persistent(&self) -> bool {
        self.persistent
    }
}

impl Resource for PipelineLayout {
    type Key = PipelineLayoutCreateInfo;
    type ExtraParams<'a> = &'a mut Cache<DescriptorSetLayout>;
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Device, key: &Self::Key, set_layout_cache: Self::ExtraParams<'_>) -> Result<Self> {
        let set_layouts = key
            .set_layouts
            .iter()
            .map(|info| unsafe { set_layout_cache.get_or_create(info, ()).unwrap().handle() })
            .collect::<Vec<_>>();

        let pc = key.push_constants.iter().map(|pc| pc.to_vk()).collect::<Vec<_>>();
        let info = vk::PipelineLayoutCreateInfo::builder()
            .flags(key.flags)
            .push_constant_ranges(pc.as_slice())
            .set_layouts(set_layouts.as_slice())
            .build();

        let handle = unsafe { device.create_pipeline_layout(&info, None)? };

        #[cfg(feature = "log-objects")]
        trace!("Created new VkPipelineLayout {handle:p}");

        Ok(Self {
            device: device.clone(),
            handle,
            set_layouts,
        })
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkPipelineLayout {:p}", self.handle);
        unsafe {
            self.device.destroy_pipeline_layout(self.handle, None);
        }
    }
}

impl PushConstantRange {
    pub fn to_vk(&self) -> vk::PushConstantRange {
        vk::PushConstantRange {
            stage_flags: self.stage_flags,
            offset: self.offset,
            size: self.size,
        }
    }
}
