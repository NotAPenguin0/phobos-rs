use std::sync::Arc;
use ash::vk;

use crate::cache::{Cache, Resource};
use crate::Device;
use crate::set_layout::{DescriptorSetLayout, DescriptorSetLayoutCreateInfo};

use super::hash::*;

use anyhow::Result;

/// A fully built Vulkan pipeline layout. This is a managed resource, so it cannot be manually
/// cloned or dropped.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct PipelineLayout {
    #[derivative(Debug="ignore")]
    pub(crate) device: Arc<Device>,
    pub(crate) handle: vk::PipelineLayout,
    pub(crate) set_layouts: Vec<vk::DescriptorSetLayout>,
}

/// Defines a range of Vulkan push constants, for manually defining a pipeline layout if you cannot
/// use shader reflection for whatever reason.
#[derive(Debug, Clone, Default, Copy, PartialEq, Eq, Hash)]
pub struct PushConstantRange {
    pub stage_flags: vk::ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}

/// Define a pipeline layout, this includes all descriptor bindings and push constant ranges used by the pipeline.
#[derive(Debug, Clone, Default)]
pub struct PipelineLayoutCreateInfo {
    pub flags: vk::PipelineLayoutCreateFlags,
    pub set_layouts: Vec<DescriptorSetLayoutCreateInfo>,
    pub push_constants: Vec<PushConstantRange>,
}


impl Resource for PipelineLayout {
    type Key = PipelineLayoutCreateInfo;
    type ExtraParams<'a> = &'a mut Cache<DescriptorSetLayout>;
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Arc<Device>, key: &Self::Key, set_layout_cache: Self::ExtraParams<'_>) -> Result<Self> {
        let set_layouts = key.set_layouts.iter().map(|info|
            set_layout_cache.get_or_create(&info, ()).unwrap().handle
        ).collect::<Vec<_>>();

        let pc = key.push_constants.iter().map(|pc| pc.to_vk()).collect::<Vec<_>>();
        let info = vk::PipelineLayoutCreateInfo::builder()
            .flags(key.flags)
            .push_constant_ranges(pc.as_slice())
            .set_layouts(set_layouts.as_slice())
            .build();
        Ok(Self {
            device: device.clone(),
            handle: unsafe { device.create_pipeline_layout(&info, None)? },
            set_layouts: set_layouts.clone()
        })
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
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