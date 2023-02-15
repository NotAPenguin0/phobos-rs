use std::sync::Arc;
use ash::vk;
use crate::{Device, Error};
use anyhow::Result;

/// Represents a vulkan sampler object.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Sampler {
    #[derivative(Debug="ignore")]
    device: Arc<Device>,
    pub handle: vk::Sampler,
}

impl Sampler {
    /// Create a new sampler with default settings.
    pub fn default(device: Arc<Device>) -> Result<Self> {
        let info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .mip_lod_bias(0.0)
            .anisotropy_enable(false)
            .max_anisotropy(0.0)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .min_lod(0.0)
            .max_lod(64.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .build();
        Ok(Self {
            device: device.clone(),
            handle: unsafe { device.create_sampler(&info, None)? },
        })
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { self.device.destroy_sampler(self.handle, None); }
    }
}