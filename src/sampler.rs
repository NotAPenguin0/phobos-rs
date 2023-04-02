use anyhow::Result;
use ash::vk;

use crate::Device;

/// Represents a vulkan sampler object.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Sampler {
    #[derivative(Debug = "ignore")]
    device: Device,
    handle: vk::Sampler,
}

impl Sampler {
    /// Create a new sampler with default settings. These settings are:
    /// - `LINEAR` min/mag filters
    /// - `LINEAR` mipmap mode
    /// - `REPEAT` address mode on all axes
    /// - `0.0` mip lod bias
    /// - Anisotropic filtering off
    /// - Sampler compare op off
    /// - Min mipmap level `0`
    /// - Unbounded max mipmap level
    /// - Normalized coordinates
    pub fn default(device: Device) -> Result<Self> {
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
            .max_lod(vk::LOD_CLAMP_NONE)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .build();
        Ok(Self {
            device: device.clone(),
            handle: unsafe { device.create_sampler(&info, None)? },
        })
    }

    /// Create a new `VkSampler` object with given settings.
    pub fn new(device: Device, info: vk::SamplerCreateInfo) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            handle: unsafe { device.create_sampler(&info, None)? },
        })
    }

    pub unsafe fn handle(&self) -> vk::Sampler {
        self.handle
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.handle, None);
        }
    }
}
