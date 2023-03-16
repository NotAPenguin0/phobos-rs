use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use ash::vk;

use crate::Device;

use anyhow::Result;
use crate::util::cache::Resource;

/// Shader resource object. This is managed by the pipeline cache internally.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Shader {
    #[derivative(Debug="ignore")]
    pub(crate) device: Arc<Device>,
    pub(crate) handle: vk::ShaderModule
}

#[derive(Debug, Clone)]
pub struct ShaderCreateInfo {
    pub stage: vk::ShaderStageFlags,
    pub(crate) code: Vec<u32>,
    pub(crate) code_hash: u64
}


impl Resource for Shader {
    type Key = ShaderCreateInfo;
    type ExtraParams<'a> = ();
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Arc<Device>, key: &Self::Key, _: Self::ExtraParams<'_>) -> Result<Self> {
        let info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            code_size: key.code.len() * 4, // code_size is in bytes, but each element of `code` is 4 bytes.
            p_code: key.code.as_ptr(),
        };

        Ok(Self {
            device: device.clone(),
            handle: unsafe { device.create_shader_module(&info, None)? },
        })
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe { self.device.destroy_shader_module(self.handle, None); }
    }
}

impl ShaderCreateInfo {
    /// Load in a spirv binary into a shader create info structure.
    pub fn from_spirv(stage: vk::ShaderStageFlags, code: Vec<u32>) -> Self {
        let mut hasher = DefaultHasher::new();
        code.hash(&mut hasher);
        Self {
            stage,
            code,
            code_hash: hasher.finish(),
        }
    }
}