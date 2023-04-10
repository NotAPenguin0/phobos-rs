use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use anyhow::Result;
use ash::vk;

use crate::Device;
use crate::util::cache::{Resource, ResourceKey};

/// Shader resource object. This is managed by the pipeline cache internally.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Shader {
    #[derivative(Debug = "ignore")]
    device: Device,
    handle: vk::ShaderModule,
}

impl Shader {
    /// Get unsafe access to the underlying `VkShaderModule` object.
    /// # Safety
    /// Any vulkan calls that mutate the shader module may put the system in an undefined state.
    pub unsafe fn handle(&self) -> vk::ShaderModule {
        self.handle
    }
}

/// Info required to create a shader. Use [`ShaderCreateInfo::from_spirv`] to construct this.
#[derive(Debug, Clone)]
pub struct ShaderCreateInfo {
    stage: vk::ShaderStageFlags,
    code: Vec<u32>,
    code_hash: u64,
    pub(crate) persistent: bool,
}

impl ShaderCreateInfo {
    /// Get the shader stage of this shader
    pub fn stage(&self) -> vk::ShaderStageFlags {
        self.stage
    }

    /// Get the SPIR-V bytecode of this shader
    pub fn code(&self) -> &[u32] {
        self.code.as_slice()
    }

    /// Get the hash of this shader's SPIR-V bytecode.
    pub fn code_hash(&self) -> u64 {
        self.code_hash
    }
}

impl ResourceKey for ShaderCreateInfo {
    /// Whether this shader is persistent.
    fn persistent(&self) -> bool {
        self.persistent
    }
}

impl Resource for Shader {
    type Key = ShaderCreateInfo;
    type ExtraParams<'a> = ();
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Device, key: &Self::Key, _: Self::ExtraParams<'_>) -> Result<Self> {
        let info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            code_size: key.code.len() * 4, // code_size is in bytes, but each element of `code` is 4 bytes.
            p_code: key.code.as_ptr(),
        };

        let handle = unsafe { device.create_shader_module(&info, None)? };
        #[cfg(feature = "log-objects")]
        trace!("Created new VkShaderModule {handle:p}");

        Ok(Self {
            device: device.clone(),
            handle,
        })
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkShaderModule {:p}", self.handle);
        unsafe {
            self.device.destroy_shader_module(self.handle, None);
        }
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
            persistent: false,
        }
    }
}
