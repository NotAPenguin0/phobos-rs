use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use ash::vk;

use crate::{ComputePipelineCreateInfo, Device, Error, PipelineCreateInfo, ShaderCreateInfo};
use crate::core::device::ExtensionID;
use crate::pipeline::{ComputePipeline, Pipeline, PipelineType};
use crate::pipeline::create_info::PipelineRenderingInfo;
use crate::pipeline::pipeline_layout::PipelineLayout;
use crate::pipeline::set_layout::DescriptorSetLayout;
use crate::pipeline::shader::Shader;
use crate::util::cache::{Cache, Resource, ResourceKey};

use super::shader_reflection::{build_pipeline_layout, reflect_shaders, ReflectionInfo};

#[derive(Debug)]
struct PipelineEntry<P>
where
    P: std::fmt::Debug, {
    pub info: P,
    #[cfg(feature = "shader-reflection")]
    pub reflection: ReflectionInfo,
}

/// The main pipeline cache struct. This stores all named pipelines and shaders.
/// To create a pipeline you should obtain a pipeline create info, and then register it using
/// [`PipelineCache::create_named_pipeline`].
/// # Example usage
/// ```
/// use phobos::prelude::*;
/// let cache = PipelineCache::new(device.clone())?;
/// let pci = PipelineBuilder::new("my_pipeline")
///     // ... options for pipeline creation
///     .build();
/// cache.lock().or_else(|_| Err(anyhow::Error::from(Error::PoisonError)))?.create_named_pipeline(pci)?;
/// ```
#[derive(Debug)]
pub struct PipelineCache {
    shaders: Cache<Shader>,
    set_layouts: Cache<DescriptorSetLayout>,
    pipeline_layouts: Cache<PipelineLayout>,
    pipelines: Cache<Pipeline>,
    compute_pipelines: Cache<ComputePipeline>,
    pipeline_infos: HashMap<String, PipelineEntry<PipelineCreateInfo>>,
    compute_pipeline_infos: HashMap<String, PipelineEntry<ComputePipelineCreateInfo>>,
}

// SAFETY: This is not automatically derived because of the pNext pointers inside the pipeline create infos.
// Since we only temporarily set those when pipelines are created, this is safe to do. No pointers to pNext
// structures are kept around.
unsafe impl Send for PipelineCache {}

macro_rules! require_extension {
    ($pci:ident, $device:ident, $state:expr, $ext:expr) => {
        if $pci.dynamic_states.contains(&$state) && !$device.is_extension_enabled($ext) {
            error!(
                "Pipeline {} requested dynamic state {:?}, but corresponding extension {:?} is not enabled. Maybe it is unsupported on the current device?",
                $pci.name, $state, $ext
            );
        }
    };
}

/// Check if dynamic states are supported by the enabled extension set
fn verify_valid_dynamic_states(device: &Arc<Device>, pci: &PipelineCreateInfo) {
    require_extension!(
        pci,
        device,
        vk::DynamicState::POLYGON_MODE_EXT,
        ExtensionID::ExtendedDynamicState3
    );
}

impl ResourceKey for PipelineCreateInfo {
    fn persistent(&self) -> bool {
        false
    }
}

impl Resource for Pipeline {
    type Key = PipelineCreateInfo;
    type ExtraParams<'a> = (
        &'a mut Cache<Shader>,
        &'a mut Cache<PipelineLayout>,
        &'a mut Cache<DescriptorSetLayout>,
    );
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Arc<Device>, info: &Self::Key, params: Self::ExtraParams<'_>) -> Result<Self> {
        let (shaders, pipeline_layouts, set_layouts) = params;
        let layout = pipeline_layouts.get_or_create(&info.layout, set_layouts)?;
        let mut pci = info.to_vk(unsafe { layout.handle() });

        verify_valid_dynamic_states(&device, &info);

        // Set shader create info
        let entry = CString::new("main")?;
        let shader_info: Vec<_> = info
            .shaders
            .iter()
            .map(|shader| -> vk::PipelineShaderStageCreateInfo {
                vk::PipelineShaderStageCreateInfo::builder()
                    .name(&entry)
                    .stage(shader.stage())
                    .module(unsafe { shaders.get_or_create(shader, ()).unwrap().handle() })
                    .build()
            })
            .collect();
        pci.stage_count = shader_info.len() as u32;
        pci.p_stages = shader_info.as_ptr();

        unsafe {
            Ok(Self {
                device: device.clone(),
                handle: device
                    .create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pci), None)
                    .or_else(|(_, e)| Err(anyhow::Error::from(Error::VkError(e))))?
                    .first()
                    .cloned()
                    .unwrap(),
                layout: layout.handle(),
                set_layouts: layout.set_layouts().to_vec(),
            })
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.handle, None);
        }
    }
}

impl ResourceKey for ComputePipelineCreateInfo {
    fn persistent(&self) -> bool {
        self.persistent
    }
}

impl Resource for ComputePipeline {
    type Key = ComputePipelineCreateInfo;
    type ExtraParams<'a> = (
        &'a mut Cache<Shader>,
        &'a mut Cache<PipelineLayout>,
        &'a mut Cache<DescriptorSetLayout>,
    );
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Arc<Device>, info: &Self::Key, params: Self::ExtraParams<'_>) -> Result<Self>
    where
        Self: Sized, {
        let (shaders, pipeline_layouts, set_layouts) = params;
        let layout = pipeline_layouts.get_or_create(&info.layout, set_layouts)?;
        let mut pci = info.to_vk(unsafe { layout.handle() });

        // Set shader create info
        let entry = CString::new("main")?;
        let shader = match &info.shader {
            None => Err(Error::Uncategorized("Compute pipeline lacks shader")),
            Some(shader) => Ok(vk::PipelineShaderStageCreateInfo::builder()
                .name(&entry)
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(unsafe { shaders.get_or_create(shader, ()).unwrap().handle() })
                .build()),
        }?;

        pci.stage = shader;

        unsafe {
            Ok(Self {
                device: device.clone(),
                handle: device
                    .create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pci), None)
                    .or_else(|(_, e)| Err(anyhow::Error::from(Error::VkError(e))))?
                    .first()
                    .cloned()
                    .unwrap(),
                layout: layout.handle(),
                set_layouts: layout.set_layouts().to_vec(),
            })
        }
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.handle, None);
        }
    }
}

// TODO: Maybe incorporate the vulkan pipeline cache api to improve startup times?

impl PipelineCache {
    /// Create a new empty pipeline cache.
    pub fn new(device: Arc<Device>) -> Result<Arc<Mutex<Self>>> {
        Ok(Arc::new(Mutex::new(Self {
            shaders: Cache::new(device.clone()),
            set_layouts: Cache::new(device.clone()),
            pipeline_layouts: Cache::new(device.clone()),
            pipelines: Cache::new(device.clone()),
            compute_pipelines: Cache::new(device.clone()),
            pipeline_infos: Default::default(),
            compute_pipeline_infos: Default::default(),
        })))
    }

    /// Create and register a new pipeline into the cache.
    #[cfg(feature = "shader-reflection")]
    pub fn create_named_pipeline(&mut self, mut info: PipelineCreateInfo) -> Result<()> {
        let refl = reflect_shaders(info.shaders.as_slice())?;
        // Using reflection, we can allow omitting the pipeline layout field.
        info.layout = build_pipeline_layout(&refl);
        let name = info.name.clone();
        self.pipeline_infos.insert(
            name.clone(),
            PipelineEntry {
                info,
                reflection: refl,
            },
        );
        self.pipeline_infos.get_mut(&name).unwrap().info.build_inner();
        Ok(())
    }

    #[cfg(not(feature = "shader-reflection"))]
    pub fn create_named_pipeline(&mut self, mut info: PipelineCreateInfo) -> Result<()> {
        info.build_inner();
        let name = info.name.clone();
        self.pipeline_infos.insert(
            name.clone(),
            PipelineEntry {
                info,
            },
        );
        self.pipeline_infos.get_mut(&name).unwrap().info.build_inner();
        Ok(())
    }

    /// Create and register a new compute pipeline into the cache
    #[cfg(feature = "shader-reflection")]
    pub fn create_named_compute_pipeline(&mut self, mut info: ComputePipelineCreateInfo) -> Result<()> {
        let refl = match &info.shader {
            None => { reflect_shaders(&[])? }
            Some(info) => { reflect_shaders(std::slice::from_ref(info))? }
        };
        // Using reflection, we can allow omitting the pipeline layout field.
        info.layout = build_pipeline_layout(&refl);
        // If this is persistent, then also make the pipeline and descriptor set layouts persistent
        if info.persistent {
            info.layout.persistent = true;
            info.layout.set_layouts.iter_mut().for_each(|set_layout| {
                set_layout.persistent = true;
            });
            match &mut info.shader {
                None => {}
                Some(shader) => {
                    shader.persistent = true;
                }
            }
        }
        let name = info.name.clone();
        self.compute_pipeline_infos.insert(
            name,
            PipelineEntry {
                info,
                reflection: refl,
            },
        );
        Ok(())
    }

    #[cfg(not(feature = "shader-reflection"))]
    pub fn create_named_compute_pipeline(&mut self, mut info: ComputePipelineCreateInfo) -> Result<()> {
        let name = info.name.clone();
        self.compute_pipeline_infos.insert(
            name,
            PipelineEntry {
                info,
                reflection: refl,
            },
        );
        Ok(())
    }

    /// Get reflection info for a previously registered pipeline.
    /// # Errors
    /// Fails if the pipeline was not found in the cache.
    #[cfg(feature = "shader-reflection")]
    pub fn reflection_info(&self, name: &str) -> Result<&ReflectionInfo> {
        Ok(&self.pipeline_infos.get(name).unwrap().reflection)
    }

    /// Get the pipeline create info associated with a pipeline
    /// # Errors
    /// Returns None if the pipeline was not found in the cache.
    pub fn pipeline_info(&self, name: &str) -> Option<&PipelineCreateInfo> {
        self.pipeline_infos.get(name).map(|entry| &entry.info)
    }

    /// Get the pipeline create info associated with a compute pipeline
    /// # Errors
    /// Returns None if the pipeline was not found in the cache.
    pub fn compute_pipeline_info(&self, name: &str) -> Option<&ComputePipelineCreateInfo> {
        self.compute_pipeline_infos.get(name).map(|entry| &entry.info)
    }

    /// Returns the pipeline type of a pipeline, or None if the pipeline does not exist.
    pub fn pipeline_type(&self, name: &str) -> Option<PipelineType> {
        if self.pipeline_infos.contains_key(name) {
            Some(PipelineType::Graphics)
        } else if self.compute_pipeline_infos.contains_key(name) {
            Some(PipelineType::Compute)
        } else {
            None
        }
    }

    /// Obtain a pipeline from the cache.
    /// # Errors
    /// - This function can fail if the requested pipeline does not exist in the cache
    /// - This function can fail if allocating the pipeline fails.
    pub(crate) fn get_pipeline(&mut self, name: &str, rendering_info: &PipelineRenderingInfo) -> Result<&Pipeline> {
        let entry = self.pipeline_infos.get_mut(name);
        let Some(entry) = entry else { return Err(anyhow::Error::from(Error::PipelineNotFound(name.to_string()))); };
        entry.info.rendering_info = rendering_info.clone();
        entry.info.build_rendering_state();
        // Also put in queries for descriptor set layouts and pipeline layout to make sure they are not destroyed.
        for layout in &entry.info.layout.set_layouts {
            self.set_layouts.get_or_create(layout, ())?;
        }
        self.pipeline_layouts.get_or_create(&entry.info.layout, &mut self.set_layouts)?;
        self.pipelines.get_or_create(
            &entry.info,
            (&mut self.shaders, &mut self.pipeline_layouts, &mut self.set_layouts),
        )
    }

    /// Obtain a compute pipeline from the cache.
    /// # Errors
    /// - This function can fail if the requested pipeline does not exist in the cache
    /// - This function can fail if allocating the pipeline fails.
    pub(crate) fn get_compute_pipeline(&mut self, name: &str) -> Result<&ComputePipeline> {
        let entry = self.compute_pipeline_infos.get_mut(name);
        let Some(entry) = entry else { return Err(anyhow::Error::from(Error::PipelineNotFound(name.to_string()))); };
        // Also put in queries for descriptor set layouts and pipeline layout to make sure they are not destroyed.
        for layout in &entry.info.layout.set_layouts {
            self.set_layouts.get_or_create(layout, ())?;
        }
        self.pipeline_layouts.get_or_create(&entry.info.layout, &mut self.set_layouts)?;
        self.compute_pipelines.get_or_create(
            &entry.info,
            (&mut self.shaders, &mut self.pipeline_layouts, &mut self.set_layouts),
        )
    }

    /// Advance cache resource time to live so resources that have not been used in a while can be cleaned up
    pub fn next_frame(&mut self) {
        self.pipelines.next_frame();
        self.pipeline_layouts.next_frame();
        self.shaders.next_frame();
        self.set_layouts.next_frame();
    }
}
