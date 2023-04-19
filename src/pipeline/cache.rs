use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use ash::vk;

use crate::{Allocator, ComputePipelineCreateInfo, DefaultAllocator, Device, Error, PipelineCreateInfo};
use crate::core::device::ExtensionID;
use crate::pipeline::{ComputePipeline, Pipeline, PipelineType, RayTracingPipeline};
use crate::pipeline::create_info::PipelineRenderingInfo;
use crate::pipeline::pipeline_layout::PipelineLayout;
use crate::pipeline::raytracing::{RayTracingPipelineCreateInfo, ShaderBindingTable, ShaderGroup};
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
    #[allow(dead_code)]
    pub reflection: ReflectionInfo,
}

#[derive(Debug)]
struct PipelineCacheInner<A: Allocator> {
    allocator: A,
    shaders: Cache<Shader>,
    set_layouts: Cache<DescriptorSetLayout>,
    pipeline_layouts: Cache<PipelineLayout>,
    pipelines: Cache<Pipeline>,
    compute_pipelines: Cache<ComputePipeline>,
    raytracing_pipelines: Cache<RayTracingPipeline<A>>,
    pipeline_infos: HashMap<String, PipelineEntry<PipelineCreateInfo>>,
    compute_pipeline_infos: HashMap<String, PipelineEntry<ComputePipelineCreateInfo>>,
    raytracing_pipeline_infos: HashMap<String, PipelineEntry<RayTracingPipelineCreateInfo>>,
}

/// The main pipeline cache struct. This stores all named pipelines and shaders.
/// To create a pipeline you should obtain a pipeline create info, and then register it using
/// [`PipelineCache::create_named_pipeline`].
///
/// This struct is `Clone`, `Send` and `Sync`.
/// # Example usage
/// ```
/// use phobos::prelude::*;
/// let mut cache = PipelineCache::new(device.clone(), allocator.clone())?;
/// let pci = PipelineBuilder::new("my_pipeline")
///     // ... options for pipeline creation
///     .build();
/// cache.create_named_pipeline(pci)?;
/// ```
#[derive(Debug, Clone)]
pub struct PipelineCache<A: Allocator = DefaultAllocator> {
    inner: Arc<RwLock<PipelineCacheInner<A>>>,
}

// SAFETY: Inner state is wrapped in an Arc<RwLock<T>>, and all pointers inside point to
// internal data
unsafe impl<A: Allocator> Send for PipelineCache<A> {}

// SAFETY: Inner state is wrapped in an Arc<RwLock<T>>, and all pointers inside point to
// internal data
unsafe impl<A: Allocator> Sync for PipelineCache<A> {}

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
fn verify_valid_dynamic_states(device: &Device, pci: &PipelineCreateInfo) {
    require_extension!(
        pci,
        device,
        vk::DynamicState::POLYGON_MODE_EXT,
        ExtensionID::ExtendedDynamicState3
    );
}

impl ResourceKey for PipelineCreateInfo {
    /// Whether this resource is persistent.
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

    fn create(device: Device, info: &Self::Key, params: Self::ExtraParams<'_>) -> Result<Self> {
        let (shaders, pipeline_layouts, set_layouts) = params;
        let layout = pipeline_layouts.get_or_create(&info.layout, set_layouts)?;
        let mut pci = info.to_vk(unsafe { layout.handle() });

        verify_valid_dynamic_states(&device, info);

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

        let handle = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pci), None)
                .map_err(|(_, e)| Error::VkError(e))?
                .first()
                .cloned()
                .unwrap()
        };

        #[cfg(feature = "log-objects")]
        trace!("Created new VkPipeline (graphics) {handle:p}");

        unsafe {
            Ok(Self {
                device: device.clone(),
                handle,
                layout: layout.handle(),
                set_layouts: layout.set_layouts().to_vec(),
            })
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkPipeline (graphics) {:p}", self.handle);
        unsafe {
            self.device.destroy_pipeline(self.handle, None);
        }
    }
}

impl ResourceKey for ComputePipelineCreateInfo {
    /// Whether this pipeline is persistent.
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

    fn create(device: Device, info: &Self::Key, params: Self::ExtraParams<'_>) -> Result<Self>
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

        let handle = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pci), None)
                .map_err(|(_, e)| Error::VkError(e))?
                .first()
                .cloned()
                .unwrap()
        };

        #[cfg(feature = "log-objects")]
        trace!("Created new VkPipeline (compute) {handle:p}");

        unsafe {
            Ok(Self {
                device: device.clone(),
                handle,
                layout: layout.handle(),
                set_layouts: layout.set_layouts().to_vec(),
            })
        }
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            #[cfg(feature = "log-objects")]
            trace!("Destroying VkPipeline (compute) {:p}", self.handle);
            self.device.destroy_pipeline(self.handle, None);
        }
    }
}

impl ResourceKey for RayTracingPipelineCreateInfo {
    fn persistent(&self) -> bool {
        false
    }
}

impl<A: Allocator> Resource for RayTracingPipeline<A> {
    type Key = RayTracingPipelineCreateInfo;
    type ExtraParams<'a> = (
        A,
        &'a mut Cache<Shader>,
        &'a mut Cache<PipelineLayout>,
        &'a mut Cache<DescriptorSetLayout>,
    );
    const MAX_TIME_TO_LIVE: u32 = 8;

    fn create(device: Device, info: &Self::Key, params: Self::ExtraParams<'_>) -> Result<Self>
        where
            Self: Sized, {
        device.require_extension(ExtensionID::RayTracingPipeline)?;
        let (alloc, shaders, pipeline_layouts, set_layouts) = params;
        let layout = pipeline_layouts.get_or_create(&info.layout, set_layouts)?;
        let mut pci = info.to_vk(unsafe { layout.handle() });

        let entry = CString::new("main")?;
        let mut shader_indices = HashMap::new();
        let shader_info: Vec<_> = info
            .shaders
            .iter()
            .enumerate()
            .map(|(idx, shader)| -> vk::PipelineShaderStageCreateInfo {
                shader_indices.insert(shader.code_hash(), idx as u32);
                vk::PipelineShaderStageCreateInfo::builder()
                    .name(&entry)
                    .stage(shader.stage())
                    .module(unsafe { shaders.get_or_create(shader, ()).unwrap().handle() })
                    .build()
            })
            .collect();
        pci.stage_count = shader_info.len() as u32;
        pci.p_stages = shader_info.as_ptr();

        let groups = info
            .shader_groups
            .iter()
            .map(|group| {
                let (general, closest_hit, any_hit, intersection, ty) = match group {
                    ShaderGroup::RayGeneration {
                        shader,
                    } => (
                        shader.index,
                        vk::SHADER_UNUSED_KHR,
                        vk::SHADER_UNUSED_KHR,
                        vk::SHADER_UNUSED_KHR,
                        vk::RayTracingShaderGroupTypeKHR::GENERAL,
                    ),
                    ShaderGroup::RayMiss {
                        shader,
                    } => (
                        shader.index,
                        vk::SHADER_UNUSED_KHR,
                        vk::SHADER_UNUSED_KHR,
                        vk::SHADER_UNUSED_KHR,
                        vk::RayTracingShaderGroupTypeKHR::GENERAL,
                    ),
                    ShaderGroup::RayHit {
                        closest_hit,
                        any_hit,
                    } => (
                        vk::SHADER_UNUSED_KHR,
                        closest_hit.map(|sh| sh.index).unwrap_or(vk::SHADER_UNUSED_KHR),
                        any_hit.map(|sh| sh.index).unwrap_or(vk::SHADER_UNUSED_KHR),
                        vk::SHADER_UNUSED_KHR,
                        vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP,
                    ),
                    ShaderGroup::Callable {
                        shader,
                    } => (
                        shader.index,
                        vk::SHADER_UNUSED_KHR,
                        vk::SHADER_UNUSED_KHR,
                        vk::SHADER_UNUSED_KHR,
                        vk::RayTracingShaderGroupTypeKHR::GENERAL,
                    ),
                };

                vk::RayTracingShaderGroupCreateInfoKHR {
                    s_type: vk::StructureType::RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                    p_next: std::ptr::null(),
                    ty,
                    general_shader: general,
                    closest_hit_shader: closest_hit,
                    any_hit_shader: any_hit,
                    intersection_shader: intersection,
                    p_shader_group_capture_replay_handle: std::ptr::null(),
                }
            })
            .collect::<Vec<_>>();
        pci.group_count = groups.len() as u32;
        pci.p_groups = groups.as_ptr();

        let fns = device.raytracing_pipeline().unwrap();
        let handle = unsafe {
            fns.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                std::slice::from_ref(&pci),
                None,
            )?
                .first()
                .cloned()
                .unwrap()
        };

        #[cfg(feature = "log-objects")]
        trace!("Created new VkPipeline (raytracing) {handle:p}");

        let sbt = ShaderBindingTable::new(device.clone(), alloc, handle, info)?;

        Ok(Self {
            device,
            handle,
            layout: unsafe { layout.handle() },
            set_layouts: layout.set_layouts().to_vec(),
            shader_binding_table: sbt,
        })
    }
}

impl<A: Allocator> Drop for RayTracingPipeline<A> {
    fn drop(&mut self) {
        unsafe {
            #[cfg(feature = "log-objects")]
            trace!("Destroying VkPipeline (raytracing) {:p}", self.handle);
            self.device.destroy_pipeline(self.handle, None);
        }
    }
}

impl<A: Allocator> PipelineCacheInner<A> {
    pub(crate) fn get_pipeline(&mut self, name: &str, rendering_info: PipelineRenderingInfo) -> Result<&Pipeline> {
        let entry = self.pipeline_infos.get_mut(name);
        let Some(entry) = entry else { return Err(anyhow::Error::from(Error::PipelineNotFound(name.to_string()))); };
        entry.info.rendering_info = rendering_info;
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

    pub(crate) fn get_raytracing_pipeline(&mut self, name: &str) -> Result<&RayTracingPipeline<A>> {
        let entry = self.raytracing_pipeline_infos.get_mut(name);
        let Some(entry) = entry else { return Err(anyhow::Error::from(Error::PipelineNotFound(name.to_string()))); };
        // Also put in queries for descriptor set layouts and pipeline layout to make sure they are not destroyed.
        for layout in &entry.info.layout.set_layouts {
            self.set_layouts.get_or_create(layout, ())?;
        }
        self.pipeline_layouts.get_or_create(&entry.info.layout, &mut self.set_layouts)?;
        self.raytracing_pipelines.get_or_create(
            &entry.info,
            (
                self.allocator.clone(),
                &mut self.shaders,
                &mut self.pipeline_layouts,
                &mut self.set_layouts,
            ),
        )
    }
}

// TODO: Maybe incorporate the vulkan pipeline cache api to improve startup times?

impl<A: Allocator> PipelineCache<A> {
    /// Create a new empty pipeline cache.
    pub fn new(device: Device, allocator: A) -> Result<Self> {
        let inner = PipelineCacheInner {
            allocator,
            shaders: Cache::new(device.clone()),
            set_layouts: Cache::new(device.clone()),
            pipeline_layouts: Cache::new(device.clone()),
            pipelines: Cache::new(device.clone()),
            compute_pipelines: Cache::new(device.clone()),
            raytracing_pipelines: Cache::new(device),
            pipeline_infos: Default::default(),
            compute_pipeline_infos: Default::default(),
            raytracing_pipeline_infos: Default::default(),
        };
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    /// Create and register a new pipeline into the cache.
    #[cfg(feature = "shader-reflection")]
    pub fn create_named_pipeline(&mut self, mut info: PipelineCreateInfo) -> Result<()> {
        let refl = reflect_shaders(info.shaders.as_slice())?;
        // Using reflection, we can allow omitting the pipeline layout field.
        info.layout = build_pipeline_layout(&refl);
        let name = info.name.clone();
        let mut inner = self.inner.write().unwrap();
        inner.pipeline_infos.insert(
            name.clone(),
            PipelineEntry {
                info,
                reflection: refl,
            },
        );
        inner.pipeline_infos.get_mut(&name).unwrap().info.build_inner();
        Ok(())
    }

    /// Create and register a new pipeline into the cache
    #[cfg(not(feature = "shader-reflection"))]
    pub fn create_named_pipeline(&mut self, mut info: PipelineCreateInfo) -> Result<()> {
        info.build_inner();
        let name = info.name.clone();
        let mut inner = self.inner.write().unwrap();
        inner.pipeline_infos.insert(
            name.clone(),
            PipelineEntry {
                info,
            },
        );
        inner.pipeline_infos.get_mut(&name).unwrap().info.build_inner();
        Ok(())
    }

    /// Create and register a new compute pipeline into the cache
    #[cfg(feature = "shader-reflection")]
    pub fn create_named_compute_pipeline(&mut self, mut info: ComputePipelineCreateInfo) -> Result<()> {
        let refl = match &info.shader {
            None => reflect_shaders(&[])?,
            Some(info) => reflect_shaders(std::slice::from_ref(info))?,
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
        let mut inner = self.inner.write().unwrap();
        inner.compute_pipeline_infos.insert(
            name,
            PipelineEntry {
                info,
                reflection: refl,
            },
        );
        Ok(())
    }

    /// Create and register a new compute pipeline into the cache
    #[cfg(not(feature = "shader-reflection"))]
    pub fn create_named_compute_pipeline(&mut self, mut info: ComputePipelineCreateInfo) -> Result<()> {
        let name = info.name.clone();
        let mut inner = self.inner.write().unwrap();
        inner.compute_pipeline_infos.insert(
            name,
            PipelineEntry {
                info,
            },
        );
        Ok(())
    }

    /// Create and register a new raytracing pipeline into the cache
    #[cfg(feature = "shader-reflection")]
    pub fn create_named_raytracing_pipeline(&mut self, mut info: RayTracingPipelineCreateInfo) -> Result<()> {
        let refl = reflect_shaders(info.shaders.as_slice())?;
        // Using reflection, we can allow omitting the pipeline layout field.
        info.layout = build_pipeline_layout(&refl);

        let name = info.name.clone();
        let mut inner = self.inner.write().unwrap();
        inner.raytracing_pipeline_infos.insert(
            name,
            PipelineEntry {
                info,
                reflection: refl,
            },
        );
        Ok(())
    }

    /// Create and register a new compute pipeline into the cache
    #[cfg(not(feature = "shader-reflection"))]
    pub fn create_named_raytracing_pipeline(&mut self, mut info: RayTracingPipelineCreateInfo) -> Result<()> {
        let name = info.name.clone();
        let mut inner = self.inner.write().unwrap();
        inner.raytracing_pipeline_infos.insert(
            name,
            PipelineEntry {
                info,
            },
        );
        Ok(())
    }

    /// Get the pipeline create info associated with a pipeline
    /// # Errors
    /// Returns None if the pipeline was not found in the cache.
    pub fn pipeline_info(&self, name: &str) -> Option<PipelineCreateInfo> {
        self.inner
            .read()
            .unwrap()
            .pipeline_infos
            .get(name)
            .map(|entry| entry.info.clone())
    }

    /// Get the pipeline create info associated with a compute pipeline
    /// # Errors
    /// Returns None if the pipeline was not found in the cache.
    pub fn compute_pipeline_info(&self, name: &str) -> Option<ComputePipelineCreateInfo> {
        self.inner
            .read()
            .unwrap()
            .compute_pipeline_infos
            .get(name)
            .map(|entry| entry.info.clone())
    }

    /// Get the pipeline create info associated with a raytracing pipeline
    /// # Errors
    /// Returns None if the pipeline was not found in the cache.
    pub fn raytracing_pipeline_info(&self, name: &str) -> Option<RayTracingPipelineCreateInfo> {
        self.inner
            .read()
            .unwrap()
            .raytracing_pipeline_infos
            .get(name)
            .map(|entry| entry.info.clone())
    }

    /// Returns the pipeline type of a pipeline, or None if the pipeline does not exist.
    pub fn pipeline_type(&self, name: &str) -> Option<PipelineType> {
        let inner = self.inner.read().unwrap();
        if inner.pipeline_infos.contains_key(name) {
            Some(PipelineType::Graphics)
        } else if inner.compute_pipeline_infos.contains_key(name) {
            Some(PipelineType::Compute)
        } else if inner.raytracing_pipeline_infos.contains_key(name) {
            Some(PipelineType::RayTracing)
        } else {
            None
        }
    }

    /// Obtain a pipeline from the cache and do some work with it.
    /// # Errors
    /// - This function can fail if the requested pipeline does not exist in the cache
    /// - This function can fail if allocating the pipeline fails.
    pub(crate) fn with_pipeline<F: FnOnce(&Pipeline) -> Result<()>>(&mut self, name: &str, rendering_info: PipelineRenderingInfo, f: F) -> Result<()> {
        let mut inner = self.inner.write().unwrap();
        let pipeline = inner.get_pipeline(name, rendering_info)?;
        f(pipeline)
    }

    /// Obtain a compute pipeline from the cache and do some work with it.
    /// # Errors
    /// - This function can fail if the requested pipeline does not exist in the cache
    /// - This function can fail if allocating the pipeline fails.
    pub(crate) fn with_compute_pipeline<F: FnOnce(&ComputePipeline) -> Result<()>>(&mut self, name: &str, f: F) -> Result<()> {
        let mut inner = self.inner.write().unwrap();
        let pipeline = inner.get_compute_pipeline(name)?;
        f(pipeline)
    }

    /// Obtain a raytracing pipeline from the cache and do some work with it.
    /// # Errors
    /// - This function can fail if the requested pipeline does not exist in the cache
    /// - This function can fail if allocating the pipeline fails.
    pub(crate) fn with_raytracing_pipeline<F: FnOnce(&RayTracingPipeline<A>) -> Result<()>>(&mut self, name: &str, f: F) -> Result<()> {
        let mut inner = self.inner.write().unwrap();
        let pipeline = inner.get_raytracing_pipeline(name)?;
        f(pipeline)
    }

    /// Advance cache resource time to live so resources that have not been used in a while can be cleaned up
    pub fn next_frame(&mut self) {
        let mut inner = self.inner.write().unwrap();
        inner.pipelines.next_frame();
        inner.compute_pipelines.next_frame();
        inner.raytracing_pipelines.next_frame();
        inner.pipeline_layouts.next_frame();
        inner.shaders.next_frame();
        inner.set_layouts.next_frame();
    }
}
