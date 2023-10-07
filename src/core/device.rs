//! Contains the Vulkan device, the main entrypoint to the Vulkan API.

use std::collections::HashSet;
use std::ffi::{CStr, CString, NulError};
use std::fmt::Formatter;
#[allow(unused_imports)]
use std::mem::ManuallyDrop;
use std::ops::Deref;
#[allow(unused_imports)]
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::Result;
use ash::extensions::{ext, khr};
use ash::vk;
#[cfg(feature = "fsr2")]
use fsr2_sys::FfxDimensions2D;

use crate::{AppSettings, Error, Instance, PhysicalDevice};
use crate::core::traits::Nameable;
use derivative::Derivative;
#[cfg(feature = "fsr2")]
use crate::fsr2::Fsr2Context;
#[cfg(feature = "fsr2")]
use crate::fsr2::Fsr2ContextCreateInfo;
use crate::util::string::unwrap_to_raw_strings;

/// Device extensions that phobos requests but might not be available.
/// # Example
/// ```
/// # use phobos::*;
/// # use phobos::core::device::ExtensionID;
///
/// fn has_dynamic_state(device: Device) -> bool {
///     device.is_extension_enabled(ExtensionID::ExtendedDynamicState3)
/// }
/// ```
#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone)]
pub enum ExtensionID {
    /// `VK_EXT_extended_dynamic_state3` provides more dynamic states to pipeline objects.
    ExtendedDynamicState3,
    /// `VK_KHR_acceleration_structure` provides acceleration structures for raytracing
    AccelerationStructure,
    /// `VK_KHR_ray_tracing_pipeline` provides raytracing pipelines and ray query objects in shaders.
    RayTracingPipeline,
}

impl std::fmt::Display for ExtensionID {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Derivative)]
#[derivative(Debug)]
struct DeviceInner {
    #[cfg(feature = "fsr2")]
    fsr2_context: Mutex<ManuallyDrop<Fsr2Context>>,
    #[derivative(Debug = "ignore")]
    handle: ash::Device,
    queue_families: Vec<u32>,
    properties: vk::PhysicalDeviceProperties,
    accel_structure_properties: Option<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>,
    rt_properties: Option<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>,
    extensions: HashSet<ExtensionID>,
    #[derivative(Debug = "ignore")]
    dynamic_state3: Option<ext::ExtendedDynamicState3>,
    #[derivative(Debug = "ignore")]
    acceleration_structure: Option<khr::AccelerationStructure>,
    #[derivative(Debug = "ignore")]
    rt_pipeline: Option<khr::RayTracingPipeline>,
    #[derivative(Debug = "ignore")]
    debug_utils: Option<ext::DebugUtils>,
}

/// Wrapper around a `VkDevice`. The device provides access to almost the entire
/// Vulkan API. Internal state is wrapped in an `Arc<DeviceInner>`, so this is safe
/// to clone
#[derive(Debug, Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

unsafe impl Send for Device {}

unsafe impl Sync for Device {}

fn add_if_supported(
    ext: ExtensionID,
    name: &CStr,
    enabled_set: &mut HashSet<ExtensionID>,
    names: &mut Vec<CString>,
    extensions: &[vk::ExtensionProperties],
) -> bool {
    // First check if extension is supported
    if extensions
        .iter()
        // SAFETY: This pointer is obtained from a c string that was returned from a Vulkan API call. We can assume the
        // Vulkan API always returns valid strings.
        .any(|ext| name == unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) })
    {
        enabled_set.insert(ext);
        names.push(CString::from(name));
        true
    } else {
        info!(
            "Requested extension {} is not available. Some features might be missing.",
            name.to_bytes().escape_ascii()
        );
        false
    }
}

impl Device {
    /// Create a new Vulkan device. This is the main interface point with the Vulkan API.
    /// # Errors
    /// * Can fail if vulkan device init fails. This is possible if an optional feature was enabled that is not supported.
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        settings: &AppSettings,
    ) -> Result<Self> {
        let mut priorities = Vec::<f32>::new();
        let queue_create_infos = physical_device
            .queue_families()
            .iter()
            .enumerate()
            .flat_map(|(index, family_info)| {
                let count = physical_device
                    .queues()
                    .iter()
                    .filter(|queue| queue.family_index == index as u32)
                    .count();
                if count == 0 {
                    return None;
                }
                let count = count.min(family_info.queue_count as usize);
                priorities.resize(usize::max(priorities.len(), count), 1.0);
                Some(vk::DeviceQueueCreateInfo {
                    queue_family_index: index as u32,
                    queue_count: count as u32,
                    p_queue_priorities: priorities.as_ptr(),
                    ..Default::default()
                })
            })
            .collect::<Vec<_>>();
        let mut extension_names: Vec<CString> = settings
            .gpu_requirements
            .device_extensions
            .iter()
            .map(|ext| CString::new(ext.clone()))
            .collect::<Result<Vec<CString>, NulError>>()?;

        // SAFETY: Vulkan API call. We have a valid reference to a PhysicalDevice, so handle() is valid.
        let available_extensions =
            unsafe { instance.enumerate_device_extension_properties(physical_device.handle())? };
        let mut enabled_extensions = HashSet::new();
        // Add the extensions we want, but that are not required.
        let dynamic_state3_supported = add_if_supported(
            ExtensionID::ExtendedDynamicState3,
            ext::ExtendedDynamicState3::name(),
            &mut enabled_extensions,
            &mut extension_names,
            available_extensions.as_slice(),
        );

        let accel_supported = if settings.raytracing {
            add_if_supported(
                ExtensionID::AccelerationStructure,
                khr::AccelerationStructure::name(),
                &mut enabled_extensions,
                &mut extension_names,
                available_extensions.as_slice(),
            )
        } else {
            false
        };

        let rt_pipeline_supported = if settings.raytracing {
            add_if_supported(
                ExtensionID::RayTracingPipeline,
                khr::RayTracingPipeline::name(),
                &mut enabled_extensions,
                &mut extension_names,
                available_extensions.as_slice(),
            )
        } else {
            false
        };

        let ray_query_name = CStr::from_bytes_with_nul(b"VK_KHR_ray_query\0")?;
        let ray_query_supported = settings.raytracing
            && available_extensions.iter().any(|ext| {
                ray_query_name == unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) }
            });
        if ray_query_supported {
            extension_names.push(CString::from(ray_query_name));
        }

        // Add required extensions
        if settings.surface_settings.is_some() {
            extension_names.push(CString::from(khr::Swapchain::name()));
        }

        if settings.raytracing {
            extension_names.push(CString::from(khr::DeferredHostOperations::name()));
        }

        info!("Enabled device extensions:");
        for ext in &extension_names {
            info!("{:?}", ext);
        }

        let mut features = settings.gpu_requirements.features;
        let mut features_1_1 = settings.gpu_requirements.features_1_1;
        let mut features_1_2 = settings.gpu_requirements.features_1_2;
        let mut features_1_3 = settings.gpu_requirements.features_1_3;
        features.pipeline_statistics_query = vk::TRUE;
        features_1_2.buffer_device_address = vk::TRUE;
        features_1_2.host_query_reset = vk::TRUE;
        features_1_2.descriptor_indexing = vk::TRUE;
        features_1_2.runtime_descriptor_array = vk::TRUE;
        features_1_2.descriptor_binding_partially_bound = vk::TRUE;
        features_1_2.shader_sampled_image_array_non_uniform_indexing = vk::TRUE;
        features_1_3.synchronization2 = vk::TRUE;
        features_1_3.dynamic_rendering = vk::TRUE;
        features_1_3.maintenance4 = vk::TRUE;
        #[cfg(feature = "fsr2")]
        {
            // Enable required and recommended features for FSR2
            features_1_2.shader_float16 = vk::TRUE;
            features.shader_int16 = vk::TRUE;
            features_1_1.storage_buffer16_bit_access = vk::TRUE;
        }

        let extension_names_raw = unwrap_to_raw_strings(extension_names.as_slice());
        let mut info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(queue_create_infos.as_slice())
            .enabled_extension_names(extension_names_raw.as_slice())
            .enabled_features(&features)
            .push_next(&mut features_1_1)
            .push_next(&mut features_1_2)
            .push_next(&mut features_1_3);

        let mut features_dynamic_state3 = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT {
            extended_dynamic_state3_polygon_mode: vk::TRUE,
            ..Default::default()
        };
        if dynamic_state3_supported {
            info = info.push_next(&mut features_dynamic_state3);
        }

        let mut features_acceleration_structure =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR {
                s_type: vk::StructureType::PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
                p_next: std::ptr::null_mut(),
                acceleration_structure: vk::TRUE,
                acceleration_structure_capture_replay: vk::FALSE,
                acceleration_structure_indirect_build: vk::FALSE,
                acceleration_structure_host_commands: vk::FALSE,
                descriptor_binding_acceleration_structure_update_after_bind: vk::FALSE,
            };

        let mut features_ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR {
            s_type: vk::StructureType::PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
            p_next: std::ptr::null_mut(),
            ray_query: vk::TRUE,
        };

        if accel_supported {
            info = info.push_next(&mut features_acceleration_structure);
        }

        if ray_query_supported {
            info = info.push_next(&mut features_ray_query);
        }

        let mut features_ray_tracing_pipeline = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR {
            s_type: vk::StructureType::PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
            p_next: std::ptr::null_mut(),
            ray_tracing_pipeline: vk::TRUE,
            ray_tracing_pipeline_shader_group_handle_capture_replay: vk::FALSE,
            ray_tracing_pipeline_shader_group_handle_capture_replay_mixed: vk::FALSE,
            ray_tracing_pipeline_trace_rays_indirect: vk::FALSE,
            ray_traversal_primitive_culling: vk::FALSE,
        };

        if rt_pipeline_supported {
            info = info.push_next(&mut features_ray_tracing_pipeline);
        }

        let info = info.build();

        let handle = unsafe { instance.create_device(physical_device.handle(), &info, None)? };
        #[cfg(feature = "log-objects")]
        trace!("Created new VkDevice {:p}", handle.handle());

        let dynamic_state3 = if dynamic_state3_supported {
            Some(ext::ExtendedDynamicState3::new(instance, &handle))
        } else {
            None
        };

        let acceleration_structure = if accel_supported {
            Some(khr::AccelerationStructure::new(instance, &handle))
        } else {
            None
        };

        let rt_pipeline = if rt_pipeline_supported {
            Some(khr::RayTracingPipeline::new(instance, &handle))
        } else {
            None
        };

        let mut properties2 = vk::PhysicalDeviceProperties2::builder();

        let mut accel_properties = if accel_supported {
            Some(vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default())
        } else {
            None
        };

        let mut rt_properties = if rt_pipeline_supported {
            Some(vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default())
        } else {
            None
        };

        let debug_utils = if settings.enable_validation {
            Some(ext::DebugUtils::new(unsafe { instance.loader() }, &instance))
        } else {
            None
        };

        match &mut accel_properties {
            None => {}
            Some(properties) => {
                properties2 = properties2.push_next(properties);
            }
        };

        match &mut rt_properties {
            None => {}
            Some(properties) => {
                properties2 = properties2.push_next(properties);
            }
        };

        unsafe {
            instance.get_physical_device_properties2(physical_device.handle(), &mut properties2)
        };

        // Create FSR2 context
        #[cfg(feature = "fsr2")]
        let fsr2 = unsafe {
            let display_size = FfxDimensions2D {
                width: settings.fsr2_settings.display_size.0,
                height: settings.fsr2_settings.display_size.1,
            };

            let max_render_size = match settings.fsr2_settings.max_render_size {
                None => display_size,
                Some((width, height)) => FfxDimensions2D {
                    width,
                    height,
                },
            };
            let info = Fsr2ContextCreateInfo {
                instance: &instance,
                physical_device: physical_device.handle(),
                device: handle.handle(),
                flags: settings.fsr2_settings.flags,
                max_render_size,
                display_size,
            };
            ManuallyDrop::new(Fsr2Context::new(info)?)
        };

        let inner = DeviceInner {
            handle,
            queue_families: queue_create_infos
                .iter()
                .map(|info| info.queue_family_index)
                .collect(),
            properties: *physical_device.properties(),
            accel_structure_properties: accel_properties,
            rt_properties,
            extensions: enabled_extensions,
            dynamic_state3,
            acceleration_structure,
            rt_pipeline,
            debug_utils,
            #[cfg(feature = "fsr2")]
            fsr2_context: Mutex::new(fsr2),
        };

        Ok(Device {
            inner: Arc::new(inner),
        })
    }

    /// Wait for the device to be completely idle.
    /// This should not be used as a synchronization measure, except on exit.
    /// # Errors
    /// This function call can be a source of `VK_ERROR_DEVICE_LOST`. This is typically due to a couple main reasons:
    /// * Previously submitted GPU work took too long, causing a driver reset. Try splitting very large workloads into multiple submits.
    /// * Previously submitted GPU work crashed the GPU due to an out of bounds memory read or related error. Try launching the program
    ///   with GPU-assisted validation through the Vulkan configurator.
    /// * Previously submitted GPU work crashed due to invalid API usage. Make sure the validation layers are on and no invalid pointers
    ///   are being passed to Vulkan calls.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use anyhow::Result;
    ///
    /// fn submit_some_gpu_work_and_wait(device: Device) -> Result<()> {
    ///     // ... gpu work
    ///     device.wait_idle()?;
    ///     // Work is now complete
    ///     Ok(())
    /// }
    /// ```
    pub fn wait_idle(&self) -> Result<()> {
        unsafe { Ok(self.inner.handle.device_wait_idle()?) }
    }

    /// Get unsafe access to the underlying `VkDevice` handle
    /// # Safety
    /// * The caller should not call `vkDestroyDevice` on this.
    /// * This handle is valid as long as there is a copy of `self` alive.
    /// * This can be used to do raw Vulkan calls. Modifying phobos objects through this
    ///   can put the system in an undefined state.
    pub unsafe fn handle(&self) -> ash::Device {
        self.inner.handle.clone()
    }

    /// Get the queue families we requested on this device. This is needed when using
    /// `VK_SHARING_MODE_CONCURRENT` on buffers and images.
    pub fn queue_families(&self) -> &[u32] {
        self.inner.queue_families.as_slice()
    }

    /// Get the physical device properties. This can be queried to check things such as the driver and GPU name,
    /// as well as API limitations.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// use std::ffi::CStr;
    /// fn list_device_info(device: Device) {
    ///     let properties = device.properties();
    ///     // SAFETY: The Vulkan API is guaranteed to return a null-terminated string.
    ///     let name = unsafe {
    ///         CStr::from_ptr(properties.device_name.as_ptr())
    ///     };
    ///     println!("Device name: {name:?}");
    ///     println!("Max bound descriptor sets: {}", properties.limits.max_bound_descriptor_sets);
    ///     // etc.
    /// }
    /// ```
    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.inner.properties
    }

    /// Get the physical device properties related to acceleration structures.
    ///
    /// # Errors
    /// - Fails if [`ExtensionID::AccelerationStructure`] is not enabled.
    pub fn acceleration_structure_properties(
        &self,
    ) -> Result<&vk::PhysicalDeviceAccelerationStructurePropertiesKHR> {
        self.require_extension(ExtensionID::AccelerationStructure)?;
        Ok(self.inner.accel_structure_properties.as_ref().unwrap())
    }

    /// Get the physical device properties related to raytracing pipelines
    /// # Errors
    /// - Fails if [`ExtensionID::RayTracingPipeline`] is not enabled.
    pub fn ray_tracing_properties(
        &self,
    ) -> Result<&vk::PhysicalDeviceRayTracingPipelinePropertiesKHR> {
        self.require_extension(ExtensionID::RayTracingPipeline)?;
        Ok(self.inner.rt_properties.as_ref().unwrap())
    }

    /// Get access to the functions of VK_EXT_debug_utils
    /// # Errors
    /// - Fails if validation layers are disabled
    pub fn debug_utils(&self) -> Result<&ext::DebugUtils> {
        self.inner
            .debug_utils
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DebugUtils does not exist"))
    }

    /// Check if a device extension is enabled.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use phobos::core::device::ExtensionID;
    /// fn has_extended_dynamic_state(device: Device) -> bool {
    ///     device.is_extension_enabled(ExtensionID::ExtendedDynamicState3)
    /// }
    /// ```
    pub fn is_extension_enabled(&self, ext: ExtensionID) -> bool {
        self.inner.extensions.contains(&ext)
    }

    /// Returns `Ok` if the given extension is enabled, `Err` otherwise.
    pub fn require_extension(&self, ext: ExtensionID) -> Result<()> {
        if self.is_extension_enabled(ext) {
            Ok(())
        } else {
            Err(Error::ExtensionNotSupported(ext).into())
        }
    }

    /// Access to the function pointers for `VK_EXT_dynamic_state_3`
    /// Access to the function pointers for `VK_EXT_dynamic_state_3`
    /// Returns `None` if the extension was not enabled or not available.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use anyhow::Result;
    /// # use ash::extensions::ext::ExtendedDynamicState3;
    /// fn set_wireframe(device: Device, cmd: vk::CommandBuffer) -> Result<()> {
    ///     match device.dynamic_state3() {
    ///         None => {
    ///             println!("VK_EXT_extended_dynamic_state3 not enabled!");
    ///             Ok(())
    ///         },
    ///         Some(ext) => {
    ///             // SAFETY: Vulkan API call.
    ///             unsafe { ext.cmd_set_polygon_mode(cmd, vk::PolygonMode::LINE) };
    ///             Ok(())
    ///         }
    ///     }
    /// }
    /// ```
    pub fn dynamic_state3(&self) -> Option<&ext::ExtendedDynamicState3> {
        self.inner.dynamic_state3.as_ref()
    }

    /// Access to the function pointers for `VK_KHR_acceleration_structure`
    ///
    /// Returns `None` if the extension is not enabled
    pub fn acceleration_structure(&self) -> Option<&khr::AccelerationStructure> {
        self.inner.acceleration_structure.as_ref()
    }

    /// Access to the function pointers for `VK_KHR_ray_tracing_pipeline`
    ///
    /// Returns `None` if the extension is not enabled
    pub fn raytracing_pipeline(&self) -> Option<&khr::RayTracingPipeline> {
        self.inner.rt_pipeline.as_ref()
    }

    /// True we only have a single queue, and thus the sharing mode for resources is always `VK_SHARING_MODE_EXCLUSIVE`.
    /// Not extremely useful on the user side, but maybe you want to know whether one physical queue is being multiplexed
    /// behind your back.
    pub fn is_single_queue(&self) -> bool {
        self.inner.queue_families.len() == 1
    }

    /// Get access to the FSR2 context. See [`Fsr2Context`](crate::fsr2::Fsr2Context)
    #[cfg(feature = "fsr2")]
    pub fn fsr2_context(&self) -> MutexGuard<ManuallyDrop<Fsr2Context>> {
        self.inner.fsr2_context.lock().unwrap()
    }

    /// Set the name of any given compatible object for debugging purposes
    pub fn set_name<T: Nameable>(&self, object: &T, name: &str) -> Result<()> {
        let object_name = CString::new(name)?;
        let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
            .object_type(<T as Nameable>::OBJECT_TYPE)
            .object_handle(unsafe { object.as_raw() })
            .object_name(&object_name)
            .build();

        unsafe {
            Ok(self
                .debug_utils()?
                .set_debug_utils_object_name(self.handle().handle(), &name_info)?)
        }
    }
}

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.inner.handle
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        #[cfg(feature = "fsr2")]
        unsafe {
            let mut fsr2 = self.fsr2_context.lock().unwrap();
            ManuallyDrop::drop(&mut fsr2);
        }
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkDevice {:p}", self.handle.handle());
        unsafe {
            self.handle.destroy_device(None);
        }
    }
}
