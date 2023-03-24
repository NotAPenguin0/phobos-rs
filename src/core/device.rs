use std::collections::HashSet;
use std::ffi::{CStr, CString, NulError};
use std::fmt::Formatter;
use std::ops::Deref;
use std::sync::Arc;

use anyhow::Result;
use ash::vk;

use crate::{AppSettings, PhysicalDevice, VkInstance, WindowInterface};
use crate::util::string::unwrap_to_raw_strings;

/// Device extensions that phobos requests but might not be available.
#[derive(Debug, Eq, PartialEq, Hash)]
pub enum ExtensionID {
    ExtendedDynamicState3,
}

impl std::fmt::Display for ExtensionID {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Wrapper around a `VkDevice`. The device provides access to almost the entire
/// Vulkan API.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Device {
    #[derivative(Debug = "ignore")]
    handle: ash::Device,
    queue_families: Vec<u32>,
    properties: vk::PhysicalDeviceProperties,
    extensions: HashSet<ExtensionID>,
    #[derivative(Debug = "ignore")]
    dynamic_state3: Option<ash::extensions::ext::ExtendedDynamicState3>,
}

fn add_if_supported(
    ext: ExtensionID,
    name: &CStr,
    enabled_set: &mut HashSet<ExtensionID>,
    names: &mut Vec<CString>,
    extensions: &Vec<vk::ExtensionProperties>,
) -> bool {
    // First check if extension is supported
    if extensions
        .iter()
        // SAFETY: This pointer is obtained from a c string that was returned from a Vulkan API call. We can assume the
        // Vulkan API always returns valid strings.
        .filter(|ext| name == unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) })
        .next()
        .is_some()
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
    /// Create a new Vulkan device. This is wrapped in an Arc because it gets passed around and stored in a
    /// lot of Vulkan-related structures.
    pub fn new<Window: WindowInterface>(
        instance: &VkInstance,
        physical_device: &PhysicalDevice,
        settings: &AppSettings<Window>,
    ) -> Result<Arc<Self>> {
        let mut priorities = Vec::<f32>::new();
        let queue_create_infos = physical_device
            .queue_families()
            .iter()
            .enumerate()
            .flat_map(|(index, _)| {
                let count = physical_device
                    .queues()
                    .iter()
                    .filter(|queue| queue.family_index == index as u32)
                    .count();
                if count == 0 {
                    return None;
                }
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
            ash::extensions::ext::ExtendedDynamicState3::name(),
            &mut enabled_extensions,
            &mut extension_names,
            &available_extensions);

        // Add required extensions
        if settings.window.is_some() {
            extension_names.push(CString::from(ash::extensions::khr::Swapchain::name()));
        }

        info!("Enabled device extensions:");
        for ext in &extension_names {
            info!("{:?}", ext);
        }

        let mut features_1_1 = settings.gpu_requirements.features_1_1;
        let mut features_1_2 = settings.gpu_requirements.features_1_2;
        let mut features_1_3 = settings.gpu_requirements.features_1_3;
        features_1_3.synchronization2 = vk::TRUE;
        features_1_3.dynamic_rendering = vk::TRUE;

        let extension_names_raw = unwrap_to_raw_strings(extension_names.as_slice());
        let mut info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(queue_create_infos.as_slice())
            .enabled_extension_names(extension_names_raw.as_slice())
            .enabled_features(&settings.gpu_requirements.features)
            .push_next(&mut features_1_1)
            .push_next(&mut features_1_2)
            .push_next(&mut features_1_3);

        let mut features_dynamic_state3 = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default();
        features_dynamic_state3.extended_dynamic_state3_polygon_mode = vk::TRUE;
        if dynamic_state3_supported {
            info = info.push_next(&mut features_dynamic_state3);
        }
        let info = info.build();

        let handle = unsafe { instance.create_device(physical_device.handle(), &info, None)? };

        let dynamic_state3 = if dynamic_state3_supported {
            Some(ash::extensions::ext::ExtendedDynamicState3::new(&instance, &handle))
        } else {
            None
        };

        Ok(Arc::new(
            Device {
                handle,
                queue_families: queue_create_infos
                    .iter()
                    .map(|info| info.queue_family_index)
                    .collect(),
                properties: *physical_device.properties(),
                extensions: enabled_extensions,
                dynamic_state3,
            }
        ))
    }

    /// Wait for the device to be completely idle.
    /// This should not be used as a synchronization measure, except on exit.
    pub fn wait_idle(&self) -> Result<()> {
        unsafe { Ok(self.device_wait_idle()?) }
    }

    /// Get unsafe access to the underlying VkDevice handle
    pub unsafe fn handle(&self) -> ash::Device {
        self.handle.clone()
    }

    /// Get the queue families we requested on this device.
    pub fn queue_families(&self) -> &[u32] {
        self.queue_families.as_slice()
    }

    /// Get the device properties
    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.properties
    }

    pub fn is_extension_enabled(&self, ext: ExtensionID) -> bool {
        self.extensions.contains(&ext)
    }

    pub fn dynamic_state3(&self) -> Option<&ash::extensions::ext::ExtendedDynamicState3> {
        self.dynamic_state3.as_ref()
    }
}

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.destroy_device(None);
        }
    }
}
