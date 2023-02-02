use std::ops::Deref;
use std::sync::Arc;
use ash::vk;
use std::ffi::{CString, NulError};
use crate::{PhysicalDevice, Error, VkInstance, AppSettings, WindowInterface};
use crate::util;

/// Wrapper around a `VkDevice`. The device provides access to almost the entire
/// Vulkan API.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Device {
    #[derivative(Debug="ignore")]
    pub(crate) handle: ash::Device,
    pub(crate) queue_families: Vec<u32>,
}

impl Device {
    /// Create a new Vulkan device. This is wrapped in an Arc because it gets passed around and stored in a
    /// lot of Vulkan-related structures.
    pub fn new<Window: WindowInterface>(instance: &VkInstance, physical_device: &PhysicalDevice, settings: &AppSettings<Window>) -> Result<Arc<Self>, Error> {
        let mut priorities = Vec::<f32>::new();
        let queue_create_infos = physical_device.queue_families.iter()
            .enumerate()
            .map(|(index, _)| {
                let count = physical_device.queues.iter().filter(|queue| queue.family_index == index as u32).count();
                priorities.resize(usize::max(priorities.len(), count), 1.0);
                vk::DeviceQueueCreateInfo {
                    queue_family_index: index as u32,
                    queue_count: count as u32,
                    p_queue_priorities: priorities.as_ptr(),
                    ..Default::default()
                }
            })
            .collect::<Vec<_>>();
        let mut extension_names: Vec<CString> = settings.gpu_requirements.device_extensions.iter()
            .map(|ext|
                CString::new(ext.clone())
            )
            .collect::<Result<Vec<CString>, NulError>>()?;

        // Add required extensions
        extension_names.push(CString::from(ash::extensions::khr::Synchronization2::name()));
        extension_names.push(CString::from(ash::extensions::khr::DynamicRendering::name()));
        if settings.window.is_some() {
            extension_names.push(CString::from(ash::extensions::khr::Swapchain::name()));
        }

        let mut features_1_1 = settings.gpu_requirements.features_1_1;
        let mut features_1_2 = settings.gpu_requirements.features_1_2;
        let mut features_1_3 = settings.gpu_requirements.features_1_3;
        features_1_3.synchronization2 = vk::TRUE;
        features_1_3.dynamic_rendering = vk::TRUE;

        let extension_names_raw = util::unwrap_to_raw_strings(extension_names.as_slice());
        let info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(queue_create_infos.as_slice())
            .enabled_extension_names(extension_names_raw.as_slice())
            .enabled_features(&settings.gpu_requirements.features)
            .push_next(&mut features_1_1)
            .push_next(&mut features_1_2)
            .push_next(&mut features_1_3)
            .build();


        Ok(Arc::new(unsafe { Device {
            handle: instance.instance.create_device(physical_device.handle, &info, None)?,
            queue_families: queue_create_infos.iter().map(|info| info.queue_family_index).collect()
        } }))
    }

    /// Wait for the device to be completely idle.
    /// This should not be used as a synchronization measure, except on exit.
    pub fn wait_idle(&self) -> Result<(), Error> {
        unsafe { Ok(self.device_wait_idle()?) }
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
        unsafe { self.destroy_device(None); }
    }
}