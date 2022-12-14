use ash::vk;
use crate::{AppSettings, Error, Surface, VkInstance, WindowInterface};
use crate::queue::{QueueInfo, QueueType};
use crate::util;

/// Stores queried properties of a Vulkan extension.
#[derive(Debug, Default)]
pub struct ExtensionProperties {
    /// Name of the extension.
    pub name: String,
    /// Specification version of the extension.
    pub spec_version: u32,
}

/// A physical device abstracts away an actual device, like a graphics card or integrated graphics card.
#[derive(Default, Debug)]
pub struct PhysicalDevice {
    /// Handle to the [`VkPhysicalDevice`](vk::PhysicalDevice).
    pub handle: vk::PhysicalDevice,
    /// [`VkPhysicalDeviceProperties`](vk::PhysicalDeviceProperties) structure with properties of this physical device.
    pub properties: vk::PhysicalDeviceProperties,
    /// [`VkPhysicalDeviceMemoryProperties`](vk::PhysicalDeviceMemoryProperties) structure with memory properties of the physical device, such as
    /// available memory types and heaps.
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Available Vulkan extensions.
    pub extension_properties: Vec<ExtensionProperties>,
    /// List of [`VkQueueFamilyProperties`](vk::QueueFamilyProperties) with properties of each queue family on the device.
    pub queue_families: Vec<vk::QueueFamilyProperties>,
    /// List of [`QueueInfo`]  with requested queues abstracted away from the physical queues.
    pub queues: Vec<QueueInfo>
}

impl PhysicalDevice {
    /// Selects the best available physical device from the given requirements and parameters.
    pub fn select<Window: WindowInterface>(instance: &VkInstance, surface: Option<&Surface>, settings: &AppSettings<Window>) -> Result<Self, Error> {
        let devices = unsafe { instance.instance.enumerate_physical_devices()? };
        if devices.is_empty() {
            return Err(Error::NoGPU);
        }

        devices.iter()
            .find_map(|device| -> Option<PhysicalDevice> {
                let mut physical_device = PhysicalDevice {
                    handle: *device,
                    properties: unsafe { instance.instance.get_physical_device_properties(*device) },
                    memory_properties: unsafe { instance.instance.get_physical_device_memory_properties(*device) },
                    extension_properties: unsafe { instance.instance.enumerate_device_extension_properties(*device).unwrap().iter().map(|vk_properties| {
                        ExtensionProperties {
                            name: util::wrap_c_str(vk_properties.extension_name.as_ptr()),
                            spec_version: vk_properties.spec_version
                        }
                    }).collect() },
                    queue_families: unsafe { instance.instance.get_physical_device_queue_family_properties(*device) },
                    ..Default::default()
                };

                if settings.gpu_requirements.dedicated && physical_device.properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU { return None; }
                if settings.gpu_requirements.min_video_memory > total_video_memory(&physical_device) { return None; }
                if settings.gpu_requirements.min_dedicated_video_memory > total_device_memory(&physical_device) { return None; }

                physical_device.queues = {
                    settings.gpu_requirements.queues.iter()
                        .filter_map(|request| -> Option<QueueInfo> {
                            let avoid = if request.dedicated {
                                match request.queue_type {
                                    QueueType::Graphics => vk::QueueFlags::COMPUTE,
                                    QueueType::Compute => vk::QueueFlags::GRAPHICS,
                                    QueueType::Transfer => vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS
                                }
                            } else { vk::QueueFlags::default() };

                            return if let Some((index, dedicated)) = get_queue_family_prefer_dedicated(physical_device.queue_families.as_slice(), request.queue_type, avoid) {
                                Some(QueueInfo {
                                    queue_type: request.queue_type,
                                    dedicated,
                                    can_present: false,
                                    family_index: index as u32,
                                    flags: physical_device.queue_families[index].queue_flags,
                                })
                            } else {
                                None
                            };
                        })
                        .collect()
                };

                // We now have a list of all the queues that were found matching our request. If this amount is smaller than the number of requested queues,
                // at least one is missing and could not be fulfilled. In this case we reject the device.
                if physical_device.queues.len() < settings.gpu_requirements.queues.len() { return None; }

                // Now check surface support (if we are not creating a headless context)
                if let Some(surface) = surface {
                    // The surface is supported if one of the queues we found can present to it.
                    let supported_queue = physical_device.queues.iter_mut().find(|queue| {
                        unsafe {
                            surface.functions.get_physical_device_surface_support(
                                    physical_device.handle,
                                    queue.family_index,
                                    surface.handle)
                                .unwrap()
                        }
                    });
                    if let Some(queue) = supported_queue {
                        // Flag that we can present to it
                        queue.can_present = true;
                    } else { // No queue to present found, reject physical device
                        return None;
                    }
                }

                // Check if all requested extensions are present
                if !settings.gpu_requirements.device_extensions.iter().all(|requested_extension| {
                    physical_device.extension_properties.iter()
                        .find(|ext|
                            ext.name == *requested_extension
                        )
                        .is_some()
                }) {
                    return None;
                }

                return Some(physical_device);
        }).ok_or(Error::NoGPU)
    }
}


fn total_video_memory(device: &PhysicalDevice) -> usize {
    device.memory_properties.memory_heaps.iter()
        .map(|heap| heap.size as usize)
        .sum()
}

fn total_device_memory(device: &PhysicalDevice) -> usize {
    device.memory_properties.memory_heaps.iter()
        .filter(|heap|
            heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        )
        .map(|heap| heap.size as usize)
        .sum()
}

fn get_queue_family_prefer_dedicated(families: &[vk::QueueFamilyProperties], queue_type: QueueType, avoid: vk::QueueFlags) -> Option<(usize, bool)> {
    let required = vk::QueueFlags::from_raw(queue_type as vk::Flags);
    families.iter().enumerate().fold(None, |current_best_match, (index, family) | -> Option<usize> {
        // Does not contain required flags, must skip
        if !family.queue_flags.contains(required) { return current_best_match; }
        // Contains required flags and does not contain *any* flags to avoid, this is an optimal match.
        // Note that to check of it doesn't contain any of the avoid flags, contains() will not work, we need to use intersects()
        if !family.queue_flags.intersects(avoid) { return Some(index) }

        // Only if we don't have a match yet, settle for a suboptimal match
        return if current_best_match.is_none() {
            Some(index)
        } else {
            current_best_match
        }

    })
    .map(|index| {
        return (index, !families[index].queue_flags.intersects(avoid));
    })
}