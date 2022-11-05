use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::str::FromStr;
use std::sync::Arc;

use ash::{vk, Entry, Instance, Device};
use ash::extensions::ext::DebugUtils;
use ash_window;

use crate::*;

extern "system" fn vk_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void) -> vk::Bool32 {

    let callback_data = unsafe { *p_callback_data };
    let message_id_number = callback_data.message_id_number as i32;
    let message_id_name = unsafe { util::wrap_c_str(callback_data.p_message_id_name) };
    let message = unsafe { util::wrap_c_str(callback_data.p_message) };

    // TODO: switch out logging with log crate: https://docs.rs/log

    println!("[{:?}] [{:?}]: {} ({}): {}",
             severity,
             msg_type,
             message_id_name,
             &message_id_number.to_string(),
             message
    );

    false as vk::Bool32
}

pub(crate) fn create_vk_instance<Window>(entry: &Entry, settings: &AppSettings<Window>) -> Option<Instance> where Window: WindowInterface {
    let app_name = CString::new(settings.name.clone()).unwrap();
    let engine_name = CString::new("Phobos").unwrap();
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 2, 0),
        p_application_name: app_name.as_ptr(),
        p_engine_name: engine_name.as_ptr(),
        engine_version: vk::make_api_version(0,
                                             u32::from_str(env!("CARGO_PKG_VERSION_MAJOR")).unwrap(),
                                             u32::from_str(env!("CARGO_PKG_VERSION_MINOR")).unwrap(),
                                             u32::from_str(env!("CARGO_PKG_VERSION_PATCH")).unwrap()),
        ..Default::default()
    };

    let mut layers = Vec::<CString>::new();
    let mut extensions = Vec::<CString>::new();

    if settings.enable_validation {
        layers.push(CString::new("VK_LAYER_KHRONOS_validation").unwrap());
        extensions.push(CString::from(DebugUtils::name()));
    }

    if let Some(window) = settings.window {
        extensions.extend(
            ash_window::enumerate_required_extensions(window.raw_display_handle())
                .unwrap()
                .to_vec()
                .iter()
                .map(|&raw_str| unsafe { CString::from(CStr::from_ptr(raw_str)) })
        );
    }

    let layers_raw = util::unwrap_to_raw_strings(layers.as_slice());
    let extensions_raw = util::unwrap_to_raw_strings(extensions.as_slice());

    let instance_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(layers_raw.as_slice())
        .enabled_extension_names(extensions_raw.as_slice())
        .build();

    return unsafe { entry.create_instance(&instance_info, None).ok() };
}

pub(crate) fn create_debug_messenger(funcs: &FuncPointers) -> vk::DebugUtilsMessengerEXT {
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
        .pfn_user_callback(Some(vk_debug_callback))
        .build();

    unsafe { funcs.debug_utils.as_ref().unwrap().create_debug_utils_messenger(&create_info, None).unwrap() }
}

pub(crate) fn create_surface<Window>(settings: &AppSettings<Window>, entry: &Entry, instance: &Instance) -> Surface where Window: WindowInterface {
    let window = settings.window.unwrap();
    let surface_handle = unsafe { ash_window::create_surface(&entry, &instance, window.raw_display_handle(), window.raw_window_handle(), None).unwrap() };
    return Surface {
        handle: surface_handle,
        // Surface capabilities will be queried by the VkPhysicalDevice.
        ..Default::default()
    };
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

pub(crate) fn select_physical_device<Window>(settings: &AppSettings<Window>, surface: &Option<Surface>, funcs: &FuncPointers, instance: &Instance) -> PhysicalDevice where Window: WindowInterface {
    let devices = unsafe { instance.enumerate_physical_devices().expect("No physical devices found.") };

    devices.iter()
        .find_map(|device| -> Option<PhysicalDevice> {
            let mut physical_device = PhysicalDevice {
                handle: *device,
                properties: unsafe { instance.get_physical_device_properties(*device) },
                memory_properties: unsafe { instance.get_physical_device_memory_properties(*device) },
                extension_properties: unsafe { instance.enumerate_device_extension_properties(*device).unwrap().iter().map(|vk_properties| {
                  ExtensionProperties {
                      name: util::wrap_c_str(vk_properties.extension_name.as_ptr()),
                      spec_version: vk_properties.spec_version
                  }
                }).collect() },
                queue_families: unsafe { instance.get_physical_device_queue_family_properties(*device) },
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
                                family_index: index as u32,
                                ..Default::default()
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
                        funcs.surface.as_ref().unwrap()
                            .get_physical_device_surface_support(
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
        })
        .expect("No physical device matching requested capabilities found.")
}

pub(crate) fn fill_surface_details(surface: &mut Surface, physical_device: &PhysicalDevice, funcs: &FuncPointers) {
    let surface_funcs = funcs.surface.as_ref().unwrap();
    unsafe {
        surface.capabilities = surface_funcs.get_physical_device_surface_capabilities(physical_device.handle, surface.handle).unwrap();
        surface.formats = surface_funcs.get_physical_device_surface_formats(physical_device.handle, surface.handle).unwrap();
        surface.present_modes = surface_funcs.get_physical_device_surface_present_modes(physical_device.handle, surface.handle).unwrap();
    }
}

pub(crate) fn create_device<Window>(settings: &AppSettings<Window>, physical_device: &PhysicalDevice, instance: &Instance) -> Device where Window: WindowInterface {
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
            CString::new(ext.clone()).unwrap()
        )
        .collect();

    // Add required extensions
    if settings.window.is_some() {
        extension_names.push(CString::from(ash::extensions::khr::Swapchain::name()));
    }

    let extension_names_raw = util::unwrap_to_raw_strings(extension_names.as_slice());
    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(queue_create_infos.as_slice())
        .enabled_extension_names(extension_names_raw.as_slice())
        .build();


    unsafe { instance.create_device(physical_device.handle, &info, None).unwrap() }
}

pub(crate) fn get_queues(physical_device: &PhysicalDevice, device: Arc<Device>) -> Vec<Queue> {
    let mut counts = HashMap::new();
    physical_device.queues.iter().map(|queue| -> Queue {
        let index = counts.entry(queue.family_index).or_insert(0 as u32);
        let handle = unsafe { device.get_device_queue(queue.family_index, *index) };
        *counts.get_mut(&queue.family_index).unwrap() += 1;
        Queue {
            handle,
            info: *queue
        }
    })
    .collect()
}

fn choose_surface_format<Window>(settings: &AppSettings<Window>, surface: &Surface) -> vk::SurfaceFormatKHR where Window: WindowInterface {
    // In case requested format isn't found, try this. If that one isn't found we fall back to the first available format.
    const FALLBACK_FORMAT: vk::SurfaceFormatKHR = vk::SurfaceFormatKHR {
        format: vk::Format::B8G8R8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR
    };

    if let Some(preferred_format) = settings.surface_format {
        if surface.formats.contains(&preferred_format) { return preferred_format; }
    }
    if surface.formats.contains(&FALLBACK_FORMAT) { return FALLBACK_FORMAT; }

    *surface.formats.first().unwrap()
}

fn choose_present_mode<Window>(settings: &AppSettings<Window>, surface: &Surface) -> vk::PresentModeKHR where Window: WindowInterface {
    if let Some(mode) = settings.present_mode {
        if surface.present_modes.contains(&mode) { return mode; }
    }
    // VSync, guaranteed to be supported
    vk::PresentModeKHR::FIFO
}

fn choose_swapchain_extent<Window>(settings: &AppSettings<Window>, surface: &Surface) -> vk::Extent2D where Window: WindowInterface {
    if surface.capabilities.current_extent.width != u32::MAX {
        return surface.capabilities.current_extent;
    }

    vk::Extent2D {
        width: settings.window.unwrap().width().clamp(surface.capabilities.min_image_extent.width, surface.capabilities.max_image_extent.width),
        height: settings.window.unwrap().height().clamp(surface.capabilities.min_image_extent.height, surface.capabilities.max_image_extent.height)
    }
}

pub(crate) fn create_swapchain<Window>(device: Arc<Device>, settings: &AppSettings<Window>, surface: &Surface, funcs: &FuncPointers) -> Swapchain where Window: WindowInterface {
    let format = choose_surface_format(settings, surface);
    let present_mode = choose_present_mode(settings, surface);
    let extent = choose_swapchain_extent(settings, surface);

    let image_count = {
        let mut count = surface.capabilities.min_image_count + 1;
        // If a maximum is set, clamp to it
        if surface.capabilities.max_image_count != 0 { count = count.clamp(0, surface.capabilities.max_image_count); }
        count
    };

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface.handle)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .present_mode(present_mode)
        .min_image_count(image_count)
        .clipped(true)
        .pre_transform(surface.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .build();

    let swapchain = unsafe { funcs.swapchain.as_ref().unwrap().create_swapchain(&info, None) }.unwrap();

    let images: Vec<ImageView> = unsafe { funcs.swapchain.as_ref().unwrap().get_swapchain_images(swapchain) }
        .unwrap()
        .iter()
        .map(move |image| {
            let image = Image {
                device: device.clone(),
                handle: *image,
                format: format.format,
                size: vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1
                },
                layers: 1,
                mip_levels: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                // Leave memory at None since this is managed by the swapchain, not our application.
                memory: None
            };
            // Create a trivial ImgView.
            let view = image.view(vk::ImageAspectFlags::COLOR);
            // Bundle them together into an owning ImageView
            ImageView::from((image, view))
        })
        .collect();
    Swapchain {
        handle: swapchain,
        format,
        present_mode,
        extent,
        images
    }
}