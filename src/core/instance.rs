//! Exposes the Vulkan instance, which represents the loaded Vulkan library

use std::ffi::{CStr, CString};
use std::ops::Deref;
use std::str::FromStr;

use anyhow::Result;
use ash;
use ash::vk;

use crate::{AppSettings, SurfaceSettings};
use crate::util::string::unwrap_to_raw_strings;

/// Represents the loaded vulkan instance.
/// You need to create this to initialize the Vulkan API. This is used
/// to create the device from.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Instance {
    #[derivative(Debug = "ignore")]
    entry: ash::Entry,
    #[derivative(Debug = "ignore")]
    instance: ash::Instance,
}

impl Instance {
    /// Initializes the Vulkan API.
    /// # Errors
    /// * Can fail if the Vulkan loader was not found. Check for valid Vulkan drivers.
    /// * Can fail if an instance extension or layer was requested that is not supported. This can happen when
    ///   validation is enabled through [`AppSettings`], but the Vulkan SDK is not installed.
    pub fn new(settings: &AppSettings) -> Result<Self> {
        let entry = unsafe { ash::Entry::load()? };
        let instance = create_vk_instance(&entry, settings)?;
        #[cfg(feature = "log-objects")]
        trace!("Created new VkInstance {:p}", instance.handle());
        Ok(Instance {
            entry,
            instance,
        })
    }

    /// Get unsafe access to the vulkan entry point.
    /// # Safety
    /// Any vulkan calls that modify the system's state may put the system in an undefined state.
    pub unsafe fn loader(&self) -> &ash::Entry {
        &self.entry
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkInstance {:p}", self.instance.handle());
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

impl Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

fn create_vk_instance(
    entry: &ash::Entry,
    settings: &AppSettings,
) -> Result<ash::Instance> {
    let app_name = CString::new(settings.name.clone())?;
    let engine_name = CString::new("Phobos")?;
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 3, 0),
        p_application_name: app_name.as_ptr(),
        p_engine_name: engine_name.as_ptr(),
        engine_version: vk::make_api_version(
            0,
            u32::from_str(env!("CARGO_PKG_VERSION_MAJOR")).unwrap(),
            u32::from_str(env!("CARGO_PKG_VERSION_MINOR")).unwrap(),
            u32::from_str(env!("CARGO_PKG_VERSION_PATCH")).unwrap(),
        ),
        ..Default::default()
    };

    let mut layers = Vec::<CString>::new();
    let mut extensions = Vec::<CString>::new();

    if settings.enable_validation {
        layers.push(CString::new("VK_LAYER_KHRONOS_validation")?);
        extensions.push(CString::from(ash::extensions::ext::DebugUtils::name()));
    }

    if let Some(SurfaceSettings { window, .. }) = settings.surface_settings {
        extensions.extend(
            ash_window::enumerate_required_extensions(window.raw_display_handle())?
                .to_vec()
                .iter()
                .map(|&raw_str| unsafe { CString::from(CStr::from_ptr(raw_str)) }),
        );
    }

    info!("Enabled instance extensions:");
    for ext in &extensions {
        info!("{:?}", ext);
    }

    let layers_raw = unwrap_to_raw_strings(layers.as_slice());
    let extensions_raw = unwrap_to_raw_strings(extensions.as_slice());

    let instance_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(layers_raw.as_slice())
        .enabled_extension_names(extensions_raw.as_slice())
        .build();

    Ok(unsafe { entry.create_instance(&instance_info, None)? })
}
