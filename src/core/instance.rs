use std::ffi::{CStr, CString};
use std::ops::Deref;
use std::str::FromStr;

use anyhow::Result;
use ash;
use ash::vk;

use crate::AppSettings;
use crate::util::string::unwrap_to_raw_strings;
use crate::WindowInterface;

/// Represents the loaded vulkan instance.
/// You need to create this to initialize the Vulkan API.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct VkInstance {
    #[derivative(Debug = "ignore")]
    entry: ash::Entry,
    #[derivative(Debug = "ignore")]
    instance: ash::Instance,
}

impl VkInstance {
    /// Initializes the Vulkan API.
    pub fn new<Window: WindowInterface>(settings: &AppSettings<Window>) -> Result<Self> {
        let entry = unsafe { ash::Entry::load()? };
        let instance = create_vk_instance(&entry, &settings)?;
        Ok(VkInstance { entry, instance })
    }

    pub unsafe fn loader(&self) -> &ash::Entry {
        &self.entry
    }
}

impl Drop for VkInstance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

impl Deref for VkInstance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

fn create_vk_instance<Window: WindowInterface>(
    entry: &ash::Entry,
    settings: &AppSettings<Window>,
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

    info!("Enabled instance extensions:");
    for ext in &extensions {
        info!("{:?}", ext);
    }

    if let Some(window) = settings.window {
        extensions.extend(
            ash_window::enumerate_required_extensions(window.raw_display_handle())?
                .to_vec()
                .iter()
                .map(|&raw_str| unsafe { CString::from(CStr::from_ptr(raw_str)) }),
        );
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
