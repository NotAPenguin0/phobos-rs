use std::ops::Deref;

use anyhow::Result;
use ash::vk;

use crate::{AppSettings, Error, PhysicalDevice, VkInstance, WindowInterface};

/// Contains all information about a [`VkSurfaceKHR`](vk::SurfaceKHR)
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Surface {
    /// Handle to the [`VkSurfaceKHR`](vk::SurfaceKHR)
    handle: vk::SurfaceKHR,
    /// [`VkSurfaceCapabilitiesKHR`](vk::SurfaceCapabilitiesKHR) structure storing information about surface capabilities.
    capabilities: vk::SurfaceCapabilitiesKHR,
    /// List of [`VkSurfaceFormatKHR`](vk::SurfaceFormatKHR) with all formats this surface supports.
    formats: Vec<vk::SurfaceFormatKHR>,
    /// List of [`VkPresentModeKHR`](vk::PresentModeKHR) with all present modes this surface supports.
    present_modes: Vec<vk::PresentModeKHR>,
    /// Vulkan extension functions for surface handling.
    #[derivative(Debug = "ignore")]
    functions: ash::extensions::khr::Surface,
}

impl Surface {
    /// Create a new surface.
    pub fn new<Window: WindowInterface>(instance: &VkInstance, settings: &AppSettings<Window>) -> Result<Self> {
        if let Some(window) = settings.window {
            let functions = ash::extensions::khr::Surface::new(unsafe { instance.loader() }, &*instance);
            let handle = unsafe {
                ash_window::create_surface(
                    instance.loader(),
                    &*instance,
                    window.raw_display_handle(),
                    window.raw_window_handle(),
                    None,
                )?
            };
            Ok(Surface {
                handle,
                functions,
                capabilities: Default::default(),
                formats: vec![],
                present_modes: vec![],
            })
        } else {
            Err(anyhow::Error::from(Error::NoWindow))
        }
    }

    /// Query support for features, capabilities and formats for this surface.
    /// Because surface support varies per physical device, this function requires one to be selected.
    pub fn query_details(&mut self, physical_device: &PhysicalDevice) -> Result<()> {
        unsafe {
            self.capabilities = self.get_physical_device_surface_capabilities(physical_device.handle(), self.handle)?;
            self.formats = self.get_physical_device_surface_formats(physical_device.handle(), self.handle)?;
            self.present_modes = self.get_physical_device_surface_present_modes(physical_device.handle(), self.handle)?;
        }
        Ok(())
    }

    pub unsafe fn handle(&self) -> vk::SurfaceKHR {
        self.handle
    }

    pub fn capabilities(&self) -> &vk::SurfaceCapabilitiesKHR {
        &self.capabilities
    }

    pub fn formats(&self) -> &[vk::SurfaceFormatKHR] {
        self.formats.as_slice()
    }

    pub fn present_modes(&self) -> &[vk::PresentModeKHR] {
        self.present_modes.as_slice()
    }
}

impl Deref for Surface {
    type Target = ash::extensions::khr::Surface;

    fn deref(&self) -> &Self::Target {
        &self.functions
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.functions.destroy_surface(self.handle, None);
        }
    }
}
