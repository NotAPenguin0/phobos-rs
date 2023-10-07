//! Wrappers around a `VkSurfaceKHR`

use std::ops::Deref;

use anyhow::Result;
use ash::vk;

use crate::{Instance, PhysicalDevice, Window};

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
    pub fn new(
        instance: &Instance,
        window: &dyn Window
    ) -> Result<Self> {
        let functions =
            ash::extensions::khr::Surface::new(unsafe { instance.loader() }, instance);
        let handle = unsafe {
            ash_window::create_surface(
                instance.loader(),
                instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )?
        };
        #[cfg(feature = "log-objects")]
        trace!("Created new VkSurfaceKHR {handle:p}");
        Ok(Surface {
            handle,
            functions,
            capabilities: Default::default(),
            formats: vec![],
            present_modes: vec![],
        })
    }

    /// Query support for features, capabilities and formats for this surface.
    /// Because surface support varies per physical device, this function requires one to be selected.
    pub fn query_details(&mut self, physical_device: &PhysicalDevice) -> Result<()> {
        unsafe {
            self.capabilities = self
                .get_physical_device_surface_capabilities(physical_device.handle(), self.handle)?;
            self.formats =
                self.get_physical_device_surface_formats(physical_device.handle(), self.handle)?;
            self.present_modes = self
                .get_physical_device_surface_present_modes(physical_device.handle(), self.handle)?;
        }
        Ok(())
    }

    /// Get unsafe access to the underlying `VkSurfaceKHR` object.
    /// # Safety
    /// Any vulkan calls that mutate the surface may put the system in an undefined state.
    pub unsafe fn handle(&self) -> vk::SurfaceKHR {
        self.handle
    }

    /// Get the surface capabilities.
    pub fn capabilities(&self) -> &vk::SurfaceCapabilitiesKHR {
        &self.capabilities
    }

    /// Get the available surface formats.
    pub fn formats(&self) -> &[vk::SurfaceFormatKHR] {
        self.formats.as_slice()
    }

    /// Get the available surface present modes.
    pub fn present_modes(&self) -> &[vk::PresentModeKHR] {
        self.present_modes.as_slice()
    }
}

impl Deref for Surface {
    type Target = ash::extensions::khr::Surface;

    /// Get access to the `VK_KHR_surface` extension functions.
    fn deref(&self) -> &Self::Target {
        &self.functions
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkSurfaceKHR {:p}", self.handle);
        unsafe {
            self.functions.destroy_surface(self.handle, None);
        }
    }
}
