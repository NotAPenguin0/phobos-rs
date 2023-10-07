//! Wrappers around a `VkSwapchainKHR`

use std::ops::Deref;

use anyhow::Result;
use ash::vk;

use crate::{Device, Error, Instance, Surface, SurfaceSettings};
use crate::image::*;

#[derive(Debug)]
pub(crate) struct SwapchainImage {
    #[allow(dead_code)]
    pub image: Image,
    pub view: ImageView,
}

/// A swapchain is an abstraction of a presentation system. It handles buffering, VSync, and acquiring images
/// to render and present frames to.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Swapchain {
    /// Handle to the [`VkSwapchainKHR`](vk::SwapchainKHR) object.
    pub(super) handle: vk::SwapchainKHR,
    /// Swapchain images to present to.
    pub(super) images: Vec<SwapchainImage>,
    /// Swapchain image format.
    pub(super) format: vk::SurfaceFormatKHR,
    /// Present mode. The only mode that is required by the spec to always be supported is `FIFO`.
    pub(super) present_mode: vk::PresentModeKHR,
    /// Size of the swapchain images. This is effectively the window render area.
    pub(super) extent: vk::Extent2D,
    /// Vulkan extension functions operating on the swapchain.
    #[derivative(Debug = "ignore")]
    pub(super) functions: ash::extensions::khr::Swapchain,
}

impl Swapchain {
    /// Create a new swapchain.
    pub fn new(
        instance: &Instance,
        device: Device,
        surface_settings: &SurfaceSettings,
        surface: &Surface,
    ) -> Result<Self> {
        let format = choose_surface_format(surface_settings, surface)?;
        let present_mode = choose_present_mode(surface_settings, surface);
        let extent = choose_swapchain_extent(surface_settings, surface);

        let image_count = {
            let mut count = surface.capabilities().min_image_count + 1;
            // If a maximum is set, clamp to it
            if surface.capabilities().max_image_count != 0 {
                count = count.clamp(0, surface.capabilities().max_image_count);
            }
            count
        };

        let info = vk::SwapchainCreateInfoKHR::builder()
            .surface(unsafe { surface.handle() })
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .present_mode(present_mode)
            .min_image_count(image_count)
            .clipped(true)
            .pre_transform(surface.capabilities().current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .build();

        let functions = ash::extensions::khr::Swapchain::new(instance, unsafe { &device.handle() });
        let swapchain = unsafe { functions.create_swapchain(&info, None)? };

        #[cfg(feature = "log-objects")]
        trace!("Created new VkSwapchainKHR {swapchain:p}");

        let images: Vec<SwapchainImage> = unsafe { functions.get_swapchain_images(swapchain)? }
            .iter()
            .map(move |image| -> Result<SwapchainImage> {
                let image = Image::new_managed(
                    device.clone(),
                    *image,
                    format.format,
                    vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    },
                    1,
                    1,
                    vk::SampleCountFlags::TYPE_1,
                );
                // Create a trivial ImgView.
                let view = image.whole_view(vk::ImageAspectFlags::COLOR)?;
                // Bundle them together into an owning ImageView
                Ok(SwapchainImage {
                    image,
                    view,
                })
            })
            .collect::<Result<Vec<SwapchainImage>>>()?;
        Ok(Swapchain {
            handle: swapchain,
            format,
            present_mode,
            extent,
            images,
            functions,
        })
    }

    /// Get unsafe access to the underlying `VkSwapchainKHR` object.
    /// # Safety
    /// Any vulkan calls that mutate the swapchain may put the system in an undefined state.
    pub unsafe fn handle(&self) -> vk::SwapchainKHR {
        self.handle
    }

    pub(crate) fn images(&self) -> &[SwapchainImage] {
        self.images.as_slice()
    }

    /// Get the available present modes of this swapchain
    pub fn present_mode(&self) -> vk::PresentModeKHR {
        self.present_mode
    }

    /// Get the current size of this swapchain
    pub fn extent(&self) -> &vk::Extent2D {
        &self.extent
    }

    /// Get the surface format of this swapchain
    pub fn format(&self) -> vk::SurfaceFormatKHR {
        self.format
    }
}

impl Deref for Swapchain {
    type Target = ash::extensions::khr::Swapchain;

    fn deref(&self) -> &Self::Target {
        &self.functions
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkSwapchainKHR {:p}", self.handle);
        // We need to manually clear this list of images *before* deleting the swapchain,
        // otherwise, the imageview handles become invalid.
        self.images.clear();
        unsafe {
            self.functions.destroy_swapchain(self.handle, None);
        }
    }
}

fn choose_surface_format(
    surface_settings: &SurfaceSettings,
    surface: &Surface,
) -> Result<vk::SurfaceFormatKHR> {
    // In case requested format isn't found, try this. If that one isn't found we fall back to the first available format.
    const FALLBACK_FORMAT: vk::SurfaceFormatKHR = vk::SurfaceFormatKHR {
        format: vk::Format::B8G8R8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };

    if let Some(preferred_format) = surface_settings.surface_format {
        if surface.formats().contains(&preferred_format) {
            return Ok(preferred_format);
        }
    }
    if surface.formats().contains(&FALLBACK_FORMAT) {
        return Ok(FALLBACK_FORMAT);
    }

    surface
        .formats()
        .first()
        .copied()
        .ok_or_else(|| anyhow::Error::from(Error::NoSurfaceFormat))
}

fn choose_present_mode(
    surface_settings: &SurfaceSettings,
    surface: &Surface,
) -> vk::PresentModeKHR {
    if let Some(mode) = surface_settings.present_mode {
        if surface.present_modes().contains(&mode) {
            return mode;
        }
    }

    // VSync, guaranteed to be supported
    vk::PresentModeKHR::FIFO
}

fn choose_swapchain_extent(
    surface_settings: &SurfaceSettings,
    surface: &Surface,
) -> vk::Extent2D {
    if surface.capabilities().current_extent.width != u32::MAX {
        return surface.capabilities().current_extent;
    }

    vk::Extent2D {
        width: surface_settings.window.width().clamp(
            surface.capabilities().min_image_extent.width,
            surface.capabilities().max_image_extent.width,
        ),
        height: surface_settings.window.height().clamp(
            surface.capabilities().min_image_extent.height,
            surface.capabilities().max_image_extent.height,
        ),
    }
}
