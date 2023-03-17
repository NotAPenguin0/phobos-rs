use std::sync::Arc;
use ash::vk;
use crate::{AppSettings, Device, Error, Surface, VkInstance, WindowInterface};
use crate::image::*;
use anyhow::Result;

#[derive(Debug)]
pub(crate) struct SwapchainImage {
    #[allow(dead_code)]
    pub image: Image,
    pub view: ImageView
}

/// A swapchain is an abstraction of a presentation system. It handles buffering, VSync, and acquiring images
/// to render and present frames to.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Swapchain {
    /// Handle to the [`VkSwapchainKHR`](vk::SwapchainKHR) object.
    pub(crate) handle: vk::SwapchainKHR,
    /// Swapchain images to present to.
    pub(crate) images: Vec<SwapchainImage>,
    /// Swapchain image format.
    pub format: vk::SurfaceFormatKHR,
    /// Present mode. The only mode that is required by the spec to always be supported is `FIFO`.
    pub present_mode: vk::PresentModeKHR,
    /// Size of the swapchain images. This is effectively the window render area.
    pub extent: vk::Extent2D,
    /// Vulkan extension functions operating on the swapchain.
    #[derivative(Debug="ignore")]
    pub(crate) functions: ash::extensions::khr::Swapchain,
}

impl Swapchain {
    /// Create a new swapchain.
    pub fn new<Window: WindowInterface>(instance: &VkInstance, device: Arc<Device>, settings: &AppSettings<Window>, surface: &Surface) -> Result<Self> {
        let format = choose_surface_format(settings, surface)?;
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

        let functions = ash::extensions::khr::Swapchain::new(&instance.instance, &device.handle);
        let swapchain = unsafe { functions.create_swapchain(&info, None)? };

        let images: Vec<SwapchainImage> = unsafe { functions.get_swapchain_images(swapchain)? }
            .iter()
            .map(move |image| -> Result<SwapchainImage> {
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
                    memory: None,
                    allocator: None
                };
                // Create a trivial ImgView.
                let view = image.view(vk::ImageAspectFlags::COLOR)?;
                // Bundle them together into an owning ImageView
                Ok(SwapchainImage {
                    image,
                    view
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

    /// Unsafe access to the swapchain extension functions.
    pub unsafe fn loader(&self) -> ash::extensions::khr::Swapchain {
        self.functions.clone()
    }

    /// Unsafe access to the underlying vulkan handle.
    pub unsafe fn handle(&self) -> vk::SwapchainKHR {
        self.handle
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        // We need to manually clear this list of images *before* deleting the swapchain,
        // otherwise, the imageview handles become invalid.
        self.images.clear();
        unsafe { self.functions.destroy_swapchain(self.handle, None); }
    }
}

fn choose_surface_format<Window: WindowInterface>(settings: &AppSettings<Window>, surface: &Surface) -> Result<vk::SurfaceFormatKHR> {
    // In case requested format isn't found, try this. If that one isn't found we fall back to the first available format.
    const FALLBACK_FORMAT: vk::SurfaceFormatKHR = vk::SurfaceFormatKHR {
        format: vk::Format::B8G8R8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR
    };

    if let Some(preferred_format) = settings.surface_format {
        if surface.formats.contains(&preferred_format) { return Ok(preferred_format); }
    }
    if surface.formats.contains(&FALLBACK_FORMAT) { return Ok(FALLBACK_FORMAT); }

    surface.formats.first().copied().ok_or(anyhow::Error::from(Error::NoSurfaceFormat))
}

fn choose_present_mode<Window : WindowInterface>(settings: &AppSettings<Window>, surface: &Surface) -> vk::PresentModeKHR {
    if let Some(mode) = settings.present_mode {
        if surface.present_modes.contains(&mode) { return mode; }
    }
    // VSync, guaranteed to be supported
    vk::PresentModeKHR::FIFO
}

fn choose_swapchain_extent<Window: WindowInterface>(settings: &AppSettings<Window>, surface: &Surface) -> vk::Extent2D {
    if surface.capabilities.current_extent.width != u32::MAX {
        return surface.capabilities.current_extent;
    }

    vk::Extent2D {
        width: settings.window.unwrap().width().clamp(surface.capabilities.min_image_extent.width, surface.capabilities.max_image_extent.width),
        height: settings.window.unwrap().height().clamp(surface.capabilities.min_image_extent.height, surface.capabilities.max_image_extent.height)
    }
}
