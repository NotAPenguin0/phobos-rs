use std::sync::Arc;
use ash::vk;
use gpu_allocator::vulkan as vk_alloc;

use crate::{Device, Error};

/// Abstraction over a [`VkImage`](vk::Image). Stores information about size, format, etc. Additionally couples the image data together
/// with a memory allocation.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Image {
    // TODO: Reconsider member visibility
    /// Reference to the [`VkDevice`](vk::Device).
    #[derivative(Debug="ignore")]
    pub device: Arc<Device>,
    /// [`VkImage`](vk::Image) handle.
    pub handle: vk::Image,
    /// Image format
    pub format: vk::Format,
    /// Size of the image. Note that this is 3D because 3D images also exist.
    /// For 2D images, `size.depth == 1`.
    pub size: vk::Extent3D,
    /// Number of array layers.
    pub layers: u32,
    /// Number of mip levels.
    pub mip_levels: u32,
    /// Number of samples. Useful for multisampled attachments
    pub samples: vk::SampleCountFlags,
    /// GPU memory allocation. If this is None, then the image is not owned by our system (for example a swapchain image) and should not be
    /// destroyed.
    pub memory: Option<vk_alloc::Allocation>,
}

/// Abstraction over a [`VkImageView`](vk::ImageView). Most functions operating on images will expect these instead of raw owning [`Image`] structs.
/// Image views can refer to one or more array layers or mip levels of an image. Given the right extension they can also interpret the image contents in a different
/// format.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct ImgView {
    /// Reference to the [`VkDevice`](vk::Device)
    #[derivative(Debug="ignore")]
    pub device: Arc<Device>,
    /// [`VkImageView`](vk::ImageView) handle
    pub handle: vk::ImageView,
    /// Reference to the [`VkImage`](vk::Image).
    pub(crate) image: vk::Image,
    /// [`VkFormat`](vk::Format) this image view uses. Note that this could be a different format than the owning [`Image`]
    pub format: vk::Format,
    /// Number of samples.
    pub samples: vk::SampleCountFlags,
    /// Image aspect.
    pub aspect: vk::ImageAspectFlags,
    /// Size of the corresponding image region.
    pub size: vk::Extent3D,
    /// First mip level in the viewed mip range.
    pub base_level: u32,
    /// Amount of mip levels in the viewed mip range.
    pub level_count: u32,
    /// First array layer in the viewed array layer range.
    pub base_layer: u32,
    /// Amount of array layers in the viewed array layer range.
    pub layer_count: u32
}

/// Reference-counted version of [`ImgView`].
pub type ImageView = Arc<ImgView>;

impl Image {
    /// Construct a trivial [`ImageView`] from this [`Image`]. This is an image view that views the
    /// entire image subresource.
    /// <br>
    /// <br>
    /// # Lifetime
    /// The returned [`ImageView`] is valid as long as `self` is valid.
    pub fn view(&self, aspect: vk::ImageAspectFlags) -> Result<ImageView, Error> {
        let info = vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::TYPE_2D) // TODO: 3D images, cubemaps, etc
            .format(self.format)
            .image(self.handle)
            .subresource_range(vk::ImageSubresourceRange::builder()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(self.mip_levels)
                .base_array_layer(0)
                .layer_count(self.layers)
                .build()
            )
            .build();
        let view_handle = unsafe { self.device.create_image_view(&info, None)? };
        Ok(ImageView::new(ImgView{
            device: self.device.clone(),
            handle: view_handle,
            image: self.handle.clone(),
            format: self.format,
            samples: self.samples,
            aspect,
            size: self.size,
            base_level: 0,
            level_count: self.mip_levels,
            base_layer: 0,
            layer_count: self.layers
        }))
    }

    /// Whether this image resource is owned by the application or an external manager (such as the swapchain).
    pub fn is_owned(&self) -> bool {
        self.memory.is_some()
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        if self.is_owned() {
            unsafe { self.device.destroy_image(self.handle, None); }
        }
    }
}

impl ImgView {
    pub fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect,
            base_mip_level: self.base_level,
            level_count: self.level_count,
            base_array_layer: self.base_layer,
            layer_count: self.layer_count
        }
    }
}

impl Drop for ImgView {
    fn drop(&mut self) {
        unsafe { self.device.destroy_image_view(self.handle, None); }
    }
}