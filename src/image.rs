use std::sync::Arc;
use ash::{Device, vk};
use gpu_allocator::vulkan as vk_alloc;

#[derive(Debug, Default, Copy, Clone)]
pub struct ImageViewInfo {
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
    pub layer_count: u32,
}

/// Abstraction over a [`VkImage`](vk::Image). Stores information about size, format, etc. Additionally couples the image data together
/// with a memory allocation.
pub struct Image {
    /// Reference to the [`VkDevice`](vk::Device).
    device: Arc<Device>,
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

/// Abstraction over a [`VkImageView`](vk::ImageView). Most functions operating on images will expect these instead of raw
/// owning [`Image`](Image) structs. Image views can refer to one or more array layers or mip levels of an image.
/// Given the right extension they can also reinterpret the image contents with a different format.
/// <br>
/// <br>
/// # Lifetime
/// - An instance of this struct must not live longer than the [`Image`] it refers to.
pub struct ImgView<'i> {
    /// Reference to the [`VkDevice`](vk::Device).
    device: Arc<Device>,
    /// Reference to the [`VkImage`](vk::Image) this image view points to.
    pub image: &'i Image,
    /// [`VkImageView`](vk::ImageView) handle.
    pub handle: vk::ImageView,
    /// Whether this ImgView owns the [`VkImageView`](vk::ImageView) it holds.
    pub owned: bool,
    /// Information about this image view's properties.
    pub info: ImageViewInfo,
}

/// Abstraction over [`VkImageView`](vk::ImageView). An [`ImageView`] owns both the [`VkImageView`](vk::ImageView) and the [`VkImage`](vk::Image).
/// It can be dereferenced into a non-owning [`ImgView`].
pub struct ImageView {
    device: Arc<Device>,
    pub image: Image,
    pub view: vk::ImageView,
    pub info: ImageViewInfo,
}

impl Image {
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

impl<'i> ImgView<'i> {
    /// Construct a non-owning [`ImgView`] from an [`ImageView`] object.
    pub fn from_owned(view: &'i ImageView) -> Self {
        ImgView {
            device: view.device.clone(),
            image: &view.image,
            handle: view.view,
            owned: false,
            info: view.info
        }
    }
}

impl<'i> Drop for ImgView<'i> {
    fn drop(&mut self) {
        if self.owned {
            unsafe { self.device.destroy_image_view(self.handle, None); }
        }
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe { self.device.destroy_image_view(self.view, None); }
    }
}