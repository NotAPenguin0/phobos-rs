//! Provides utilities to manage [`VkImage`](vk::Image) and [`VkImageView`](vk::ImageView) objects.
//!
//! # Images
//!
//! Images are managed through the [`Image`] struct. These images are usually backed by a memory allocation, except when
//! they are swapchain images managed by the OS.
//!
//! # Image views
//!
//! Using [`Image::view`] you can create an [`ImageView`] that covers the entire image. Note that [`ImageView`] is in fact an
//! `Arc<ImgView>`. The relationship between [`ImageView`] and [`ImgView`] is similar to `String` vs `str`, except that an
//! [`ImgView`] also owns a full Vulkan resource. For this reason, we wrap it in a reference-counted `Arc` so we can safely treat it as if it were
//! a `str` to a `String`. Most API functions will ask for an [`ImageView`].

use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use ash::vk;
use ash::vk::Handle;

use crate::{Allocation, Allocator, DefaultAllocator, Device, MemoryType};
use crate::core::traits::{AsRaw, Nameable};

/// Abstraction over a [`VkImage`](vk::Image). Stores information about size, format, etc. Additionally couples the image data together
/// with a memory allocation.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Image<A: Allocator = DefaultAllocator> {
    /// Reference to the [`VkDevice`](vk::Device).
    #[derivative(Debug = "ignore")]
    device: Device,
    /// [`VkImage`](vk::Image) handle.
    handle: vk::Image,
    /// GPU memory allocation. If this is None, then the image is not owned by our system (for example a swapchain image) and should not be
    /// destroyed.
    #[derivative(Debug = "ignore")]
    memory: Option<A::Allocation>,
    /// Image format
    format: vk::Format,
    /// Size of the image. Note that this is 3D because 3D images also exist.
    /// For 2D images, `size.depth == 1`.
    size: vk::Extent3D,
    /// Number of array layers.
    layers: u32,
    /// Number of mip levels.
    mip_levels: u32,
    /// Number of samples. Useful for multisampled attachments
    samples: vk::SampleCountFlags,
}

unsafe impl<A: Allocator> Send for Image<A> {}

unsafe impl<A: Allocator> Sync for Image<A> {}

/// Abstraction over a [`VkImageView`](vk::ImageView). Most functions operating on images will expect these instead of raw owning [`Image`] structs.
/// Image views can refer to one or more array layers or mip levels of an image. Given the right extension they can also interpret the image contents in a different
/// format.
#[derive(Derivative)]
#[derivative(Debug, Hash, PartialEq, Eq)]
pub struct ImgView {
    /// Reference to the [`VkDevice`](vk::Device)
    #[derivative(Debug = "ignore")]
    #[derivative(Hash = "ignore")]
    #[derivative(PartialEq = "ignore")]
    device: Device,
    /// [`VkImageView`](vk::ImageView) handle
    handle: vk::ImageView,
    /// Reference to the [`VkImage`](vk::Image).
    image: vk::Image,
    /// [`VkFormat`](vk::Format) this image view uses. Note that this could be a different format than the owning [`Image`]
    format: vk::Format,
    /// Number of samples.
    samples: vk::SampleCountFlags,
    /// Image aspect.
    aspect: vk::ImageAspectFlags,
    /// Size of the corresponding image region.
    size: vk::Extent3D,
    /// First mip level in the viewed mip range.
    base_level: u32,
    /// Amount of mip levels in the viewed mip range.
    level_count: u32,
    /// First array layer in the viewed array layer range.
    base_layer: u32,
    /// Amount of array layers in the viewed array layer range.
    layer_count: u32,
    /// Unique ID for this image view, because vk handles may be reused.
    id: u64,
}

/// Settings that describe how an image view should be created from its Image
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct ImageViewCreateInfo {
    /// Image aspect
    pub aspect: vk::ImageAspectFlags,
    /// View type of the image view
    pub view_type: vk::ImageViewType,
    /// Base mip level (starting mip level)
    pub base_mip_level: u32,
    /// Number of mip levels to use in the view (set to None to use the rest)
    pub level_count: Option<u32>,
    /// Base layer (starting layer)
    pub base_layer: u32,
    /// Number of layers to use in the view (set to None to use the rest)
    pub layers: Option<u32>,
}


/// Reference-counted version of [`ImgView`].
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ImageView(pub Arc<ImgView>);

impl Deref for ImageView {
    type Target = Arc<ImgView>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl Send for ImageView {}

unsafe impl Sync for ImageView {}

/// Settings that describe how an image should be created
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct ImageCreateInfo {
    /// Width in pixels of the image
    pub width: u32,
    /// Height in pixels of the image
    pub height: u32,
    /// Depth in pixels of the image, set to 1 for 2D images.
    pub depth: u32,
    /// Image usage flags
    pub usage: vk::ImageUsageFlags,
    /// Pixel format of the image
    pub format: vk::Format,
    /// MSAA samples
    pub samples: vk::SampleCountFlags,
    /// Number of mip levels. Set to 1 if not using mipmapping
    pub mip_levels: u32,
    /// Number of array layers. Set to 1 for non-array textures.
    pub layers: u32,
}

impl<A: Allocator> Image<A> {
    /// Create a new simple [`VkImage`] and allocate some memory to it.
    pub fn new(
        device: Device,
        alloc: &mut A,
        info: ImageCreateInfo,
    ) -> Result<Self> {
        let sharing_mode = if device.is_single_queue()
            || info.usage.intersects(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ) {
            vk::SharingMode::EXCLUSIVE
        } else {
            vk::SharingMode::CONCURRENT
        };

        let image_type = if info.height == 1 && info.depth == 1 {
            vk::ImageType::TYPE_1D
        } else if info.depth > 1 {
            vk::ImageType::TYPE_3D
        } else if info.height > 1 {
            vk::ImageType::TYPE_2D
        } else {
            return Err(anyhow::anyhow!("Image extents invalid"));
        };

        let extent = vk::Extent3D {
            width: info.width,
            height: info.height,
            depth: info.depth
        };
        let handle = unsafe {
            device.create_image(
                &vk::ImageCreateInfo {
                    s_type: vk::StructureType::IMAGE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: Default::default(),
                    image_type,
                    format: info.format,
                    extent,
                    mip_levels: info.mip_levels,
                    array_layers: info.layers,
                    samples: info.samples,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: info.usage,
                    sharing_mode,
                    queue_family_index_count: if sharing_mode == vk::SharingMode::CONCURRENT {
                        device.queue_families().len() as u32
                    } else {
                        0
                    },
                    p_queue_family_indices: if sharing_mode == vk::SharingMode::CONCURRENT {
                        device.queue_families().as_ptr()
                    } else {
                        std::ptr::null()
                    },
                    initial_layout: vk::ImageLayout::UNDEFINED,
                },
                None,
            )?
        };
        #[cfg(feature = "log-objects")]
        trace!("Created new VkImage {handle:p}");

        let requirements = unsafe { device.get_image_memory_requirements(handle) };

        // TODO: Proper memory location configuration
        let memory = alloc.allocate("image_", &requirements, MemoryType::GpuOnly)?;
        unsafe {
            device.bind_image_memory(handle, memory.memory(), memory.offset())?;
        }

        Ok(Self {
            device,
            handle,
            format: info.format,
            size: extent,
            layers: info.layers,
            mip_levels: info.mip_levels,
            samples: info.samples,
            memory: Some(memory),
        })
    }

    pub(crate) fn new_managed(
        device: Device,
        handle: vk::Image,
        format: vk::Format,
        size: vk::Extent3D,
        layers: u32,
        mip_levels: u32,
        samples: vk::SampleCountFlags,
    ) -> Self {
        Self {
            device,
            handle,
            memory: None,
            format,
            size,
            layers,
            mip_levels,
            samples,
        }
    }

    /// Construct a trivial [`ImageView`] from this [`Image`]. This is an image view that views the
    /// whole image subresource.
    /// * `aspect` - The image aspect flags that will be used to create the image view
    /// <br>
    /// <br>
    /// # Lifetime
    /// The returned [`ImageView`] is valid as long as `self` is valid.
    pub fn whole_view(&self, aspect: vk::ImageAspectFlags) -> Result<ImageView> {
        let view_type = if self.size.width == 1 && self.size.width == 1 && self.size.depth == 1 {
            vk::ImageViewType::TYPE_1D
        } else if self.size.depth > 1 {
            vk::ImageViewType::TYPE_3D
        } else if self.size.height > 1 {
            vk::ImageViewType::TYPE_2D
        } else {
            anyhow::bail!("Image extents invalid");
        };

        self.view(
            ImageViewCreateInfo {
                aspect,
                view_type,
                base_mip_level: 0,
                level_count: None,
                base_layer: 0,
                layers: None,
            },
        )
    }

    /// Construct an [`ImageView`] from this [`Image`]. This is an image view that views the
    /// image subresource specified by the given arguments.
    /// * `aspect` - The image aspect flags that will be used to create the image view
    /// * `view_type` - The image type of the view. Must be supported by the image type of the current image
    /// * `base_mip_level` - Starting mip level that the view sub-resource will look from
    /// * `level_count` - End mip level that the view sub-resource will look to. Set to None to get the remainder of the mip levels
    /// * `base_layer` - Starting layer that the view sub-resource will look from
    /// * `layers` - End layer that the view sub-resource will look to. Set to None to get the remainder of the layer
    /// <br>
    /// <br>
    /// # Lifetime
    /// The returned [`ImageView`] is valid as long as `self` is valid.
    pub fn view(
        &self,
        create_info: ImageViewCreateInfo,
    ) -> Result<ImageView> {
        let ImageViewCreateInfo {
            aspect,
            view_type,
            base_mip_level,
            level_count,
            base_layer,
            layers
        } = create_info;

        // TODO: Validate args
        let info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            image: self.handle,
            view_type,
            format: self.format,
            components: vk::ComponentMapping::default(),
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level,
                level_count: match level_count {
                    Some(count) => count,
                    None => self.mip_levels - base_mip_level,
                },
                base_array_layer: base_layer,
                layer_count: match layers {
                    Some(count) => count,
                    None => self.layers - base_layer,
                },
            },
        };

        let view_handle = unsafe { self.device.create_image_view(&info, None)? };
        #[cfg(feature = "log-objects")]
        trace!("Created new VkImageView {view_handle:p}");
        Ok(ImageView(Arc::new(ImgView {
            device: self.device.clone(),
            handle: view_handle,
            image: self.handle,
            format: self.format,
            samples: self.samples,
            aspect,
            size: self.size,
            base_level: base_mip_level,
            level_count: info.subresource_range.level_count,
            base_layer,
            layer_count: info.subresource_range.layer_count,
            id: ImgView::get_new_id(),
        })))
    }

    /// Whether this image resource is owned by the application or an external manager (such as the swapchain).
    pub fn is_owned(&self) -> bool {
        self.memory.is_some()
    }

    /// Get unsafe access to the underlying `VkImage` handle.
    /// # Safety
    /// Any vulkan calls that mutate this image's state may put the system into an undefined state.
    pub unsafe fn handle(&self) -> vk::Image {
        self.handle
    }

    /// Get the image format
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Get the image size
    pub fn size(&self) -> vk::Extent3D {
        self.size
    }

    /// Get the image width
    pub fn width(&self) -> u32 {
        self.size().width
    }

    /// Get the image height
    pub fn height(&self) -> u32 {
        self.size().height
    }

    /// Get the image depth (for 2D images, this is 1)
    pub fn depth(&self) -> u32 {
        self.size().depth
    }

    /// Get the number of layers in the image
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// Get the number of mip levels in the image
    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    /// Get the number of MSAA samples for this image.
    pub fn samples(&self) -> vk::SampleCountFlags {
        self.samples
    }
}

unsafe impl AsRaw for Image {
    unsafe fn as_raw(&self) -> u64 {
        self.handle().as_raw()
    }
}

impl Nameable for Image {
    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::IMAGE;
}

impl<A: Allocator> Drop for Image<A> {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkImage {:p}", self.handle);
        if self.is_owned() {
            unsafe {
                self.device.destroy_image(self.handle, None);
            }
        }
    }
}

impl ImgView {
    fn get_new_id() -> u64 {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    /// Returns the subresource range of the original image that this image view covers.
    pub fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect,
            base_mip_level: self.base_level,
            level_count: self.level_count,
            base_array_layer: self.base_layer,
            layer_count: self.layer_count,
        }
    }

    /// Get unsafe access to the underlying `VkImageView` handle.
    /// # Safety
    /// Any vulkan calls that mutate this image view's state may put the system in an undefined state.
    pub unsafe fn handle(&self) -> vk::ImageView {
        self.handle
    }

    /// Get unsafe access to the underlying `VkImage` handle.
    /// # Safety
    /// Any vulkan calls that mutate this image's state may put the system in an undefined state.
    pub unsafe fn image(&self) -> vk::Image {
        self.image
    }

    /// Get the image format
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Get the number of MSAA samples for this image.
    pub fn samples(&self) -> vk::SampleCountFlags {
        self.samples
    }

    /// Get the image aspect that this view was built from
    pub fn aspect(&self) -> vk::ImageAspectFlags {
        self.aspect
    }

    /// Get the image size
    pub fn size(&self) -> vk::Extent3D {
        self.size
    }

    /// Get the image width
    pub fn width(&self) -> u32 {
        self.size().width
    }

    /// Get the image height
    pub fn height(&self) -> u32 {
        self.size().height
    }

    /// Get the image depth. For 2D images, this is 1
    pub fn depth(&self) -> u32 {
        self.size().depth
    }

    /// Get the first layer this view was made from
    pub fn base_layer(&self) -> u32 {
        self.base_layer
    }

    /// Get the number of layers this view was made from
    pub fn layer_count(&self) -> u32 {
        self.layer_count
    }

    /// Get the first mip level this view was made from
    pub fn base_level(&self) -> u32 {
        self.base_level
    }

    /// Get the number of mip levels this view was made from
    pub fn level_count(&self) -> u32 {
        self.level_count
    }

    /// Get the unique ID of this image view. Every unique image view
    /// is guaranteed to have its own unique id.
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl Drop for ImgView {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkImageView {:p}", self.handle);
        unsafe {
            self.device.destroy_image_view(self.handle, None);
        }
    }
}
