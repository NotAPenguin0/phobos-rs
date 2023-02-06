use std::sync::{Arc, Mutex};
use ash::vk;
use gpu_allocator::{MemoryLocation, vulkan as vk_alloc};
use gpu_allocator::vulkan::{AllocationCreateDesc, Allocator};
use anyhow::Result;

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
    #[derivative(Debug="ignore")]
    pub allocator: Option<Arc<Mutex<Allocator>>>,
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
#[derivative(Debug, Hash, PartialEq, Eq)]
pub struct ImgView {
    /// Reference to the [`VkDevice`](vk::Device)
    #[derivative(Debug="ignore")]
    #[derivative(Hash="ignore")]
    #[derivative(PartialEq="ignore")]
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
    // TODO: Allow specifying an initial layout for convenience
    // TODO: Full wrapper around the allocator for convenience
    /// Create a new simple [`VkImage`] and allocate some memory to it.
    pub fn new(device: Arc<Device>, alloc: Arc<Mutex<Allocator>>, width: u32, height: u32, usage: vk::ImageUsageFlags, format: vk::Format) -> Result<Self> {
        let sharing_mode = if usage.intersects(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
            vk::SharingMode::EXCLUSIVE
        }
        else {
            vk::SharingMode::CONCURRENT
        };

        let handle = unsafe { device.create_image(&vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode,
            queue_family_index_count: if sharing_mode == vk::SharingMode::CONCURRENT { device.queue_families.len() as u32 } else { 0 },
            p_queue_family_indices: if sharing_mode == vk::SharingMode::CONCURRENT { device.queue_families.as_ptr() } else { std::ptr::null() },
            initial_layout: vk::ImageLayout::UNDEFINED,
        }, None)? };

        let requirements = unsafe { device.get_image_memory_requirements(handle) };

        let memory = alloc.lock().or_else(|_| Err(anyhow::Error::from(Error::PoisonError)))?.allocate(&AllocationCreateDesc {
            name: "image_",
            requirements,
            // TODO: Proper memory location configuration
            location: MemoryLocation::GpuOnly,
            linear: false,
        })?;

        unsafe { device.bind_image_memory(handle, memory.memory(), memory.offset())?; }

        Ok(Self {
            device: device.clone(),
            allocator: Some(alloc.clone()),
            handle,
            format,
            size: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            layers: 1,
            mip_levels: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            memory: Some(memory),
        })
    }

    /// Construct a trivial [`ImageView`] from this [`Image`]. This is an image view that views the
    /// entire image subresource.
    /// <br>
    /// <br>
    /// # Lifetime
    /// The returned [`ImageView`] is valid as long as `self` is valid.
    pub fn view(&self, aspect: vk::ImageAspectFlags) -> Result<ImageView> {
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
            if let Some(memory) = &mut self.memory {
                let memory = std::mem::take(memory);
                if let Some(allocator) = &mut self.allocator {
                    allocator.lock().unwrap().free(memory).unwrap();
                }
            }
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