use ash::vk;

use crate::core::window::WindowInterface;
use crate::QueueType;

/// Structure holding a queue with specific capabilities to request from the physical device.
#[derive(Debug)]
pub struct QueueRequest {
    /// Whether this queue should be dedicated if possible. For example, requesting a dedicated queue of type `QueueType::Transfer` will try to
    /// match this to a queue that does not have graphics or compute capabilities. On the other hand, requesting a dedicated graphics queue will not
    /// try to exclude transfer capabilities, as this is not possible per spec guarantees (a graphics queue must have transfer support)
    pub dedicated: bool,
    /// Capabilities that are requested from the queue.
    pub queue_type: QueueType
}

/// Minimum requirements for the GPU. This will be used to determine what physical device is selected.
#[derive(Default, Debug)]
pub struct GPURequirements {
    /// Whether a dedicated GPU is required. Setting this to true will discard integrated GPUs.
    pub dedicated: bool,
    /// Minimum amount of video memory required, in bytes. Note that this might count shared memory if RAM is shared.
    pub min_video_memory: usize,
    /// Minimum amount of dedicated video memory, in bytes. This only counts memory that is on the device.
    pub min_dedicated_video_memory: usize,
    /// Command queue types requested from the physical device.
    pub queues: Vec<QueueRequest>,
    /// Vulkan 1.0 features that are required from the physical device.
    pub features: vk::PhysicalDeviceFeatures,
    /// Vulkan 1.1 features that are required from the physical device.
    pub features_1_1: vk::PhysicalDeviceVulkan11Features,
    /// Vulkan 1.2 features that are required from the physical device.
    pub features_1_2: vk::PhysicalDeviceVulkan12Features,
    /// Vulkan 1.3 features that are required from the physical device.
    pub features_1_3: vk::PhysicalDeviceVulkan13Features,
    /// Vulkan extensions that should be present and enabled.
    pub device_extensions: Vec<String>,
}

/// Application settings used to initialize the phobos context.
#[derive(Debug)]
pub struct AppSettings<'a, Window> where Window: WindowInterface {
    /// Application name. Possibly displayed in debugging tools, task manager, etc.
    pub name: String,
    /// Application version.
    pub version: (u32, u32, u32),
    /// Enable Vulkan validation layers for additional debug output. For developing this should almost always be on.
    pub enable_validation: bool,
    /// Optionally a reference to an object implementing a windowing system. If this is not None, it will be used to create a [`VkSurfaceKHR`](vk::SurfaceKHR) to present to.
    pub window: Option<&'a Window>,
    /// Optionally a preferred surface format. This is ignored for a headless context. If set to None, a fallback surface format will be chosen.
    /// This format is `{BGRA8_SRGB, NONLINEAR_SRGB}` if it is available. Otherwise, the format is implementation-defined.
    pub surface_format: Option<vk::SurfaceFormatKHR>,
    /// Optionally a preferred present mode. This is ignored for a headless context. If set to None, this will fall back to
    /// `vk::PresentModeKHR::FIFO`, as this is guaranteed to always be supported.
    pub present_mode: Option<vk::PresentModeKHR>,
    /// Minimum requirements the selected physical device should have.
    pub gpu_requirements: GPURequirements,
    /// Maximum per-frame size of scratch vertex buffer objects.
    pub scratch_vbo_size: vk::DeviceSize,
    /// Maximum per-frame size of scratch index buffer objects.
    pub scratch_ibo_size: vk::DeviceSize,
    /// Maximum per-frame size of scratch uniform buffer objects.
    pub scratch_ubo_size: vk::DeviceSize,
    /// Maximum per-frame size of scratch shader storage buffer objects.
    pub scratch_ssbo_size: vk::DeviceSize,
}

impl<'a, Window> Default for AppSettings<'a, Window> where Window: WindowInterface {
    fn default() -> Self {
        AppSettings {
            name: String::from(""),
            version: (0, 0, 0),
            enable_validation: false,
            window: None,
            surface_format: None,
            present_mode: None,
            gpu_requirements: GPURequirements::default(),
            scratch_vbo_size: 1,
            scratch_ibo_size: 1,
            scratch_ubo_size: 1,
            scratch_ssbo_size: 1,
        }
    }
}

pub struct AppBuilder<'a, Window> where Window: WindowInterface {
    inner: AppSettings<'a, Window>,
}

impl<'a, Window> AppBuilder<'a, Window> where Window: WindowInterface {
    pub fn new() -> Self {
        AppBuilder { inner: AppSettings::default() }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.inner.name = name.into();
        self
    }

    pub fn version(mut self, ver: (u32, u32, u32)) -> Self {
        self.inner.version = ver;
        self
    }

    pub fn validation(mut self, val: bool) -> Self {
        self.inner.enable_validation = val;
        self
    }

    pub fn window(mut self, window: &'a Window) -> Self {
        self.inner.window = Some(window);
        self
    }

    pub fn surface_format(mut self, format: vk::SurfaceFormatKHR) -> Self {
        self.inner.surface_format = Some(format);
        self
    }

    pub fn present_mode(mut self, mode: vk::PresentModeKHR) -> Self {
        self.inner.present_mode = Some(mode);
        self
    }

    pub fn gpu(mut self, gpu: GPURequirements) -> Self {
        self.inner.gpu_requirements = gpu;
        self
    }

    pub fn scratch_size(mut self, size: vk::DeviceSize) -> Self {
        self.inner.scratch_vbo_size = size;
        self.inner.scratch_ibo_size = size;
        self.inner.scratch_ubo_size = size;
        self.inner.scratch_ssbo_size = size;

        self
    }

    pub fn build(self) -> AppSettings<'a, Window> {
        self.inner
    }
}