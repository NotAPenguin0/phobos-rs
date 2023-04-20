//! Exposes all structs needed to store initialization parameters.

use ash::vk;

use crate::core::queue::QueueType;
use crate::WindowInterface;

/// Structure holding a queue with specific capabilities to request from the physical device.
///
/// See also: [`GPURequirements`](crate::GPURequirements), [`QueueType`](crate::core::queue::QueueType), [`Queue`](crate::core::queue::Queue)
///
/// # Example
/// ```
/// # use phobos::*;
/// let transfer = QueueRequest {
///     dedicated: true,
///     queue_type: QueueType::Transfer
/// };
///
/// let graphics = QueueRequest {
///     dedicated: false,
///     queue_type: QueueType::Graphics
/// };
/// ```
#[derive(Debug)]
pub struct QueueRequest {
    /// Whether this queue should be dedicated if possible. For example, requesting a dedicated queue of type [`QueueType::Transfer`] will try to
    /// match this to a queue that does not have graphics or compute capabilities.
    ///
    /// Note that some queues might still expose other features even if they are dedicated. Most notably, graphics and compute queues will always
    /// support transfer operations (this is guaranteed by the Vulkan spec).
    ///
    /// Generally, there is little reason to request a dedicated graphics queue. Transfer and compute are possibly useful to have dedicated queues
    /// for.
    pub dedicated: bool,
    /// Capabilities that are requested from the queue. Graphics and compute both imply transfer capabilities.
    ///
    /// The main graphics queue usually also has compute support (from the Vulkan spec: If there is a queue family that exposes
    /// graphics capabilities, there is at least one queue family that exposes both graphics and compute capabilities).
    pub queue_type: QueueType,
}

/// Minimum requirements for the GPU. This will be used to determine what physical device is selected, and enable
/// optional Vulkan features and extensions.
/// # Example
/// ```
/// # use phobos::*;
/// let mut requirements = GPURequirements {
///     dedicated: true,
///     min_video_memory: 1024  * 1024 * 1024,
///     min_dedicated_video_memory: 1024  * 1024 * 1024,
///     queues: vec![
///         QueueRequest {
///             dedicated: false,
///             queue_type: QueueType::Graphics,
///         }
///     ],
///     ..Default::default()
/// };
/// // Enable an optional Vulkan feature.
/// requirements.features.sampler_anisotropy = vk::TRUE;
/// ```
#[derive(Default, Debug)]
pub struct GPURequirements {
    /// Whether a dedicated GPU is required. Setting this to true will discard integrated GPUs.
    pub dedicated: bool,
    /// Minimum amount of video memory required, in bytes. Note that this might count shared memory if RAM is shared.
    pub min_video_memory: usize,
    /// Minimum amount of dedicated video memory, in bytes. This only counts memory that is on the device.
    pub min_dedicated_video_memory: usize,
    /// Command queue types requested from the physical device. For more information, see
    /// [`QueueRequest`]
    pub queues: Vec<QueueRequest>,
    /// Optional Vulkan 1.0 features that are required from the physical device.
    /// See also: [`VkPhysicalDeviceFeatures`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceFeatures.html)
    pub features: vk::PhysicalDeviceFeatures,
    /// Optional Vulkan 1.1 features that are required from the physical device.
    /// See also: [`VkPhysicalDeviceVulkan11Features`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceVulkan11Features.html)
    pub features_1_1: vk::PhysicalDeviceVulkan11Features,
    /// Optional Vulkan 1.2 features that are required from the physical device.
    /// See also: [`VkPhysicalDeviceVulkan12Features`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceVulkan12Features.html)
    pub features_1_2: vk::PhysicalDeviceVulkan12Features,
    /// Optional Vulkan 1.3 features that are required from the physical device.
    /// See also: [`VkPhysicalDeviceVulkan13Features`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceVulkan13Features.html)
    pub features_1_3: vk::PhysicalDeviceVulkan13Features,
    /// Vulkan device extensions that should be present and enabled.
    pub device_extensions: Vec<String>,
}

/// Application settings used to initialize the phobos context.
#[derive(Debug)]
pub struct AppSettings<'a, Window: WindowInterface> {
    /// Application name. Possibly displayed in debugging tools, task manager, etc.
    pub name: String,
    /// Application version.
    pub version: (u32, u32, u32),
    /// Enable Vulkan validation layers for additional debug output. For developing this should almost always be on.
    pub enable_validation: bool,
    /// Optionally a reference to an object implementing a windowing system. If this is not `None`, it will be used to create a
    /// [`VkSurfaceKHR`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSurfaceKHR.html) to present to.
    pub window: Option<&'a Window>,
    /// Optionally a preferred surface format. This is ignored for a headless context. If set to None, a fallback surface format will be chosen.
    /// This format is `{BGRA8_SRGB, NONLINEAR_SRGB}` if it is available. Otherwise, the format is implementation-defined.
    pub surface_format: Option<vk::SurfaceFormatKHR>,
    /// Optionally a preferred present mode. This is ignored for a headless context. If set to None, this will fall back to
    /// [`VK_PRESENT_MODE_FIFO_KHR`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPresentModeKHR.html),
    /// as this is guaranteed to always be supported.
    pub present_mode: Option<vk::PresentModeKHR>,
    /// Minimum requirements the selected physical device should have.
    pub gpu_requirements: GPURequirements,
    /// Maximum per-frame size of scratch vertex buffer objects. This is the size of each
    /// frame context's [`ScratchAllocator`](crate::ScratchAllocator)
    pub scratch_vbo_size: vk::DeviceSize,
    /// Maximum per-frame size of scratch index buffer objects. This is the size of each
    /// frame context's [`ScratchAllocator`](crate::ScratchAllocator)
    pub scratch_ibo_size: vk::DeviceSize,
    /// Maximum per-frame size of scratch uniform buffer objects. This is the size of each
    /// frame context's [`ScratchAllocator`](crate::ScratchAllocator)
    pub scratch_ubo_size: vk::DeviceSize,
    /// Maximum per-frame size of scratch shader storage buffer objects. This is the size of each
    /// frame context's [`ScratchAllocator`](crate::ScratchAllocator)
    pub scratch_ssbo_size: vk::DeviceSize,
    /// Whether to enable raytracing extensions.
    pub raytracing: bool,
}

impl<'a, Window: WindowInterface> Default for AppSettings<'a, Window> {
    /// Create some default app settings. One thing to note is that this sets scratch allocator size to 1 byte.
    /// This is because passing 0 as the size of a scratch allocator is invalid, and wrapping them in `Option<T>` is
    /// also not great for convenience.
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
            raytracing: false,
        }
    }
}

/// The app builder is a convenience struct to easily create [`AppSettings`](crate::AppSettings).
///
/// For information about each of the fields, see [`AppSettings`](crate::AppSettings)
/// # Example
/// ```
/// # use phobos::*;
/// # use phobos::wsi::window::HeadlessWindowInterface;
///
/// // No window, so we have to specify the headless window interface type.
/// let info: AppSettings<_> = AppBuilder::<HeadlessWindowInterface>::new()
///     .name("My phobos application")
///     .present_mode(vk::PresentModeKHR::FIFO)
///     .scratch_size(1024)
///     .validation(true)
///     .build();
/// ```
pub struct AppBuilder<'a, Window: WindowInterface> {
    inner: AppSettings<'a, Window>,
}

impl<'a, Window: WindowInterface> Default for AppBuilder<'a, Window> {
    /// Create a new app builder with default settings.
    fn default() -> Self {
        Self {
            inner: AppSettings::default(),
        }
    }
}

impl<'a, Window: WindowInterface> AppBuilder<'a, Window> {
    /// Create a new app builder with default settings.
    pub fn new() -> Self {
        AppBuilder {
            inner: AppSettings::default(),
        }
    }

    /// Sets the application name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.inner.name = name.into();
        self
    }

    /// Sets the application version.
    pub fn version(mut self, ver: impl Into<(u32, u32, u32)>) -> Self {
        self.inner.version = ver.into();
        self
    }

    /// Enable the Vulkan validation layers.
    pub fn validation(mut self, val: bool) -> Self {
        self.inner.enable_validation = val;
        self
    }

    /// Set the window interface.
    pub fn window(mut self, window: &'a Window) -> Self {
        self.inner.window = Some(window);
        self
    }

    /// The surface format to use (if using a window context).
    pub fn surface_format(mut self, format: vk::SurfaceFormatKHR) -> Self {
        self.inner.surface_format = Some(format);
        self
    }

    /// The present mode to use (if using a window context).
    pub fn present_mode(mut self, mode: vk::PresentModeKHR) -> Self {
        self.inner.present_mode = Some(mode);
        self
    }

    /// The gpu requirements that the physical device must satisfy.
    pub fn gpu(mut self, gpu: GPURequirements) -> Self {
        self.inner.gpu_requirements = gpu;
        self
    }

    /// Scratch allocator size for each of the buffer types.
    pub fn scratch_size(mut self, size: impl Into<vk::DeviceSize>) -> Self {
        let size = size.into();
        self.inner.scratch_vbo_size = size;
        self.inner.scratch_ibo_size = size;
        self.inner.scratch_ubo_size = size;
        self.inner.scratch_ssbo_size = size;

        self
    }

    /// Enable as many raytracing extensions as possible.
    /// Will try to enable the following extensions if they are available
    /// - `VK_KHR_acceleration_structure`
    /// - `VK_KHR_ray_query`
    /// - `VK_KHR_ray_tracing_pipeline`
    pub fn raytracing(mut self, enabled: bool) -> Self {
        self.inner.raytracing = enabled;
        self
    }

    /// Build the resulting application settings.
    pub fn build(self) -> AppSettings<'a, Window> {
        self.inner
    }
}
