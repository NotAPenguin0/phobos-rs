//! Exposes the phobos error type

use std::ffi::NulError;
use std::sync::PoisonError;

use ash;
use gpu_allocator::AllocationError;
use thiserror::Error;

use crate::core::device::ExtensionID;

/// Error type that phobos can return.
#[derive(Error, Debug)]
pub enum Error {
    /// Failed to load the Vulkan library.
    #[error("Failed to load Vulkan.")]
    LoadFailed(ash::LoadingError),
    /// Could not convert rust string to C-String because it has null bytes
    #[error("Invalid C string")]
    InvalidString(NulError),
    /// Generic Vulkan error type.
    #[error("Vulkan error: `{0}`")]
    VkError(ash::vk::Result),
    /// No window context specified where one was expected.
    #[error("Expected a window context.")]
    NoWindow,
    /// No suitable GPU found.
    #[error("No physical device found matching requirements.")]
    NoGPU,
    /// No supported surface formats found.
    #[error("No supported surface formats found.")]
    NoSurfaceFormat,
    /// No queue was found that supports presentation.
    #[error("No queue found that supports presentation. Only headless mode is supported.")]
    NoPresentQueue,
    /// No queue was found for requested domain. Did you forget to request it?
    #[error("No queue found for requested domain. Did you forget a queue request on initialization?")]
    NoCapableQueue,
    /// Vulkan allocation error.
    #[error("Vulkan allocation error: `{0}`")]
    AllocationError(AllocationError),
    /// Task graph contains a cycle and is impossible to resolve.
    #[error("Task graph contains cycle.")]
    GraphHasCycle,
    /// Node not found in graph. Generally this should not happen.
    #[error("Implementation error. Node not found. Please open an issue.")]
    NodeNotFound,
    /// Task graph contains two nodes that act on the same resource with different usage flags.
    /// This is impossible to resolve in an unambiguous way.
    #[error("Illegal task graph using the same resource in different ways.")]
    IllegalTaskGraph,
    /// No resource was bound to a virtual resource
    #[error("No resource bound to virtual resource `{0}`")]
    NoResourceBound(String),
    /// Named pipeline not registered in the pipeline cache.
    #[error("Named pipeline `{0}` not found.")]
    PipelineNotFound(String),
    /// Tried to add a vertex attribute to a vertex binding that does not exist.
    #[error("Tried to add a vertex attribute to a vertex binding that does not exist.")]
    NoVertexBinding,
    /// Tried to allocate an empty descriptor set.
    #[error("Empty descriptor set.")]
    EmptyDescriptorBinding,
    /// No descriptor set layout was given, probably because it was not obtained through a command buffer with a valid pipeline bound.
    #[error("No descriptor set layout was given. Always create descriptor sets through a command buffer after binding a pipeline.")]
    NoDescriptorSetLayout,
    /// No clear value was specified even though one was required.
    #[error("No clear value specified for an attachment with `VK_LOAD_OP_CLEAR`")]
    NoClearValue,
    /// Poisoned mutex
    #[error("Poisoned mutex")]
    PoisonError,
    /// Buffer view out of range of original buffer
    #[error("Buffer view is not a valid range in the parent buffer.")]
    BufferViewOutOfRange,
    /// Buffer copy between views of different sizes is not allowed.
    #[error("Buffer copy has invalid buffer views as range.")]
    InvalidBufferCopy,
    /// Mappable buffer expected
    #[error("Requested mappable buffer, but buffer does not have a memory map")]
    UnmappableBuffer,
    /// Shader needs an entry point named `main`.
    #[error("Shader does not have an entry point.")]
    NoEntryPoint,
    /// Shader uses descriptor sets with the same binding but different types. This is legal, but currently unsupported.
    #[error("Shader uses aliased descriptor `{0}`, which is currently not supported.")]
    AliasedDescriptor(String),
    /// Call requires reflection information, but was not given.
    #[error("Missing shader reflection information in call that requires it.")]
    NoReflectionInformation,
    /// Tried to look up descriptor binding in reflection info, but does not exist.
    #[error("Descriptor `{0}` does not exist.")]
    NoBinding(String),
    /// Returned from [`ExecutionManager::try_on_domain()`](crate::ExecutionManager::try_on_domain) to indicate the queue is currently locked.
    #[error("Returned as a result from ExecutionManager::try_on_domain to indicate the queue is currently locked.")]
    QueueLocked,
    /// Tried to bind descriptors to command buffer with no descriptor cache. Give the descriptor cache in
    /// [`ExecutionManager::on_domain()`](crate::ExecutionManager::on_domain)
    #[error("Tried to bind descriptors to command buffer with no descriptor cache")]
    NoDescriptorCache,
    /// Tried to bind a pipeline to command buffer with no pipeline cache. Give the pipeline cache in
    /// [`ExecutionManager::on_domain()`](crate::ExecutionManager::on_domain)
    #[error("Tried to bind pipeline to command buffer with no pipeline cache")]
    NoPipelineCache,
    /// Cannot bind a graphics pipeline outside for a renderpass. Make sure the current pass was created using
    /// [`PassBuilder::render()`](crate::PassBuilder::render)
    #[error("Tried to obtain a graphics pipeline outside of a render pass.")]
    NoRenderpass,
    /// Function call requires extension to be enabled, but this extension was not requested or not available.
    #[error("Extension {0} required for this feature, but not enabled.")]
    ExtensionNotSupported(ExtensionID),
    /// Uncategorized error.
    #[error("Uncategorized error: `{0}`")]
    Uncategorized(&'static str),
}

impl From<ash::LoadingError> for Error {
    fn from(value: ash::LoadingError) -> Self {
        Error::LoadFailed(value)
    }
}

impl From<NulError> for Error {
    fn from(value: NulError) -> Self {
        Error::InvalidString(value)
    }
}

impl From<ash::vk::Result> for Error {
    fn from(value: ash::vk::Result) -> Self {
        Error::VkError(value)
    }
}

impl From<AllocationError> for Error {
    fn from(value: AllocationError) -> Self {
        Error::AllocationError(value)
    }
}

impl From<(Vec<ash::vk::Pipeline>, ash::vk::Result)> for Error {
    fn from(_: (Vec<ash::vk::Pipeline>, ash::vk::Result)) -> Self {
        Error::Uncategorized("Pipeline creation failed")
    }
}

impl<T> From<PoisonError<T>> for Error {
    fn from(_: PoisonError<T>) -> Self {
        Error::PoisonError
    }
}
