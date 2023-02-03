use std::ffi::NulError;
use std::sync::PoisonError;
use ash;
use ash::vk::Pipeline;
use gpu_allocator::AllocationError;

#[derive(Debug)]
pub enum Error {
    /// Failed to load the Vulkan library
    LoadFailed(ash::LoadingError),
    /// Could not convert rust string to C-String because it has null bytes
    InvalidString(NulError),
    /// Generic Vulkan error type
    VkError(ash::vk::Result),
    /// No window context specified where one was expected.
    NoWindow,
    /// No suitable GPU found.
    NoGPU,
    /// No supported surface formats found.
    NoSurfaceFormat,
    /// No queue was found that supports presentation.
    NoPresentQueue,
    /// No queue was found for requested domain. Did you forget to request it?
    NoCapableQueue,
    /// Vulkan allocation error.
    AllocationError(AllocationError),
    /// Task graph contains a cycle and is impossible to resolve.
    GraphHasCycle,
    /// Node not found in graph. Generally this should not happen.
    NodeNotFound,
    /// Task graph contains two nodes that act on the same resource with different usage flags.
    /// This is impossible to resolve in an unambiguous way.
    IllegalTaskGraph,
    /// No resource was bound to a virtual resource
    NoResourceBound(String),
    /// Named pipeline not registered in the pipeline cache.
    PipelineNotFound(String),
    /// Tried to add a vertex attribute to a vertex binding that does not exist.
    NoVertexBinding,
    /// Tried to allocate an empty descriptor set.
    EmptyDescriptorBinding,
    /// No descriptor set layout was given, probably because it was not obtained through a command buffer with a valid pipeline bound.
    NoDescriptorSetLayout,
    /// No clear value was specified even though one was required.
    NoClearValue,
    /// Poisoned mutex
    PoisonError,
    /// Uncategorized error.
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
    fn from(value: AllocationError) -> Self { Error::AllocationError(value) }
}

impl From<(Vec<ash::vk::Pipeline>, ash::vk::Result)> for Error {
    fn from(_: (Vec<Pipeline>, ash::vk::Result)) -> Self {
        Error::Uncategorized("Pipeline creation failed")
    }
}

impl<T> From<PoisonError<T>> for Error {
    fn from(_: PoisonError<T>) -> Self {
        Error::PoisonError
    }
}