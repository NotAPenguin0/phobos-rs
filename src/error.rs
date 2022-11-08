use std::ffi::NulError;
use ash;
use crate::QueueType;

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
    /// No queue was found for a specific queue type. Did you forget to request it?
    NoCapableQueue(QueueType),
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