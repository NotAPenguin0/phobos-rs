//! Utility traits to convert objects into their corresponding vulkan types

/// Convert an object into a vulkan type
pub trait IntoVulkanType {
    /// Output Vulkan type
    type Output;

    /// Consume self and return a vulkan type
    fn into_vulkan(self) -> Self::Output;
}

/// Get a reference to the object, as a vulkan type
pub trait AsVulkanType {
    /// Output Vulkan type
    type Output;

    /// Return a vulkan type that lives as long as self
    fn as_vulkan(&self) -> Self::Output;
}