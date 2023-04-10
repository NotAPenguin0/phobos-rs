pub trait IntoVulkanType {
    type Output;

    fn into_vulkan(self) -> Self::Output;
}

pub trait AsVulkanType {
    type Output;

    fn as_vulkan(&self) -> Self::Output;
}