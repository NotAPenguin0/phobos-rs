//! Wrapper around `VkTransformMatrixKHR` for easy usage

use ash::vk;

use crate::util::to_vk::IntoVulkanType;

/// Represents a row-major affine transformation matrix.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct TransformMatrix(vk::TransformMatrixKHR);

impl TransformMatrix {
    /// Create an identity matrix
    pub fn identity() -> Self {
        Self(vk::TransformMatrixKHR {
            matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        })
    }
}

impl Default for TransformMatrix {
    fn default() -> Self {
        Self {
            0: vk::TransformMatrixKHR {
                matrix: [0.0; 12],
            },
        }
    }
}

impl TransformMatrix {
    /// Create a new TransformMatrix using TransformMatrixKHR
    pub fn new(in_transform: &vk::TransformMatrixKHR) -> Self {
        Self(in_transform)
    }
}

impl IntoVulkanType for TransformMatrix {
    type Output = vk::TransformMatrixKHR;

    fn into_vulkan(self) -> Self::Output {
        self.0
    }
}
