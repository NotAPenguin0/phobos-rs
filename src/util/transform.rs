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

    /// Build a 4x3 transform matrix from 12 elements, specified in row-major order.
    pub fn from_elements(elements: &[f32; 12]) -> Self {
        Self(vk::TransformMatrixKHR {
            matrix: *elements,
        })
    }

    /// Build a 4x3 transform matrix from 3 rows of 4 elements.
    pub fn from_rows(rows: &[[f32; 4]; 3]) -> Self {
        Self(vk::TransformMatrixKHR {
            matrix: [
                rows[0][0], rows[0][1], rows[0][2], rows[0][3], rows[1][0], rows[1][1], rows[1][2],
                rows[1][3], rows[2][0], rows[2][1], rows[2][2], rows[2][3],
            ],
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

impl IntoVulkanType for TransformMatrix {
    type Output = vk::TransformMatrixKHR;

    fn into_vulkan(self) -> Self::Output {
        self.0
    }
}
