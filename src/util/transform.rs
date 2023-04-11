/// Represents a row-major affine transformation matrix.
#[derive(Default, Copy, Clone)]
#[repr(transparent)]
pub struct TransformMatrix([[f32; 4]; 3]);

impl TransformMatrix {
    pub fn identity() -> Self {
        Self([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    }
}