use std::mem::size_of;
use ash::vk;

/// Simple trait to get the size of one element in bytes of a `vk::Format`.
pub trait ByteSize {
    /// Returns the size, in bytes, of one element of this thing.
    fn byte_size(&self) -> usize;
}

impl ByteSize for vk::Format {
    /// If an image is created with this format, then the return value of this function is the size in bytes of one pixel.
    fn byte_size(&self) -> usize {
        match *self {
            vk::Format::R32G32_SFLOAT => 2 * size_of::<f32>(),
            vk::Format::R32G32B32_SFLOAT => 3 * size_of::<f32>(),
            vk::Format::R32G32B32A32_SFLOAT => 4 * size_of::<f32>(),
            vk::Format::R8_UNORM => 1,
            vk::Format::R8G8_UNORM => 2,
            vk::Format::R8G8B8_UNORM => 3,
            vk::Format::R8G8B8A8_UNORM => 4,
            _ => { todo!() }
        }
    }
}