use std::mem::size_of;
use ash::vk;

pub trait ByteSize {
    fn byte_size(&self) -> usize;
}

impl ByteSize for vk::Format {
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