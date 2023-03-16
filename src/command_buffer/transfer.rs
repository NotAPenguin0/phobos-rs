use crate::domain::ExecutionDomain;
use crate::{BufferView, Error, ImageView, TransferCmdBuffer, TransferSupport};

use anyhow::Result;
use ash::vk;
use crate::command_buffer::IncompleteCommandBuffer;

impl<D: TransferSupport + ExecutionDomain> TransferCmdBuffer for IncompleteCommandBuffer<'_, D> {
    /// Copy one buffer to the other.
    /// # Errors
    /// Fails if the buffer views do not have the same size.
    fn copy_buffer(self, src: &BufferView, dst: &BufferView) -> Result<Self> {
        if src.size != dst.size {
            return Err(Error::InvalidBufferCopy.into());
        }

        let copy = vk::BufferCopy {
            src_offset: src.offset,
            dst_offset: dst.offset,
            size: src.size
        };

        unsafe { self.device.cmd_copy_buffer(self.handle, src.handle, dst.handle, std::slice::from_ref(&copy)); }

        Ok(self)
    }

    /// Copy a buffer to the base mip level of the specified image.
    fn copy_buffer_to_image(self, src: &BufferView, dst: &ImageView) -> Result<Self> where Self: Sized {
        let copy = vk::BufferImageCopy {
            buffer_offset: src.offset,
            buffer_row_length: dst.size.width,
            buffer_image_height: dst.size.height,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: dst.aspect,
                mip_level: dst.base_level,
                base_array_layer: dst.base_layer,
                layer_count: dst.layer_count,
            },
            image_offset: Default::default(),
            image_extent: dst.size,
        };

        unsafe { self.device.cmd_copy_buffer_to_image(self.handle, src.handle, dst.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, std::slice::from_ref(&copy)); }

        Ok(self)
    }
}