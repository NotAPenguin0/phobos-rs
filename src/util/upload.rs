use std::sync::Arc;
use crate::{Allocator, Buffer, Device, domain, ExecutionManager, Fence, IncompleteCmdBuffer, MemoryType, TransferCmdBuffer};

use anyhow::Result;
use ash::vk;

/// Perform a staged upload to a GPU buffer. Returns a future that can be awaited to obtain the resulting buffer.
pub async fn staged_buffer_upload<T: Copy, A: Allocator>(device: Arc<Device>, mut allocator: A, exec: Arc<ExecutionManager>, data: &[T]) -> Result<Buffer<A>> {
    let staging = Buffer::new(device.clone(), &mut allocator, (data.len() * std::mem::size_of::<T>()) as vk::DeviceSize, vk::BufferUsageFlags::TRANSFER_SRC, MemoryType::CpuToGpu)?;
    let mut staging = staging.view_full();
    staging.mapped_slice()?.copy_from_slice(data);

    let buffer = Buffer::new_device_local(device.clone(), &mut allocator, staging.size, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER)?;
    let view = buffer.view_full();

    let cmd =
        exec.on_domain::<domain::Transfer>(None, None)?
            .copy_buffer(&staging, &view)?
            .finish()?;

    ExecutionManager::submit(exec.clone(), cmd)?
        .with_cleanup(move || {
            drop(staging);
        })
        .attach_value(Ok(buffer))
        .await
}