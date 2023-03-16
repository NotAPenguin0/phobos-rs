use std::sync::Arc;
use crate::{Allocator, Buffer, Device, domain, ExecutionManager, GpuFuture, IncompleteCmdBuffer, MemoryType, PassBuilder, PassGraph, PhysicalResourceBindings, RecordGraphToCommandBuffer, ThreadContext, TransferCmdBuffer};

use anyhow::Result;
use ash::vk;

/// Perform a staged upload to a GPU buffer. Returns a future that can be awaited to obtain the resulting buffer.
pub fn staged_buffer_upload<T, A: Allocator>(device: Arc<Device>, mut allocator: A, exec: Arc<ExecutionManager>, data: &[T]) -> Result<GpuFuture<Buffer<A>>> where T: Copy {
    let staging = Buffer::new(device.clone(), &mut allocator, (data.len() * std::mem::size_of::<T>()) as vk::DeviceSize, vk::BufferUsageFlags::TRANSFER_SRC, MemoryType::CpuToGpu)?;
    let mut staging = staging.view_full();
    staging.mapped_slice()?.copy_from_slice(data);

    let buffer = Buffer::new_device_local(device.clone(), &mut allocator, staging.size, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER)?;
    let view = buffer.view_full();

    let mut ctx = ThreadContext::new(device.clone(), allocator, None)?;

    // TODO: Figure out a way to share these graphs, safely.
    let mut graph = PassGraph::new(None);
    let pass = PassBuilder::new("copy".to_owned())
        .execute(|cmd, _, _| {
            cmd.copy_buffer(&staging, &view)
        })
        .build();

    graph = graph.add_pass(pass)?;
    let mut graph = graph.build()?;

    let cmd = exec.on_domain::<domain::Transfer>(None, None)?;
    let mut ifc = ctx.get_ifc();
    let bindings = PhysicalResourceBindings::new();
    let cmd = graph.record(cmd, &bindings, &mut ifc, None)?.finish()?;

    let fence = ExecutionManager::submit(exec.clone(), cmd)?
        .with_cleanup(move || {
            drop(staging);
        });
    Ok(fence.attach_value(buffer))
}