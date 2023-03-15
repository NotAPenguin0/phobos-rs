use std::ffi::{c_char, CStr, CString};
use std::mem::size_of;
use std::sync::{Arc, Mutex};
use ash::vk;
use anyhow::Result;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;
use crate::{Buffer, Device, domain, ExecutionManager, GpuFuture, IncompleteCmdBuffer, PassBuilder, PassGraph, PhysicalResourceBindings, record_graph, ThreadContext, TransferCmdBuffer};

/// Wraps a c string into a string, or an empty string if the provided c string was null.
/// Assumes the provided c string is null terminated.
pub(crate) unsafe fn wrap_c_str(s: *const c_char) -> String {
    return if s.is_null() {
        String::default()
    } else {
        CStr::from_ptr(s).to_string_lossy().to_owned().to_string()
    }
}

/// Safely unwraps a slice of strings into a vec of raw c strings.
pub(crate) fn unwrap_to_raw_strings(strings: &[CString]) -> Vec<*const c_char> {
    strings.iter().map(|string| string.as_ptr()).collect()
}

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

/// Perform a staged upload to a GPU buffer. Returns a future that can be awaited to obtain the resulting buffer.
pub fn staged_buffer_upload<T>(device: Arc<Device>, allocator: Arc<Mutex<Allocator>>, exec: Arc<ExecutionManager>, data: &[T]) -> Result<GpuFuture<Buffer>> where T: Copy {
    let staging = Buffer::new(device.clone(), allocator.clone(), (data.len() * std::mem::size_of::<T>()) as vk::DeviceSize, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu)?;
    let mut staging = staging.view_full();
    staging.mapped_slice()?.copy_from_slice(data);

    let buffer = Buffer::new_device_local(device.clone(), allocator.clone(), staging.size, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER)?;
    let view = buffer.view_full();

    let mut ctx = ThreadContext::new(device.clone(), allocator.clone(), None)?;

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
    let cmd = record_graph(&mut graph, &bindings, &mut ifc, cmd, None)?.finish()?;

    let fence = ExecutionManager::submit(exec.clone(), cmd)?
        .with_cleanup(move || {
            drop(staging);
        });
    Ok(fence.attach_value(buffer))
}