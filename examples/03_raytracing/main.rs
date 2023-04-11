use anyhow::Result;
use ash::vk;

use phobos::{Buffer, CommandBuffer, ComputeCmdBuffer, IncompleteCmdBuffer, InFlightContext, MemoryType, PassBuilder, PassGraph, PhysicalResourceBindings, RecordGraphToCommandBuffer, VirtualResource};
use phobos::acceleration_structure::{
    AccelerationStructure, AccelerationStructureBuildInfo, AccelerationStructureBuildType, AccelerationStructureGeometryTrianglesData,
    AccelerationStructureType,
};
use phobos::domain::{All, Compute};
use phobos::util::address::{DeviceOrHostAddress, DeviceOrHostAddressConst};

use crate::example_runner::{Context, ExampleApp, ExampleRunner, WindowContext};

#[path = "../example_runner/lib.rs"]
mod example_runner;

struct RaytracingSample {
    acceleration_structure: AccelerationStructure,
    buffer: Buffer,
}

impl ExampleApp for RaytracingSample {
    fn new(mut ctx: Context) -> Result<Self>
        where
            Self: Sized, {
        // Create a vertex buffer and fill it with vertex data.
        let vertices: [f32; 6] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let vtx_buffer = Buffer::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            (vertices.len() * std::mem::size_of::<f32>()) as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryType::CpuToGpu,
        )?;
        vtx_buffer.view_full().mapped_slice::<f32>()?.copy_from_slice(&vertices);

        // Create our initial acceleration structure build info to query the size of scratch buffers and the acceleration structure.
        // We only need to set the build mode, flags and all geometry.
        // src and dst acceleration structures can be left empty
        let build_info = AccelerationStructureBuildInfo::new_build()
            .flags(vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION | vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .set_type(AccelerationStructureType::BottomLevel)
            .push_triangles(
                AccelerationStructureGeometryTrianglesData {
                    format: vk::Format::R32G32_SFLOAT,
                    vertex_data: vtx_buffer.address().into(),
                    stride: 2 * std::mem::size_of::<f32>() as vk::DeviceSize,
                    max_vertex: 2,
                    index_type: vk::IndexType::NONE_KHR,
                    index_data: DeviceOrHostAddressConst::null_host(),
                    transform_data: DeviceOrHostAddressConst::null_host(),
                },
                vk::GeometryFlagsKHR::OPAQUE,
            )
            .push_range(1, 0, 0, 0);
        // Query acceleration structure and scratch buffer size.
        let sizes = AccelerationStructure::build_sizes(&ctx.device, AccelerationStructureBuildType::Device, &build_info, &[1])?;
        // Allocate backing buffer for acceleration structure
        let buffer = Buffer::new_device_local(
            ctx.device.clone(),
            &mut ctx.allocator,
            sizes.size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        )?;
        // Allocate scratch buffer for building the acceleration structure
        let scratch_buffer = Buffer::new_device_local(
            ctx.device.clone(),
            &mut ctx.allocator,
            sizes.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let acceleration_structure = AccelerationStructure::new(
            ctx.device.clone(),
            AccelerationStructureType::BottomLevel,
            buffer.view_full(),
            vk::AccelerationStructureCreateFlagsKHR::default(),
        )?;
        // We can now fill the rest of the build info (source and destination acceleration structures, and the scratch data).
        let build_info = build_info
            .src(&acceleration_structure)
            .dst(&acceleration_structure)
            .scratch_data(scratch_buffer.address().into());

        // Create a command buffer. Building acceleration structures is done on a compute command buffer.
        let cmd = ctx.exec.on_domain::<Compute>(None, None)?
            // Building an acceleration structure is just a single command
            .build_acceleration_structure(&build_info)?
            .finish()?;
        // Submit the command buffer and wait for its completion.
        ctx.exec.submit(cmd)?.wait()?;

        // Store resources so they do not get dropped
        Ok(Self {
            acceleration_structure,
            buffer,
        })
    }

    fn frame(&mut self, ctx: Context, mut ifc: InFlightContext) -> Result<CommandBuffer<All>> {
        let swap = VirtualResource::image("swapchain");
        let pass = PassBuilder::present("present", &swap);
        let mut graph = PassGraph::new(Some(&swap))
            .add_pass(pass)?
            .build()?;
        let mut bindings = PhysicalResourceBindings::new();
        bindings.bind_image("swapchain", ifc.swapchain_image.as_ref().unwrap());
        let cmd = ctx.exec.on_domain::<All>(None, None)?;
        let cmd = graph.record(cmd, &bindings, &mut ifc, None, &mut ())?;
        cmd.finish()
    }
}

fn main() -> Result<()> {
    let window = WindowContext::new("03_raytracing")?;
    ExampleRunner::new("03_raytracing", Some(&window), |settings| settings.raytracing(true).build())?.run::<RaytracingSample>(Some(window));
}
