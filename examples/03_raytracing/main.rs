use anyhow::Result;
use ash::vk;
use log::info;

use phobos::{
    Buffer, CommandBuffer, ComputeCmdBuffer, IncompleteCmdBuffer, InFlightContext, MemoryType, PassBuilder, PassGraph, PhysicalResourceBindings,
    RecordGraphToCommandBuffer, VirtualResource,
};
use phobos::acceleration_structure::{
    AccelerationStructure, AccelerationStructureBuildInfo, AccelerationStructureBuildType, AccelerationStructureGeometryInstancesData,
    AccelerationStructureGeometryTrianglesData, AccelerationStructureInstance, AccelerationStructureType,
};
use phobos::domain::{All, Compute};
use phobos::query_pool::{AccelerationStructureCompactedSizeQuery, QueryPool, QueryPoolCreateInfo};
use phobos::util::address::{DeviceOrHostAddress, DeviceOrHostAddressConst};
use phobos::util::align::align;
use phobos::util::transform::TransformMatrix;

use crate::example_runner::{Context, ExampleApp, ExampleRunner, WindowContext};

#[path = "../example_runner/lib.rs"]
mod example_runner;

struct RaytracingSample {
    acceleration_structure: AccelerationStructure,
    buffer: Buffer,
    instance_as: AccelerationStructure,
    instance_buffer: Buffer,
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
                AccelerationStructureGeometryTrianglesData::default()
                    .format(vk::Format::R32G32_SFLOAT)
                    .vertex_data(vtx_buffer.address())
                    .stride((2 * std::mem::size_of::<f32>()) as u64)
                    .max_vertex(2)
                    .flags(vk::GeometryFlagsKHR::OPAQUE),
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

        // Create a query pool to query the compacted size.
        let mut qp = QueryPool::<AccelerationStructureCompactedSizeQuery>::new(
            ctx.device.clone(),
            QueryPoolCreateInfo {
                count: 1,
                statistic_flags: None,
            },
        )?;

        info!("Acceleration structure size before compacting: {} bytes", sizes.size);

        // Create a command buffer. Building acceleration structures is done on a compute command buffer.
        let cmd = ctx
            .exec
            .on_domain::<Compute>(None, None)?
            // Building an acceleration structure is just a single command
            .build_acceleration_structure(&build_info)?
            // Query the compacted size properties. Note that the query type is inferred from the query pool type.
            .write_acceleration_structure_properties(&acceleration_structure, &mut qp)?
            .finish()?;
        // Submit the command buffer and wait for its completion.
        ctx.exec.submit(cmd)?.wait()?;

        // Use our compacted size query to compact this acceleration structure
        let compacted_size = align(qp.wait_for_single_result(0)?, AccelerationStructure::alignment());
        info!("Acceleration structure size after compacting: {} bytes", compacted_size);

        // Create final compacted acceleration structures
        let compact_buffer = Buffer::new_device_local(
            ctx.device.clone(),
            &mut ctx.allocator,
            compacted_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        )?;
        let compact_as = AccelerationStructure::new(
            ctx.device.clone(),
            AccelerationStructureType::BottomLevel,
            compact_buffer.view_full(),
            vk::AccelerationStructureCreateFlagsKHR::default(),
        )?;

        // Lets also create a TLAS. This is quite similar to the BLAS, except we wont do compaction here
        let instances: [AccelerationStructureInstance; 1] = [AccelerationStructureInstance::default()
            .mask(0xFF)
            .flags(vk::GeometryInstanceFlagsKHR::TRIANGLE_FRONT_COUNTERCLOCKWISE)
            .sbt_record_offset(0)?
            .custom_index(0)?
            .transform(TransformMatrix::identity())
            // This instance points at the compacted BLAS.
            .acceleration_structure(&compact_as, AccelerationStructureBuildType::Device)?];
        let instance_buffer = Buffer::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            (instances.len() * std::mem::size_of::<AccelerationStructureInstance>()) as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryType::CpuToGpu,
        )?;
        instance_buffer
            .view_full()
            .mapped_slice::<AccelerationStructureInstance>()?
            .copy_from_slice(&instances);

        let build_info = AccelerationStructureBuildInfo::new_build()
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .set_type(AccelerationStructureType::TopLevel)
            .push_instances(AccelerationStructureGeometryInstancesData {
                data: instance_buffer.address().into(),
                flags: vk::GeometryFlagsKHR::OPAQUE | vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION,
            })
            .push_range(1, 0, 0, 0);

        let instance_build_sizes = AccelerationStructure::build_sizes(&ctx.device, AccelerationStructureBuildType::Device, &build_info, &[1])?;
        let instance_scratch_data = Buffer::new_device_local(
            ctx.device.clone(),
            &mut ctx.allocator,
            instance_build_sizes.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        let instance_as_buffer = Buffer::new_device_local(
            ctx.device.clone(),
            &mut ctx.allocator,
            instance_build_sizes.size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        )?;
        let instance_as = AccelerationStructure::new(
            ctx.device.clone(),
            AccelerationStructureType::TopLevel,
            instance_as_buffer.view_full(),
            Default::default(),
        )?;
        // Set scratch buffer and src/dst acceleration structures
        let build_info = build_info
            .src(&instance_as)
            .dst(&instance_as)
            .scratch_data(instance_scratch_data.address().into());

        // Submit compacting and TLAS build command
        let cmd = ctx
            .exec
            .on_domain::<Compute>(None, None)?
            // Build instance TLAS
            .build_acceleration_structure(&build_info)?
            // Compact triangle BLAS
            .compact_acceleration_structure(&acceleration_structure, &compact_as)?
            .finish()?;
        ctx.exec.submit(cmd)?.wait()?;

        // Store resources so they do not get dropped
        Ok(Self {
            acceleration_structure: compact_as,
            buffer: compact_buffer,
            instance_as,
            instance_buffer: instance_as_buffer,
        })
    }

    fn frame(&mut self, ctx: Context, mut ifc: InFlightContext) -> Result<CommandBuffer<All>> {
        let swap = VirtualResource::image("swapchain");
        let pass = PassBuilder::present("present", &swap);
        let mut graph = PassGraph::new(Some(&swap)).add_pass(pass)?.build()?;
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
