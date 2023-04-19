use std::path::Path;

use anyhow::Result;
use ash::vk;
use glam::{Mat4, Vec3};
use log::{info, trace};

use phobos::domain::{All, Compute};
use phobos::prelude::*;
use phobos::util::align::align;

use crate::example_runner::{Context, ExampleApp, ExampleRunner, load_spirv_file, WindowContext};

#[path = "../example_runner/lib.rs"]
mod example_runner;

struct RaytracingSample {
    idx: Buffer,
    vtx: Buffer,
    inst: Buffer,
    blas: AccelerationStructure,
    blas_buffer: Buffer,
    tlas: AccelerationStructure,
    tlas_buffer: Buffer,
}

impl ExampleApp for RaytracingSample {
    fn new(mut ctx: Context) -> Result<Self> {
        // Create a vertex buffer and fill it with vertex data.
        let vertices: [f32; 18] = [-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0];
        let vtx_buffer = Buffer::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            (vertices.len() * std::mem::size_of::<f32>()) as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryType::CpuToGpu,
        )?;
        vtx_buffer.view_full().mapped_slice::<f32>()?.copy_from_slice(&vertices);

        let indices = (0..5).collect::<Vec<u32>>();
        let idx_buffer = Buffer::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            (indices.len() * std::mem::size_of::<u32>()) as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            MemoryType::CpuToGpu,
        )?;
        idx_buffer
            .view_full()
            .mapped_slice::<u32>()?
            .copy_from_slice(indices.as_slice());

        // Create our initial acceleration structure build info to query the size of scratch buffers and the acceleration structure.
        // We only need to set the build mode, flags and all geometry.
        // src and dst acceleration structures can be left empty
        let mut build_info = AccelerationStructureBuildInfo::new_build()
            .flags(vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION | vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .set_type(AccelerationStructureType::BottomLevel)
            .push_triangles(
                AccelerationStructureGeometryTrianglesData::default()
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .vertex_data(vtx_buffer.address())
                    .stride((3 * std::mem::size_of::<f32>()) as u64)
                    .max_vertex(5)
                    .index_data(vk::IndexType::UINT32, idx_buffer.address())
                    .flags(vk::GeometryFlagsKHR::OPAQUE | vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION),
            )
            .push_range(2, 0, 0, 0);
        // Query acceleration structure and scratch buffer size.
        let sizes = query_build_size(&ctx.device, AccelerationStructureBuildType::Device, &build_info, &[2])?;
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
        build_info = build_info.dst(&acceleration_structure).scratch_data(scratch_buffer.address());

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
            // This barrier is required!
            .memory_barrier(
                PipelineStage::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                PipelineStage::ALL_COMMANDS,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
            )
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
        let instance = AccelerationStructureInstance::default()
            .mask(0xFF)
            // Nvidia best practices recommend disabling face culling!
            .flags(vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE)
            .sbt_record_offset(0)?
            .custom_index(0)?
            .transform(TransformMatrix::identity())
            // This instance points at the compacted BLAS.
            .acceleration_structure(&compact_as, AccelerationStructureBuildType::Device)?;
        let instance_buffer = Buffer::new_aligned(
            ctx.device.clone(),
            &mut ctx.allocator,
            std::mem::size_of::<AccelerationStructureInstance>() as u64,
            // The Vulkan spec states: For any element of pInfos[i].pGeometries or pInfos[i].ppGeometries with a geometryType of VK_GEOMETRY_TYPE_INSTANCES_KHR,
            // if geometry.arrayOfPointers is VK_FALSE, geometry.instances.data.deviceAddress must be aligned to 16 bytes
            16u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryType::CpuToGpu,
        )?;
        *instance_buffer
            .view_full()
            .mapped_slice::<AccelerationStructureInstance>()?
            .first_mut()
            .unwrap() = instance;

        let mut tlas_build_info = AccelerationStructureBuildInfo::new_build()
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .set_type(AccelerationStructureType::TopLevel)
            .push_instances(AccelerationStructureGeometryInstancesData {
                data: instance_buffer.address().into(),
                flags: vk::GeometryFlagsKHR::OPAQUE | vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION,
            })
            .push_range(1, 0, 0, 0);

        let instance_build_sizes = query_build_size(&ctx.device, AccelerationStructureBuildType::Device, &tlas_build_info, &[1])?;
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
        tlas_build_info = tlas_build_info.dst(&instance_as).scratch_data(instance_scratch_data.address());

        // Submit compacting and TLAS build command
        let cmd = ctx
            .exec
            .on_domain::<Compute>(None, None)?
            // Build instance TLAS
            // Compact triangle BLAS
            .compact_acceleration_structure(&acceleration_structure, &compact_as)?
            .memory_barrier(
                PipelineStage::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
                PipelineStage::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ,
            )
            .build_acceleration_structure(&tlas_build_info)?
            .finish()?;
        ctx.exec.submit(cmd)?.wait()?;

        // Create our sample pipeline
        let vtx_code = load_spirv_file(Path::new("examples/data/vert.spv"));
        let frag_code = load_spirv_file(Path::new("examples/data/trace.spv"));

        let vertex = ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::VERTEX, vtx_code);
        let fragment = ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, frag_code);
        let pci = PipelineBuilder::new("rt")
            .vertex_input(0, vk::VertexInputRate::VERTEX)
            .vertex_attribute(0, 0, vk::Format::R32G32_SFLOAT)?
            .vertex_attribute(0, 1, vk::Format::R32G32_SFLOAT)?
            .attach_shader(vertex)
            .attach_shader(fragment)
            .cull_mask(vk::CullModeFlags::NONE)
            .blend_attachment_none()
            .depth(false, false, false, vk::CompareOp::ALWAYS)
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .build();
        ctx.pipelines.create_named_pipeline(pci)?;

        // Store resources so they do not get dropped
        Ok(Self {
            idx: idx_buffer,
            vtx: vtx_buffer,
            inst: instance_buffer,
            blas: compact_as,
            blas_buffer: compact_buffer,
            tlas: instance_as,
            tlas_buffer: instance_as_buffer,
        })
    }

    fn frame(&mut self, ctx: Context, mut ifc: InFlightContext) -> Result<CommandBuffer<All>> {
        let swap = VirtualResource::image("swapchain");
        let render_pass = PassBuilder::render("render")
            .color_attachment(
                &swap,
                vk::AttachmentLoadOp::CLEAR,
                Some(vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                }),
            )?
            .execute_fn(|cmd, ifc, bindings, _| {
                let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 0.0));
                let projection = Mat4::perspective_rh(90.0_f32.to_radians(), 800.0 / 600.0, 0.001, 100.0);
                let vertices: [f32; 24] =
                    [-1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
                let mut buffer = ifc.allocate_scratch_vbo((vertices.len() * std::mem::size_of::<f32>()) as vk::DeviceSize)?;
                buffer.mapped_slice::<f32>()?.copy_from_slice(vertices.as_slice());
                cmd.bind_graphics_pipeline("rt")?
                    .full_viewport_scissor()
                    .bind_vertex_buffer(0, &buffer)
                    .push_constant(vk::ShaderStageFlags::FRAGMENT, 0, &view)
                    .push_constant(vk::ShaderStageFlags::FRAGMENT, std::mem::size_of::<Mat4>() as u32, &projection)
                    .bind_acceleration_structure(0, 0, &self.tlas)?
                    .draw(6, 1, 0, 0)
            })
            .build();

        let present = PassBuilder::present("present", render_pass.output(&swap).unwrap());
        let mut graph = PassGraph::new(Some(&swap)).add_pass(render_pass)?.add_pass(present)?.build()?;

        let mut bindings = PhysicalResourceBindings::new();
        bindings.bind_image("swapchain", ifc.swapchain_image.as_ref().unwrap());
        let cmd = ctx
            .exec
            .on_domain::<All>(Some(ctx.pipelines.clone()), Some(ctx.descriptors.clone()))?;
        let cmd = graph.record(cmd, &bindings, &mut ifc, None, &mut ())?;
        cmd.finish()
    }
}

fn main() -> Result<()> {
    let window = WindowContext::new("03_raytracing")?;
    ExampleRunner::new("03_raytracing", Some(&window), |settings| settings.raytracing(true).build())?.run::<RaytracingSample>(Some(window));
}
