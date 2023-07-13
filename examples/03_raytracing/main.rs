use std::path::Path;

use anyhow::Result;
use ash::vk;
use glam::{Mat4, Vec3};
use log::{info, trace};
use phobos::graph::pass::ClearColor;
use phobos::image;
use phobos::image::ImageCreateInfo;
use phobos::pipeline::raytracing::RayTracingPipelineBuilder;
use phobos::pool::LocalPool;
use phobos::prelude::*;
use phobos::sync::domain::{All, Compute};
use phobos::sync::submit_batch::SubmitBatch;
use phobos::util::align::align;

use crate::example_runner::{
    create_shader, load_spirv_file, Context, ExampleApp, ExampleRunner, WindowContext,
};

#[path = "../example_runner/lib.rs"]
mod example_runner;

struct BackedAccelerationStructure {
    pub accel: AccelerationStructure,
    pub buffer: Buffer,
    pub scratch: Buffer,
    pub sizes: AccelerationStructureBuildSize,
}

struct RaytracingSample {
    idx: Buffer,
    vtx: Buffer,
    instances: Buffer,
    blas: BackedAccelerationStructure,
    tlas: BackedAccelerationStructure,
    attachment: Image,
    attachment_view: ImageView,
    sampler: Sampler,
}

fn make_input_buffer<T: Copy>(
    ctx: &mut Context,
    data: &[T],
    alignment: Option<u64>,
    name: &str,
) -> Result<Buffer> {
    let buffer = match alignment {
        None => Buffer::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            (data.len() * std::mem::size_of::<T>()) as u64,
            MemoryType::CpuToGpu,
        )?,
        Some(alignment) => Buffer::new_aligned(
            ctx.device.clone(),
            &mut ctx.allocator,
            (data.len() * std::mem::size_of::<T>()) as u64,
            alignment,
            MemoryType::CpuToGpu,
        )?,
    };
    buffer
        .view_full()
        .mapped_slice::<T>()?
        .copy_from_slice(data);
    ctx.device.set_name(&buffer, name)?;
    Ok(buffer)
}

fn make_vertex_buffer(ctx: &mut Context) -> Result<Buffer> {
    let vertices: [f32; 18] = [
        -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
        1.0,
    ];
    make_input_buffer(ctx, &vertices, None, "Vertex Buffer")
}

fn make_index_buffer(ctx: &mut Context) -> Result<Buffer> {
    let indices = (0..=5).collect::<Vec<u32>>();
    make_input_buffer(ctx, indices.as_slice(), None, "Index Buffer")
}

fn make_instance_buffer(ctx: &mut Context, blas: &AccelerationStructure) -> Result<Buffer> {
    let instance = AccelerationStructureInstance::default()
        .mask(0xFF)
        // Nvidia best practices recommend disabling face culling!
        .flags(vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE)
        .sbt_record_offset(0)?
        .custom_index(0)?
        .transform(TransformMatrix::identity())
        .acceleration_structure(&blas, AccelerationStructureBuildType::Device)?;
    // The Vulkan spec states: For any element of pInfos[i].pGeometries or pInfos[i].ppGeometries with a geometryType of VK_GEOMETRY_TYPE_INSTANCES_KHR,
    // if geometry.arrayOfPointers is VK_FALSE, geometry.instances.data.deviceAddress must be aligned to 16 bytes
    make_input_buffer(ctx, std::slice::from_ref(&instance), Some(16), "Instance Buffer")
}

fn blas_build_info<'a>(vertices: &Buffer, indices: &Buffer) -> AccelerationStructureBuildInfo<'a> {
    AccelerationStructureBuildInfo::new_build()
        .flags(
            vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION
                | vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        )
        .set_type(AccelerationStructureType::BottomLevel)
        .push_triangles(
            AccelerationStructureGeometryTrianglesData::default()
                .format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vertices.address())
                .stride((3 * std::mem::size_of::<f32>()) as u64)
                .max_vertex(5)
                .index_data(vk::IndexType::UINT32, indices.address())
                .flags(
                    vk::GeometryFlagsKHR::OPAQUE
                        | vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION,
                ),
        )
        .push_range(2, 0, 0, 0)
}

fn tlas_build_info<'a>(instances: &Buffer) -> AccelerationStructureBuildInfo<'a> {
    AccelerationStructureBuildInfo::new_build()
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .set_type(AccelerationStructureType::TopLevel)
        .push_instances(AccelerationStructureGeometryInstancesData {
            data: instances.address().into(),
            flags: vk::GeometryFlagsKHR::OPAQUE
                | vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION,
        })
        .push_range(1, 0, 0, 0)
}

fn make_acceleration_structure(
    ctx: &mut Context,
    build_info: &AccelerationStructureBuildInfo,
    prim_counts: &[u32],
    name: &str,
) -> Result<BackedAccelerationStructure> {
    let sizes = query_build_size(
        &ctx.device,
        AccelerationStructureBuildType::Device,
        build_info,
        prim_counts,
    )?;
    let buffer = Buffer::new_device_local(ctx.device.clone(), &mut ctx.allocator, sizes.size)?;
    // Allocate scratch buffer for building the acceleration structure
    let scratch_buffer =
        Buffer::new_device_local(ctx.device.clone(), &mut ctx.allocator, sizes.build_scratch_size)?;
    let acceleration_structure = AccelerationStructure::new(
        ctx.device.clone(),
        build_info.ty(),
        buffer.view_full(),
        vk::AccelerationStructureCreateFlagsKHR::default(),
    )?;
    ctx.device.set_name(&acceleration_structure, name)?;
    Ok(BackedAccelerationStructure {
        accel: acceleration_structure,
        buffer,
        scratch: scratch_buffer,
        sizes,
    })
}

fn make_compacted(
    ctx: &mut Context,
    accel: &AccelerationStructure,
    size: u64,
) -> Result<(AccelerationStructure, Buffer)> {
    // Create final compacted acceleration structures
    let compact_buffer = Buffer::new_device_local(ctx.device.clone(), &mut ctx.allocator, size)?;
    let compact_as = AccelerationStructure::new(
        ctx.device.clone(),
        accel.ty(),
        compact_buffer.view_full(),
        vk::AccelerationStructureCreateFlagsKHR::default(),
    )?;
    Ok((compact_as, compact_buffer))
}

impl ExampleApp for RaytracingSample {
    fn new(mut ctx: Context) -> Result<Self> {
        let vtx_buffer = make_vertex_buffer(&mut ctx)?;
        let idx_buffer = make_index_buffer(&mut ctx)?;

        // Create our initial acceleration structure build info to query the size of scratch buffers and the acceleration structure.
        // We only need to set the build mode, flags and all geometry.
        // src and dst acceleration structures can be left empty
        let mut blas_build_info = blas_build_info(&vtx_buffer, &idx_buffer);
        let blas = make_acceleration_structure(&mut ctx, &blas_build_info, &[2], "BLAS")?;
        // We can now fill the rest of the build info (source and destination acceleration structures, and the scratch data).
        blas_build_info = blas_build_info
            .dst(&blas.accel)
            .scratch_data(blas.scratch.address());

        // Create a query pool to query the compacted size.
        let mut qp = QueryPool::<AccelerationStructureCompactedSizeQuery>::new(
            ctx.device.clone(),
            QueryPoolCreateInfo {
                count: 1,
                statistic_flags: None,
            },
        )?;

        info!("Acceleration structure size before compacting: {} bytes", blas.sizes.size);

        // Create a command buffer. Building acceleration structures is done on a compute command buffer.
        let cmd = ctx
            .exec
            .on_domain::<Compute>()?
            // Building an acceleration structure is just a single command
            .build_acceleration_structure(&blas_build_info)?
            // This barrier is required!
            .memory_barrier(
                PipelineStage::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                PipelineStage::ALL_COMMANDS,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
            )
            // Query the compacted size properties. Note that the query type is inferred from the query pool type.
            .write_acceleration_structure_properties(&blas.accel, &mut qp)?
            .finish()?;
        // Submit the command buffer and wait for its completion.
        ctx.exec.submit(cmd)?.wait()?;

        // Use our compacted size query to compact this acceleration structure
        let compacted_size =
            align(qp.wait_for_single_result(0)?, AccelerationStructure::alignment());
        info!("Acceleration structure size after compacting: {} bytes", compacted_size);
        let (compact_as, compact_buffer) = make_compacted(&mut ctx, &blas.accel, compacted_size)?;

        let instance_buffer = make_instance_buffer(&mut ctx, &compact_as)?;
        let mut tlas_build_info = tlas_build_info(&instance_buffer);
        let tlas = make_acceleration_structure(&mut ctx, &tlas_build_info, &[1], "TLAS")?;
        tlas_build_info = tlas_build_info
            .dst(&tlas.accel)
            .scratch_data(tlas.scratch.address());

        // Submit compacting and TLAS build command
        let cmd = ctx
            .exec
            .on_domain::<Compute>()?
            // Build instance TLAS
            // Compact triangle BLAS
            .compact_acceleration_structure(&blas.accel, &compact_as)?
            .memory_barrier(
                PipelineStage::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
                PipelineStage::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ,
            )
            .build_acceleration_structure(&tlas_build_info)?
            .finish()?;
        ctx.exec.submit(cmd)?.wait()?;

        let rgen = create_shader("examples/data/raygen.spv", vk::ShaderStageFlags::RAYGEN_KHR);
        let rchit =
            create_shader("examples/data/rayhit.spv", vk::ShaderStageFlags::CLOSEST_HIT_KHR);
        let rmiss = create_shader("examples/data/raymiss.spv", vk::ShaderStageFlags::MISS_KHR);

        // Create the raytracing pipeline
        let pci = RayTracingPipelineBuilder::new("rt")
            .max_recursion_depth(1)
            .add_ray_gen_group(rgen)
            .add_ray_hit_group(Some(rchit), None)
            .add_ray_miss_group(rmiss)
            .build();
        ctx.pool.pipelines.create_named_raytracing_pipeline(pci)?;

        // Create the pipeline for drawing the raytraced result to the screen
        let vertex = create_shader("examples/data/vert.spv", vk::ShaderStageFlags::VERTEX);
        let fragment = create_shader("examples/data/frag.spv", vk::ShaderStageFlags::FRAGMENT);

        let pci = PipelineBuilder::new("sample")
            .vertex_input(0, vk::VertexInputRate::VERTEX)
            .vertex_attribute(0, 0, vk::Format::R32G32_SFLOAT)?
            .vertex_attribute(0, 1, vk::Format::R32G32_SFLOAT)?
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .blend_attachment_none()
            .cull_mask(vk::CullModeFlags::NONE)
            .attach_shader(vertex.clone())
            .attach_shader(fragment)
            .build();
        ctx.pool.pipelines.create_named_pipeline(pci)?;

        let attachment = Image::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            ImageCreateInfo {
                width: 800,
                height: 600,
                depth: 1,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
                format: vk::Format::R32G32B32A32_SFLOAT,
                samples: vk::SampleCountFlags::TYPE_1,
                mip_levels: 1,
                layers: 1,
            },
        )?;
        let view = attachment.whole_view(vk::ImageAspectFlags::COLOR)?;

        let sampler = Sampler::default(ctx.device.clone())?;

        Ok(Self {
            idx: idx_buffer,
            vtx: vtx_buffer,
            instances: instance_buffer,
            // swap out buffers for compact buffers
            blas: BackedAccelerationStructure {
                accel: compact_as,
                buffer: compact_buffer,
                scratch: blas.scratch,
                sizes: blas.sizes,
            },
            tlas,
            attachment,
            attachment_view: view,
            sampler,
        })
    }

    fn frame(&mut self, ctx: Context, ifc: InFlightContext) -> Result<SubmitBatch<All>> {
        let swap = image!("swapchain");
        let rt_image = image!("rt_out");

        let mut pool = LocalPool::new(ctx.pool.clone())?;

        let rt_pass = PassBuilder::new("raytrace")
            .write_storage_image(&rt_image, PipelineStage::RAY_TRACING_SHADER_KHR)
            .execute_fn(|cmd, ifc, bindings, _| {
                let view = Mat4::look_at_rh(
                    Vec3::new(0.0, 0.0, -1.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    Vec3::new(0.0, 1.0, 0.0),
                );
                let projection =
                    Mat4::perspective_rh(90.0_f32.to_radians(), 800.0 / 600.0, 0.001, 100.0);
                cmd.bind_ray_tracing_pipeline("rt")?
                    .push_constant(vk::ShaderStageFlags::RAYGEN_KHR, 0, &view)
                    .push_constant(vk::ShaderStageFlags::RAYGEN_KHR, 64, &projection)
                    .bind_acceleration_structure(0, 0, &self.tlas.accel)?
                    .resolve_and_bind_storage_image(0, 1, &rt_image, bindings)?
                    .trace_rays(800, 600, 1)
            })
            .build();

        let render_pass = PassBuilder::render("copy")
            .clear_color_attachment(&swap, ClearColor::Float([0.0, 0.0, 0.0, 0.0]))?
            .sample_image(rt_pass.output(&rt_image).unwrap(), PipelineStage::FRAGMENT_SHADER)
            .execute_fn(|cmd, pool, bindings, _| {
                let vertices: Vec<f32> = vec![
                    -1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0,
                    1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
                ];
                let mut vtx_buffer = pool.allocate_scratch_buffer(
                    (vertices.len() * std::mem::size_of::<f32>()) as vk::DeviceSize,
                )?;
                let slice = vtx_buffer.mapped_slice::<f32>()?;
                slice.copy_from_slice(vertices.as_slice());
                cmd.full_viewport_scissor()
                    .bind_graphics_pipeline("sample")?
                    .bind_vertex_buffer(0, &vtx_buffer)
                    .resolve_and_bind_sampled_image(0, 0, &rt_image, &self.sampler, bindings)?
                    .draw(6, 1, 0, 0)
            })
            .build();

        let present = PassBuilder::present("present", render_pass.output(&swap).unwrap());
        let mut graph = PassGraph::new()
            .add_pass(rt_pass)?
            .add_pass(render_pass)?
            .add_pass(present)?
            .build()?;

        let mut bindings = PhysicalResourceBindings::new();
        bindings.bind_image("swapchain", &ifc.swapchain_image);
        bindings.bind_image("rt_out", &self.attachment_view);
        let cmd = ctx.exec.on_domain::<All>()?;
        let cmd = graph.record(cmd, &bindings, &mut pool, None, &mut ())?;
        let cmd = cmd.finish()?;
        let mut batch = ctx.exec.start_submit_batch()?;
        batch.submit_for_present(cmd, ifc, pool)?;
        Ok(batch)
    }
}

fn main() -> Result<()> {
    let window = WindowContext::new("03_raytracing")?;
    ExampleRunner::new("03_raytracing", Some(&window), |settings| {
        settings.raytracing(true).build()
    })?
    .run::<RaytracingSample>(Some(window));
}
