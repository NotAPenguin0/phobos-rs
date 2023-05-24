use std::time::{Duration, Instant};

use anyhow::Result;
use ash::vk;
use fsr2_sys::FfxFloatCoords2D;
use glam::{Mat4, Vec3};
use winit::event::Event;

use phobos::{
    GraphicsCmdBuffer, image, Image, ImageView, IncompleteCmdBuffer, InFlightContext, Pass, PassBuilder, PassGraph, PhysicalResourceBindings, PipelineBuilder,
    PipelineStage, RecordGraphToCommandBuffer, Sampler,
};
use phobos::domain::All;
use phobos::fsr2::Fsr2DispatchDescription;
use phobos::graph::pass::Fsr2DispatchVirtualResources;
use phobos::sync::submit_batch::SubmitBatch;

use crate::example_runner::{Camera, Context, create_shader, ExampleApp, ExampleRunner, WindowContext};

#[path = "../example_runner/lib.rs"]
mod example_runner;

struct Attachment {
    pub image: Image,
    pub view: ImageView,
}

impl Attachment {
    pub fn new(ctx: &mut Context, format: vk::Format, width: u32, height: u32, extra_usage: vk::ImageUsageFlags) -> Result<Self> {
        let (usage, aspect) = if format == vk::Format::D32_SFLOAT {
            (vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::ImageAspectFlags::DEPTH)
        } else {
            (vk::ImageUsageFlags::COLOR_ATTACHMENT, vk::ImageAspectFlags::COLOR)
        };
        let image = Image::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            width,
            height,
            usage | extra_usage,
            format,
            vk::SampleCountFlags::TYPE_1,
        )?;
        let view = image.view(aspect)?;
        Ok(Self {
            image,
            view,
        })
    }
}

struct Fsr2Sample {
    pub color: Attachment,
    pub depth: Attachment,
    pub motion_vectors: Attachment,
    pub color_upscaled: Attachment,
    pub sampler: Sampler,
    pub previous_time: Instant,
    pub previous_matrix: Mat4,
    pub camera: Camera,
}

impl ExampleApp for Fsr2Sample {
    fn new(mut ctx: Context) -> Result<Self>
        where
            Self: Sized, {
        // Create pipelines

        let vertex = create_shader("examples/data/fsr_render_vert.spv", vk::ShaderStageFlags::VERTEX);
        let fragment = create_shader("examples/data/fsr_render_frag.spv", vk::ShaderStageFlags::FRAGMENT);

        let pci = PipelineBuilder::new("render")
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .blend_attachment_none()
            .blend_attachment_none()
            .cull_mask(vk::CullModeFlags::NONE)
            .attach_shader(vertex)
            .attach_shader(fragment)
            .build();
        ctx.pipelines.create_named_pipeline(pci)?;

        let vertex = create_shader("examples/data/vert.spv", vk::ShaderStageFlags::VERTEX);
        let fragment = create_shader("examples/data/frag.spv", vk::ShaderStageFlags::FRAGMENT);

        let pci = PipelineBuilder::new("sample")
            .vertex_input(0, vk::VertexInputRate::VERTEX)
            .vertex_attribute(0, 0, vk::Format::R32G32_SFLOAT)?
            .vertex_attribute(0, 1, vk::Format::R32G32_SFLOAT)?
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .blend_attachment_none()
            .cull_mask(vk::CullModeFlags::NONE)
            .attach_shader(vertex)
            .attach_shader(fragment)
            .build();
        ctx.pipelines.create_named_pipeline(pci)?;

        let sampler = Sampler::default(ctx.device.clone())?;

        Ok(Self {
            color: Attachment::new(
                &mut ctx,
                vk::Format::R32G32B32A32_SFLOAT,
                512,
                512,
                vk::ImageUsageFlags::SAMPLED,
            )?,
            depth: Attachment::new(&mut ctx, vk::Format::D32_SFLOAT, 512, 512, vk::ImageUsageFlags::SAMPLED)?,
            motion_vectors: Attachment::new(&mut ctx, vk::Format::R16G16_SFLOAT, 512, 512, vk::ImageUsageFlags::SAMPLED)?,
            color_upscaled: Attachment::new(
                &mut ctx,
                vk::Format::R32G32B32A32_SFLOAT,
                512,
                512,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            )?,
            sampler,
            previous_time: Instant::now(),
            previous_matrix: Mat4::IDENTITY,
            camera: Camera::new(Vec3::new(-3.0, 2.0, 0.0), Vec3::new(-20.0f32.to_radians(), 0.0, 0.0)),
        })
    }

    fn handle_event(&mut self, event: &Event<()>) -> Result<()> {
        self.camera.controls(event);
        Ok(())
    }

    fn frame(&mut self, ctx: Context, mut ifc: InFlightContext) -> Result<SubmitBatch<All>> {
        let swapchain = image!("swapchain");
        let color = image!("color");
        let depth = image!("depth");
        let motion_vectors = image!("motion_vectors");
        let color_upscaled = image!("color_upscaled");

        let time = Instant::now();
        let frame_time = time - self.previous_time;
        self.previous_time = time;

        let near = 0.1f32;
        let far = 100.0f32;
        let fov = 90.0f32.to_radians();

        let (jitter_x, jitter_y) = ctx.device.fsr2_context().jitter_offset(512, 512)?;

        let transform = Mat4::IDENTITY;
        let view = self.camera.matrix();
        let mut projection = Mat4::perspective_rh(fov, 1.0, near, far);
        // Jitter projection matrix
        let jitter_x = 2.0 * jitter_x / 512.0f32;
        let jitter_y = -2.0 * jitter_y / 512.0f32;
        let jitter_translation_matrix = Mat4::from_translation(Vec3::new(jitter_x, jitter_y, 0.0));
        projection = jitter_translation_matrix * projection;
        // Flip y because Vulkan
        let v = projection.col_mut(1).y;
        projection.col_mut(1).y = v * -1.0;
        let previous_matrix = self.previous_matrix;
        self.previous_matrix = projection * view * transform;

        let render_pass = PassBuilder::render("main_render")
            .color_attachment(
                &color,
                vk::AttachmentLoadOp::CLEAR,
                Some(vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                }),
            )?
            .color_attachment(
                &motion_vectors,
                vk::AttachmentLoadOp::CLEAR,
                Some(vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                }),
            )?
            .depth_attachment(
                &depth,
                vk::AttachmentLoadOp::CLEAR,
                Some(vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                }),
            )?
            .execute_fn(|cmd, ifc, _, _| {
                ubo_struct_assign!(data, ifc,
                    struct Data {
                        transform: Mat4 = transform,
                        view: Mat4 = view,
                        projection: Mat4 = projection,
                        previous_matrix: Mat4 = previous_matrix,
                    }
                );

                let cmd = cmd
                    .full_viewport_scissor()
                    .bind_graphics_pipeline("render")?
                    .bind_uniform_buffer(0, 0, &data_buffer)?
                    .draw(36, 1, 0, 0)?;
                Ok(cmd)
            })
            .build();

        let fsr2_dispatch = Fsr2DispatchDescription {
            jitter_offset: FfxFloatCoords2D {
                x: jitter_x,
                y: jitter_y,
            },
            motion_vector_scale: FfxFloatCoords2D {
                x: 512.0,
                y: 512.0,
            },
            enable_sharpening: false,
            sharpness: 0.0,
            frametime_delta: frame_time,
            pre_exposure: 1.0,
            reset: false,
            camera_near: near,
            camera_far: far,
            camera_fov_vertical: fov,
            viewspace_to_meters_factor: 1.0,
            auto_reactive: None,
        };

        let fsr2_resources = Fsr2DispatchVirtualResources {
            color: render_pass.output(&color).cloned().unwrap(),
            depth: render_pass.output(&depth).cloned().unwrap(),
            motion_vectors: render_pass.output(&motion_vectors).cloned().unwrap(),
            exposure: None,
            reactive: None,
            transparency_and_composition: None,
            output: color_upscaled.clone(),
        };

        let fsr2_pass = PassBuilder::fsr2(ctx.device.clone(), fsr2_dispatch, fsr2_resources);

        let output_pass = PassBuilder::render("sample")
            .color_attachment(
                &swapchain,
                vk::AttachmentLoadOp::CLEAR,
                Some(vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                }),
            )?
            .sample_image(fsr2_pass.output(&color_upscaled).unwrap(), PipelineStage::FRAGMENT_SHADER)
            .execute_fn(|cmd, ifc, bindings, _| {
                let vertices: Vec<f32> =
                    vec![-1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
                let mut vtx_buffer = ifc.allocate_scratch_vbo((vertices.len() * std::mem::size_of::<f32>()) as vk::DeviceSize)?;
                let slice = vtx_buffer.mapped_slice::<f32>()?;
                slice.copy_from_slice(vertices.as_slice());
                cmd.full_viewport_scissor()
                    .bind_graphics_pipeline("sample")?
                    .bind_vertex_buffer(0, &vtx_buffer)
                    .resolve_and_bind_sampled_image(0, 0, &color_upscaled, &self.sampler, bindings)?
                    .draw(6, 1, 0, 0)
            })
            .build();

        let graph = PassGraph::new(Some(&swapchain));
        let graph = graph
            .add_pass(PassBuilder::present("present", output_pass.output(&swapchain).unwrap()))?
            .add_pass(render_pass)?
            .add_pass(fsr2_pass)?
            .add_pass(output_pass)?;
        let mut bindings = PhysicalResourceBindings::new();
        bindings.bind_image("swapchain", ifc.swapchain_image.as_ref().unwrap());
        bindings.bind_image("color", &self.color.view);
        bindings.bind_image("depth", &self.depth.view);
        bindings.bind_image("motion_vectors", &self.motion_vectors.view);
        bindings.bind_image("color_upscaled", &self.color_upscaled.view);
        let mut graph = graph.build()?;
        let cmd = ctx
            .exec
            .on_domain::<All, _>(Some(ctx.pipelines.clone()), Some(ctx.descriptors.clone()))?;
        let cmd = graph.record(cmd, &bindings, &mut ifc, None, &mut ())?;
        let cmd = cmd.finish()?;
        let mut batch = ctx.exec.start_submit_batch()?;
        batch.submit_for_present(cmd, &ifc)?;
        Ok(batch)
    }
}

fn main() -> Result<()> {
    let window = WindowContext::with_size("04_fsr2", 512.0, 512.0)?;
    ExampleRunner::new("04_fsr2", Some(&window), |settings| settings.build())?.run::<Fsr2Sample>(Some(window));
}