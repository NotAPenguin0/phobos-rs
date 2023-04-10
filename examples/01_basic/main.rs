use std::path::Path;

use anyhow::Result;
use futures::executor::block_on;

use phobos::command_buffer::traits::*;
use phobos::domain::All;
use phobos::prelude::*;
use phobos::vk;

use crate::example_runner::{Context, ExampleApp, ExampleRunner, load_spirv_file, WindowContext};

#[path = "../example_runner/lib.rs"]
mod example_runner;

struct Resources {
    #[allow(dead_code)]
    pub offscreen: Image,
    pub offscreen_view: ImageView,
    pub sampler: Sampler,
    #[allow(dead_code)]
    pub vertex_buffer: Buffer,
}

struct Basic {
    resources: Resources,
}

impl ExampleApp for Basic {
    fn new(mut ctx: Context) -> Result<Self>
    where
        Self: Sized, {
        // create some pipelines
        // First, we need to load shaders
        let vtx_code = load_spirv_file(Path::new("examples/data/vert.spv"));
        let frag_code = load_spirv_file(Path::new("examples/data/frag.spv"));

        let vertex = ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::VERTEX, vtx_code);
        let fragment = ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, frag_code);

        // Now we can start using the pipeline builder to create our full pipeline.
        let pci = PipelineBuilder::new("sample".to_string())
            .vertex_input(0, vk::VertexInputRate::VERTEX)
            .vertex_attribute(0, 0, vk::Format::R32G32_SFLOAT)?
            .vertex_attribute(0, 1, vk::Format::R32G32_SFLOAT)?
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .blend_attachment_none()
            .cull_mask(vk::CullModeFlags::NONE)
            .attach_shader(vertex.clone())
            .attach_shader(fragment)
            .build();

        // Store the pipeline in the pipeline cache
        ctx.pipelines.create_named_pipeline(pci)?;

        let frag_code = load_spirv_file(Path::new("examples/data/blue.spv"));
        let fragment = ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, frag_code);

        let pci = PipelineBuilder::new("offscreen".to_string())
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

        // Define some resources we will use for rendering
        let image = Image::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            800,
            600,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::Format::R8G8B8A8_SRGB,
            vk::SampleCountFlags::TYPE_1,
        )?;
        let data: Vec<f32> = vec![-1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let resources = Resources {
            offscreen_view: image.view(vk::ImageAspectFlags::COLOR)?,
            offscreen: image,
            sampler: Sampler::default(ctx.device.clone())?,
            vertex_buffer: block_on(async {
                staged_buffer_upload(ctx.device.clone(), ctx.allocator.clone(), ctx.exec.clone(), data.as_slice())
                    .unwrap()
                    .await
            }),
        };

        Ok(Self {
            resources,
        })
    }

    fn frame(&mut self, ctx: Context, mut ifc: InFlightContext) -> Result<CommandBuffer<All>> {
        // Define a virtual resource pointing to the swapchain
        let swap_resource = VirtualResource::image("swapchain");
        let offscreen = VirtualResource::image("offscreen");

        let vertices: Vec<f32> =
            vec![-1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        // Define a render graph with one pass that clears the swapchain image
        let graph = PassGraph::new(Some(&swap_resource));

        // Render pass that renders to an offscreen attachment
        let offscreen_pass = PassBuilder::render("offscreen")
            .color([1.0, 0.0, 0.0, 1.0])
            .color_attachment(
                &offscreen,
                vk::AttachmentLoadOp::CLEAR,
                Some(vk::ClearColorValue {
                    float32: [1.0, 0.0, 0.0, 1.0],
                }),
            )?
            .execute_fn(|mut cmd, ifc, _bindings, _| {
                // Our pass will render a fullscreen quad that 'clears' the screen, just so we can test pipeline creation
                let mut buffer = ifc.allocate_scratch_vbo((vertices.len() * std::mem::size_of::<f32>()) as vk::DeviceSize)?;
                let slice = buffer.mapped_slice::<f32>()?;
                slice.copy_from_slice(vertices.as_slice());
                cmd = cmd
                    .bind_vertex_buffer(0, &buffer)
                    .bind_graphics_pipeline("offscreen")?
                    .full_viewport_scissor()
                    .draw(6, 1, 0, 0)?;
                Ok(cmd)
            })
            .build();

        // Render pass that samples the offscreen attachment, and possibly does some postprocessing to it
        let sample_pass = PassBuilder::render(String::from("sample"))
            .color([0.0, 1.0, 0.0, 1.0])
            .color_attachment(
                &swap_resource,
                vk::AttachmentLoadOp::CLEAR,
                Some(vk::ClearColorValue {
                    float32: [1.0, 0.0, 0.0, 1.0],
                }),
            )?
            .sample_image(offscreen_pass.output(&offscreen).unwrap(), PipelineStage::FRAGMENT_SHADER)
            .execute_fn(|cmd, _ifc, bindings, _| {
                cmd.full_viewport_scissor()
                    .bind_graphics_pipeline("sample")?
                    .resolve_and_bind_sampled_image(0, 0, &offscreen, &self.resources.sampler, bindings)?
                    .draw(6, 1, 0, 0)
            })
            .build();
        // Add another pass to handle presentation to the screen
        let present_pass = PassBuilder::present(
            "present",
            // This pass uses the output from the clear pass on the swap resource as its input
            sample_pass.output(&swap_resource).unwrap(),
        );
        let mut graph = graph
            .add_pass(offscreen_pass)?
            .add_pass(sample_pass)?
            .add_pass(present_pass)?
            // Build the graph, now we can bind physical resources and use it.
            .build()?;

        let mut bindings = PhysicalResourceBindings::new();
        bindings.bind_image("swapchain", ifc.swapchain_image.as_ref().unwrap());
        bindings.bind_image("offscreen", &self.resources.offscreen_view);
        // create a command buffer capable of executing graphics commands
        let cmd = ctx
            .exec
            .on_domain::<All>(Some(ctx.pipelines.clone()), Some(ctx.descriptors.clone()))
            .unwrap();
        // record render graph to this command buffer
        let cmd = graph.record(cmd, &bindings, &mut ifc, None, &mut ())?.finish();
        cmd
    }
}

fn main() -> Result<()> {
    let window = WindowContext::new("01_basic")?;
    ExampleRunner::new("01_basic", Some(&window), |s| s.build())?.run::<Basic>(Some(window));
}
