use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex};
use ash::vk;
use phobos as ph;

use winit;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoopBuilder};
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::{WindowBuilder};

use ph::IncompleteCmdBuffer; // TODO: Probably add this as a pub use to lib.rs

use futures::executor::block_on;
use phobos::{GraphicsCmdBuffer, PipelineStage};

use anyhow::Result;

// TODO:

// 1. CLion rust formatting

struct Resources {
    pub offscreen: ph::Image,
    pub offscreen_view: ph::ImageView,
    pub sampler: ph::Sampler,
}

fn main_loop(frame: &mut ph::FrameManager,
             resources: &Resources,
             pipelines: Arc<Mutex<ph::PipelineCache>>,
             descriptors: Arc<Mutex<ph::DescriptorCache>>,
             debug: &ph::DebugMessenger,
             exec: &ph::ExecutionManager,
             surface: &ph::Surface,
             window: &winit::window::Window) -> Result<()> {
    // Define a virtual resource pointing to the swapchain
    let swap_resource = ph::VirtualResource::new("swapchain".to_string());
    let offscreen = ph::VirtualResource::new("offscreen".to_string());

    let vertices: Vec<f32> = vec![
        -1.0, 1.0, 0.0, 1.0,
        -1.0, -1.0, 0.0, 0.0,
        1.0, -1.0, 1.0, 0.0,
        -1.0, 1.0, 0.0, 1.0,
        1.0, -1.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0
    ];

    // Define a render graph with one pass that clears the swapchain image
    let mut graph = ph::GpuTaskGraph::new();

    // Render pass that renders to an offscreen attachment
    let offscreen_pass = ph::PassBuilder::render(String::from("offscreen"))
        .color([1.0, 0.0, 0.0, 1.0])
        .color_attachment(offscreen.clone(), vk::AttachmentLoadOp::CLEAR, Some(vk::ClearColorValue{ float32: [1.0, 0.0, 0.0, 1.0] }))?
        .execute(|mut cmd, ifc, bindings| {
            // Our pass will render a fullscreen quad that 'clears' the screen, just so we can test pipeline creation
            let mut buffer = ifc.allocate_scratch_vbo((vertices.len() * std::mem::size_of::<f32>()) as vk::DeviceSize)?;
            let slice = buffer.mapped_slice::<f32>()?;
            slice.copy_from_slice(vertices.as_slice());
            cmd = cmd.bind_vertex_buffer(0, buffer)
                     .bind_graphics_pipeline("offscreen", pipelines.clone())?
                     .viewport(vk::Viewport{
                            x: 0.0,
                            y: 0.0,
                            width: 800.0,
                            height: 600.0,
                            min_depth: 0.0,
                            max_depth: 0.0,
                    })
                     .scissor(vk::Rect2D { offset: Default::default(), extent: vk::Extent2D { width: 800, height: 600 } })
                     .draw(6, 1, 0, 0);
            Ok(cmd)
        })
        .build();

    // Render pass that samples the offscreen attachment, and possibly does some postprocessing to it
    let sample_pass = ph::PassBuilder::render(String::from("sample"))
        .color([0.0, 1.0, 0.0, 1.0])
        .color_attachment(swap_resource.clone(),
                          vk::AttachmentLoadOp::CLEAR,
                        Some(vk::ClearColorValue{ float32: [1.0, 0.0, 0.0, 1.0] }))?
        .sample_image(offscreen_pass.output(&offscreen).unwrap(), PipelineStage::FRAGMENT_SHADER)
        .execute(|mut cmd, ifc, bindings| {
            cmd = cmd.bind_graphics_pipeline("sample", pipelines.clone()).unwrap()
                    .viewport(vk::Viewport{
                        x: 0.0,
                        y: 0.0,
                        width: 800.0,
                        height: 600.0,
                        min_depth: 0.0,
                        max_depth: 0.0,
                    })
                    .scissor(vk::Rect2D { offset: Default::default(), extent: vk::Extent2D { width: 800, height: 600 } });
            let ph::PhysicalResource::Image(offscreen_attachment) = bindings.resolve(&offscreen).unwrap() else { panic!() };
            let set = {
                let pipelines = pipelines.lock().unwrap();
                let reflection = pipelines.reflection_info("sample")?;
                ph::DescriptorSetBuilder::with_reflection(&reflection)
                    .bind_named_sampled_image("tex", offscreen_attachment.clone(), &resources.sampler)?
                    .build()
            };
            Ok(cmd.bind_new_descriptor_set(0, descriptors.clone(), set)?
                .draw(6, 1, 0, 0))
        })
        .build();
    // Add another pass to handle presentation to the screen
    let present_pass = ph::PassBuilder::present(
        "present".to_string(),
        // This pass uses the output from the clear pass on the swap resource as its input
        sample_pass.output(&swap_resource).unwrap());
    graph.add_pass(offscreen_pass)?;
    graph.add_pass(sample_pass)?;
    graph.add_pass(present_pass)?;
    // Build the graph, now we can bind physical resources and use it.
    graph.build()?;

    block_on(frame.new_frame(&exec, window, &surface, |mut ifc| {
        // create physical bindings for the render graph resources
        let mut bindings = ph::PhysicalResourceBindings::new();
        bindings.bind_image("swapchain".to_string(), ifc.swapchain_image.as_ref().unwrap().clone());
        bindings.bind_image("offscreen".to_string(), resources.offscreen_view.clone());
        // create a command buffer capable of executing graphics commands
        let cmd = exec.on_domain::<ph::domain::Graphics>()?;
        // record render graph to this command buffer
        ph::record_graph(&mut graph, &bindings, &mut ifc, cmd, Some(debug))?
            .finish()
    }))?;

    Ok(())
}

fn load_spirv_file(path: &Path) -> Vec<u32> {
    let mut f = File::open(&path).expect("no file found");
    let metadata = fs::metadata(&path).expect("unable to read metadata");
    let mut buffer = vec![0; metadata.len() as usize];
    f.read(&mut buffer).expect("buffer overflow");
    let (_, binary, _) = unsafe { buffer.align_to::<u32>() };
    Vec::from(binary)
}

fn main() -> Result<()> {
    let event_loop = EventLoopBuilder::new().with_any_thread(true).build();
    let window = WindowBuilder::new()
        .with_title("Phobos test app")
        .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0))
        .build(&event_loop)
        .unwrap();

    let settings = ph::AppBuilder::new()
        .version((1, 0, 0))
        .name(String::from("Phobos test app"))
        .validation(true)
        .window(&window) // TODO: pass window information instead of window interface to remove dependency
        .present_mode(vk::PresentModeKHR::MAILBOX)
        .scratch_size(1 * 1024) // 1 KiB scratch memory per buffer type per frame
        .gpu(ph::GPURequirements {
            dedicated: true,
            min_video_memory: 1 * 1024 * 1024 * 1024, // 1 GiB.
            min_dedicated_video_memory: 1 * 1024 * 1024 * 1024,
            queues: vec![
                ph::QueueRequest { dedicated: false, queue_type: ph::QueueType::Graphics },
                ph::QueueRequest { dedicated: true, queue_type: ph::QueueType::Transfer },
                ph::QueueRequest { dedicated: true, queue_type: ph::QueueType::Compute }
            ],
            ..Default::default()
        })
        .build();

    let instance = ph::VkInstance::new(&settings)?;
    let debug_messenger = ph::DebugMessenger::new(&instance)?;
    let (surface, physical_device) = {
        let mut surface = ph::Surface::new(&instance, &settings)?;
        let physical_device = ph::PhysicalDevice::select(&instance, Some(&surface), &settings)?;
        surface.query_details(&physical_device)?;
        (surface, physical_device)
    };
    let device = ph::Device::new(&instance, &physical_device, &settings)?;
    let mut alloc = ph::create_allocator(&instance, device.clone(), &physical_device)?;
    let exec = ph::ExecutionManager::new(device.clone(), &physical_device)?;
    let mut frame = {
        let swapchain = ph::Swapchain::new(&instance, device.clone(), &settings, &surface)?;
         ph::FrameManager::new(device.clone(), alloc.clone(), &settings, swapchain)?
    };

    // Let's build a graphics pipeline!

    // We create a pipeline cache to store our pipeline and associated resources in.
    let mut cache = ph::PipelineCache::new(device.clone())?;

    // First, we need to load shaders
    let vtx_code = load_spirv_file(Path::new("examples/data/vert.spv"));
    let frag_code = load_spirv_file(Path::new("examples/data/frag.spv"));

    let vertex = ph::ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::VERTEX, vtx_code);
    let fragment = ph::ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, frag_code);

    // Now we can start using the pipeline builder to create our full pipeline.
    let pci = ph::PipelineBuilder::new("sample".to_string())
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
    cache.lock().unwrap().create_named_pipeline(pci)?;

    let frag_code = load_spirv_file(Path::new("examples/data/blue.spv"));
    let fragment = ph::ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, frag_code);

    let pci = ph::PipelineBuilder::new("offscreen".to_string())
        .vertex_input(0, vk::VertexInputRate::VERTEX)
        .vertex_attribute(0, 0, vk::Format::R32G32_SFLOAT)?
        .vertex_attribute(0, 1, vk::Format::R32G32_SFLOAT)?
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
        .blend_attachment_none()
        .cull_mask(vk::CullModeFlags::NONE)
        .attach_shader(vertex)
        .attach_shader(fragment)
        .build();
    cache.lock().unwrap().create_named_pipeline(pci)?;
    // Define some resources we will use for rendering
    let image = ph::Image::new(device.clone(), alloc.clone(), 800, 600, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED, vk::Format::R8G8B8A8_SRGB)?;
    let mut resources = Resources {
        offscreen_view: image.view(vk::ImageAspectFlags::COLOR)?,
        offscreen: image,
        sampler: ph::Sampler::default(device.clone())?
    };

    let descriptor_cache = ph::DescriptorCache::new(device.clone())?;

    event_loop.run(move |event, _, control_flow| {
        // Do not render a frame if Exit control flow is specified, to avoid
        // sync issues.
        if let ControlFlow::ExitWithCode(_) = *control_flow { return; }

        main_loop(&mut frame, &resources, cache.clone(), descriptor_cache.clone(), &debug_messenger, &exec, &surface, &window).unwrap();

        *control_flow = ControlFlow::Poll;

        cache.lock().unwrap().next_frame();
        descriptor_cache.lock().unwrap().next_frame();

        // Note that we want to handle events after processing our current frame, so that
        // requesting an exit doesn't attempt to render another frame, which causes
        // sync issues.
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id
            } if window_id == window.id() => {
                *control_flow = ControlFlow::Exit;
                device.wait_idle().unwrap();
            },
            _ => (),
        }
    })
}