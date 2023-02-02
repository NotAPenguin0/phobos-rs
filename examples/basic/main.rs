use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use ash::vk;
use phobos as ph;

use winit;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoopBuilder};
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::{WindowBuilder};

use ph::IncompleteCmdBuffer; // TODO: Probably add this as a pub use to lib.rs

use futures::executor::block_on;
use phobos::GraphicsCmdBuffer;

// TODO:

// 1. CLion rust formatting

fn main_loop(frame: &mut ph::FrameManager, pipelines: &mut ph::PipelineCache, exec: &ph::ExecutionManager,
             surface: &ph::Surface, window: &winit::window::Window) -> Result<(), ph::Error> {
    // Define a virtual resource pointing to the swapchain
    let swap_resource = ph::VirtualResource::new("swapchain".to_string());
    // Define a render graph with one pass that clears the swapchain image
    let mut graph = ph::GpuTaskGraph::<ph::domain::Graphics>::new();
    let clear_pass = ph::PassBuilder::render(String::from("clear"))
        .color_attachment(swap_resource.clone(),
                          vk::AttachmentLoadOp::CLEAR,
                        Some(vk::ClearColorValue{ float32: [1.0, 0.0, 0.0, 1.0] }))
        // Our pass will render a fullscreen quad that 'clears' the screen, just so we can test pipeline creation
        .execute(|cmd| {
            cmd.bind_graphics_pipeline("simple", pipelines).unwrap()
        })
        .get();
    // Add another pass to handle presentation to the screen
    let present_pass = ph::PassBuilder::present(
        "present".to_string(),
        // This pass uses the output from the clear pass on the swap resource as its input
        clear_pass.output(&swap_resource).unwrap());
    graph.add_pass(clear_pass)?;
    graph.add_pass(present_pass)?;
    // Build the graph, now we can bind physical resources and use it.
    graph.build()?;

    block_on(frame.new_frame(&exec, window, &surface, |ifc| {
        // create physical bindings for the render graph resources
        let mut bindings = ph::PhysicalResourceBindings::new();
        bindings.bind_image("swapchain".to_string(), ifc.swapchain_image.clone());
        // create a command buffer capable of executing graphics commands
        let cmd = exec.on_domain::<ph::domain::Graphics>()?;
        // record render graph to this command buffer
        ph::record_graph(&mut graph, &bindings, cmd)?
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

fn main() -> Result<(), ph::Error> {
    let event_loop = EventLoopBuilder::new().with_any_thread(true).build();
    let window = WindowBuilder::new()
        .with_title("Phobos test app")
        .build(&event_loop)
        .unwrap();

    let settings = ph::AppBuilder::new()
        .version((1, 0, 0))
        .name(String::from("Phobos test app"))
        .validation(true)
        .window(&window) // TODO: pass window information instead of window interface to remove dependency
        .present_mode(vk::PresentModeKHR::MAILBOX)
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
    let _debug_messenger = ph::DebugMessenger::new(&instance)?;
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
         ph::FrameManager::new(device.clone(), swapchain)?
    };

    // Let's build a graphics pipeline!

    // We create a pipeline cache to store our pipeline and associated resources in.
    let mut cache = ph::PipelineCache::new(device.clone())?;

    // First, we need to load shaders
    let vtx_code = load_spirv_file(Path::new("data/vert.spv"));
    let frag_code = load_spirv_file(Path::new("data/frag.s~pv"));

    let vertex = ph::ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::VERTEX, vtx_code);
    let fragment = ph::ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, frag_code);

    // Now we can start using the pipeline builder to create our full pipeline.
    let mut pci = ph::PipelineBuilder::new("simple".to_string())
        .vertex_input(0, vk::VertexInputRate::VERTEX)
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
        .blend_attachment_none()
        .cull_mask(vk::CullModeFlags::NONE)
        .attach_shader(vertex)
        .attach_shader(fragment)
        .build();
    // For now we have to manually create descriptor set and pipeline layouts, but this will be fully automated using
    // shader reflection in a future version
    // Luckily, this pipeline does not use any descriptors, so we can simply do nothing!

    // Store the pipeline in the pipeline cache
    cache.create_named_pipeline(pci)?;

    event_loop.run(|event, _, control_flow| {
        // Do not render a frame if Exit control flow is specified, to avoid
        // sync issues.
        if let ControlFlow::ExitWithCode(_) = *control_flow { return; }
        
        main_loop(&mut frame, &mut cache, &exec, &surface, &window).unwrap();

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