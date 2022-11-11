use ash::vk::PresentModeKHR;
use ash::vk;
use phobos as ph;

use winit;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoopBuilder};
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::{WindowBuilder};

use futures::executor::block_on;
use phobos::{IncompleteCmdBuffer};

// TODO:

// 1. CLion rust formatting

fn main_loop() -> Result<(), ph::Error> {
    // TODO: implement
    Ok(())
}


fn main() -> Result<(), ph::Error> {
    let event_loop = EventLoopBuilder::new().with_any_thread(true).build();
    let window = WindowBuilder::new()
        .with_title("Phobos test app")
        .build(&event_loop)
        .unwrap();

    // TODO: Use builder pattern
    // TODO: Better defaults (add custom Default impl)
    let settings = ph::AppSettings {
        version: (1, 0, 0), // TODO: Use env! instead
        name: String::from("Phobos test app"),
        enable_validation: true,
        // TODO: pass relevant window data instead of 
        // interface. This would also entirely remove the dependency on
        // winit.
        window: Some(&window),
        surface_format: None, // Use default fallback format.
        present_mode: Some(PresentModeKHR::MAILBOX),
        gpu_requirements: ph::GPURequirements {
            dedicated: true,
            min_video_memory: 1 * 1024 * 1024 * 1024, // 1 GiB.
            min_dedicated_video_memory: 1 * 1024 * 1024 * 1024,
            queues: vec![
                ph::QueueRequest { dedicated: false, queue_type: ph::QueueType::Graphics },
                ph::QueueRequest { dedicated: true, queue_type: ph::QueueType::Transfer },
                ph::QueueRequest { dedicated: true, queue_type: ph::QueueType::Compute }
            ],
            ..Default::default()
        }
    };

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

    event_loop.run(move |event, _, control_flow| {
        // Do not render a frame if Exit control flow is specified, to avoid
        // sync issues.
        if let ControlFlow::ExitWithCode(_) = *control_flow { return; }
        // *control_flow = ControlFlow::Wait;

        // Acquire new frame
        // TODO: add swapchain image reference to ifc.
        let ifc = block_on(frame.new_frame(&exec, &window, &surface)).unwrap();
        // Do some work for this frame

        // new_frame() takes closure
        // give ifc to closure
        // closure returns command buffer
        // submit + present
        // frame.new_frame(&exec, &window, &surface, async |ifc| /* -> CommandBuffer */ { ... }).await.unwrap();
        // task.new_task(&exec, &window, &surface, async |ctx| /* -> CommandBuffer */ { ... }).await.unwrap();
        // task.new_task(&exec, async |ctx| /* -> CommandBuffer */ { ... })
        // auto sync tasks w semaphore/fence
        // .then(&exec, |ctx| ... etc).await.unwrap();
        
        // graph.add_pass(|cmdbuf| -> cmdbuf {...})

        // This is actually safe, but not great.
        // Move to IFC for no more issue.
        let swapchain = unsafe { frame.get_swapchain_image(&ifc).unwrap() };
        let commands = exec.on_domain::<ph::domain::Graphics>().unwrap()
            // not final, just a command to run
            .transition_image(&swapchain, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags::empty(), vk::AccessFlags::empty())
            .finish()
            .unwrap();

        // submit() to queue outside of frame

        // Submit this frame's commands
        // TODO: move this to IFC.
        frame.submit(commands, &exec).unwrap();
        // Present
        frame.present(&exec).unwrap();

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