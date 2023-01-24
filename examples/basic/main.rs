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

fn main_loop(frame: &mut ph::FrameManager, exec: &ph::ExecutionManager, 
             surface: &ph::Surface, window: &winit::window::Window) -> Result<(), ph::Error> {

    block_on(frame.new_frame(&exec, window, &surface, |ifc| {
        exec.on_domain::<ph::domain::Graphics>()?
        .transition_image(&ifc.swapchain_image, 
            vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags::empty(), vk::AccessFlags::empty())
        .finish()
    }))?;

    Ok(())
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
        .present_mode(PresentModeKHR::MAILBOX)
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

    event_loop.run(move |event, _, control_flow| {
        // Do not render a frame if Exit control flow is specified, to avoid
        // sync issues.
        if let ControlFlow::ExitWithCode(_) = *control_flow { return; }
        
        main_loop(&mut frame, &exec, &surface, &window).unwrap();

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