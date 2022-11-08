use ash::vk::PresentModeKHR;
use phobos as ph;

use winit;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoopBuilder};
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::{WindowBuilder};

use futures::executor::block_on;
use phobos::{GraphicsCmdBuffer, IncompleteCmdBuffer};

#[test]
fn main() -> Result<(), ph::Error> {
    let event_loop = EventLoopBuilder::new().with_any_thread(true).build();
    let window = WindowBuilder::new()
        .with_title("Phobos test app")
        .build(&event_loop)
        .unwrap();

    let settings = ph::AppSettings {
        version: (1, 0, 0),
        name: String::from("Phobos test app"),
        enable_validation: true,
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
            features: Default::default(),
            features_1_1: Default::default(),
            features_1_2: Default::default(),
            device_extensions: Default::default()
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
    let exec = ph::ExecutionManager::new(device.clone(), &physical_device)?;
    let mut frame = {
        let swapchain = ph::Swapchain::new(&instance, device.clone(), &settings, &surface)?;
         ph::FrameManager::new(device.clone(), swapchain)?
    };

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id
            } if window_id == window.id() => *control_flow = ControlFlow::Exit,
            _ => (),
        }

        // Acquire new frame
        let ifc = block_on(frame.new_frame()).unwrap();
        // Do some work for this frame
        let commands =
            exec.on_domain::<ph::domain::Graphics>().unwrap()
            // doesn't do anything yet, just a sample function
            .draw()
            .finish()
            .unwrap();
        // Submit this frame's commands
        frame.submit(commands, &exec).unwrap();
        // Present
        frame.present(&exec).unwrap();
    });
}