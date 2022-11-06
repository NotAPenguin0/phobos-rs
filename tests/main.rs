use ash::vk::PresentModeKHR;
use phobos as ph;

use winit;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop, EventLoopBuilder};
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::{Window, WindowBuilder};

use futures::executor::block_on;

#[test]
fn create_context() {
    let event_loop = EventLoopBuilder::new().with_any_thread(true).build();
    let window = WindowBuilder::new()
        .with_title("Phobos test app")
        .build(&event_loop)
        .unwrap();

    let ctx = {
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

        ph::Context::new(settings).unwrap()
    };

    event_loop.run(move |event, _, control_flow| {
        block_on(async {
            *control_flow = ControlFlow::Wait;

            let ifc = ctx.frame.new_frame().await;
            println!("{:?}", ifc);

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id
                } if window_id == window.id() => *control_flow = ControlFlow::Exit,
                _ => (),
            }
        });
    });
}