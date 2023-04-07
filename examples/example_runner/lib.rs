use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{bail, Result};
use futures::executor::block_on;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop, EventLoopBuilder};
use winit::window::{Window, WindowBuilder};

use phobos::prelude::*;

pub fn load_spirv_file(path: &Path) -> Vec<u32> {
    let mut f = File::open(&path).expect("no file found");
    let metadata = fs::metadata(&path).expect("unable to read metadata");
    let mut buffer = vec![0; metadata.len() as usize];
    f.read(&mut buffer).expect("buffer overflow");
    let (_, binary, _) = unsafe { buffer.align_to::<u32>() };
    Vec::from(binary)
}

#[derive(Debug)]
pub struct WindowContext {
    pub event_loop: EventLoop<()>,
    pub window: Window,
}

impl WindowContext {
    #[allow(dead_code)]
    pub fn new(title: impl Into<String>) -> Result<Self> {
        let event_loop = EventLoopBuilder::new().build();
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0))
            .build(&event_loop)?;
        Ok(Self {
            event_loop,
            window,
        })
    }
}

pub struct VulkanContext {
    pub frame: Option<FrameManager>,
    pub exec: ExecutionManager,
    pub allocator: DefaultAllocator,
    pub device: Device,
    pub physical_device: PhysicalDevice,
    pub surface: Option<Surface>,
    pub debug_messenger: DebugMessenger,
    pub instance: VkInstance,
}

pub struct Context {
    pub device: Device,
    pub exec: ExecutionManager,
    pub allocator: DefaultAllocator,
    pub pipelines: PipelineCache,
    pub descriptors: DescriptorCache,
}

pub trait ExampleApp {
    fn new(ctx: Context) -> Result<Self>
    where
        Self: Sized;

    // Implement this for a windowed application
    fn frame(&mut self, _ctx: Context, _ifc: InFlightContext) -> Result<CommandBuffer<domain::All>> {
        bail!("frame() not implemented for non-headless example app");
    }

    // Implement this for a headless application
    fn run(&mut self, _ctx: Context, _thread: ThreadContext) -> Result<()> {
        bail!("run() not implemented for headless example app");
    }
}

pub struct ExampleRunner {
    vk: VulkanContext,
    pipelines: PipelineCache,
    descriptors: DescriptorCache,
}

impl ExampleRunner {
    pub fn new(name: impl Into<String>, window: Option<&WindowContext>) -> Result<Self> {
        std::env::set_var("RUST_LOG", "trace");
        pretty_env_logger::init();
        let mut settings = AppBuilder::new()
            .version((1, 0, 0))
            .name(name)
            .validation(true)
            .present_mode(vk::PresentModeKHR::MAILBOX)
            .scratch_size(1 * 1024u64) // 1 KiB scratch memory per buffer type per frame
            .gpu(GPURequirements {
                dedicated: false,
                min_video_memory: 1 * 1024 * 1024 * 1024, // 1 GiB.
                min_dedicated_video_memory: 1 * 1024 * 1024 * 1024,
                queues: vec![
                    QueueRequest {
                        dedicated: false,
                        queue_type: QueueType::Graphics,
                    },
                    QueueRequest {
                        dedicated: true,
                        queue_type: QueueType::Transfer,
                    },
                    QueueRequest {
                        dedicated: true,
                        queue_type: QueueType::Compute,
                    },
                ],
                ..Default::default()
            });

        match window {
            None => {}
            Some(window) => {
                settings = settings.window(&window.window);
            }
        };
        let settings = settings.build();

        let (instance, physical_device, surface, device, allocator, exec, frame, Some(debug_messenger)) = initialize(&settings, window.is_none())? else {
            panic!("Asked for debug messenger but didnt get one")
        };
        let vk = VulkanContext {
            frame,
            exec,
            allocator,
            device,
            physical_device,
            surface,
            debug_messenger,
            instance,
        };

        let pipelines = PipelineCache::new(vk.device.clone())?;
        let descriptors = DescriptorCache::new(vk.device.clone())?;

        Ok(Self {
            vk,
            pipelines,
            descriptors,
        })
    }

    fn run_headless<E: ExampleApp + 'static>(self, mut app: E) -> ! {
        let ctx = self.make_context();
        let thread = ThreadContext::new(ctx.device.clone(), ctx.allocator.clone(), Some(1024 * 1024u64)).unwrap();
        app.run(self.make_context(), thread).unwrap();
        self.vk.device.wait_idle().unwrap();
        drop(app);
        std::process::exit(0);
    }

    fn make_context(&self) -> Context {
        Context {
            device: self.vk.device.clone(),
            exec: self.vk.exec.clone(),
            allocator: self.vk.allocator.clone(),
            pipelines: self.pipelines.clone(),
            descriptors: self.descriptors.clone(),
        }
    }

    fn frame<E: ExampleApp + 'static>(&mut self, app: &mut E, window: &Window) -> Result<()> {
        let ctx = self.make_context();
        let frame = self.vk.frame.as_mut().unwrap();
        let surface = self.vk.surface.as_ref().unwrap();
        block_on(frame.new_frame(self.vk.exec.clone(), window, surface, |ifc| app.frame(ctx, ifc)))?;

        Ok(())
    }

    fn run_windowed<E: ExampleApp + 'static>(mut self, app: E, window: WindowContext) -> ! {
        let event_loop = window.event_loop;
        let window = window.window;
        let mut app = Some(app);
        event_loop.run(move |event, _, control_flow| {
            // Do not render a frame if Exit control flow is specified, to avoid
            // sync issues.
            if let ControlFlow::ExitWithCode(_) = *control_flow {
                return;
            }
            *control_flow = ControlFlow::Poll;

            // Note that we want to handle events after processing our current frame, so that
            // requesting an exit doesn't attempt to render another frame, which causes
            // sync issues.
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window_id == window.id() => {
                    *control_flow = ControlFlow::Exit;
                    self.vk.device.wait_idle().unwrap();
                    let app = app.take();
                    match app {
                        None => {}
                        Some(app) => {
                            drop(app);
                        }
                    }
                }
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(_) => match app.as_mut() {
                    None => {}
                    Some(app) => {
                        self.frame(app, &window).unwrap();
                        self.pipelines.next_frame();
                        self.descriptors.next_frame();
                    }
                },
                _ => (),
            }
        })
    }

    pub fn run<E: ExampleApp + 'static>(self, window: Option<WindowContext>) -> ! {
        let app = E::new(self.make_context()).unwrap();
        match window {
            None => self.run_headless(app),
            Some(window) => self.run_windowed(app, window),
        }
    }
}
