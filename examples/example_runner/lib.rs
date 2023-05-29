use std::fs;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{bail, Result};
use futures::executor::block_on;
use glam::{Mat4, Vec3};
use layout::backends::svg::SVGWriter;
use layout::gv;
use layout::gv::GraphBuilder;
use winit::event::{
    ElementState, Event, MouseButton, MouseScrollDelta, VirtualKeyCode, WindowEvent,
};
use winit::event_loop::{ControlFlow, EventLoop, EventLoopBuilder};
use winit::window::{Window, WindowBuilder};

use phobos::pool::ResourcePool;
use phobos::prelude::*;
use phobos::sync::submit_batch::SubmitBatch;

pub fn front_direction(rotation: Vec3) -> Vec3 {
    let cos_pitch = rotation.x.cos();
    let cos_yaw = rotation.y.cos();
    let sin_pitch = rotation.x.sin();
    let sin_yaw = rotation.y.sin();

    Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize()
}

#[derive(Debug, Copy, Clone)]
pub struct Camera {
    pub position: Vec3,
    pub rotation: Vec3,
    middle: bool,
    shift: bool,
    prev_position: (f32, f32),
}

impl Camera {
    pub fn new(position: Vec3, rotation: Vec3) -> Self {
        Self {
            position,
            rotation,
            middle: false,
            shift: false,
            prev_position: (0.0, 0.0),
        }
    }

    fn clamp_rotation(rot: Vec3) -> Vec3 {
        const MAX_ANGLE: f32 = std::f32::consts::PI / 2.0 - 0.0001;
        const UNBOUNDED: f32 = f32::MAX;
        rot.clamp(Vec3::new(-MAX_ANGLE, -UNBOUNDED, 0.0), Vec3::new(MAX_ANGLE, UNBOUNDED, 0.0))
    }

    pub fn update_position(&mut self, pos: Vec3) {
        self.position += pos;
    }

    pub fn update_rotation(&mut self, rot: Vec3) {
        self.rotation += rot;
        self.rotation = Self::clamp_rotation(self.rotation);
    }

    fn handle_move(&mut self, dx: f32, dy: f32) {
        const SPEED: f32 = 0.1;
        let delta = self.up() * (dy as f32) + self.right() * (-dx as f32);
        self.update_position(delta * SPEED);
    }

    fn handle_rotate(&mut self, dx: f32, dy: f32) {
        const SPEED: f32 = 0.01;
        let delta = Vec3::new(-dy as f32, dx as f32, 0.0);
        self.update_rotation(delta * SPEED);
    }

    fn handle_scroll(&mut self, dy: f32) {
        const SPEED: f32 = 0.2;
        let delta = self.front() * dy;
        self.update_position(delta * SPEED);
    }

    pub fn controls(&mut self, event: &Event<()>) {
        match event {
            Event::WindowEvent {
                event,
                ..
            } => match event {
                WindowEvent::MouseWheel {
                    delta,
                    ..
                } => {
                    if let MouseScrollDelta::PixelDelta(pos) = delta {
                        self.handle_scroll(pos.y as f32);
                    } else if let MouseScrollDelta::LineDelta(_, y) = delta {
                        self.handle_scroll(*y);
                    }
                }
                WindowEvent::KeyboardInput {
                    input,
                    ..
                } => {
                    if let Some(keycode) = input.virtual_keycode {
                        if keycode == VirtualKeyCode::LShift {
                            self.shift = input.state == ElementState::Pressed;
                        }
                    }
                }
                WindowEvent::CursorMoved {
                    position,
                    ..
                } => {
                    let x = position.x as f32;
                    let y = position.y as f32;
                    let dx = x - self.prev_position.0;
                    let dy = y - self.prev_position.1;
                    if self.middle {
                        if self.shift {
                            self.handle_move(dx, dy);
                        } else {
                            self.handle_rotate(dx, dy);
                        }
                    }
                    self.prev_position.0 = x;
                    self.prev_position.1 = y;
                }
                WindowEvent::MouseInput {
                    button,
                    state,
                    ..
                } => {
                    if *button == MouseButton::Middle {
                        self.middle = *state == ElementState::Pressed;
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    pub fn front(&self) -> Vec3 {
        front_direction(self.rotation)
    }

    pub fn right(&self) -> Vec3 {
        self.front().cross(Vec3::new(0.0, 1.0, 0.0)).normalize()
    }

    pub fn up(&self) -> Vec3 {
        self.right().cross(self.front())
    }

    pub fn matrix(&self) -> Mat4 {
        let front = self.front();
        let up = self.up();
        Mat4::look_at_rh(self.position, self.position + front, up)
    }
}

#[macro_export]
macro_rules! ubo_struct_assign {
    (
        $var:ident,
        $pool:ident,
        struct $name:ident {
            $(
                $fname:ident:$ftype:ty = $finit:expr,
            )*
        }) => {
        concat_idents::concat_idents!(buffer_name = $var, _, buffer {
            #[repr(C)]
            struct $name {
                $($fname:$ftype,)*
            }

            let mut buffer_name = $pool.allocate_scratch_ubo(std::mem::size_of::<$name>() as vk::DeviceSize)?;
            let $var = buffer_name.mapped_slice::<$name>()?;
            let mut $var = $var.get_mut(0).unwrap();

            $(
                $var.$fname = $finit;
            )*
        });
    };
}

#[allow(dead_code)]
pub fn load_spirv_file(path: &Path) -> Vec<u32> {
    let mut f = File::open(&path).expect("no file found");
    let metadata = fs::metadata(&path).expect("unable to read metadata");
    let mut buffer = vec![0; metadata.len() as usize];
    f.read(&mut buffer).expect("buffer overflow");
    let (_, binary, _) = unsafe { buffer.align_to::<u32>() };
    Vec::from(binary)
}

#[allow(dead_code)]
pub fn create_shader(path: &str, stage: vk::ShaderStageFlags) -> ShaderCreateInfo {
    let code = load_spirv_file(Path::new(path));
    ShaderCreateInfo::from_spirv(stage, code)
}

#[allow(dead_code)]
pub fn save_dotfile<G>(graph: &G, path: &str)
where
    G: GraphViz, {
    let dot = graph.dot().unwrap();
    let dot = format!("{}", dot);
    let mut parser = gv::DotParser::new(&dot);
    match parser.process() {
        Ok(g) => {
            let mut svg = SVGWriter::new();
            let mut builder = GraphBuilder::new();
            builder.visit_graph(&g);
            let mut vg = builder.get();
            vg.do_it(false, false, false, &mut svg);
            let svg = svg.finalize();
            let mut f = File::create(Path::new(path)).unwrap();
            f.write(&svg.as_bytes()).unwrap();
        }
        Err(e) => {
            parser.print_error();
            println!("dot render error: {}", e);
        }
    }
}

#[allow(dead_code)]
pub fn staged_buffer_upload<T: Copy>(
    mut ctx: Context,
    data: &[T],
    usage: vk::BufferUsageFlags,
) -> Result<Buffer> {
    let staging = Buffer::new(
        ctx.device.clone(),
        &mut ctx.allocator,
        data.len() as u64 * std::mem::size_of::<T>() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryType::CpuToGpu,
    )?;

    let mut staging_view = staging.view_full();
    staging_view.mapped_slice()?.copy_from_slice(data);

    let buffer = Buffer::new_device_local(
        ctx.device,
        &mut ctx.allocator,
        staging.size(),
        vk::BufferUsageFlags::TRANSFER_DST | usage,
    )?;
    let view = buffer.view_full();

    let cmd = ctx
        .exec
        .on_domain::<domain::Transfer>()?
        .copy_buffer(&staging_view, &view)?
        .finish()?;

    ctx.exec.submit(cmd)?.wait()?;
    Ok(buffer)
}

#[derive(Debug)]
pub struct WindowContext {
    pub event_loop: EventLoop<()>,
    pub window: Window,
}

impl WindowContext {
    #[allow(dead_code)]
    pub fn new(title: impl Into<String>) -> Result<Self> {
        Self::with_size(title, 800.0, 600.0)
    }

    pub fn with_size(title: impl Into<String>, width: f32, height: f32) -> Result<Self> {
        let event_loop = EventLoopBuilder::new().build();
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
            .build(&event_loop)?;
        Ok(Self {
            event_loop,
            window,
        })
    }
}

pub struct VulkanContext {
    pub frame: Option<FrameManager>,
    pub pool: ResourcePool,
    pub exec: ExecutionManager,
    pub allocator: DefaultAllocator,
    pub device: Device,
    pub physical_device: PhysicalDevice,
    pub surface: Option<Surface>,
    pub debug_messenger: DebugMessenger,
    pub instance: VkInstance,
}

#[derive(Clone)]
pub struct Context {
    pub device: Device,
    pub exec: ExecutionManager,
    pub allocator: DefaultAllocator,
    pub pool: ResourcePool,
}

pub trait ExampleApp {
    fn new(ctx: Context) -> Result<Self>
    where
        Self: Sized;

    // Implement this for a windowed application
    fn frame(&mut self, _ctx: Context, _ifc: InFlightContext) -> Result<SubmitBatch<domain::All>> {
        bail!("frame() not implemented for non-headless example app");
    }

    // Implement this for a headless application
    fn run(&mut self, _ctx: Context) -> Result<()> {
        bail!("run() not implemented for headless example app");
    }

    fn handle_event(&mut self, _event: &Event<()>) -> Result<()> {
        Ok(())
    }
}

pub struct ExampleRunner {
    vk: VulkanContext,
}

impl ExampleRunner {
    pub fn new(
        name: impl Into<String>,
        window: Option<&WindowContext>,
        make_settings: impl Fn(AppBuilder<Window>) -> AppSettings<Window>,
    ) -> Result<Self> {
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
        let settings = make_settings(settings);

        let (instance, physical_device, surface, device, allocator, pool, exec, frame, Some(debug_messenger)) = initialize(&settings, window.is_none())? else {
            panic!("Asked for debug messenger but didnt get one")
        };

        let vk = VulkanContext {
            frame,
            pool,
            exec,
            allocator,
            device,
            physical_device,
            surface,
            debug_messenger,
            instance,
        };

        Ok(Self {
            vk,
        })
    }

    fn run_headless<E: ExampleApp + 'static>(self, mut app: E) -> ! {
        app.run(self.make_context()).unwrap();
        self.vk.device.wait_idle().unwrap();
        drop(app);
        std::process::exit(0);
    }

    fn make_context(&self) -> Context {
        Context {
            device: self.vk.device.clone(),
            exec: self.vk.exec.clone(),
            allocator: self.vk.allocator.clone(),
            pool: self.vk.pool.clone(),
        }
    }

    fn frame<E: ExampleApp + 'static>(&mut self, app: &mut E, window: &Window) -> Result<()> {
        let ctx = self.make_context();
        let frame = self.vk.frame.as_mut().unwrap();
        let surface = self.vk.surface.as_ref().unwrap();
        block_on(
            frame.new_frame(self.vk.exec.clone(), window, surface, |ifc| app.frame(ctx, ifc)),
        )?;

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
                self.vk.device.wait_idle().unwrap();
                return;
            }
            *control_flow = ControlFlow::Poll;

            match &mut app {
                None => {}
                Some(app) => {
                    app.handle_event(&event).unwrap();
                }
            }

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
                        self.vk.pool.next_frame();
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
