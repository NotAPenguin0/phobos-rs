use anyhow::Result;
use ash::vk;

use phobos::{Buffer, CommandBuffer, IncompleteCmdBuffer, InFlightContext};
use phobos::acceleration_structure::{AccelerationStructure, AccelerationStructureType};
use phobos::domain::All;

use crate::example_runner::{Context, ExampleApp, ExampleRunner, WindowContext};

#[path = "../example_runner/lib.rs"]
mod example_runner;

struct RaytracingSample {}

impl ExampleApp for RaytracingSample {
    fn new(mut ctx: Context) -> Result<Self>
        where
            Self: Sized, {
        let buffer = Buffer::new_device_local(ctx.device.clone(), &mut ctx.allocator, 128u64, vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR)?;
        let view = buffer.view_full();
        let accel = AccelerationStructure::new(ctx.device, AccelerationStructureType::TopLevel, view, Default::default())?;
        Ok(Self {})
    }

    fn frame(&mut self, ctx: Context, _ifc: InFlightContext) -> Result<CommandBuffer<All>> {
        let cmd = ctx.exec.on_domain::<All>(None, None)?;
        cmd.finish()
    }
}

fn main() -> Result<()> {
    let window = WindowContext::new("01_basic")?;
    ExampleRunner::new("03_raytracing", Some(&window), |settings| settings.raytracing(true).build())?.run::<RaytracingSample>(Some(window));
}
