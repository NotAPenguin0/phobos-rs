use anyhow::Result;

use phobos::{CommandBuffer, IncompleteCmdBuffer, InFlightContext};
use phobos::domain::All;

use crate::example_runner::{Context, ExampleApp, ExampleRunner, WindowContext};

#[path = "../example_runner/lib.rs"]
mod example_runner;

struct RaytracingSample {}

impl ExampleApp for RaytracingSample {
    fn new(ctx: Context) -> Result<Self>
        where
            Self: Sized, {
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
