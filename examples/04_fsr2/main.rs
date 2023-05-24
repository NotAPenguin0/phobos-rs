use anyhow::Result;

use phobos::{IncompleteCmdBuffer, InFlightContext};
use phobos::domain::All;
use phobos::sync::submit_batch::SubmitBatch;

use crate::example_runner::{Context, ExampleApp, ExampleRunner, WindowContext};

#[path = "../example_runner/lib.rs"]
mod example_runner;

struct Fsr2Sample {}

impl ExampleApp for Fsr2Sample {
    fn new(ctx: Context) -> Result<Self>
        where
            Self: Sized, {
        Ok(Self {})
    }

    fn frame(&mut self, ctx: Context, ifc: InFlightContext) -> Result<SubmitBatch<All>> {
        let cmd = ctx
            .exec
            .on_domain::<All, _>(Some(ctx.pipelines.clone()), Some(ctx.descriptors.clone()))?;
        let cmd = cmd.finish()?;
        let mut batch = ctx.exec.start_submit_batch()?;
        batch.submit_for_present(cmd, &ifc)?;
        Ok(batch)
    }
}

fn main() -> Result<()> {
    let window = WindowContext::with_size("04_fsr2", 512.0, 512.0)?;
    ExampleRunner::new("04_fsr2", Some(&window), |settings| settings.build())?.run::<Fsr2Sample>(Some(window));
}
