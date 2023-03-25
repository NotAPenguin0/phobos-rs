/// This example illustrates using compute shaders in phobos, as well as
/// how phobos can be used without creating a window.

#[path = "../example_runner/lib.rs"]
mod example_runner;

use crate::example_runner::{Context, ExampleApp, ExampleRunner, WindowContext};
use anyhow::Result;
use phobos::prelude::*;

struct Compute {}

impl ExampleApp for Compute {
    fn new(_ctx: Context) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {})
    }
}

fn main() -> Result<()> {
    ExampleRunner::new("02_headless_compute", None)?.run::<Compute>(None);
}
