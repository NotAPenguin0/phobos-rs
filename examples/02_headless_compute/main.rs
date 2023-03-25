/// This example illustrates using compute shaders in phobos, as well as
/// how phobos can be used without creating a window.

#[path = "../example_runner/lib.rs"]
mod example_runner;

use std::path::Path;

use anyhow::Result;
use phobos::prelude::*;

use crate::example_runner::{load_spirv_file, Context, ExampleApp, ExampleRunner};

struct Compute {
    buffer: Buffer,
}

impl ExampleApp for Compute {
    fn new(mut ctx: Context) -> Result<Self>
    where
        Self: Sized, {
        // Create our output buffer, we make this CpuToGpu so it is both DEVICE_LOCAL and HOST_VISIBLE
        let buffer = Buffer::new(
            ctx.device,
            &mut ctx.allocator,
            (16 * std::mem::size_of::<f32>()) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryType::CpuToGpu,
        )?;

        // Create our compute pipeline
        let shader_code = load_spirv_file(Path::new("examples/data/compute.spv"));
        let pci = ComputePipelineBuilder::new("compute")
            .set_shader(ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::COMPUTE, shader_code))
            .build();
        ctx.pipelines.lock().unwrap().create_named_compute_pipeline(pci)?;

        Ok(Self {
            buffer,
        })
    }

    fn run(&mut self, ctx: Context, _thread: ThreadContext) -> Result<()> {
        // Allocate a command buffer on the compute domain
        let cmd = ctx
            .exec
            .on_domain::<domain::Compute>(Some(ctx.pipelines.clone()), Some(ctx.descriptors.clone()))?;

        // Record some commands, then obtain a finished command buffer to submit
        let cmd = cmd
            .bind_compute_pipeline("compute")?
            .bind_storage_buffer(0, 0, &self.buffer.view_full())?
            .dispatch(4, 1, 1)?
            .finish()?;

        // Submit our command buffer and wait for its completion.
        // We could also use async-await, but we dont need to here.
        ctx.exec.submit(cmd)?.wait()?;

        let mut view = self.buffer.view_full();
        let data = view.mapped_slice::<f32>()?;
        println!("Output data: {:?}", data);

        Ok(())
    }
}

fn main() -> Result<()> {
    ExampleRunner::new("02_headless_compute", None)?.run::<Compute>(None);
}
