use std::path::Path;

use anyhow::Result;

use phobos::prelude::*;
use phobos::query_pool::{PipelineStatisticsQuery, QueryPool, QueryPoolCreateInfo, TimestampQuery};

use crate::example_runner::{Context, ExampleApp, ExampleRunner, load_spirv_file};

/// This example illustrates using compute shaders in phobos, as well as
/// how phobos can be used without creating a window.

#[path = "../example_runner/lib.rs"]
mod example_runner;

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
            (1024 * 4 * std::mem::size_of::<f32>()) as u64,
            MemoryType::CpuToGpu,
        )?;

        // Create our compute pipeline
        let shader_code = load_spirv_file(Path::new("examples/data/compute.spv"));
        let pci = ComputePipelineBuilder::new("compute")
            .set_shader(ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::COMPUTE, shader_code))
            .build();
        ctx.pool.pipelines.create_named_compute_pipeline(pci)?;

        Ok(Self {
            buffer,
        })
    }

    fn run(&mut self, ctx: Context) -> Result<()> {
        // Allocate a command buffer on the compute domain
        let cmd = ctx.exec.on_domain::<domain::Compute>()?;

        // Create a query pool to record timestamps
        let mut timestamps = QueryPool::<TimestampQuery>::new(
            ctx.device.clone(),
            QueryPoolCreateInfo {
                count: 2,
                statistic_flags: None,
            },
        )?;

        // Create a query pool to record pipeline statistics
        let mut stats = QueryPool::<PipelineStatisticsQuery>::new(
            ctx.device,
            QueryPoolCreateInfo {
                count: 1,
                statistic_flags: Some(vk::QueryPipelineStatisticFlags::COMPUTE_SHADER_INVOCATIONS),
            },
        )?;

        let multiplier: f32 = 2.0;
        let query = stats.next().unwrap();

        // Record some commands, then obtain a finished command buffer to submit
        let cmd = cmd
            .write_timestamp(&mut timestamps, PipelineStage::TOP_OF_PIPE)?
            .begin_query(&stats, query)
            .bind_compute_pipeline("compute")?
            .bind_storage_buffer(0, 0, &self.buffer.view_full())?
            .push_constant(vk::ShaderStageFlags::COMPUTE, 0, &multiplier)
            .dispatch(1024, 1, 1)?
            .end_query(&stats, query)
            .write_timestamp(&mut timestamps, PipelineStage::COMPUTE_SHADER)?
            .finish()?;

        // Submit our command buffer and wait for its completion.
        // We could also use async-await, but we dont need to here.
        ctx.exec.submit(cmd)?.wait()?;

        let mut view = self.buffer.view_full();
        let data = view.mapped_slice::<f32>()?;
        println!("Output data: {data:?}");

        // Get our timestamp data
        let times = timestamps.wait_for_all_results()?;
        let start = *times.first().unwrap();
        let end = *times.last().unwrap();
        println!("Entire command buffer took {} nanoseconds", (end - start).as_nanos());
        let stats = stats.wait_for_single_result(query)?;
        println!(
            "Number of compute shader invocations: {}",
            stats.compute_shader_invocations.unwrap()
        );

        Ok(())
    }
}

fn main() -> Result<()> {
    ExampleRunner::new("02_headless_compute", None, |s| s.build())?.run::<Compute>(None);
}
