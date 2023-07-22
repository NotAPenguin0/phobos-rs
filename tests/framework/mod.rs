use std::sync::Arc;

use anyhow::Result;

use phobos::{
    Allocator, AppBuilder, DefaultAllocator, Device, ExecutionManager, GPURequirements, Instance,
    PhysicalDevice, QueueRequest, QueueType,
};
use phobos::pool::ResourcePool;
use phobos::wsi::window::HeadlessWindowInterface;

#[derive(Clone, Debug)]
pub struct Context<A: Allocator> {
    pub exec: ExecutionManager<A>,
    pub pool: ResourcePool<A>,
    pub allocator: A,
    pub device: Device,
    pub phys_device: Arc<PhysicalDevice>,
    pub instance: Arc<Instance>,
}

/// Creates a headless phobos context ready for automated tests
pub fn make_context() -> Result<Context<DefaultAllocator>> {
    make_context_with_queues([QueueRequest {
        dedicated: false,
        queue_type: QueueType::Graphics,
    }])
}

/// Create a headless phobos context and request some queues
pub fn make_context_with_queues(
    queues: impl Into<Vec<QueueRequest>>,
) -> Result<Context<DefaultAllocator>> {
    let settings = AppBuilder::<HeadlessWindowInterface>::new()
        .name("phobos test framework")
        .version((0, 0, 1))
        .validation(false)
        .scratch_size(1024 as u64)
        .gpu(GPURequirements {
            dedicated: false,
            min_video_memory: 0,
            min_dedicated_video_memory: 0,
            queues: queues.into(),
            features: Default::default(),
            features_1_1: Default::default(),
            features_1_2: Default::default(),
            features_1_3: Default::default(),
            device_extensions: vec![],
        })
        .build();
    let (instance, phys_device, None, device, allocator, pool, exec, None, None) =
        phobos::initialize(&settings, true)? else {
        panic!("test framework: requested headless non-debug context but got debug context or a window.");
    };

    Ok(Context {
        instance: Arc::new(instance),
        phys_device: Arc::new(phys_device),
        device,
        allocator,
        pool,
        exec,
    })
}

pub fn make_context_with_settings<
    F: FnOnce(AppBuilder<HeadlessWindowInterface>) -> AppBuilder<HeadlessWindowInterface>,
>(
    callback: F,
) -> Result<Context<DefaultAllocator>> {
    let builder = AppBuilder::<HeadlessWindowInterface>::new()
        .name("phobos test framework")
        .version((0, 0, 1))
        .validation(false)
        .scratch_size(1024 as u64)
        .gpu(GPURequirements {
            dedicated: false,
            min_video_memory: 0,
            min_dedicated_video_memory: 0,
            queues: vec![QueueRequest {
                dedicated: false,
                queue_type: QueueType::Graphics,
            }],
            features: Default::default(),
            features_1_1: Default::default(),
            features_1_2: Default::default(),
            features_1_3: Default::default(),
            device_extensions: vec![],
        });

    let settings = callback(builder).build();
    let (instance, phys_device, None, device, allocator, pool, exec, None, None) =
        phobos::initialize(&settings, true)? else {
        panic!("test framework: requested headless non-debug context but got debug context or a window.");
    };

    Ok(Context {
        instance: Arc::new(instance),
        phys_device: Arc::new(phys_device),
        device,
        allocator,
        pool,
        exec,
    })
}
