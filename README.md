# Phobos

![build](https://github.com/NotAPenguin0/phobos-rs/.github/workflows/rust.yml/badge.svg)

Phobos is a fast, powerful Vulkan abstraction library. It provides abstractions to automatically
manage common Vulkan problems like synchronization and resource management. At the same time, it aims to 
expose the full Vulkan API without major limitations.

At the moment, the project is highly WIP, and not all these goals have been fully achieved yet. It is developed
together with a rendering engine using it, so features are currently added as needed.

The abstraction level of Phobos sits a bit above [Vulkano](https://crates.io/crates/vulkano). While the full API 
is exposed, Phobos provides many powerful quality-of-life features sitting on top of it (see below) that Vulkano does not implement.
If you are simply looking for a safe, low-level wrapper around Vulkan, Vulkano is the better choice.

## What does Phobos do?

- All Vulkan initialization from a single configuration structure.
- Manage per-frame synchronization with the presentation engine.
- GPU futures, fully integrated with Rust futures.
  - More formally, `Future` is implemented for `phobos::Fence`.
  - There is also a `GpuFuture<T>` which can be used to attach a future value to a fence.
- Provide a task graph that can be used to automatically synchronize resources in your renderer.
  - Automatic image layout transitions.
  - Automatic renderpass declarations.
  - Automatic memory barriers for buffers.
  - Virtual resources, meaning actual resources are only bound to a graph at record time. This allows general-purpose graphs to be re-used if desired.
- Safe wrappers for Vulkan objects.
- Automatic descriptor set management.
- Automatic pipeline and pipeline layout management.
- Shader reflection to make binding descriptors easy.
- A linear allocator for per-frame allocations like uniform buffers.
- Typed command buffers per queue type.
- Automatically thread safe command buffer recording.

## What does Phobos not do?

Phobos is not a renderer, it does not implement any visual features. It's intended as a library to help you 
write a Vulkan renderer more easily and correctly, without hiding important API details.

## Example

For more elaborate examples, please check the [examples](examples) folder.

```rust 
use phobos as ph;

fn main() {
    // Fill out app settings for initialization
    let settings = ph::AppBuilder::new()
        .version((1, 0, 0))
        .name("Phobos example app")
        .validation(true)
        .window(&window) // Your winit window, or some other interface.
        .present_mode(vk::PresentModeKHR::MAILBOX)
        .scratch_size(1024)
        .gpu(ph::GPURequirements {
          dedicated: true,
          queues: vec![
            ph::QueueRequest { dedicated: false, queue_type: ph::QueueType::Graphics },
            ph::QueueRequest { dedicated: true, queue_type: ph::QueueType::Transfer },
            ph::QueueRequest { dedicated: true, queue_type: ph::QueueType::Compute }
          ],
          ..Default::default()
        })
        .build();
  
    // Initialize Vulkan. This is generally always going to be the same for every project, but it is 
    // not abstracted away to allow keeping each created object separately.
    let instance = ph::VkInstance::new(&settings)?;
    let debug_messenger = ph::DebugMessenger::new(&instance)?;
    let (surface, physical_device) = {
      let mut surface = ph::Surface::new(&instance, &settings)?;
      let physical_device = ph::PhysicalDevice::select(&instance, Some(&surface), &settings)?;
      surface.query_details(&physical_device)?;
      (surface, physical_device)
    };
    let device = ph::Device::new(&instance, &physical_device, &settings)?;
    let mut alloc = ph::create_allocator(&instance, device.clone(), &physical_device)?;
    let exec = ph::ExecutionManager::new(device.clone(), &physical_device)?;
    let mut frame = {
      let swapchain = ph::Swapchain::new(&instance, device.clone(), &settings, &surface)?;
      ph::FrameManager::new(device.clone(), alloc.clone(), &settings, swapchain)?
    };

    // Create a new pass graph for rendering. Note how we only do this once, as 
    // we are using virtual resources that do not depend on the frame.
    let swapchain = ph::VirtualResource::image("swapchain");
    let clear_pass = ph::PassBuilder::render("clear")
            .color_attachment(swapchain.clone(), vk::AttachmentLoadOp::CLEAR,
                              // Clear the swapchain to red.
                              Some(vk::ClearColorValue{ float32: [1.0, 0.0, 0.0, 1.0] }))?
            .build();
    let present_pass = ph::PassBuilder::present("present", clear_pass.output(&swapchain).unwrap());
    let graph = ph::PassGraph::new()
            .add_pass(clear_pass)?
            .add_pass(present_pass)?
            .build()?;
    // Your event loop goes here
    while event_loop {
      // Wait for a new frame to be available. Once there is one, the provided
      // callback will be called.
      futures::executor::block_on(frame.new_frame(exec.clone(), window, &surface, |mut ifc| {
            // Bind some physical resources to the render graph.
            let mut bindings = ph::PhysicalResourceBindings::new();
            bindings.bind_image("swapchain", ifc.swapchain_image.as_ref().unwrap().clone());
            let cmd = exec.on_domain::<ph::domain::Graphics>()?;
            // Record render graph to our command buffer
            ph::record_graph(&mut graph, &bindings, &mut ifc, cmd, None).finish()
      }))?;
    }
}
```

## Support

Visit the [docs.rs](https://docs.rs/phobos/latest) page, or open an issue.

## Planned features

- Compute shader support
- Raytracing support
- Automatic semaphore synchronization
- Expose more Vulkan API features.
