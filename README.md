# Phobos

![build](https://github.com/NotAPenguin0/phobos-rs/actions/workflows/rust.yml/badge.svg)

Phobos is a Vulkan abstraction library aiming to create Vulkan applications more easily. It provides abstractions to automatically
manage common Vulkan problems like synchronization and resource management. At the same time, it aims to 
expose the full Vulkan API without major limitations.

At the moment, the project is highly WIP, and not all these goals have been fully achieved yet. It is developed
together with a rendering engine using it, so features are currently added as needed.

## What does Phobos do?

- All Vulkan initialization from a single configuration structure.
- Manage per-frame synchronization with the presentation engine.
- GPU futures, fully integrated with Rust futures.
  - More formally, `Future<Output = T>` is implemented for `phobos::Fence<T>`.
- Provide a task graph that can be used to automatically synchronize resources in your renderer.
  - Automatic image layout transitions.
  - Automatic renderpass declarations.
  - Automatic memory barriers for buffers.
  - Virtual resources, meaning actual resources are only bound to a graph at record time. This allows general-purpose graphs to be re-used if desired.
- Safe wrappers for Vulkan objects.
- No `unsafe` in the public API, except when accessing raw Vulkan handles.
- Descriptor sets are completely hidden. Simply bind resources directly to the command buffer.
- Automatic pipeline management.
- Shader reflection to automatically generate pipeline layouts.
- Automatic double buffering of resources that need it.
- A linear allocator for per-frame allocations like uniform buffers.
- Typed command buffers per queue type.
- Automatically thread safe command buffer recording.
- Easily batch together submits into one `vkQueueSubmit` call and synchronize them with semaphores using
  the `SubmitBatch` utility.
- Automatically create a shader binding table for your ray tracing pipeline.

## What does Phobos not do?

- Implement a renderer for you. It simply exposes the Vulkan API.
- Support mobile GPUs. Phobos is optimized for desktop GPUs and makes no effort to support mobile GPUs.

## Example

For more elaborate examples, please check the [examples](examples) folder.

```rust 
use phobos::prelude::*;

fn main() {
    // Fill out app settings for initialization
    let settings = AppBuilder::new()
        .version((1, 0, 0))
        .name("Phobos example app")
        .validation(true)
        .window(&window) // Your winit window, or some other interface.
        .present_mode(vk::PresentModeKHR::MAILBOX)
        .scratch_size(1024u64)
        .gpu(ph::GPURequirements {
          dedicated: true,
          queues: vec![
            QueueRequest { dedicated: false, queue_type: QueueType::Graphics },
            QueueRequest { dedicated: true, queue_type: QueueType::Transfer },
            QueueRequest { dedicated: true, queue_type: QueueType::Compute }
          ],
          ..Default::default()
        })
            .build();

  // Initialize Vulkan. There are other ways to initialize, for example
  // with a custom allocator, or without a window context. See the core::init module for this 
  // functionality.
  use phobos::prelude::*;
  let (
    instance,
    physical_device,
    surface,
    device
    allocator,
    exec,
    frame,
    Some(debug_messenger)
  ) = WindowedContext::init(&settings)? else {
    panic!("Asked for debug messenger but didn't get one.")
  };

  // Create a new pass graph for rendering. Note how we only do this once, as 
  // we are using virtual resources that do not depend on the frame.
  let swapchain = VirtualResource::image("swapchain");
  let clear_pass = PassBuilder::render("clear")
          .color_attachment(&swapchain,
                            vk::AttachmentLoadOp::CLEAR,
                            // Clear the swapchain to red.
                            Some(vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0] }))?
            .build();
    let present_pass = PassBuilder::present("present", clear_pass.output(&swapchain).unwrap());
    let graph = PassGraph::new()
            .add_pass(clear_pass)?
            .add_pass(present_pass)?
            .build()?;
    // Your event loop goes here
    while event_loop {
      // Wait for a new frame to be available. Once there is one, the provided
      // callback will be called.
      futures::executor::block_on(frame.new_frame(exec.clone(), window, &surface, |mut ifc| {
            // Bind some physical resources to the render graph.
            let mut bindings = PhysicalResourceBindings::new();
            bindings.bind_image("swapchain", &ifc.swapchain_image.as_ref().unwrap());
            let cmd = exec.on_domain::<domain::Graphics, DefaultAllocator>(None, None)?;
            // Record render graph to our command buffer
            graph.record(cmd, &bindings, &mut ifc, None, &mut ()).finish()
      }))?;
    }
}
```

## Support

Visit the [docs.rs](https://docs.rs/phobos/latest) page, or open an issue.

## Planned features

- Expose more Vulkan API features.
