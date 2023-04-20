//! The pass graph system is a powerful abstraction that allows you to automatically manage synchronization **within a single queue**,
//! and automatically transition image layouts based on usage.
//!
//! Each pass needs to declare its inputs and outputs, and then the graph
//! is built and barriers are inserted where needed. All resources are specified as [`VirtualResource`](crate::VirtualResource)s, referenced by a string ID.
//! This means that it's possible to not have to rebuild the graph every frame.
//!
//! Actual resources need to be bound to each virtual resource before recording the graph into a command buffer.
//! This is done using the [`PhysicalResourceBindings`](crate::PhysicalResourceBindings) struct.
//!
//! Through the [`GraphViz`](task_graph::GraphViz) trait, it's possible to export a graphviz-compatible dot file to display the task graph.
//!
//! # Example
//!
//! ```
//! use phobos::prelude::*;
//!
//! // Define a virtual resource for the swapchain
//! let swap_resource = VirtualResource::image("swapchain");
//! // Define a pass that will handle the layout transition to `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`.
//! // This is required in your main frame graph.
//! let present_pass = PassBuilder::present("present", &swap_resource);
//! // Create the graph. Note that we need to pass the swapchain resource to it as well.
//! let mut graph = PassGraph::<domain::Graphics>::new(Some(&swap_resource));
//! // Add our pass
//! graph.add_pass(present_pass)?;
//! // Build the graph and obtain a BuiltPassGraph.
//! let mut graph = graph.build()?;
//! // To record, check the next example.
//! ```
//!
//! For more complex passes, see the [`pass`] module documentation.
//!
//! # Recording
//!
//! Once a graph has been built it can be recorded to a compatible command buffer (one over the same [`ExecutionDomain`](crate::domain::ExecutionDomain) as the task graph. The
//! type system enforces this.).
//! To do this, first bind physical resources to each virtual resource used, and then call [`BuiltPassGraph::record()`](crate::graph::pass_graph::BuiltPassGraph::record).
//!
//! Using the graph from the previous example:
//! ```
//! use phobos::prelude::*;
//!
//! // Bind swapchain virtual resource to this frame's swapchain image.
//! let mut bindings = PhysicalResourceBindings::new();
//! bindings.bind_image("swapchain", ifc.swapchain_image.as_ref().unwrap());
//! let cmd = exec.on_domain::<domain::Graphics>(None, None)?;
//! // Debug messenger not required, but recommended together with the `debug-markers` feature.
//! let final_cmd = graph.record(cmd, &bindings, &mut ifc, Some(debug_messenger))?
//!                 .finish();
//! ```

pub mod pass;
pub mod pass_graph;
pub mod physical_resource;
pub mod record;
pub mod resource;
pub mod virtual_resource;

pub(crate) mod task_graph;
