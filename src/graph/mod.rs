//! The task graph system is a powerful abstraction that allows you to automatically manage synchronization **within a single queue**,
//! and automatically transition image layouts based on usage. Each pass needs to declare its inputs and outputs, and then the graph
//! is built and barriers are inserted where needed. All resources are specified as [`VirtualResource`]s, referenced by a string ID.
//! This means that it's possible to not have to rebuild the graph every frame.
//!
//! Actual resources need to be bound to each virtual resource before recording the graph into a command buffer.
//! This is done using the [`PhysicalResourceBindings`] struct.
//!
//! Through the [`GraphViz`] trait, it's possible to export a graphviz-compatible dot file to display the task graph.
//!
//! # Example
//!
//! ```
//! use phobos as ph;
//!
//! // Define a virtual resource for the swapchain
//! let swap_resource = ph::VirtualResource::image("swapchain".to_string());
//! // Define a pass that will handle the layout transition to `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`.
//! // This is required in your main frame graph.
//! let present_pass = ph::PassBuilder::present("present".to_string(), swap_resource);
//! let mut graph = ph::PassGraph::<ph::domain::Graphics>::new(swap_resource.clone());
//! graph.add_pass(present_pass)?;
//! // Build the graph and obtain a BuiltPassGraph.
//! let mut graph = graph.build()?;
//! ```
//!
//! For more complex passes, see the [`pass`] module documentation.
//!
//! # Recording
//!
//! Once a graph has been built it can be recorded to a compatible command buffer (one over the same [`ExecutionDomain`] as the task graph).
//! To do this, first bind physical resources to each virtual resource used, and then call [`ph::record_graph`].
//!
//! Using the graph from the previous example:
//! ```
//! use phobos as ph;
//! use phobos::IncompleteCmdBuffer;
//!
//! // Bind swapchain virtual resource to this frame's swapchain image.
//! let mut bindings = ph::PhysicalResourceBindings::new();
//! bindings.bind_image("swapchain".to_string(), ifc.swapchain_image.as_ref().unwrap().clone());
//! let cmd = exec.on_domain::<ph::domain::Graphics>()?;
//! // Debug messenger not required, but recommended together with the `debug-markers` feature.
//! let final_cmd = graph.record(cmd, &bindings, &mut ifc, Some(debug_messenger))?
//!                 .finish();
//! ```

pub mod record;
pub mod virtual_resource;
pub mod physical_resource;
pub mod resource;
pub mod pass_graph;
pub mod pass;

pub(crate) mod task_graph;

