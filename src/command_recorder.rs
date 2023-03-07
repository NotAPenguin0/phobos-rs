use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fmt::Debug;
use ash::vk;
use petgraph::graph::NodeIndex;
use petgraph::{Incoming, Outgoing};
use petgraph::prelude::EdgeRef;
use crate::domain::{ExecutionDomain};
use crate::{DebugMessenger, Error, GpuBarrier, GpuResource, GpuTask, PassGraph, ImageView, IncompleteCommandBuffer, InFlightContext, ResourceUsage, task_graph::Node, VirtualResource, BuiltPassGraph, BufferView};
use crate::task_graph::Resource;

use anyhow::Result;
use petgraph::data::DataMapMut;
use crate::command_buffer::{RenderingAttachmentInfo, RenderingInfo};

// Traversal
// =============
// Algorithm as follows:
// - Start with only the source node as an active node.
// - Each iteration:
//      - For each node that is a child of an 'active' node
//          - If all parents of this node are in the active set, then they have all been recorded already.
//              - Record this node, and add it to the active set.
// - Continue this until the active set contains all nodes.

// To implement this we will keep track of a set of children.
// Any time a node is added to the active set, it must come from the children set.
// Then we can:
//  1) Remove this node from the children set.
//  2) Add it to the active set.
//  3) Add its children to the children set.

macro_rules! children {
    ($node:ident, $graph:ident) => {
        $graph.task_graph().graph.edges_directed($node.clone(), Outgoing).map(|edge| edge.target())
    }
}

macro_rules! parents {
    ($node:ident, $graph:ident) => {
        $graph.task_graph().graph.edges_directed($node.clone(), Incoming).map(|edge| edge.source())
    }
}

fn insert_in_active_set<'a, 'e, 'q, D>(
    node: NodeIndex,
    graph: &'a PassGraph<'e, 'q, D>,
    active: &mut HashSet<NodeIndex>,
    children: &mut HashSet<NodeIndex>) where D: ExecutionDomain {
    children.remove(&node);
    active.insert(node);
    for child in children!(node, graph) {
        children.insert(child);
    }
}

fn color_attachments<D>(pass: &GpuTask<GpuResource, D>, bindings: &PhysicalResourceBindings) -> Result<Vec<RenderingAttachmentInfo>>
    where D: ExecutionDomain {
    Ok(pass.outputs.iter().filter_map(|resource| -> Option<RenderingAttachmentInfo> {
        if resource.layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL { return None; }
        let Some(PhysicalResource::Image(image)) = bindings.resolve(&resource.resource) else {
            // TODO: handle or report this error better
            panic!("No resource bound");
        };
        // Attachment should always have a load op set, or our library is bugged
        let info = RenderingAttachmentInfo {
            image_view: image.clone(),
            image_layout: resource.layout,
            resolve_mode: None,
            resolve_image_view: None,
            resolve_image_layout: None,
            load_op: resource.load_op.unwrap(),
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: resource.clear_value.unwrap_or(vk::ClearValue::default()),
        };
        Some(info)
    }).collect())
}

fn depth_attachment<D>(pass: &GpuTask<GpuResource, D>, bindings: &PhysicalResourceBindings)
    -> Option<RenderingAttachmentInfo> where D: ExecutionDomain {
    pass.outputs.iter().filter_map(|resource| -> Option<RenderingAttachmentInfo> {
        if resource.layout != vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL { return None; }

        let Some(PhysicalResource::Image(image)) = bindings.resolve(&resource.resource) else {
            // TODO: handle or report this error better
            panic!("No resource bound");
        };
        let info = RenderingAttachmentInfo {
            image_view: image.clone(),
            image_layout: resource.layout,
            resolve_mode: None,
            resolve_image_view: None,
            resolve_image_layout: None,
            load_op: resource.load_op.unwrap(),
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: resource.clear_value.unwrap_or(vk::ClearValue::default()),
        };
        Some(info)
    }).next()
}

fn render_area<D>(pass: &GpuTask<GpuResource, D>, bindings: &PhysicalResourceBindings) -> vk::Rect2D where D: ExecutionDomain {
    let resource = pass.outputs.iter().filter(|resource|
        match resource.usage {
            ResourceUsage::Attachment(_) => true,
            _ => false,
        }
    ).next().unwrap();
    let Some(PhysicalResource::Image(image)) = bindings.resolve(&resource.resource) else {
        // TODO: handle or report this error better
        panic!("No resource bound");
    };
    vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        // TODO: properly set size of current level?
        extent: vk::Extent2D { width: image.size.width, height: image.size.height },
    }
}

#[cfg(feature="debug-markers")]
fn annotate_pass<'q, D>(pass: &GpuTask<GpuResource, D>, debug: &DebugMessenger, mut cmd: IncompleteCommandBuffer<'q, D>) -> Result<IncompleteCommandBuffer<'q, D>> where D: ExecutionDomain {
    let name = CString::new(pass.identifier.clone())?;
    let label = vk::DebugUtilsLabelEXT::builder()
        .label_name(&name)
        .color(pass.color.unwrap_or([1.0, 1.0, 1.0, 1.0]))
        .build();
    Ok(cmd.begin_label(label, debug))
}

#[cfg(not(feature="debug-markers"))]
fn annotate_pass<D>(_: &GpuTask<GpuResource, D>, _: &DebugMessenger, cmd: IncompleteCommandBuffer<D>) -> Result<IncompleteCommandBuffer<D>> where D: ExecutionDomain { Ok(cmd) }

fn record_pass<'exec, 'q, D>(pass: &mut GpuTask<'exec, 'q, GpuResource, D>, bindings: &PhysicalResourceBindings, ifc: &mut InFlightContext, mut cmd: IncompleteCommandBuffer<'q, D>, debug: Option<&DebugMessenger>)
    -> Result<IncompleteCommandBuffer<'q, D>> where D: ExecutionDomain  {

    if let Some(debug) = debug {
        cmd = annotate_pass(&pass, debug, cmd)?;
    }

    if pass.is_renderpass {
        let info = RenderingInfo {
            flags: Default::default(),
            render_area: render_area(&pass, &bindings),
            layer_count: 1, // TODO: Multilayer rendering fix
            view_mask: 0,
            color_attachments: color_attachments(&pass, &bindings)?,
            depth_attachment: depth_attachment(&pass, &bindings),
            stencil_attachment: None, // TODO: Stencil
        };
        let depth_info = depth_attachment(&pass, &bindings);
        cmd = cmd.begin_rendering(&info);
    }

    cmd = pass.execute.call_mut((cmd, ifc, bindings))?;

    if pass.is_renderpass {
        cmd = cmd.end_rendering()
    }

    if let Some(debug) = debug {
        if cfg!(feature="debug-markers") {
            cmd = cmd.end_label(debug);
        }
    }

    Ok(cmd)
}

fn record_image_barrier<'q, D>(barrier: &GpuBarrier, image: &ImageView, dst_resource: &GpuResource, cmd: IncompleteCommandBuffer<'q, D>)
    -> Result<IncompleteCommandBuffer<'q, D>>
    where D: ExecutionDomain {

    // Image layouts:
    // barrier.resource has information on srcLayout
    // dst_resource(barrier) has information on dstLayout

    let info = vk::DependencyInfo::builder()
        .dependency_flags(vk::DependencyFlags::BY_REGION);
    let vk_barrier = vk::ImageMemoryBarrier2::builder()
        .image(image.image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .src_access_mask(barrier.src_access)
        .dst_access_mask(barrier.dst_access)
        .src_stage_mask(barrier.src_stage)
        .dst_stage_mask(barrier.dst_stage)
        .old_layout(barrier.resource.layout)
        .new_layout(dst_resource.layout)
        .subresource_range(image.subresource_range())
        .build();
    let dependency = info.image_memory_barriers(std::slice::from_ref(&vk_barrier)).build();
    Ok(cmd.pipeline_barrier_2(&dependency))
}

fn record_buffer_barrier<'q, D>(barrier: &GpuBarrier, buffer: &BufferView, dst_resource: &GpuResource, cmd: IncompleteCommandBuffer<'q, D>)
    -> Result<IncompleteCommandBuffer<'q, D>>
    where D: ExecutionDomain {

    let info = vk::DependencyInfo::builder()
        .dependency_flags(vk::DependencyFlags::BY_REGION);
    // Since every driver implements buffer barriers as global memory barriers, we will do the same.
    let vk_barrier = vk::MemoryBarrier2::builder()
        .src_access_mask(barrier.src_access)
        .dst_access_mask(barrier.dst_access)
        .src_stage_mask(barrier.src_stage)
        .dst_stage_mask(barrier.dst_stage)
        .build();
    let dependency = info.memory_barriers(std::slice::from_ref(&vk_barrier)).build();

    Ok(cmd.pipeline_barrier_2(&dependency))
}

fn record_barrier<'q, D>(barrier: &GpuBarrier, dst_resource: &GpuResource, bindings: &PhysicalResourceBindings,
                     cmd: IncompleteCommandBuffer<'q, D>) -> Result<IncompleteCommandBuffer<'q, D>> where D: ExecutionDomain{
    let physical_resource = bindings.resolve(&barrier.resource.resource);
    let Some(resource) = physical_resource else { return Err(anyhow::Error::from(Error::NoResourceBound(barrier.resource.uid().clone()))) };
    match resource {
        PhysicalResource::Image(image) => { record_image_barrier(&barrier, image, dst_resource, cmd) }
        PhysicalResource::Buffer(buffer) => { record_buffer_barrier(&barrier, buffer, dst_resource, cmd) }
    }
}

fn record_node<'exec, 'q, D>(graph: &mut BuiltPassGraph<'exec, 'q, D>, node: NodeIndex, bindings: &PhysicalResourceBindings, ifc: &mut InFlightContext,
                             cmd: IncompleteCommandBuffer<'q, D>, debug: Option<&DebugMessenger>) -> Result<IncompleteCommandBuffer<'q, D>> where D: ExecutionDomain {
    let graph = &mut graph.graph.graph;
    let dst_resource_res = PassGraph::barrier_dst_resource(&graph, node).cloned();
    let weight = graph.node_weight_mut(node).unwrap();
    match weight {
        Node::Task(pass) => { record_pass(pass, &bindings, ifc, cmd, debug) }
        Node::Barrier(barrier) => {
            // Find destination resource in graph
            record_barrier(&barrier, &dst_resource_res?, &bindings, cmd)
        }
        Node::_Unreachable(_) => { unreachable!() }
    }
}

/// Describes any physical resource handle on the GPU.
#[derive(Debug, Clone)]
pub enum PhysicalResource {
    Image(ImageView),
    Buffer(BufferView),
}

/// Stores bindings from virtual resources to physical resources.
/// # Example usage
/// ```
/// use ash::vk;
/// use phobos::{Error, Image, PhysicalResourceBindings, VirtualResource};
///
/// let resource = VirtualResource::new(String::from("image"));
/// let image = Image::new(/*...*/);
/// let view = image.view(vk::ImageAspectFlags::COLOR)?;
/// let mut bindings = PhysicalResourceBindings::new();
/// // Bind the virtual resource to the image
/// bindings.bind_image(String::from("image"), view.clone());
/// // ... Later, lookup the physical image handle from a virtual resource handle
/// let view = bindings.resolve(&resource).ok_or(Error::NoResourceBound)?;
/// ```
#[derive(Debug)]
pub struct PhysicalResourceBindings {
    bindings: HashMap<String, PhysicalResource>
}

impl PhysicalResourceBindings {
    /// Create a new physical resource binding map.
    pub fn new() -> Self {
        PhysicalResourceBindings { bindings: Default::default() }
    }

    /// Bind an image to all virtual resources with `name(+*)` as their uid.
    pub fn bind_image(&mut self, name: impl Into<String>, image: ImageView) {
        self.bindings.insert(name.into(), PhysicalResource::Image(image));
    }

    /// Bind a buffer to all virtual resources with this name as their uid.
    pub fn bind_buffer(&mut self, name: impl Into<String>, buffer: BufferView) { self.bindings.insert(name.into(), PhysicalResource::Buffer(buffer)); }

    /// Alias a resource by giving it an alternative name
    pub fn alias(&mut self, new_name: impl Into<String>, resource: &str) -> Result<()> {
        self.bindings.insert(new_name.into(), self.bindings.get(resource).ok_or(Error::NoResourceBound(resource.to_owned()))?.clone());
        Ok(())
    }

    /// Resolve a virtual resource to a physical resource. Returns `None` if the resource was not found.
    pub fn resolve(&self, resource: &VirtualResource) -> Option<&PhysicalResource> {
        self.bindings.get(&resource.name())
    }
}

/// Records a render graph to a command buffer. This also takes in a set of physical bindings to resolve virtual resource names
/// to actual resources.
/// # Errors
/// - This function can error if a virtual resource used in the graph is lacking an physical binding.
pub fn record_graph<'a, 'exec, 'q, D>(graph: &'a mut BuiltPassGraph<'exec, 'q, D>, bindings: &PhysicalResourceBindings, ifc: &mut InFlightContext, mut cmd: IncompleteCommandBuffer<'q, D>, debug: Option<&DebugMessenger>)
                                      -> Result<IncompleteCommandBuffer<'q, D>> where D: ExecutionDomain {
    let mut active = HashSet::new();
    let mut children = HashSet::new();
    for start in graph.graph.sources() {
        insert_in_active_set(start, &graph, &mut active, &mut children);
    }
    // Record each initial active node.
    for node in &active {
        cmd = record_node(graph, node.clone(), &bindings, ifc, cmd, debug)?;
    }

    while active.len() != graph.num_nodes() {
        // For each node that is a child of an active node
        let mut recorded_nodes = Vec::new();
        for child in &children {
            // If all parents of this child node are in the active set, record it.
            if parents!(child, graph).all(|parent| active.contains(&parent)) {
                cmd = record_node(graph, child.clone(), &bindings, ifc, cmd, debug)?;
                recorded_nodes.push(child.clone());
            }
        }
        // Now we swap all recorded nodes to the active set
        for node in recorded_nodes {
            insert_in_active_set(node.clone(), &graph, &mut active, &mut children);
        }
    }

    Ok(cmd)
}