use std::collections::HashSet;
use std::ffi::CString;
use std::sync::Arc;

use anyhow::Result;
use ash::vk;
use petgraph::{Incoming, Outgoing};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;

use crate::{Allocator, BufferView, DebugMessenger, Error, ImageView, InFlightContext, PassGraph, PhysicalResourceBindings};
use crate::command_buffer::IncompleteCommandBuffer;
use crate::command_buffer::state::{RenderingAttachmentInfo, RenderingInfo};
use crate::domain::ExecutionDomain;
use crate::graph::pass_graph::{BuiltPassGraph, PassNode, PassResource, PassResourceBarrier};
use crate::graph::physical_resource::PhysicalResource;
use crate::graph::resource::{AttachmentType, ResourceUsage};
use crate::graph::task_graph::{Node, Resource};

pub trait RecordGraphToCommandBuffer<'q, D: ExecutionDomain, U, A: Allocator> {
    /// Records a render graph to a command buffer. This also takes in a set of physical bindings to resolve virtual resource names
    /// to actual resources.
    /// # Errors
    /// - This function can error if a virtual resource used in the graph is lacking an physical binding.
    fn record(
        &mut self,
        cmd: IncompleteCommandBuffer<'q, D>,
        bindings: &PhysicalResourceBindings,
        ifc: &mut InFlightContext<A>,
        debug: Option<Arc<DebugMessenger>>,
        user_data: &U,
    ) -> Result<IncompleteCommandBuffer<'q, D>>
    where
        Self: Sized;
}

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
        $graph
            .task_graph()
            .graph
            .edges_directed($node.clone(), Outgoing)
            .map(|edge| edge.target())
    };
}

macro_rules! parents {
    ($node:ident, $graph:ident) => {
        $graph
            .task_graph()
            .graph
            .edges_directed($node.clone(), Incoming)
            .map(|edge| edge.source())
    };
}

fn insert_in_active_set<D: ExecutionDomain, U, A: Allocator>(
    node: NodeIndex,
    graph: &PassGraph<'_, '_, D, U, A>,
    active: &mut HashSet<NodeIndex>,
    children: &mut HashSet<NodeIndex>,
) {
    children.remove(&node);
    active.insert(node);
    for child in children!(node, graph) {
        children.insert(child);
    }
}

fn find_resolve_attachment<D: ExecutionDomain, U, A: Allocator>(
    pass: &PassNode<PassResource, D, U, A>,
    bindings: &PhysicalResourceBindings,
    resource: &PassResource,
) -> Option<ImageView> {
    pass.outputs
        .iter()
        .find(|output| match &output.usage {
            ResourceUsage::Attachment(AttachmentType::Resolve(resolve)) => resource.resource.is_associated_with(resolve),
            _ => false,
        })
        .map(|resolve| {
            let Some(PhysicalResource::Image(image)) = bindings.resolve(&resolve.resource) else {
                // TODO: handle or report this error better
                panic!("No resource bound");
            };
            image
        })
        .cloned()
}

fn color_attachments<D: ExecutionDomain, U, A: Allocator>(
    pass: &PassNode<PassResource, D, U, A>,
    bindings: &PhysicalResourceBindings,
) -> Result<Vec<RenderingAttachmentInfo>> {
    Ok(pass
        .outputs
        .iter()
        .filter_map(|resource| -> Option<RenderingAttachmentInfo> {
            if !matches!(resource.usage, ResourceUsage::Attachment(AttachmentType::Color)) {
                return None;
            }
            let Some(PhysicalResource::Image(image)) = bindings.resolve(&resource.resource) else {
                // TODO: handle or report this error better
                panic!("No resource bound");
            };
            let resolve = find_resolve_attachment(pass, bindings, resource);
            // Attachment should always have a load op set, or our library is bugged
            let info = RenderingAttachmentInfo {
                image_view: image.clone(),
                image_layout: resource.layout,
                resolve_mode: resolve.is_some().then_some(vk::ResolveModeFlags::AVERAGE),
                resolve_image_layout: resolve.is_some().then_some(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                resolve_image_view: resolve,
                load_op: resource.load_op.unwrap(),
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value: resource.clear_value.unwrap_or(vk::ClearValue::default()),
            };
            Some(info)
        })
        .collect())
}

fn depth_attachment<D: ExecutionDomain, U, A: Allocator>(
    pass: &PassNode<PassResource, D, U, A>,
    bindings: &PhysicalResourceBindings,
) -> Option<RenderingAttachmentInfo> {
    pass.outputs
        .iter()
        .filter_map(|resource| -> Option<RenderingAttachmentInfo> {
            if resource.layout != vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
                return None;
            }

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
        })
        .next()
}

fn render_area<D: ExecutionDomain, U, A: Allocator>(pass: &PassNode<PassResource, D, U, A>, bindings: &PhysicalResourceBindings) -> vk::Rect2D {
    let resource = pass
        .outputs
        .iter()
        .find(|resource| matches!(resource.usage, ResourceUsage::Attachment(_)))
        .unwrap();
    let Some(PhysicalResource::Image(image)) = bindings.resolve(&resource.resource) else {
        // TODO: handle or report this error better
        panic!("No resource bound");
    };
    vk::Rect2D {
        offset: vk::Offset2D {
            x: 0,
            y: 0,
        },
        // TODO: properly set size of current level?
        extent: vk::Extent2D {
            width: image.width(),
            height: image.height(),
        },
    }
}

#[cfg(feature = "debug-markers")]
fn annotate_pass<'q, D: ExecutionDomain, U, A: Allocator>(
    pass: &PassNode<PassResource, D, U, A>,
    debug: &Arc<DebugMessenger>,
    cmd: IncompleteCommandBuffer<'q, D>,
) -> Result<IncompleteCommandBuffer<'q, D>> {
    let name = CString::new(pass.identifier.clone())?;
    let label = vk::DebugUtilsLabelEXT {
        s_type: vk::StructureType::DEBUG_UTILS_LABEL_EXT,
        p_next: std::ptr::null(),
        p_label_name: name.as_ptr(),
        color: pass.color.unwrap_or([1.0, 1.0, 1.0, 1.0]),
    };
    Ok(cmd.begin_label(label, debug))
}

#[cfg(not(feature = "debug-markers"))]
fn annotate_pass<D: ExecutionDomain>(_: &PassNode<PassResource, D>, _: &Arc<DebugMessenger>, cmd: IncompleteCommandBuffer<D>) -> Result<IncompleteCommandBuffer<D>> {
    Ok(cmd)
}

fn record_pass<'q, D: ExecutionDomain, U, A: Allocator>(
    pass: &mut PassNode<'_, 'q, PassResource, D, U, A>,
    bindings: &PhysicalResourceBindings,
    ifc: &mut InFlightContext<A>,
    mut cmd: IncompleteCommandBuffer<'q, D>,
    debug: Option<Arc<DebugMessenger>>,
    user_data: &U,
) -> Result<IncompleteCommandBuffer<'q, D>> {
    if let Some(debug) = debug.clone() {
        cmd = annotate_pass(pass, &debug, cmd)?;
    }

    if pass.is_renderpass {
        let info = RenderingInfo {
            flags: Default::default(),
            render_area: render_area(pass, bindings),
            layer_count: 1, // TODO: Multilayer rendering fix
            view_mask: 0,
            color_attachments: color_attachments(pass, bindings)?,
            depth_attachment: depth_attachment(pass, bindings),
            stencil_attachment: None, // TODO: Stencil
        };
        cmd = cmd.begin_rendering(&info);
    }

    cmd = pass.execute.call_mut((cmd, ifc, bindings, user_data))?;

    if pass.is_renderpass {
        cmd = cmd.end_rendering()
    }

    if let Some(debug) = debug {
        if cfg!(feature = "debug-markers") {
            cmd = cmd.end_label(&debug);
        }
    }

    Ok(cmd)
}

fn record_image_barrier<'q, D: ExecutionDomain>(
    barrier: &PassResourceBarrier,
    image: &ImageView,
    dst_resource: &PassResource,
    cmd: IncompleteCommandBuffer<'q, D>,
) -> Result<IncompleteCommandBuffer<'q, D>> {
    // Image layouts:
    // barrier.resource has information on srcLayout
    // dst_resource(barrier) has information on dstLayout

    let vk_barrier = vk::ImageMemoryBarrier2 {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
        p_next: std::ptr::null(),
        src_stage_mask: barrier.src_stage,
        src_access_mask: barrier.src_access,
        dst_stage_mask: barrier.dst_stage,
        dst_access_mask: barrier.dst_access,
        old_layout: barrier.resource.layout,
        new_layout: dst_resource.layout,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: unsafe { image.image() },
        subresource_range: image.subresource_range(),
    };

    let dependency = vk::DependencyInfo {
        s_type: vk::StructureType::DEPENDENCY_INFO,
        p_next: std::ptr::null(),
        dependency_flags: vk::DependencyFlags::BY_REGION,
        memory_barrier_count: 0,
        p_memory_barriers: std::ptr::null(),
        buffer_memory_barrier_count: 0,
        p_buffer_memory_barriers: std::ptr::null(),
        image_memory_barrier_count: 1,
        p_image_memory_barriers: &vk_barrier,
    };

    Ok(cmd.pipeline_barrier_2(&dependency))
}

fn record_buffer_barrier<'q, D: ExecutionDomain>(
    barrier: &PassResourceBarrier,
    _buffer: &BufferView,
    _dst_resource: &PassResource,
    cmd: IncompleteCommandBuffer<'q, D>,
) -> Result<IncompleteCommandBuffer<'q, D>> {
    // Since every driver implements buffer barriers as global memory barriers, we will do the same.
    let vk_barrier = vk::MemoryBarrier2 {
        s_type: vk::StructureType::MEMORY_BARRIER_2,
        p_next: std::ptr::null(),
        src_stage_mask: barrier.src_stage,
        src_access_mask: barrier.src_access,
        dst_stage_mask: barrier.dst_stage,
        dst_access_mask: barrier.dst_access,
    };

    let dependency = vk::DependencyInfo {
        s_type: vk::StructureType::DEPENDENCY_INFO,
        p_next: std::ptr::null(),
        dependency_flags: vk::DependencyFlags::BY_REGION,
        memory_barrier_count: 1,
        p_memory_barriers: &vk_barrier,
        buffer_memory_barrier_count: 0,
        p_buffer_memory_barriers: std::ptr::null(),
        image_memory_barrier_count: 0,
        p_image_memory_barriers: std::ptr::null(),
    };

    Ok(cmd.pipeline_barrier_2(&dependency))
}

fn record_barrier<'q, D: ExecutionDomain>(
    barrier: &PassResourceBarrier,
    dst_resource: &PassResource,
    bindings: &PhysicalResourceBindings,
    cmd: IncompleteCommandBuffer<'q, D>,
) -> Result<IncompleteCommandBuffer<'q, D>> {
    let physical_resource = bindings.resolve(&barrier.resource.resource);
    let Some(resource) = physical_resource else { return Err(anyhow::Error::from(Error::NoResourceBound(barrier.resource.uid().clone()))) };
    match resource {
        PhysicalResource::Image(image) => record_image_barrier(barrier, image, dst_resource, cmd),
        PhysicalResource::Buffer(buffer) => record_buffer_barrier(barrier, buffer, dst_resource, cmd),
    }
}

fn record_node<'q, D: ExecutionDomain, U, A: Allocator>(
    graph: &mut BuiltPassGraph<'_, 'q, D, U, A>,
    node: NodeIndex,
    bindings: &PhysicalResourceBindings,
    ifc: &mut InFlightContext<A>,
    cmd: IncompleteCommandBuffer<'q, D>,
    debug: Option<Arc<DebugMessenger>>,
    user_data: &U,
) -> Result<IncompleteCommandBuffer<'q, D>> {
    let graph = &mut graph.graph.graph;
    let dst_resource_res = PassGraph::barrier_dst_resource(graph, node).cloned();
    let weight = graph.node_weight_mut(node).unwrap();
    match weight {
        Node::Task(pass) => record_pass(pass, bindings, ifc, cmd, debug, user_data),
        Node::Barrier(barrier) => {
            // Find destination resource in graph
            record_barrier(barrier, &dst_resource_res?, bindings, cmd)
        }
        Node::_Unreachable(_) => {
            unreachable!()
        }
    }
}

impl<'q, 'exec, D: ExecutionDomain, U, A: Allocator> RecordGraphToCommandBuffer<'q, D, U, A> for BuiltPassGraph<'exec, 'q, D, U, A> {
    fn record(
        &mut self,
        mut cmd: IncompleteCommandBuffer<'q, D>,
        bindings: &PhysicalResourceBindings,
        ifc: &mut InFlightContext<A>,
        debug: Option<Arc<DebugMessenger>>,
        user_data: &U,
    ) -> Result<IncompleteCommandBuffer<'q, D>>
        where
            Self: Sized, {
        let mut active = HashSet::new();
        let mut children = HashSet::new();
        for start in self.graph.sources() {
            insert_in_active_set(start, self, &mut active, &mut children);
        }
        // Record each initial active node.
        for node in &active {
            cmd = record_node(self, *node, bindings, ifc, cmd, debug.clone(), user_data)?;
        }

        while active.len() != self.num_nodes() {
            // For each node that is a child of an active node
            let mut recorded_nodes = Vec::new();
            for child in &children {
                // If all parents of this child node are in the active set, record it.
                if parents!(child, self).all(|parent| active.contains(&parent)) {
                    cmd = record_node(self, *child, bindings, ifc, cmd, debug.clone(), user_data)?;
                    recorded_nodes.push(*child);
                }
            }
            // Now we swap all recorded nodes to the active set
            for node in recorded_nodes {
                insert_in_active_set(node, self, &mut active, &mut children);
            }
        }

        Ok(cmd)
    }
}
