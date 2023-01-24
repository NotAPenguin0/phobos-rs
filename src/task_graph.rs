/// # Task graph system
///
/// The task graph system exposes a powerful system for managing gpu-gpu synchronization of resources.
/// Note that the task graph deals in purely virtual resources. There are no physical resources bound to the task graph.

use std::collections::HashSet;
use std::fmt::{Debug, Display};

use petgraph::graph::*;
use petgraph::visit::NodeRef;
use petgraph;
use petgraph::dot;
use petgraph::dot::Dot;

use crate::error::Error;
use crate::pass::Pass;

// Current issues:
// - If there is a barrier node with two dependent nodes, but both use the resource in a different way (e.g. layout), we should split this barrier in two barrier nodes and then serialize it.
//      => solving this should probably be a responsibility of the translation layer.

pub trait Resource {
    fn is_dependency_of(&self, lhs: &Self) -> bool;
    fn uid(&self) -> &String;
}

/// Task in a task dependency graph. This is parametrized on a resource type.
#[derive(Debug, Clone)]
pub struct Task<R> where R: Resource {
    pub identifier: String,
    pub inputs: Vec<R>,
    pub outputs: Vec<R>
}

/// Represents a barrier in the task graph.
#[derive(Debug, Clone)]
pub struct Barrier<R> where R: Resource {
    pub resource: R
}

#[derive(Debug, Clone)]
pub enum Node<R> where R: Resource {
    Task(Task<R>),
    Barrier(Barrier<R>)
}

pub struct TaskGraph<R> where R: Resource {
    graph: Graph<Node<R>, String>,
}


pub struct GpuTask {

}

/// Represents a virtual resource in the system, uniquely identified by a string.
#[derive(Debug, Clone)]
pub struct VirtualResource {
    pub uid: String,
}

// temporary
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ImageResourceUsage {
    Attachment,
    Sample,
    Write,
}

#[derive(Debug, Clone)]
pub struct ImageResource {
    pub usage: ImageResourceUsage,
    pub resource: VirtualResource,
}

#[derive(Debug, Clone)]
pub enum GpuResource {
    Image(ImageResource)
}

pub struct GpuTaskGraph {
    graph: TaskGraph<GpuResource>,
}

impl VirtualResource {
    pub fn new(uid: String) -> Self {
        VirtualResource { uid }
    }

    /// 'Upgrades' the resource to a new version of itself. This is used to obtain the virtual resource name of an input resource after
    /// a task completes.
    pub fn upgrade(&self) -> Self {
        VirtualResource {
            uid: self.uid.clone() + "*"
        }
    }

    /// Returns the full, original name of the resource (without potential version star symbols)
    pub fn name(&self) -> String {
        let mut name = self.uid.clone();
        name.retain(|c| c != '*');
        name
    }

    /// Two virtual resources are associated if and only if their uid's only differ by "*" symbols.
    pub fn are_associated(lhs: &VirtualResource, rhs: &VirtualResource) -> bool {
        // Since virtual resource uid's are constructed by appending * symbols, we can simply check whether the largest of the two strings starts with the shorter one
        let larger = if lhs.uid.len() >= rhs.uid.len() { lhs } else { rhs };
        let smaller = if lhs.uid.len() < rhs.uid.len() { lhs } else { rhs };
        larger.uid.starts_with(&smaller.uid)
    }

    /// One virtual resource is older than another if it has less '*' symbols.
    pub fn is_older(lhs: &VirtualResource, rhs: &VirtualResource) -> bool {
        if !VirtualResource::are_associated(&lhs, &rhs) { return false; }
        lhs.uid.len() < rhs.uid.len()
    }

    /// Note that this is not the same as inverting the result of as_older(), for the same exact state of the resource,
    /// both of these functions should return false (they decide whether resources are strictly older or younger than each other).
    pub fn is_younger(lhs: &VirtualResource, rhs: &VirtualResource) -> bool {
        if !VirtualResource::are_associated(&lhs, &rhs) { return false; }
        rhs.uid.len() < lhs.uid.len()
    }
}
impl GpuResource {
    pub fn virtual_resource(&self) -> &VirtualResource {
        match self {
            GpuResource::Image(i) => { &i.resource }
        }
    }
}

impl Resource for GpuResource {
    fn is_dependency_of(&self, lhs: &Self) -> bool {
        self.virtual_resource().uid == lhs.virtual_resource().uid
    }

    fn uid(&self) -> &String {
        &self.virtual_resource().uid
    }
}

impl GpuTaskGraph {
    pub fn new() -> Self {
        GpuTaskGraph {
            graph: TaskGraph::new()
        }
    }
    
    // note: pass api highly experimental and will be changed a lot.
    pub fn add_pass(&mut self, pass: Pass) -> Result<(), Error> {
        self.graph.add_task(Task {
            identifier: pass.name,
            inputs: pass.inputs,
            outputs: pass.outputs
        })?;

        Ok(())
    }

    pub fn build(&mut self) {
        self.graph.create_barrier_nodes();
    }

    pub fn task_graph(&self) -> &TaskGraph<GpuResource> {
        &self.graph
    }
}

impl<R> TaskGraph<R> where R: Debug + Clone + Resource {
    pub fn new() -> Self {
        TaskGraph {
            graph: Graph::new()
        }
    }

    /// Outputs graphviz-compatible dot file for displaying the graph.
    pub fn as_dot(&self) -> Result<Dot<&Graph<Node<R>, String>>, Error> {
        Ok(petgraph::dot::Dot::with_config(&self.graph, &[]))
    }

    fn is_dependent(&self, graph: &Graph<Node<R>, String>, child: NodeIndex, parent: NodeIndex) -> Result<Option<R>, Error> {
        let child = graph.node_weight(child).ok_or(Error::NodeNotFound)?;
        let parent = graph.node_weight(parent).ok_or(Error::NodeNotFound)?;
        if let Node::Task(child) = child {
            if let Node::Task(parent) = parent {
                return Ok(child.inputs.iter().find(|input| {
                    parent.outputs.iter().any(|output| input.is_dependency_of(&output))
                })
                .cloned());
            }
        }

        Ok(None)
    }

    fn is_task_node(graph: &Graph<Node<R>, String>, node: NodeIndex) -> Result<bool, Error> {
        Ok(matches!(graph.node_weight(node).ok_or(Error::NodeNotFound)?, Node::Task(_)))
    }

    /// Add a task to the task graph.
    pub fn add_task(&mut self, task: Task<R>) -> Result<(), Error> {
        let node = self.graph.add_node(Node::Task(task));
        // When adding a node, we need to update edges in the graph.
        // X = The newly added node
        // For every node Y:
        //      1. If Y produces an output used by X
        //          Add a connection Y -> X
        //      2. If Y consumes an input produced by X
        //          Add a connection X -> Y
        // Check for cycles in the graph. If there is a cycle, adding this node results in an illegal state.

        // Note that we unwrap here as this must never fail.
        self.graph.node_indices().for_each(|other_node| {
            // task depends on other task, add an edge other_task -> task
            if let Some(dependency) = self.is_dependent(&self.graph, node, other_node).unwrap() {
                self.graph.add_edge(other_node, node, dependency.uid().clone());
            }

            // Note: no else here, since we will detect cycles and error on them,
            // which is better than silently ignoring some cycles.
            if let Some(dependency) = self.is_dependent(&self.graph, other_node, node).unwrap() {
                self.graph.add_edge(node, other_node, dependency.uid().clone());
            }
        });

        match petgraph::algo::is_cyclic_directed(&self.graph) {
            true => Err(Error::GraphHasCycle),
            false => Ok(())
        }
    }

    /// Create a maximum set of barrier nodes for the task graph. This means that we will assume every resource that is being consumed needs its own barrier.
    /// These barriers are not yet serialized, as we only want to do that after we know which barriers are equivalent.
    pub fn create_barrier_nodes(&mut self) {
        // We create barrier nodes as follows:
        // For each task node P:
        //      - For each resource R that P produces:
        //          - If there are no nodes that depend directly on this resource R, do nothing.
        //          - Otherwise, add a new barrier node B, acting on the resource R.
        //          - Then add an edge from P to B, and edges from B to each node Q that consumes the resource R directly.
        //          - Finally, remove the edges from P to each node Q.
        //
        // Note that this algorithm creates too many barriers for practical usage.
        // We will compact the amount of dependency barriers when translating this graph to a render graph

        self.graph.node_indices().clone().for_each(|node| {
            if !Self::is_task_node(&self.graph, node).unwrap() { return; }

            let Node::Task(task) = self.graph.node_weight(node).cloned().unwrap() else { unimplemented!() };
            for resource in &task.outputs {
                // Find all nodes in the graph that depend directly on this resource
                let consumers = self.graph.node_indices().filter(|&consumer| -> bool {
                    let consumer = self.graph.node_weight(consumer).unwrap();
                    match consumer {
                        Node::Task(t) => { t.inputs.iter().any(|input| input.is_dependency_of(&resource)) }
                        Node::Barrier(_) => false
                    }
                }).collect::<Vec<NodeIndex>>();

                if consumers.is_empty() { return; }
                let barrier = self.graph.add_node(Node::Barrier(Barrier{resource: resource.clone()}));
                self.graph.update_edge(node, barrier, resource.uid().clone());
                for consumer in consumers {
                    self.graph.update_edge(barrier, consumer, resource.uid().clone());
                    self.graph.remove_edge(self.graph.find_edge(node, consumer).unwrap());
                }
            }
        })
    }
}

impl<R> Display for Node<R> where R: Debug + Resource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Task(task) => f.write_fmt(format_args!("Task: {}", &task.identifier)),
            Node::Barrier(barrier) => { f.write_fmt(format_args!("Barrier"))}
        }
    }
}

impl<R> Display for TaskGraph<R> where R: Debug + Resource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dot = petgraph::dot::Dot::new(&self.graph);
        std::fmt::Display::fmt(&dot, f)
    }
}


