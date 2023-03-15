use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use petgraph::{Graph, Incoming};
use petgraph::graph::{EdgeReference, NodeIndex};

use anyhow::Result;
use petgraph::dot::Dot;
use crate::domain::ExecutionDomain;
use crate::Error;
use crate::graph::pass_graph::{PassNode, PassResource, PassResourceBarrier};

/// Represents a resource in a task graph.
pub trait Resource {
    fn is_dependency_of(&self, lhs: &Self) -> bool;
    fn uid(&self) -> &String;
}

/// Task in a task dependency graph. This is parametrized on a resource type.
pub trait Task<R> where R: Resource {
    fn inputs(&self) -> &Vec<R>;
    fn outputs(&self) -> &Vec<R>;
}

/// Represents a barrier in the task graph.
pub trait Barrier<R> where R: Resource {
    fn new(resource: R) -> Self;
    fn resource(&self) -> &R;
}

/// Represents a node in a task graph.
#[derive(Debug)]
pub enum Node<R, B, T> where R: Resource, B: Barrier<R>, T: Task<R> {
    Task(T),
    Barrier(B),
    _Unreachable((!, PhantomData<R>)),
}

/// Task graph structure, used for automatic synchronization of resource accesses.
pub struct TaskGraph<R, B, T> where R: Resource + Default, B: Barrier<R> + Clone, T: Task<R> {
    pub(crate) graph: Graph<Node<R, B, T>, String>,
}


impl<R, B, T> TaskGraph<R, B, T> where R: Clone + Default + Resource, B: Barrier<R> + Clone, T: Task<R> {
    pub fn new() -> Self {
        TaskGraph {
            graph: Graph::new()
        }
    }

    fn is_dependent(&self, graph: &Graph<Node<R, B, T>, String>, child: NodeIndex, parent: NodeIndex) -> Result<Option<R>> {
        let child = graph.node_weight(child).ok_or(Error::NodeNotFound)?;
        let parent = graph.node_weight(parent).ok_or(Error::NodeNotFound)?;
        if let Node::Task(child) = child {
            if let Node::Task(parent) = parent {
                return Ok(child.inputs().iter().find(|&input| {
                    parent.outputs().iter().any(|output| input.is_dependency_of(&output))
                })
                    .cloned());
            }
        }

        Ok(None)
    }

    fn is_task_node(graph: &Graph<Node<R, B, T>, String>, node: NodeIndex) -> Result<bool> {
        Ok(matches!(graph.node_weight(node).ok_or(Error::NodeNotFound)?, Node::Task(_)))
    }

    fn get_edge_attributes(_: &Graph<Node<R, B, T>, String>, _: EdgeReference<String>) -> String {
        String::from("")
    }

    fn get_node_attributes(_: &Graph<Node<R, B, T>, String>, node: (NodeIndex, &Node<R, B, T>)) -> String {
        match node.1 {
            Node::Task(_) => { String::from("fillcolor = \"#5e6df7\"") }
            Node::Barrier(_) => { String::from("fillcolor = \"#f75e70\" shape=box") }
            Node::_Unreachable(_) => { unreachable!() }
        }
    }

    /// Return all source nodes in the graph, these are the nodes with no parent node.
    pub fn sources<'a>(&'a self) -> impl Iterator<Item = NodeIndex> + 'a {
        self.graph.node_indices().filter(|node| {
            self.graph.edges_directed(node.clone(), Incoming).next().is_none()
        })
    }

    /// Add a task to the task graph.
    pub fn add_task(&mut self, task: T) -> Result<()> {
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
            true => Err(anyhow::Error::from(Error::GraphHasCycle)),
            false => Ok(())
        }
    }

    fn task_outputs(&self, node: NodeIndex) -> &Vec<R> {
        let Node::Task(task) = self.graph.node_weight(node).unwrap() else { unimplemented!() };
        task.outputs()
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

            for resource in self.task_outputs(node).clone() {
                // Find all nodes in the graph that depend directly on this resource
                let consumers = self.graph.node_indices().filter(|&consumer| -> bool {
                    let consumer = self.graph.node_weight(consumer).unwrap();
                    match consumer {
                        Node::Task(t) => { t.inputs().iter().any(|input| input.is_dependency_of(&resource)) }
                        Node::Barrier(_) => false,

                        Node::_Unreachable(_) => { unreachable!() }
                    }
                }).collect::<Vec<NodeIndex>>();

                if consumers.is_empty() { continue; }
                for consumer in consumers {
                    let barrier = self.graph.add_node(Node::Barrier(B::new(resource.clone())));
                    self.graph.update_edge(node, barrier, resource.uid().clone());
                    self.graph.update_edge(barrier, consumer, resource.uid().clone());
                    if let Some(edge) = self.graph.find_edge(node, consumer) {
                        self.graph.remove_edge(edge);
                    }
                }
            }
        })
    }
}

pub trait GraphViz {
    fn dot(&self) -> Result<String>;
}

impl<D> GraphViz for TaskGraph<PassResource, PassResourceBarrier, PassNode<'_, '_, PassResource, D>> where D: ExecutionDomain {
    fn dot(&self) -> Result<String> {
        Ok(format!("{}", Dot::with_attr_getters(&self.graph, &[], &Self::get_edge_attributes, &Self::get_node_attributes)))
    }
}

impl<D> Display for Node<PassResource, PassResourceBarrier, PassNode<'_, '_, PassResource, D>> where D: ExecutionDomain {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Task(task) => f.write_fmt(format_args!("Task: {}", &task.identifier)),
            Node::Barrier(barrier) => { f.write_fmt(format_args!("{}({:#?} => {:#?})\n({:#?} => {:#?})", &barrier.resource.uid(), barrier.src_access, barrier.dst_access, barrier.src_stage, barrier.dst_stage))}
            Node::_Unreachable(_) => { unreachable!() }
        }
    }
}

