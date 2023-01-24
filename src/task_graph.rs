use std::collections::HashSet;
use std::fmt::Display;

use petgraph::graph::*;
use petgraph::visit::NodeRef;
use petgraph;
use petgraph::dot;
use petgraph::dot::Dot;

use crate::error::Error;

#[derive(Debug, Clone)]
pub struct Task {
    // TODO: Actual inputs and outputs instead of this.
    // We just use strings for testing the algorithms.
    pub inputs: Vec<String>,
    pub outputs: Vec<String>
}

// Problem:
// We specify tasks in an unknown order.
// Each task may require inputs from previous tasks's outputs.
// Suppose multiple task produce the same output
// We do not know which version of the output we need
// Possible solution: 
// - Outputting a resource produces a unique identifier, even if
//   other tasks also output to the same physical resource.
//   This way we can match the used identifier to the task it is produced/used in.
// - This allows multiple logical bindings (tags/resource identifiers) to map to 
//   a single physical resource (Image/Buffer).

#[derive(Debug, Clone)]
pub struct Barrier {
    pub resource: String
}

#[derive(Debug, Clone)]
pub enum Node {
    Task(Task),
    Barrier(Barrier)
}

pub struct TaskGraph {
    // u32 is the associated data with each edge.
    // We can change this to whatever we want later.
    graph: Graph<Node, u32>,

    // We must also detect cycles in the graph as errors.
}

impl TaskGraph {
    pub fn new() -> Self {
        TaskGraph {
            graph: Graph::new()
        }
    }

    /// Outputs graphviz-compatible dot file for displaying the graph.
    pub fn as_dot(&self) -> Result<Dot<&Graph<Node, u32>>, Error> {
        Ok(petgraph::dot::Dot::with_config(&self.graph, &[petgraph::dot::Config::EdgeNoLabel]))
    }

    // Tests if a task is dependent on another task.
    // Will return true if child consumes an input that parent produces
    fn is_dependent(graph: &Graph<Node, u32>, child: NodeIndex, parent: NodeIndex) -> Result<bool, Error> {
        let child = graph.node_weight(child).ok_or(Error::NodeNotFound)?;
        let parent = graph.node_weight(parent).ok_or(Error::NodeNotFound)?;
        if let Node::Task(child) = child {
            if let Node::Task(parent) = parent {
                return Ok(child.inputs.iter().any(|input| {
                    parent.outputs.contains(&input)
                }));
            }
        }

        Ok(false)
    }

    fn is_task_node(graph: &Graph<Node, u32>, node: NodeIndex) -> Result<bool, Error> {
        Ok(matches!(graph.node_weight(node).ok_or(Error::NodeNotFound)?, Node::Task(_)))
    }

    pub fn add_task(&mut self, task: Task) -> Result<(), Error> {
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
            if Self::is_dependent(&self.graph, node, other_node).unwrap() {
                self.graph.add_edge(other_node, node, 0);
            }

            // Note: no else here, since we will detect cycles and error on them,
            // which is better than silently ignoring some cycles.
            if Self::is_dependent(&self.graph, other_node, node).unwrap() {
                self.graph.add_edge(node, other_node, 0);
            }
        });

        match petgraph::algo::is_cyclic_directed(&self.graph) {
            true => Err(Error::GraphHasCycle),
            false => Ok(())
        }
    }

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
                        Node::Task(t) => { t.inputs.contains(&resource) }
                        Node::Barrier(_) => false
                    }
                }).collect::<Vec<NodeIndex>>();

                if consumers.is_empty() { return; }
                let barrier = self.graph.add_node(Node::Barrier(Barrier{resource: resource.clone()}));
                self.graph.update_edge(node, barrier, 0);
                for consumer in consumers {
                    self.graph.update_edge(barrier, consumer, 0);
                    self.graph.remove_edge(self.graph.find_edge(node, consumer).unwrap());
                }
            }
        })
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Task(task) => f.write_fmt(format_args!("Task\nin: {:#?}\nout: {:#?}", &task.inputs, &task.outputs)),
            Node::Barrier(barrier) => { f.write_fmt(format_args!("Barrier on {:#?}", &barrier.resource))}
        }
    } 
}

impl Display for TaskGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dot = petgraph::dot::Dot::new(&self.graph);
        dot.fmt(f)
    }
}


