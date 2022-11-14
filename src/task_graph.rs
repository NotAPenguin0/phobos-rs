use std::fmt::Display;

use petgraph::graph::*;
use petgraph::visit::NodeRef;
use petgraph;
use petgraph::dot;

use crate::error::Error;

#[derive(Debug)]
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

pub struct TaskGraph {
    // u32 is the associated data with each edge.
    // We can change this to whatever we want later.
    graph: Graph<Task, u32>,

    // We must also detect cycles in the graph as errors.
}

impl TaskGraph {
    pub fn new() -> Self {
        TaskGraph {
            graph: Graph::new()
        }
    }

    // Tests if a task is dependent on another task.
    // Will return true if child consumes an input that parent produces
    fn is_dependent(graph: &Graph<Task, u32>, child: NodeIndex, parent: NodeIndex) -> Result<bool, Error> {
        let child = graph.node_weight(child).ok_or(Error::NodeNotFound)?;
        let parent = graph.node_weight(parent).ok_or(Error::NodeNotFound)?;
        Ok(child.inputs.iter().any(|input| {
            parent.outputs.contains(&input)
        }))
    }

    pub fn add_task(&mut self, task: Task) -> Result<(), Error> {
        let node = self.graph.add_node(task);
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
}

impl Display for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("in: {:#?}\nout: {:#?}", &self.inputs, &self.outputs))
    } 
}

impl Display for TaskGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dot = petgraph::dot::Dot::new(&self.graph);
        dot.fmt(f)
    }
}


