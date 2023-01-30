use std::collections::HashSet;
use petgraph::graph::NodeIndex;
use petgraph::{Incoming, Outgoing};
use petgraph::prelude::EdgeRef;
use petgraph::visit::IntoEdgesDirected;
use crate::domain::{All, ExecutionDomain};
use crate::{GpuTaskGraph, IncompleteCommandBuffer};

// Implementation plan:
// 1. [Check] Get traversal working
// 2. Record commands for barriers
// 3. Verify correctness
// 4. Possibly optimize away unnecessary barriers

// 1. Traversal
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

fn node_children<D>(node: NodeIndex, graph: &GpuTaskGraph<D>) -> impl Iterator<Item = NodeIndex> + '_  where D: ExecutionDomain {
    let graph = &graph.task_graph().graph;
    graph.edges_directed(node, Outgoing).map(|edge| edge.target())
}

fn node_parents<D>(node: NodeIndex, graph: &GpuTaskGraph<D>) -> impl Iterator<Item = NodeIndex> + '_ where D: ExecutionDomain {
    let graph = &graph.task_graph().graph;
    graph.edges_directed(node, Incoming).map(|edge| edge.source())
}

pub fn insert_in_active_set<D>(node: NodeIndex, graph: &GpuTaskGraph<D>, active: &mut HashSet<NodeIndex>, children: &mut HashSet<NodeIndex>) where D: ExecutionDomain {
    children.remove(&node);
    active.insert(node);
    for child in node_children(node, &graph) {
        children.insert(child);
    }
}

pub fn record_graph<D>(graph: &GpuTaskGraph<D>/*, cmd: &IncompleteCommandBuffer<D>*/) where D: ExecutionDomain {
    let start = graph.source();
    let mut active = HashSet::new();
    let mut children = HashSet::new();
    insert_in_active_set(start, &graph, &mut active, &mut children);

    while active.len() != graph.num_nodes() {
        // For each node that is a child of an active node
        let mut recorded_nodes = Vec::new();
        for child in &children {
            // If all parents of this child node are in the active set, record it.
            if node_parents(child.clone(), &graph).all(|parent| active.contains(&parent)) {
                println!("Recording node: {}", graph.task_graph().graph.node_weight(child.clone()).unwrap());
                // TODO: RECORD NODE HERE
                recorded_nodes.push(child.clone());
            }
        }
        // Now we swap all recorded nodes to the active set
        for node in recorded_nodes {
            insert_in_active_set(node.clone(), &graph, &mut active, &mut children);
        }
    }
}