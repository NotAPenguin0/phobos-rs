use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use ash::vk;
use crate::domain::ExecutionDomain;
use crate::graph::task_graph::{Barrier, Node, Resource, Task, TaskGraph};
use crate::graph::resource::{AttachmentType, ResourceUsage};
use crate::graph::virtual_resource::VirtualResource;
use crate::{Error, IncompleteCommandBuffer, InFlightContext, Pass, PhysicalResourceBindings, PipelineStage};

use anyhow::Result;
use petgraph::{Direction, Graph};
use petgraph::graph::NodeIndex;
use petgraph::prelude::EdgeRef;

/// Virtual GPU resource in a task graph.
#[derive(Derivative, Default, Clone)]
#[derivative(Debug)]
pub struct PassResource {
    pub usage: ResourceUsage,
    pub resource: VirtualResource,
    pub stage: PipelineStage,
    pub layout: vk::ImageLayout,
    #[derivative(Debug="ignore")]
    pub clear_value: Option<vk::ClearValue>,
    pub load_op: Option<vk::AttachmentLoadOp>
}

/// GPU barrier in a task graph. Directly translates to `vkCmdPipelineBarrier()`.
#[derive(Debug, Clone)]
pub struct PassResourceBarrier {
    pub resource: PassResource,
    pub src_access: vk::AccessFlags2,
    pub dst_access: vk::AccessFlags2,
    pub src_stage: PipelineStage,
    pub dst_stage: PipelineStage,
}


/// A task in a pass graph. Either a render pass, or a compute pass, etc.
pub struct PassNode<'exec, 'q, R, D> where R: Resource, D: ExecutionDomain {
    pub identifier: String,
    pub color: Option<[f32; 4]>,
    pub inputs: Vec<R>,
    pub outputs: Vec<R>,
    pub execute: Box<dyn FnMut(IncompleteCommandBuffer<'q, D>, &mut InFlightContext, &PhysicalResourceBindings) -> Result<IncompleteCommandBuffer<'q, D>>  + 'exec>,
    pub is_renderpass: bool
}

/// Pass graph, used for synchronizing resources over a single queue.
pub struct PassGraph<'exec, 'q, D> where D: ExecutionDomain {
    pub(crate) graph: TaskGraph<PassResource, PassResourceBarrier, PassNode<'exec, 'q, PassResource, D>>,
    // Note that this is guaranteed to be stable.
    // This is because the only time indices are invalidated is when deleting a node, and even then only the last
    // index is invalidated. Since the source is always the first node, this is never invalidated.
    source: NodeIndex,
    swapchain: Option<VirtualResource>,
    last_usages: HashMap<String, (usize, PipelineStage)>,
}

pub struct BuiltPassGraph<'exec, 'q, D> where D: ExecutionDomain {
    graph: PassGraph<'exec, 'q, D>,
}

impl<'exec, 'q, D> Deref for BuiltPassGraph<'exec, 'q, D> where D: ExecutionDomain {
    type Target = PassGraph<'exec, 'q, D>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl<'exec, 'q, D> DerefMut for BuiltPassGraph<'exec, 'q, D> where D: ExecutionDomain {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl PassResource {
    pub fn virtual_resource(&self) -> &VirtualResource {
        &self.resource
    }
}

impl<> Barrier<PassResource> for PassResourceBarrier {
    fn new(resource: PassResource) -> Self {
        Self {
            src_access: resource.usage.access(),
            dst_access: vk::AccessFlags2::NONE,
            src_stage: resource.stage.clone(),
            dst_stage: PipelineStage::NONE,
            resource,
        }
    }

    fn resource(&self) -> &PassResource {
        &self.resource
    }
}

impl Resource for PassResource {
    fn is_dependency_of(&self, lhs: &Self) -> bool {
        self.virtual_resource().uid == lhs.virtual_resource().uid
    }

    fn uid(&self) -> &String {
        &self.virtual_resource().uid
    }
}

impl<R, D> Task<R> for PassNode<'_, '_, R, D> where R: Resource, D: ExecutionDomain {
    fn inputs(&self) -> &Vec<R> {
        &self.inputs
    }

    fn outputs(&self) -> &Vec<R> {
        &self.outputs
    }
}

impl ResourceUsage {
    pub fn access(&self) -> vk::AccessFlags2 {
        match self {
            ResourceUsage::Nothing => { vk::AccessFlags2::NONE }
            ResourceUsage::Present => { vk::AccessFlags2::NONE }
            ResourceUsage::Attachment(AttachmentType::Color) => { vk::AccessFlags2::COLOR_ATTACHMENT_WRITE }
            ResourceUsage::Attachment(AttachmentType::Depth) => { vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE }
            ResourceUsage::Attachment(AttachmentType::Resolve(_)) => { vk::AccessFlags2::COLOR_ATTACHMENT_WRITE }
            ResourceUsage::ShaderRead => { vk::AccessFlags2::SHADER_READ }
            ResourceUsage::ShaderWrite => { vk::AccessFlags2::SHADER_WRITE }
        }
    }

    pub fn is_read(&self) -> bool {
        match self {
            ResourceUsage::Nothing => { true }
            ResourceUsage::Present => { false }
            ResourceUsage::Attachment(_) => { false }
            ResourceUsage::ShaderRead => { true }
            ResourceUsage::ShaderWrite => { false }
        }
    }
}

macro_rules! barriers {
    ($graph:ident) => {
        $graph.node_indices().filter_map(|node| match $graph.node_weight(node).unwrap() {
            Node::Task(_) => { None }
            Node::Barrier(barrier) => { Some((node, barrier.clone())) }
            Node::_Unreachable(_) => { unreachable!() }
        })
    }
}

impl<'exec, 'q, D> PassGraph<'exec, 'q, D> where D: ExecutionDomain {
    /// Create a new task graph. If rendering to a swapchain, also give it the virtual resource you are planning to use for this.
    /// This is necessary for proper sync
    pub fn new(swapchain: Option<VirtualResource>) -> Self {
        let mut graph = PassGraph {
            graph: TaskGraph::new(),
            source: NodeIndex::default(),
            swapchain,
            last_usages: Default::default(),
        };

        // insert dummy 'source' node. This node produces all initial inputs and is used for start of frame sync.
        graph.graph.add_task(PassNode {
            identifier: "_source".to_string(),
            color: None,
            inputs: vec![],
            outputs: vec![],
            execute: Box::new(|c, _, _| Ok(c)),
            is_renderpass: false,
        }).unwrap();
        graph.source = graph.graph.graph.node_indices().next().unwrap();
        graph
    }

    /// Add a pass to a task graph.
    /// # Errors
    /// - This function can fail if adding the pass results in a cyclic dependency in the graph.
    pub fn add_pass(mut self, pass: Pass<'exec, 'q, D>) -> Result<Self> {
        {
            // Before adding this pass, we need to add every initial input (one with no '+' signs in its uid) to the output of the source node.
            // Note that we dont actually fill the pipeline stages yet, we do that later
            let Node::Task(source) = self.graph.graph.node_weight_mut(self.source).unwrap() else { panic!("Graph does not have a source node"); };
            for input in &pass.inputs {
                if input.resource.is_source() {
                    source.outputs.push(
                        PassResource {
                            usage: ResourceUsage::Nothing,
                            resource: input.resource.clone(),
                            stage: PipelineStage::NONE, // We will set this later!
                            layout: vk::ImageLayout::UNDEFINED,
                            clear_value: None,
                            load_op: None
                        }
                    )
                }
            }
        }

        for input in &pass.inputs {
            self.update_last_usage(&input.resource, input.stage)?;
        }

        for output in &pass.outputs {
            self.update_last_usage(&output.resource, output.stage)?;
        }

        self.graph.add_task(PassNode {
            identifier: pass.name,
            color: pass.color,
            inputs: pass.inputs,
            outputs: pass.outputs,
            execute: pass.execute,
            is_renderpass: pass.is_renderpass
        })?;

        Ok(self)
    }

    /// Builds the task graph so it can be recorded into a command buffer.
    pub fn build(mut self) -> Result<BuiltPassGraph<'exec, 'q, D>> {
        self.set_source_stages()?;
        self.graph.create_barrier_nodes();
        self.merge_identical_barriers()?;

        Ok(BuiltPassGraph {
            graph: self,
        })
    }

    /// Returns the task graph built by the GPU task graph system, useful for outputting dotfiles.
    pub fn task_graph(&self) -> &TaskGraph<PassResource, PassResourceBarrier, PassNode<'exec, 'q, PassResource, D>> {
        &self.graph
    }

    /// Returns the total amount of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.graph.graph.node_count()
    }

    #[allow(dead_code)]
    pub(crate) fn source(&self) -> NodeIndex {
        self.source
    }

    fn update_last_usage(&mut self, resource: &VirtualResource, stage: PipelineStage) -> Result<()> {
        let entry = self.last_usages.entry(resource.name());
        match entry {
            Entry::Occupied(mut entry) => {
                let version = resource.version();
                if version > entry.get().0 {
                    entry.insert((version, stage));
                }
            }
            Entry::Vacant(entry) => {
                entry.insert((resource.version(), stage));
            }
        };
        Ok(())
    }

    #[allow(dead_code)]
    fn barrier_src_resource<'a>(graph: &'a Graph<Node<PassResource, PassResourceBarrier, PassNode<PassResource, D>>, String>, node: NodeIndex) -> Result<&'a PassResource> {
        let Node::Barrier(barrier) = graph.node_weight(node).unwrap() else { return Err(Error::NodeNotFound.into()) };
        let edge = graph.edges_directed(node, Direction::Incoming).next().unwrap();
        let src_node = edge.source();
        // An edge from a barrier always points to a task.
        let Node::Task(task) = graph.node_weight(src_node).unwrap() else { unimplemented!() };
        // This unwrap() cannot fail, or the graph was constructed incorrectly.
        Ok(task.inputs.iter().find(|&input| input.uid() == barrier.resource.uid()).unwrap())
    }

    pub(crate) fn barrier_dst_resource<'a>(graph: &'a Graph<Node<PassResource, PassResourceBarrier, PassNode<PassResource, D>>, String>, node: NodeIndex) -> Result<&'a PassResource> {
        // We know that:
        // 1) Each barrier has at least one outgoing edge
        // 2) During the merge, each outgoing edge from a barrier will have the same resource usage
        // Knowing this, we can simply pick the first edge in the list to determine the resource usage
        let Node::Barrier(barrier) = graph.node_weight(node).unwrap() else { return Err(anyhow::Error::from(Error::NodeNotFound)) };
        let edge = graph.edges(node).next().unwrap();
        let dst_node = edge.target();
        // An edge from a barrier always points to a task.
        let Node::Task(task) = graph.node_weight(dst_node).unwrap() else { unimplemented!() };
        // This unwrap() cannot fail, or the graph was constructed incorrectly.
        Ok(task.inputs.iter().find(|&input| input.uid() == barrier.resource.uid()).unwrap())
    }

    /// Set source barrier stages to the *last* usage in the frame, for cross-frame sync
    fn set_source_stages(&mut self) -> Result<()> {
        let Node::Task(source) = self.graph.graph.node_weight_mut(self.source).unwrap() else { panic!("Graph does not have a source node"); };
        // For each output, look for the last usage of this resource in the frame.
        for output in &mut source.outputs {
            // Will only succeed if swapchain is set and this resource is the swapchain
            let default = VirtualResource::image("__none__internal__");
            if VirtualResource::are_associated(&output.resource, self.swapchain.as_ref().unwrap_or(&default)) {
                output.stage = PipelineStage::COLOR_ATTACHMENT_OUTPUT;
            }
            else {
                let (_, stage) = self.last_usages.get(&output.resource.name()).unwrap();
                output.stage = *stage;
            }
        }
        Ok(())
    }

    // Pass in the build step where identical barriers are merged into one for efficiency reasons.
    fn merge_identical_barriers(&mut self) -> Result<()> {
        let graph: &mut Graph<_, _> = &mut self.graph.graph;
        // Find a barrier that has duplicates
        let mut to_remove = Vec::new();
        let mut edges_to_add = Vec::new();
        let mut barrier_flags: HashMap<NodeIndex, _> = HashMap::new();

        for (node, barrier) in barriers!(graph) {
            let dst_resource = &Self::barrier_dst_resource(&graph, node)?;
            let dst_usage = dst_resource.usage.clone();
            barrier_flags.insert(node, (dst_resource.stage.clone(), dst_usage.access()));
            // Now we know the usage of this barrier, we can find all other barriers with the exact same resource usage and
            // merge those with this one
            for (other_node, other_barrier) in barriers!(graph) {
                if other_node == node { continue; }
                if to_remove.contains(&node) { continue; }
                let other_resource = Self::barrier_dst_resource(&graph, other_node)?;
                let other_usage = &other_resource.usage;
                if other_barrier.resource.uid() == barrier.resource.uid() {
                    if !other_usage.is_read() && !dst_usage.is_read() && other_usage != &dst_usage {
                        return Err(anyhow::Error::from(Error::IllegalTaskGraph));
                    }
                    to_remove.push(other_node);
                    edges_to_add.push((node, graph.edges(other_node).next().unwrap().target(), other_resource.uid().clone()));
                    let (stage, access) = barrier_flags.get(&node).cloned().unwrap();
                    barrier_flags.insert(node, (other_resource.stage | stage, other_resource.usage.access() | access));
                }
            }
        }

        for (src, dst, uid) in edges_to_add {
            graph.update_edge(src, dst, uid);
        }
        for node in graph.node_indices() {
            if let Node::Barrier(barrier) = graph.node_weight_mut(node).unwrap() {
                let (stage, access) = barrier_flags.get(&node).cloned().unwrap();
                barrier.dst_stage = stage;
                barrier.dst_access = access;
            }
        }
        graph.retain_nodes(|_, node| { !to_remove.contains(&node) });

        Ok(())
    }
}