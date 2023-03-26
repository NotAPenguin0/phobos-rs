use std::fmt::Debug;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use ash::vk;
use layout::backends::svg::SVGWriter;
use layout::gv;
use layout::gv::GraphBuilder;
use ph::task_graph::*;
use phobos as ph;
use phobos::pass::{Pass, PassBuilder};
use phobos::pipeline::PipelineStage;
use phobos::{domain, record_graph, IncompleteCommandBuffer};

pub fn display_dot<G>(graph: &G, path: &str)
where
    G: GraphViz, {
    let dot = graph.dot().unwrap();
    let dot = format!("{}", dot);
    let mut parser = gv::DotParser::new(&dot);
    match parser.process() {
        Ok(g) => {
            let mut svg = SVGWriter::new();
            let mut builder = GraphBuilder::new();
            builder.visit_graph(&g);
            let mut vg = builder.get();
            vg.do_it(false, false, false, &mut svg);
            let svg = svg.finalize();
            let mut f = File::create(Path::new(path)).unwrap();
            f.write(&svg.as_bytes()).unwrap();
        }
        Err(e) => {
            parser.print_error();
            println!("dot render error: {}", e);
        }
    }
}

#[test]
fn associated_virtual_resources() -> Result<(), ph::Error> {
    let v1 = VirtualResource {
        uid: String::from("abc"),
    };
    let v2 = VirtualResource {
        uid: String::from("def*"),
    };
    let v3 = VirtualResource {
        uid: String::from("abc*"),
    };
    let v4 = VirtualResource {
        uid: String::from("def**"),
    };

    assert!(VirtualResource::are_associated(&v1, &v3));
    assert!(VirtualResource::are_associated(&v3, &v1));

    assert!(!VirtualResource::are_associated(&v1, &v2));
    assert!(!VirtualResource::are_associated(&v1, &v4));

    assert!(VirtualResource::are_associated(&v2, &v4));

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StringResource(String);

impl Resource for StringResource {
    fn is_dependency_of(&self, lhs: &Self) -> bool {
        self == lhs
    }

    fn uid(&self) -> &String {
        &self.0
    }
}

#[test]
fn test_graph() -> Result<(), ph::Error> {
    let mut graph = ph::PassGraph::<domain::All>::new();

    let offscreen = VirtualResource::new(String::from("offscreen"));
    let depth = VirtualResource::new(String::from("depth"));
    let swap = VirtualResource::new(String::from("swapchain"));

    // Sample graph, not a model of a real render pass system.

    let p1 = PassBuilder::render(String::from("Offscreen render"))
        .color_attachment(offscreen.clone(), vk::AttachmentLoadOp::CLEAR)
        .depth_attachment(depth.clone(), vk::AttachmentLoadOp::CLEAR)
        .get();
    let p2 = PassBuilder::render(String::from("Postprocess render"))
        .color_attachment(p1.output(&offscreen).unwrap(), vk::AttachmentLoadOp::LOAD)
        .sample_image(p1.output(&depth).unwrap(), PipelineStage::VERTEX_SHADER)
        .get();
    let p3 = PassBuilder::render(String::from("Finalize output"))
        .color_attachment(swap.clone(), vk::AttachmentLoadOp::CLEAR)
        .sample_image(p2.output(&offscreen).unwrap(), PipelineStage::FRAGMENT_SHADER)
        .sample_image(p1.output(&depth).unwrap(), PipelineStage::FRAGMENT_SHADER)
        .get();

    let p4 = PassBuilder::present(String::from("present"), p3.output(&swap).unwrap());

    graph.add_pass(p1)?;
    graph.add_pass(p2)?;
    graph.add_pass(p3)?;
    graph.add_pass(p4)?;

    graph.build()?;

    display_dot(graph.task_graph(), "output/2.svg");

    Ok(())
}
