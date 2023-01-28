use std::fmt::Debug;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use layout::backends::svg::SVGWriter;
use phobos as ph;
use ph::task_graph::*;
use layout::gv;
use layout::gv::GraphBuilder;
use phobos::pass::Pass;
use phobos::pipeline::PipelineStage;

pub fn display_dot<R, B>(graph: &ph::TaskGraph<R, B>, path: &str) where R: Debug + Default + Resource + Clone, B: Barrier<R> + Clone {
    let dot = graph.as_dot().unwrap();
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
        },
        Err(e) => {
            parser.print_error();
            println!("dot render error: {}", e);
        }
    }
}

#[test]
fn associated_virtual_resources() -> Result<(), ph::Error> {
    let v1 = VirtualResource { uid: String::from("abc") };
    let v2 = VirtualResource { uid: String::from("def*") };
    let v3 = VirtualResource { uid: String::from("abc*") };
    let v4 = VirtualResource { uid: String::from("def**") };

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
    let mut graph = ph::GpuTaskGraph::new();

    let offscreen = VirtualResource::new(String::from("offscreen"));
    let depth = VirtualResource::new(String::from("depth"));
    let swap = VirtualResource::new(String::from("swapchain"));

    // Sample graph, not a model of a real render pass system.

    let p1 = Pass {
        name: "Offscreen render".to_string(),
        inputs: vec![],
        outputs: vec![
            GpuResource {
                usage: ResourceUsage::Attachment,
                resource: offscreen.clone(),
                stage: PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            },
            GpuResource {
                usage: ResourceUsage::Attachment,
                resource: depth.clone(),
                stage: PipelineStage::COLOR_ATTACHMENT_OUTPUT
            }
        ]
    };

    let p2 = Pass {
        name: "Postprocess render".to_string(),
        inputs: vec![
            GpuResource {
                usage: ResourceUsage::ShaderRead,
                resource: offscreen.clone(),
                stage: PipelineStage::FRAGMENT_SHADER
            },
            GpuResource{
                usage: ResourceUsage::ShaderRead,
                resource: depth.clone(),
                stage: PipelineStage::VERTEX_SHADER,
            }
        ],
        outputs: vec![
            GpuResource{
                usage: ResourceUsage::Attachment,
                resource: offscreen.upgrade(),
                stage: PipelineStage::COLOR_ATTACHMENT_OUTPUT
            }
        ]
    };

    // Todo: we can abstract this concept away.
    let p2_output = &p2.outputs[0] else { unimplemented!() };
    let p2_output = p2_output.resource.clone();


    let p3 = Pass {
        name: "Finalize output".to_string(),
        inputs: vec![
            GpuResource{
                usage: ResourceUsage::ShaderRead,
                resource: p2_output,
                stage: PipelineStage::FRAGMENT_SHADER
            },
            GpuResource{
                usage: ResourceUsage::ShaderRead,
                resource: depth.clone(),
                stage: PipelineStage::FRAGMENT_SHADER,
            }
        ],
        outputs: vec![
            GpuResource {
                usage: ResourceUsage::Attachment,
                resource: swap.clone(),
                stage: PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            }
        ]
    };

    graph.add_pass(p1)?;
    graph.add_pass(p2)?;
    graph.add_pass(p3)?;

    graph.build()?;

    display_dot(&graph.task_graph(), "output/2.svg");

    Ok(())
}