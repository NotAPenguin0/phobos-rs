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

pub fn display_dot<R>(graph: &ph::TaskGraph<R>, path: &str) where R: Debug + Resource + Clone {
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
        Err(_) => {
            parser.print_error();
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

    let p1 = Pass {
        name: "Offscreen render".to_string(),
        inputs: vec![],
        outputs: vec![
            GpuResource::Image(ImageResource{
                usage: ImageResourceUsage::Attachment,
                resource: offscreen.clone()
            }),
            GpuResource::Image(ImageResource{
                usage: ImageResourceUsage::Attachment,
                resource: depth.clone()
            })
        ]
    };

    let p2 = Pass {
        name: "Postprocess render".to_string(),
        inputs: vec![
            GpuResource::Image(ImageResource{
                usage: ImageResourceUsage::Sample,
                resource: offscreen.clone()
            })],
        outputs: vec![
            GpuResource::Image(ImageResource{
                usage: ImageResourceUsage::Attachment,
                resource: offscreen.upgrade()
            })
        ]
    };

    // Todo: we can abstract this concept away.
    let GpuResource::Image(p2_output) = &p2.outputs[0] else { unimplemented!() };
    let p2_output = p2_output.resource.clone();


    let p3 = Pass {
        name: "Finalize output".to_string(),
        inputs: vec![
            GpuResource::Image(ImageResource{
                usage: ImageResourceUsage::Sample,
                resource: p2_output
            }),
            GpuResource::Image(ImageResource{
                usage: ImageResourceUsage::Sample,
                resource: depth.clone(),
            })
        ],
        outputs: vec![
            GpuResource::Image(ImageResource {
                usage: ImageResourceUsage::Attachment,
                resource: swap.clone()
            })
        ]
    };

    graph.add_pass(p1)?;
    graph.add_pass(p2)?;
    graph.add_pass(p3)?;

    graph.build();

    display_dot(&graph.task_graph(), "output/2.svg");

    Ok(())
}