use std::fmt::Debug;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use layout::backends::svg::SVGWriter;
use phobos as ph;
use ph::task_graph::*;
use layout::gv;
use layout::gv::GraphBuilder;

pub fn display_dot<R>(graph: &ph::TaskGraph<R>) where R: Debug + Eq + Clone {
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
            let mut f = File::create(Path::new("output/graph.svg")).unwrap();
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

#[test]
fn test_graph() -> Result<(), ph::Error> {
    let mut graph = ph::TaskGraph::<String>::new();
    
    let t1 = Task {
        identifier: String::from("Task 1"),
        inputs: vec![],
        outputs: vec![String::from("t1_out")],
    };

    let t2 = Task {
        identifier: String::from("Task 2"),
        inputs: vec![String::from("t1_out")],
        outputs: vec![String::from("t2_out")]
    };

    let t3 = Task {
        identifier: String::from("Task 3"),
        inputs: vec![String::from("t1_out"), String::from("t2_out")],
        outputs: vec![String::from("final")]
    };
    graph.add_task(t1)?;
    graph.add_task(t2)?;
    graph.add_task(t3)?;
    graph.create_barrier_nodes();

    display_dot(&graph);

    Ok(())
}