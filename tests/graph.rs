use std::fs::File;
use std::io::Write;
use std::path::Path;
use layout::backends::svg::SVGWriter;
use phobos as ph;
use ph::task_graph::*;
use layout::gv;
use layout::gv::GraphBuilder;
use layout::topo::layout::VisualGraph;

pub fn display_dot(graph: &ph::TaskGraph) {
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
fn test_graph() -> Result<(), ph::Error> {
    let mut graph = ph::TaskGraph::new();
    
    let t1 = Task {
        inputs: vec![],
        outputs: vec![String::from("t1_out")],
    };

    let t2 = Task {
        inputs: vec![String::from("t1_out")],
        outputs: vec![String::from("t2_out")]
    };

    let t3 = Task {
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