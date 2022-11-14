use phobos as ph;
use ph::task_graph::*;

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

    println!("{}", &graph);

    Ok(())
}