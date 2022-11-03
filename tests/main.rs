use phobos as ph;

#[test]
fn create_context() {
    let settings = ph::AppSettings {
        version: (1, 0, 0),
        name: String::from("Phobos test app"),
        enable_validation: true
    };
    let ctx = ph::Context::new(settings).unwrap();
}