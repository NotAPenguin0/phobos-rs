use phobos as ph;

#[test]
fn create_context() {
    let settings = ph::AppSettings::<ph::HeadlessWindowInterface> {
        version: (1, 0, 0),
        name: String::from("Phobos test app"),
        enable_validation: true,
        window: None
    };
    let ctx = ph::Context::new(settings).unwrap();
}