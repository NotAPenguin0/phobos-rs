fn main() {
    println!("cargo:rerun-if-changed=examples/data/vert.glsl");
    println!("cargo:rerun-if-changed=examples/data/frag.glsl");
}