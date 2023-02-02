use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

extern crate shaderc;

fn load_file(path: &Path) -> String {
    let mut out = String::new();
    File::open(path).unwrap().read_to_string(&mut out).unwrap();
    out
}

fn save_file(path: &Path, binary: &[u8]) {
    File::create(path).unwrap().write_all(binary).unwrap();
}

fn compile_shader(path: &Path, kind: shaderc::ShaderKind, output: &Path) {
    let compiler = shaderc::Compiler::new().unwrap();
    let binary = compiler.compile_into_spirv(&load_file(path), kind, path.as_os_str().to_str().unwrap(), "main", None).unwrap();
    save_file(output, binary.as_binary_u8());
}

fn main() {
    println!("cargo:rerun-if-changed=examples/data/vert.glsl");
    println!("cargo:rerun-if-changed=examples/data/frag.glsl");
    println!("cargo:rerun-if-changed=examples/data/blue.glsl");

    compile_shader(Path::new("examples/data/vert.glsl"), shaderc::ShaderKind::Vertex, Path::new("examples/data/vert.spv"));
    compile_shader(Path::new("examples/data/frag.glsl"), shaderc::ShaderKind::Fragment, Path::new("examples/data/frag.spv"));
    compile_shader(Path::new("examples/data/blue.glsl"), shaderc::ShaderKind::Fragment, Path::new("examples/data/blue.spv"));
}