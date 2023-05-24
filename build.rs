#[cfg(feature = "shaderc")]
extern crate shaderc;

#[allow(unused_imports)]
use std::fs::File;
#[allow(unused_imports)]
use std::io::{Read, Write};
#[allow(unused_imports)]
use std::path::Path;

#[cfg(feature = "shaderc")]
use shaderc::{CompileOptions, EnvVersion, TargetEnv};

#[cfg(feature = "shaderc")]
fn load_file(path: &Path) -> String {
    let mut out = String::new();
    File::open(path).unwrap().read_to_string(&mut out).unwrap();
    out
}

#[cfg(feature = "shaderc")]
fn save_file(path: &Path, binary: &[u8]) {
    File::create(path).unwrap().write_all(binary).unwrap();
}

#[cfg(feature = "shaderc")]
fn compile_shader(path: &Path, kind: shaderc::ShaderKind, output: &Path) {
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = CompileOptions::new().unwrap();
    options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_2 as u32);
    let binary = compiler
        .compile_into_spirv(&load_file(path), kind, path.as_os_str().to_str().unwrap(), "main", Some(&options))
        .unwrap();
    save_file(output, binary.as_binary_u8());
}

#[cfg(feature = "shaderc")]
fn compile_shaders() {
    println!("cargo:rerun-if-changed=examples/data/vert.glsl");
    println!("cargo:rerun-if-changed=examples/data/frag.glsl");
    println!("cargo:rerun-if-changed=examples/data/blue.glsl");
    println!("cargo:rerun-if-changed=examples/data/compute.glsl");
    println!("cargo:rerun-if-changed=examples/data/raygen.rgen");
    println!("cargo:rerun-if-changed=examples/data/rayhit.rchit");
    println!("cargo:rerun-if-changed=examples/data/raymiss.rmiss");
    println!("cargo:rerun-if-changed=examples/data/fsr_render_frag.glsl");
    println!("cargo:rerun-if-changed=examples/data/fsr_render_vert.glsl");

    compile_shader(
        Path::new("examples/data/vert.glsl"),
        shaderc::ShaderKind::Vertex,
        Path::new("examples/data/vert.spv"),
    );
    compile_shader(
        Path::new("examples/data/frag.glsl"),
        shaderc::ShaderKind::Fragment,
        Path::new("examples/data/frag.spv"),
    );
    compile_shader(
        Path::new("examples/data/blue.glsl"),
        shaderc::ShaderKind::Fragment,
        Path::new("examples/data/blue.spv"),
    );
    compile_shader(
        Path::new("examples/data/compute.glsl"),
        shaderc::ShaderKind::Compute,
        Path::new("examples/data/compute.spv"),
    );
    compile_shader(
        Path::new("examples/data/raygen.rgen"),
        shaderc::ShaderKind::RayGeneration,
        Path::new("examples/data/raygen.spv"),
    );
    compile_shader(
        Path::new("examples/data/rayhit.rchit"),
        shaderc::ShaderKind::ClosestHit,
        Path::new("examples/data/rayhit.spv"),
    );
    compile_shader(
        Path::new("examples/data/raymiss.rmiss"),
        shaderc::ShaderKind::Miss,
        Path::new("examples/data/raymiss.spv"),
    );
    compile_shader(
        Path::new("examples/data/fsr_render_frag.glsl"),
        shaderc::ShaderKind::Fragment,
        Path::new("examples/data/fsr_render_frag.spv"),
    );
    compile_shader(
        Path::new("examples/data/fsr_render_vert.glsl"),
        shaderc::ShaderKind::Vertex,
        Path::new("examples/data/fsr_render_vert.spv"),
    );
}

fn main() {
    #[cfg(feature = "shaderc")]
    compile_shaders();
}
