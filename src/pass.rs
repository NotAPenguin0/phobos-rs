use crate::GpuResource;

pub struct Pass {
    pub name: String,
    pub inputs: Vec<GpuResource>,
    pub outputs: Vec<GpuResource>,
}