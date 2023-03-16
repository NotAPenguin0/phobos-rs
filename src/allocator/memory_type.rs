/// The memory type of an allocation indicates where it should live.
#[derive(Debug)]
pub enum MemoryType {
    /// Store the allocation in GPU only accessible memory - typically this is the faster GPU resource and this should be
    /// where most of the allocations live.
    GpuOnly,
    /// Memory useful for uploading data to the GPU and potentially for constant buffers
    CpuToGpu,
    /// Memory useful for CPU readback of data
    GpuToCpu,
}

impl From<MemoryType> for gpu_allocator::MemoryLocation {
    fn from(value: MemoryType) -> Self {
        match value {
            MemoryType::GpuOnly => { gpu_allocator::MemoryLocation::GpuOnly }
            MemoryType::CpuToGpu => { gpu_allocator::MemoryLocation::CpuToGpu }
            MemoryType::GpuToCpu => { gpu_allocator::MemoryLocation::GpuToCpu }
        }
    }
}