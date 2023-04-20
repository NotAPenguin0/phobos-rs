//! Exposes different memory types that determine where memory allocations should live.

/// The memory type of an allocation indicates where it should live.
/// Give this to an [`Allocator`](crate::Allocator) to let it decide
/// where your allocation should live.
///
/// See also: [`Allocator::allocate()`](crate::Allocator::allocate()), [`DefaultAllocator::allocate()`](crate::DefaultAllocator::allocate())
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum MemoryType {
    /// Store the allocation in GPU only accessible memory - typically this is the faster GPU resource and this should be
    /// where most of the allocations live.
    GpuOnly,
    /// Memory useful for uploading data to the GPU and potentially for constant buffers. On most implementations,
    /// this maps to the PCIe BAR (See <https://stackoverflow.com/questions/30190050/what-is-the-base-address-register-bar-in-pcie>).
    CpuToGpu,
    /// Memory useful for CPU readback of data.
    GpuToCpu,
}

impl From<MemoryType> for gpu_allocator::MemoryLocation {
    fn from(value: MemoryType) -> Self {
        match value {
            MemoryType::GpuOnly => gpu_allocator::MemoryLocation::GpuOnly,
            MemoryType::CpuToGpu => gpu_allocator::MemoryLocation::CpuToGpu,
            MemoryType::GpuToCpu => gpu_allocator::MemoryLocation::GpuToCpu,
        }
    }
}
