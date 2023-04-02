use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use ash::vk::{DeviceMemory, DeviceSize, MemoryRequirements};
use gpu_allocator::vulkan as vk_alloc;
use gpu_allocator::vulkan::AllocationScheme;

use crate::{Allocator, Device, Error, PhysicalDevice, VkInstance};
use crate::allocator::memory_type::MemoryType;
use crate::allocator::traits;

/// The default allocator. This calls into the `gpu_allocator` crate.
/// It's important to note that this allocator is `Clone`, `Send` and `Sync`. All its internal state is safely
/// wrapped inside an `Arc<Mutex<T>>`. This is to facilitate passing it around everywhere.
///
/// See also: [`Allocator`](traits::Allocator), [`Allocation`](traits::Allocation)
///
/// # Example
/// ```
/// # use phobos::*;
///
/// let mut allocator = DefaultAllocator::new(&instance, &device, &physical_device)?;
/// let requirements: vk::MemoryRequirements = vkGetMemoryRequirements(device, buffer);
/// let memory = allocator.allocate("buffer_memory", &requirements, MemoryType::GpuOnly)?;
/// // SAFETY: We are passing `memory.offset()` correctly.
/// vkBindBufferMemory(device, buffer, unsafe { memory.memory() }, memory.offset());
/// ```
#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct DefaultAllocator {
    #[derivative(Debug = "ignore")]
    alloc: Arc<Mutex<vk_alloc::Allocator>>,
}

/// Allocation returned from the default allocator.
#[derive(Derivative)]
#[derivative(Default, Debug)]
pub struct Allocation {
    allocator: Option<DefaultAllocator>,
    allocation: Option<vk_alloc::Allocation>,
}

impl DefaultAllocator {
    /// Create a new default allocator.
    pub fn new(instance: &VkInstance, device: &Device, physical_device: &PhysicalDevice) -> Result<Self> {
        Ok(Self {
            alloc: Arc::new(Mutex::new(vk_alloc::Allocator::new(&vk_alloc::AllocatorCreateDesc {
                instance: (*instance).clone(),
                device: unsafe { device.handle() },
                physical_device: unsafe { physical_device.handle() },
                debug_settings: Default::default(),
                buffer_device_address: false, // We might change this if the bufferDeviceAddress feature gets enabled.
            })?)),
        })
    }
}

impl DefaultAllocator {
    fn free_impl(&mut self, allocation: &mut <Self as Allocator>::Allocation) -> Result<()> {
        let mut alloc = self.alloc.lock().map_err(|_| Error::PoisonError)?;
        let memory = allocation.allocation.take().unwrap();
        alloc.free(memory)?;
        Ok(())
    }
}

impl Allocator for DefaultAllocator {
    type Allocation = Allocation;

    /// Allocates raw memory of a specific memory type. The given name is used for internal tracking.
    fn allocate(&mut self, name: &'static str, requirements: &MemoryRequirements, ty: MemoryType) -> Result<Self::Allocation> {
        let mut alloc = self.alloc.lock().map_err(|_| Error::PoisonError)?;
        let allocation = alloc.allocate(&vk_alloc::AllocationCreateDesc {
            name,
            requirements: *requirements,
            location: gpu_allocator::MemoryLocation::from(ty),
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        Ok(Allocation {
            allocator: Some(self.clone()),
            allocation: Some(allocation),
        })
    }

    /// Free some memory allocated from this allocator.
    fn free(&mut self, mut allocation: Self::Allocation) -> Result<()> {
        self.free_impl(&mut allocation)
    }
}

impl traits::Allocation for Allocation {
    unsafe fn memory(&self) -> DeviceMemory {
        self.allocation.as_ref().unwrap().memory()
    }

    fn offset(&self) -> DeviceSize {
        self.allocation.as_ref().unwrap().offset()
    }

    fn mapped_ptr(&self) -> Option<NonNull<c_void>> {
        self.allocation.as_ref().unwrap().mapped_ptr()
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        let mut allocator = self.allocator.clone().unwrap();
        allocator.free_impl(self).unwrap();
    }
}