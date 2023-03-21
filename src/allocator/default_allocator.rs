use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use ash::vk::{DeviceMemory, DeviceSize, MemoryRequirements};
use gpu_allocator::vulkan as vk_alloc;
use gpu_allocator::vulkan::AllocationScheme;

use crate::{Device, Error, PhysicalDevice, VkInstance};
use crate::allocator::memory_type::MemoryType;
use crate::allocator::traits;

/// The default allocator. This calls into the `gpu_allocator` crate.
/// It's important to note that this allocator is `Clone`, `Send` and `Sync`. All its internal state is safely
/// wrapped inside an `Arc<Mutex<T>>`. This is to facilitate passing it around everywhere.
#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct DefaultAllocator {
    #[derivative(Debug="ignore")]
    alloc: Arc<Mutex<vk_alloc::Allocator>>,
}

/// Allocation returned from the default allocator. This must be freed explicitly by calling [`DefaultAllocator::free()`](crate::DefaultAllocator::free)
#[derive(Default, Derivative)]
#[derivative(Debug)]
pub struct Allocation {
    allocation: vk_alloc::Allocation,
}

impl DefaultAllocator {
    /// Create a new default allocator.
    pub fn new(instance: &VkInstance, device: &Arc<Device>, physical_device: &PhysicalDevice) -> Result<Self> {
        Ok(Self {
                alloc: Arc::new(Mutex::new(vk_alloc::Allocator::new(
                &vk_alloc::AllocatorCreateDesc {
                    instance: instance.instance.clone(),
                    device: unsafe { device.handle() },
                    physical_device: physical_device.handle.clone(),
                    debug_settings: Default::default(),
                    buffer_device_address: false // We might change this if the bufferDeviceAddress feature gets enabled.
                    }
                )?))
            }
        )
    }
}

impl traits::Allocator for DefaultAllocator {
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
            allocation,
        })
    }

    /// Free some memory allocated from this allocator.
    fn free(&mut self, allocation: Self::Allocation) -> Result<()> {
        let mut alloc = self.alloc.lock().map_err(|_| Error::PoisonError)?;
        alloc.free(allocation.allocation)?;
        Ok(())
    }
}

impl traits::Allocation for Allocation {
    unsafe fn memory(&self) -> DeviceMemory {
        self.allocation.memory()
    }

    fn offset(&self) -> DeviceSize {
        self.allocation.offset()
    }

    fn mapped_ptr(&self) -> Option<NonNull<c_void>> {
        self.allocation.mapped_ptr()
    }
}