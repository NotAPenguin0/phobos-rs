use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use crate::allocator::traits;

use gpu_allocator::vulkan as vk_alloc;

use anyhow::Result;
use ash::vk::{DeviceMemory, DeviceSize, MemoryRequirements};
use gpu_allocator::vulkan::AllocationScheme;
use crate::{Device, Error, PhysicalDevice, VkInstance};
use crate::allocator::memory_type::MemoryType;

#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct DefaultAllocator {
    #[derivative(Debug="ignore")]
    alloc: Arc<Mutex<vk_alloc::Allocator>>,
}

#[derive(Default, Derivative)]
#[derivative(Debug)]
pub struct Allocation {
    allocation: vk_alloc::Allocation,
}

impl DefaultAllocator {
    pub fn new(instance: &VkInstance, device: Arc<Device>, physical_device: &PhysicalDevice) -> Result<Self> {
        Ok(Self {
                alloc: Arc::new(Mutex::new(vk_alloc::Allocator::new(
                &vk_alloc::AllocatorCreateDesc {
                    instance: instance.instance.clone(),
                    device: device.handle.clone(),
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