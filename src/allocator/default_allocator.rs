//! Contains a default allocator type based on the [`gpu_allocator`] crate that is good for most needs.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use ash::vk::{DeviceMemory, DeviceSize, MemoryRequirements};
use gpu_allocator::vulkan as vk_alloc;
use gpu_allocator::vulkan::AllocationScheme;

use crate::{Allocator, Device, Error, Instance, PhysicalDevice};
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
/// # use anyhow::Result;
/// # fn vk_get_memory_requirements(device: &Device, buffer: &Buffer) -> vk::MemoryRequirements { unimplemented!() }
/// # fn vk_bind_buffer_memory(device: &Device, buffer: &Buffer, memory: vk::DeviceMemory, offset: u64) { unimplemented!() }
/// fn use_allocator(
///     instance: Instance,
///     physical_device: PhysicalDevice,
///     device: Device,
///     allocator: DefaultAllocator,
///     buffer: Buffer)
///     -> Result<()> {
///     let mut allocator = DefaultAllocator::new(&instance, &device, &physical_device)?;
///     let requirements = vk_get_memory_requirements(&device, &buffer);
///     let memory = allocator.allocate("buffer_memory", &requirements, MemoryType::GpuOnly)?;
///     // SAFETY: We are passing `memory.offset()` correctly.
///     vk_bind_buffer_memory(&device, &buffer, unsafe { memory.memory() }, memory.offset());
///     Ok(())
/// }
/// ```
#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct DefaultAllocator {
    #[derivative(Debug = "ignore")]
    alloc: Arc<Mutex<vk_alloc::Allocator>>,
}

/// Allocation returned from the default allocator.
/// Can be obtained by calling using [`DefaultAllocator::allocate()`]. This allocation is automatically freed
/// when it is dropped, so it's not strictly necessary to call [`DefaultAllocator::free()`].
///
/// See also: [`DefaultAllocator`], [Allocation](traits::Allocation)
/// # Example
/// ```
/// # use phobos::*;
/// let mut allocator = DefaultAllocator::new(&instance, &device, &physical_device)?;
/// // Note: Supply memory_requirements.
/// let allocation = allocator.allocate("buffer_memory", &memory_requirements, MemoryType::GpuOnly)?;
/// ```
#[derive(Derivative)]
#[derivative(Default, Debug)]
pub struct Allocation {
    // These are wrapped in `Option`s so we can "move" out of them in `Drop`.
    // They are always Some(_)
    allocator: Option<DefaultAllocator>,
    allocation: Option<vk_alloc::Allocation>,
}

impl DefaultAllocator {
    /// Create a new default allocator.
    /// # Errors
    /// * May fail if creating the internal `gpu_allocator` fails.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// let mut allocator = DefaultAllocator::new(&instance, &device, &physical_device)?;
    /// // Use allocator.
    /// ```
    pub fn new(
        instance: &Instance,
        device: &Device,
        physical_device: &PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            alloc: Arc::new(Mutex::new(vk_alloc::Allocator::new(
                &vk_alloc::AllocatorCreateDesc {
                    instance: (*instance).clone(),
                    // SAFETY: The user passed in a valid Device reference.
                    device: unsafe { device.handle() },
                    // SAFETY: The user passed in a valid PhysicalDevice reference.
                    physical_device: unsafe { physical_device.handle() },
                    debug_settings: Default::default(),
                    buffer_device_address: true,
                },
            )?)),
        })
    }
}

impl DefaultAllocator {
    fn free_impl(&mut self, allocation: &mut <Self as Allocator>::Allocation) -> Result<()> {
        let mut alloc = self.alloc.lock().map_err(|_| Error::PoisonError)?;
        match allocation.allocation.take() {
            None => {}
            Some(allocation) => {
                alloc.free(allocation)?;
            }
        }
        Ok(())
    }
}

impl Allocator for DefaultAllocator {
    /// The allocation type that is returned from calling [`DefaultAllocator::allocate()`]
    type Allocation = Allocation;

    /// Allocates raw memory of a specific memory type. The given name is used for internal tracking and
    /// debug logging. To get proper [`MemoryRequirements`], call [`vkGetBufferMemoryRequirements`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetBufferMemoryRequirements.html) or
    /// [`vkGetImageMemoryRequirements`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetImageMemoryRequirements.html) with your buffer or image.
    /// # Errors
    /// * May fail if the device is out of memory
    /// * May fail if invalid [`MemoryRequirements`] were passed in.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use anyhow::Result;
    /// # fn vk_get_memory_requirements(device: &Device, buffer: &Buffer) -> vk::MemoryRequirements { unimplemented!() }
    /// # fn vk_bind_buffer_memory(device: &Device, buffer: &Buffer, memory: vk::DeviceMemory, offset: u64) { unimplemented!() }
    /// fn use_allocator(device: Device, allocator: DefaultAllocator, buffer: Buffer) -> Result<()> {
    ///     let mut allocator = DefaultAllocator::new(&instance, &device, &physical_device)?;
    ///     let requirements: vk_get_memory_requirements(&device, &buffer);
    ///     let memory = allocator.allocate("buffer_memory", &requirements, MemoryType::GpuOnly)?;
    ///     // SAFETY: We are passing `memory.offset()` correctly.
    ///     vk_bind_buffer_memory(&device, &buffer, unsafe { memory.memory() }, memory.offset());
    ///     Ok(())
    /// }
    /// ```
    fn allocate(
        &mut self,
        name: &str,
        requirements: &MemoryRequirements,
        ty: MemoryType,
    ) -> Result<Self::Allocation> {
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

    /// Explicitly free memory owned by this allocator. This is generally not needed,
    /// since the implementation of [`Drop`] for [`DefaultAllocator::Allocation`](Allocation)
    /// already handles this. This function is still present to satisfy the [`Allocator`] trait.
    /// # Errors
    /// * May fail if the vulkan context is no longer valid.
    /// * May fail if the allocation was not valid.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// let mut allocator = DefaultAllocator::new(&instance, &device, &physical_device)?;
    /// let allocation = allocator.allocate("buffer_memory", &get_memory_requirements(buffer), MemoryType::GpuOnly)?;
    /// // ... use allocation
    ///
    /// // Or just drop the allocation.
    /// allocator.free(allocation)?;
    /// ```
    fn free(&mut self, mut allocation: Self::Allocation) -> Result<()> {
        self.free_impl(&mut allocation)
    }
}

impl traits::Allocation for Allocation {
    /// Get unsafe access to the underlying [`VkDeviceMemory`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDeviceMemory.html).
    /// Should always be used together with [`Allocation::offset()`](crate::traits::Allocation::offset()).
    /// # Example
    /// This is useful when binding memory to a buffer or image. For [`Buffer`](crate::Buffer) and [`Image`](crate::Image) this is already
    /// done internally.
    /// ```
    /// # use phobos::*;
    /// # use anyhow::Result;
    /// # use ash::prelude::VkResult;
    /// fn bind_allocation_to_buffer<A: Allocation>(device: Device, allocation: &mut A, buffer: vk::Buffer) -> VkResult<()> {
    ///     // SAFETY:
    ///     // * User passed in a valid Vulkan device.
    ///     // * User passed in a valid allocation.
    ///     // * We offset into allocation.memory() using allocation.offset()
    ///     unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }
    /// }
    /// ```
    /// # Safety
    /// The user must not free this memory or access a range outside of (`allocation.offset()..allocation.offset() + allocation.size())`.
    unsafe fn memory(&self) -> DeviceMemory {
        self.allocation.as_ref().unwrap().memory()
    }

    /// Get the offset in this `VkDeviceMemory` this allocation refers to. This is exposed because the allocator implementation may choose
    /// to subdivide large memory blocks into smaller allocations.
    /// # Example
    /// This is useful when binding memory to a buffer or image.
    /// ```
    /// # use phobos::*;
    /// # use anyhow::Result;
    /// # use ash::prelude::VkResult;
    /// fn bind_allocation_to_buffer<A: Allocation>(device: Device, allocation: &mut A, buffer: vk::Buffer) -> VkResult<()> {
    ///     // SAFETY:
    ///     // * User passed in a valid Vulkan device.
    ///     // * User passed in a valid allocation.
    ///     // * We offset into allocation.memory() using allocation.offset()
    ///     unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }
    /// }
    /// ```
    fn offset(&self) -> DeviceSize {
        self.allocation.as_ref().unwrap().offset()
    }

    /// Obtain a mapped pointer to this allocation. This pointer can be used to directly write into the owned memory.
    /// This pointer already points into the exact memory region of the suballocation, so no offset must be applied.
    ///
    /// Note: generally you want to write through a buffer instead of directly through the allocation, though internally
    /// it goes through this path anyway.
    ///
    /// * Returns `None` if this memory was not mappable (not [`HOST_VISIBLE`](ash::vk::MemoryPropertyFlags::HOST_VISIBLE)). Memory allocated with [`MemoryType::CpuToGpu`] is always mappable.
    /// * If this memory comes from a [`HOST_VISIBLE`](ash::vk::MemoryPropertyFlags::HOST_VISIBLE) heap, this returns `Some(ptr)`, with `ptr` a non-null pointer pointing to the allocation memory.
    /// # Example
    /// ```
    /// use phobos::*;
    /// // Writes the integer '5' into the first std::mem::size_of::<i32>() bytes of the allocation.
    /// // Assumes this allocation is mappable and at least std::mem::size_of::<i32>() bytes large.
    /// unsafe fn write_five<A: Allocation>(allocation: &A) {
    ///     let memory = allocation.mapped_ptr().expect("Expected allocation to be HOST_VISIBLE");
    ///     // SAFETY: Assume this allocation is at least std::mem::size_of::<i32>() bytes large.
    ///     *memory.cast::<i32>().as_mut() = 5;
    /// }
    /// ```
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
