//! Allocator traits to implement for using your own custom allocator with phobos

use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::Result;
use ash::vk;

use crate::allocator::memory_type::MemoryType;

/// To supply custom allocators to phobos, this trait must be implemented.
/// Note that all allocators must be `Clone`, `Send` and `Sync`. To do this, wrap internal state in
/// `Arc<Mutex<T>>` or similar where applicable.
pub trait Allocator: Clone + Send + Sync {
    /// Allocation type for this allocator. Must implement [`Allocation`]
    type Allocation: Allocation;

    /// Allocates raw memory of a specific memory type. The given name is used for internal tracking and
    /// debug logging. To get proper [`VkMemoryRequirements`](crate::vk::MemoryRequirements),
    /// call [`vkGetBufferMemoryRequirements`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetBufferMemoryRequirements.html) or
    /// [`vkGetImageMemoryRequirements`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetImageMemoryRequirements.html) with your buffer or image.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use anyhow::Result;
    /// # fn vk_get_memory_requirements(device: &Device, buffer: &Buffer) -> vk::MemoryRequirements { unimplemented!() }
    /// # fn vk_bind_buffer_memory(device: &Device, buffer: &Buffer, memory: vk::DeviceMemory, offset: u64) { unimplemented!() }
    /// fn use_allocator<A: Allocator>(device: Device, allocator: &mut A, buffer: Buffer) -> Result<()> {
    ///     let requirements = vk_get_memory_requirements(&device, &buffer);
    ///     let memory = allocator.allocate("buffer_memory", &requirements, MemoryType::GpuOnly)?;
    ///     // SAFETY: We are passing `memory.offset()` correctly.
    ///     vk_bind_buffer_memory(&device, &buffer, unsafe { memory.memory() }, memory.offset());
    ///     Ok(())
    /// }
    /// ```
    fn allocate(
        &mut self,
        name: &str,
        requirements: &vk::MemoryRequirements,
        ty: MemoryType,
    ) -> Result<Self::Allocation>;

    /// Free some memory allocated from this allocator. It's allowed for this function to do nothing, and instead
    /// use [`Drop`] to do this. Note that in this case, the allocation is still dropped. because it is moved into the function.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// # use anyhow::Result;
    /// # fn vk_get_memory_requirements(device: &Device, buffer: &Buffer) -> vk::MemoryRequirements { unimplemented!() }    ///
    /// // Admittedly, this is function kind of silly.
    /// fn use_allocator<A: Allocator>(device: Device, allocator: &mut A, buffer: Buffer) -> Result<()> {
    ///     let requirements = vk_get_memory_requirements(&device, &buffer);
    ///     let memory = allocator.allocate("buffer_memory", &requirements, MemoryType::GpuOnly)?;
    ///     allocator.free(memory)
    /// }
    /// ```
    fn free(&mut self, allocation: Self::Allocation) -> Result<()>;
}

/// Represents an allocation. This trait exposes methods for accessing the underlying device memory, obtain a mapped pointer, etc.
pub trait Allocation: Default {
    /// Get unsafe access to the underlying [`VkDeviceMemory`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDeviceMemory.html).
    /// Should always be used together with [`Allocation::offset()`].
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
    unsafe fn memory(&self) -> vk::DeviceMemory;

    /// Get the offset in this [`VkDeviceMemory`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDeviceMemory.html) this allocation refers to.
    /// This is exposed because the allocator implementation may choose to subdivide large memory blocks into smaller allocations.
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
    fn offset(&self) -> vk::DeviceSize;

    /// Obtain a mapped pointer to this allocation. This pointer can be used to directly write into the owned memory.
    /// This pointer already points into the exact memory region of the suballocation, so no offset must be applied.
    ///
    /// * Returns `None` if this memory was not mappable (not [`HOST_VISIBLE`](ash::vk::MemoryPropertyFlags::HOST_VISIBLE)). Memory allocated with [`MemoryType::CpuToGpu`] is always mappable.
    /// * If this memory comes from a [`HOST_VISIBLE`](ash::vk::MemoryPropertyFlags::HOST_VISIBLE) heap, this returns `Some(ptr)`, with `ptr` a non-null pointer pointing to the allocation memory.
    /// # Example
    /// ```
    /// # use phobos::*;
    /// // Writes the integer '5' into the first std::mem::size_of::<i32>() bytes of the allocation.
    /// // Assumes this allocation is mappable and at least std::mem::size_of::<i32>() bytes large.
    /// unsafe fn write_five<A: Allocation>(allocation: &A) {
    ///     let memory = allocation.mapped_ptr().expect("Expected allocation to be HOST_VISIBLE");
    ///     // SAFETY: Assume this allocation is at least std::mem::size_of::<i32>() bytes large.
    ///     *memory.cast::<i32>().as_mut() = 5;
    /// }
    /// ```
    fn mapped_ptr(&self) -> Option<NonNull<c_void>>;
}
