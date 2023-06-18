//! Abstraction for `VkFence` objects.

use std::pin::Pin;
use std::slice;
use std::task::{Context, Poll};
use std::time::Duration;

use anyhow::Result;
use ash::prelude::VkResult;
use ash::vk;

use crate::Device;
use crate::pool::Poolable;

struct CleanupFnLink<'f> {
    pub f: Box<dyn FnOnce() + 'f>,
    pub next: Option<Box<CleanupFnLink<'f>>>,
}

/// Trait that allows accessing the value of a fence.
pub trait FenceValue<T> {
    /// Get the value of this fence. Note that using this without an attached value will panic.
    /// Using this before the fence was awaited may result in undefined behaviour.
    fn value(&mut self) -> Option<T>;
}

/// Wrapper around a [`VkFence`](vk::Fence) object. Fences are used for CPU-GPU sync.
/// The most powerful feature of fences is that they have [`Future<Output = T>`](std::future::Future)
/// implemented for them. This allows you to wait for GPU work using `.await` like any normal
/// Rust future.
/// # Example
/// ```
/// use phobos::prelude::*;
///
/// let exec = ExecutionManager::new(device, &physical_device, pool)?;
/// // Obtain some command buffer
/// let cmd = exec.on_domain::<domain::All>()?.finish()?;
/// let fence = exec.submit(cmd)?;
/// // We can now await this fence, or attach a resulting value to it to make the future
/// // a little more useful
/// async {
///     fence.attach_value(5) // This would usually be some kind of GPU resource, like an image that was just written to
///          .await?;
/// }
/// ```
/// # Caveats
/// Since returning a fence and awaiting it later would make objects
/// local to the function go out of scope and drop them, this is a problem when you consider the fact
/// that the GPU might still be using those resources. Unfortunately, the compiler cannot catch this.
/// Consider the following case
/// ```
/// use std::mem::size_of;
/// use std::sync::Arc;
/// use anyhow::Result;
///
/// use phobos::prelude::*;
///
/// async fn upload_buffer<T: Copy>(device: Device, mut allocator: DefaultAllocator, exec: ExecutionManager, src: &[T]) -> Result<Buffer> {
///     // Create our result buffer
///     let size = (src.len() * size_of::<T>()) as u64;
///     let buffer = Buffer::new_device_local(device.clone(), &mut allocator, size, vk::BufferUsageFlags::TRANSFER_DST)?;
///     let view = buffer.view_full();
///     // Create a staging buffer and copy our data to it
///     let staging = Buffer::new(device.clone(), &mut allocator, size, vk::BufferUsageFlags::TRANSFER_SRC, MemoryType::CpuToGpu)?;
///     let mut staging_view = staging.view_full();
///     staging_view.mapped_slice()?.copy_from_slice(src);
///     // Create a command buffer to copy the buffers
///     let cmd =
///         exec.on_domain::<domain::Transfer>()?
///             .copy_buffer(&staging_view, &view)?
///             .finish()?;
///     // Submit our command buffer and obtain a fence
///     let fence = exec.submit(cmd)?;
///     // Attach our resulting buffer and await the fence.
///     fence.attach_value(Ok(buffer)).await
/// }
/// ```
/// This has a major problem in that the staging buffer is dropped when the future is returned,
/// but the fence is still not done so the gpu is still accessing it. To fix this, we can use
/// [`Fence::with_cleanup`] as follows:
/// ```
/// use std::mem::size_of;
/// use std::sync::Arc;
/// use anyhow::Result;
///
/// use phobos::prelude::*;
///
///
/// async fn upload_buffer<T: Copy>(device: Device, mut allocator: DefaultAllocator, exec: ExecutionManager, src: &[T]) -> Result<Buffer> {
///     // ... snip
///     // Submit our command buffer and obtain a fence
///     let mut fence = exec.submit(cmd)?;
///     // Attach our resulting buffer and await the fence.
///     // To do this we have to use fence.replace() to replace the value inside the pooled object.
///     fence
///         .replace(|fence| {
///             // Add a cleanup function which will take ownership of any data that needs to be freed
///             // after the fence completes.
///             // The future will call these functions when the fence is ready.
///             fence.with_cleanup(move || {
///                 drop(staging);
///             })
///         }).await?;
///     Ok(buffer)
/// }
/// ```
///
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Fence<T = ()> {
    device: Device,
    #[derivative(Debug = "ignore")]
    first_cleanup_fn: Option<Box<CleanupFnLink<'static>>>,
    value: Option<T>,
    poll_rate: Duration,
    handle: vk::Fence,
}

// SAFETY: Fences refer to a VkFence object on the gpu, which is not dropped when it goes out of scope and can
// safely be sent between threads.
unsafe impl<T> Send for Fence<T> {}

impl<T> Unpin for Fence<T> {}

/// Type alias that more expressively conveys the intent that this Fence is a Future.
pub type GpuFuture<T> = Fence<T>;

impl<T> FenceValue<T> for Fence<T> {
    /// Get the value of this fence. Note that using this without an attached value will panic.
    /// Using this before the fence was awaited may result in undefined behaviour.
    fn value(&mut self) -> Option<T> {
        self.value.take()
    }
}

impl Fence<()> {
    /// Attach a value to the fence that is returned from the future
    /// when it completes.
    pub fn attach_value<T>(mut self, value: T) -> Fence<T> {
        let mut handle = vk::Fence::null();
        std::mem::swap(&mut self.handle, &mut handle);
        Fence::<T> {
            handle,
            first_cleanup_fn: self.first_cleanup_fn.take(),
            device: self.device.clone(),
            value: Some(value),
            poll_rate: self.poll_rate,
        }
    }
}

impl<T> Fence<T> {
    /// Create a new fence, possibly in the signaled status.
    pub fn new(device: Device, signaled: bool) -> Result<Self, vk::Result> {
        Self::new_with_poll_rate(device, signaled, Duration::from_millis(5))
    }

    /// Create a new fence with the specified poll rate for awaiting it as a future.
    pub fn new_with_poll_rate(
        device: Device,
        signaled: bool,
        poll_rate: Duration,
    ) -> Result<Self, vk::Result> {
        let info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: if signaled {
                vk::FenceCreateFlags::SIGNALED
            } else {
                vk::FenceCreateFlags::empty()
            },
        };

        let handle = unsafe { device.create_fence(&info, None)? };

        #[cfg(feature = "log-objects")]
        trace!("Created new VkFence {handle:p}");

        Ok(Fence {
            handle,
            device,
            poll_rate,
            first_cleanup_fn: None,
            value: None,
        })
    }

    fn call_cleanup_chain(&mut self) {
        let mut f = self.first_cleanup_fn.take();
        while f.is_some() {
            let func = f.take().unwrap();
            (func.f)();
            f = func.next
        }
    }

    fn poll_status(&self) -> VkResult<bool> {
        unsafe { self.device.get_fence_status(self.handle) }
    }

    pub(crate) unsafe fn wait_without_cleanup(&self) -> VkResult<()> {
        self.device
            .wait_for_fences(slice::from_ref(&self.handle), true, u64::MAX)
    }

    /// Waits for the fence by polling repeatedly and yielding execution to the OS. This is useful if you don't care about quickly knowing the fence is
    /// available and just want to wait with minimal overhead.
    /// <br>
    /// <br>
    /// ## rayon
    /// If the rayon feature is enabled, this will first yield to rayon and then yield to the OS if there is no rayon work.
    pub fn wait_and_yield(&mut self) -> Result<Option<T>> {
        loop {
            if self.poll_status()? {
                break;
            }

            #[cfg(feature = "rayon")]
            {
                match rayon::yield_now() {
                    // If rayon found no work, yield to the OS scheduler.
                    Some(rayon::Yield::Idle) => {
                        std::thread::yield_now();
                    }
                    _ => {}
                }
            }

            #[cfg(not(feature = "rayon"))]
            {
                std::thread::yield_now();
            }
        }
        self.call_cleanup_chain();
        Ok(self.value())
    }

    /// Waits for the fence to be signaled with no timeout. Note that this is a blocking call. For the nonblocking version, use the `Future` implementation by calling
    /// `.await`.
    pub fn wait(&mut self) -> Result<Option<T>> {
        let result = unsafe { self.wait_without_cleanup() };
        self.call_cleanup_chain();
        // Return previous result
        Ok(result.map(|_| self.value())?)
    }

    /// Resets a fence to the unsignaled status.
    pub fn reset(&self) -> VkResult<()> {
        unsafe { self.device.reset_fences(slice::from_ref(&self.handle)) }
    }

    /// Add a function to the front of the chain of functions to be called when this fence is completed, so either after
    /// wait() or after .await
    pub fn with_cleanup(mut self, f: impl FnOnce() + 'static) -> Self {
        if self.first_cleanup_fn.is_some() {
            let mut head = Box::new(CleanupFnLink {
                f: Box::new(f),
                next: None,
            });
            let fun = self.first_cleanup_fn.take().unwrap();
            head.next = Some(fun);
            self.first_cleanup_fn = Some(head);
            self
        } else {
            self.first_cleanup_fn = Some(Box::new(CleanupFnLink {
                f: Box::new(f),
                next: None,
            }));
            self
        }
    }

    /// Get unsafe access to the `VkFence` handle.
    /// # Safety
    /// Any vulkan calls that mutate the fence's state may put the system in an undefined state.
    pub unsafe fn handle(&self) -> vk::Fence {
        self.handle
    }
}

// Note that the future implementation for Fence works by periodically polling the fence.
// This could not be desirable depending on the timeout chosen by the implementation.
impl<T> std::future::Future for Fence<T> {
    type Output = Option<T>;

    fn poll(mut self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        let status = unsafe { self.device.get_fence_status(self.handle).unwrap() };

        if status {
            self.call_cleanup_chain();
            Poll::Ready(self.as_mut().value())
        } else {
            let waker = ctx.waker().clone();
            let poll_rate = self.poll_rate;
            std::thread::spawn(move || {
                std::thread::sleep(poll_rate);
                waker.wake();
            });
            Poll::Pending
        }
    }
}

impl<T> Drop for Fence<T> {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkFence {:p}", self.handle);
        unsafe {
            self.device.destroy_fence(self.handle, None);
        }
    }
}

impl<T> Poolable for Fence<T> {
    /// All fences are created equal
    type Key = ();

    fn on_release(&mut self) {
        self.reset().unwrap();
        self.value = None;
        self.first_cleanup_fn = None;
    }
}
