use std::pin::Pin;
use std::slice;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use ash::prelude::VkResult;
use ash::vk;

use crate::Device;

struct CleanupFnLink<'f> {
    pub f: Box<dyn FnOnce() -> () + 'f>,
    pub next: Option<Box<CleanupFnLink<'f>>>
}

trait FenceValue<T> {
    fn value(&mut self) -> T;
}

/// Wrapper around a [`VkFence`](vk::Fence) object. Fences are used for CPU-GPU sync.
/// The most powerful feature of fences is that they have [`Future<Output = T>`](std::future::Future)
/// implemented for them. This allows you to wait for GPU work using `.await` like any normal
/// Rust future.
/// # Example
/// ```
/// use phobos::prelude::*;
///
/// let exec = ExecutionManager::new(device, &physical_device)?;
/// // Obtain some command buffer
/// let cmd = exec.on_domain::<domain::All>(None, None)?.finish()?;
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
/// that the GPU might still be using those resources.
/// Consider the following case
/// ```
/// use std::mem::size_of;
/// use std::sync::Arc;
/// use anyhow::Result;
///
/// use phobos::prelude::*;
///
/// async fn upload_buffer<T: Copy>(device: Arc<Device>, mut allocator: DefaultAllocator, exec: ExecutionManager, src: &[T]) -> Result<Buffer> {
///     // Create our result buffer
///     let size = (src.len() * size_of::<T>()) as u64;
///     let buffer = Buffer::new_device_local(device.clone(), &allocator, size, vk::BufferUsageFlags::TRANSFER_DST)?;
///     let view = buffer.view_full();
///     // Create a staging buffer and copy our data to it
///     let staging = Buffer::new(device.clone(), &allocator, size, vk::BufferUsageFlags::TRANSFER_SRC, MemoryType::CpuToGpu)?;
///     let mut staging_view = staging.view_full();
///     staging_view.mapped_slice()?.copy_from_slice(src);
///     // Create a command buffer to copy the buffers
///     let cmd =
///         exec.on_domain::<domain::Transfer>(None, None)?
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
/// async fn upload_buffer<T: Copy>(device: Arc<Device>, mut allocator: DefaultAllocator, exec: ExecutionManager, src: &[T]) -> Result<Buffer> {
///     // ... snip
///     // Submit our command buffer and obtain a fence
///     let fence = exec.submit(cmd)?;
///     // Attach our resulting buffer and await the fence.
///     fence
///         // Add a cleanup function which will take ownership of any data that needs to be freed
///         // after the fence completes.
///         // The future will call these functions when the fence is ready.
///         .with_cleanup(move || {
///             drop(staging);
///         })
///         .attach_value(Ok(buffer)).await
/// }
/// ```
///
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Fence<T = ()> {
    device: Arc<Device>,
    #[derivative(Debug="ignore")]
    first_cleanup_fn: Option<Box<CleanupFnLink<'static>>>,
    value: Option<T>,
    pub handle: vk::Fence,
}

pub type GpuFuture<T> = Fence<T>;

impl<T> FenceValue<T> for Fence<T> {
    default fn value(&mut self) -> T {
        self.value.take().unwrap()
    }
}

impl FenceValue<()> for Fence<()> {
    fn value(&mut self) -> () {}
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
        }
    }
}

impl<T> Unpin for Fence<T> {}

impl<T> Fence<T> {
    /// Create a new fence, possibly in the singaled status.
    pub fn new(device: Arc<Device>, signaled: bool) -> Result<Self, vk::Result> {
        let info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: if signaled { vk::FenceCreateFlags::SIGNALED } else { vk::FenceCreateFlags::empty() },
        };
        Ok(Fence {
            handle: unsafe {
                device.create_fence(&info, None)?
            },
            device,
            first_cleanup_fn: None,
            value: None,
        })
    }

    /// Waits for the fence to be signaled with no timeout. Note that this is a blocking call. For the nonblocking version, use the `Future` implementation by calling
    /// `.await`.
    pub fn wait(&self) -> VkResult<()> {
        unsafe { self.device.wait_for_fences(slice::from_ref(&self.handle), true, u64::MAX) }
    }

    /// Resets a fence to the unsignaled status.
    pub fn reset(&self) -> VkResult<()> {
        unsafe { self.device.reset_fences(slice::from_ref(&self.handle)) }
    }

    /// Add a function to the front of the chain of functions to be called when this fence runs to completion ***AS A FUTURE***.
    /// TODO: Possibly also call this after Self::wait()
    pub fn with_cleanup(mut self, f: impl FnOnce() -> () + 'static) -> Self {
        if self.first_cleanup_fn.is_some() {
            let mut head = Box::new(CleanupFnLink {
                f: Box::new(f),
                next: None
            });
            let fun = self.first_cleanup_fn.take().unwrap();
            head.next = Some(fun);
            self.first_cleanup_fn = Some(head);
            self
        } else {
            self.first_cleanup_fn = Some(Box::new(CleanupFnLink {
                f: Box::new(f),
                next: None
            }));
            self
        }
    }
}

// Note that the future implementation for Fence works by periodically polling the fence.
// This could not be desirable depending on the timeout chosen by the implementation.
impl<T> std::future::Future for Fence<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        let status = unsafe { self.device.get_fence_status(self.handle).unwrap() };

        if status {
            // Call the whole chain of cleanup functions.
            let mut f = self.first_cleanup_fn.take();
            while let Some(_) = f {
                let func = f.take().unwrap();
                func.f.call_once(());
                f = func.next
            }
            Poll::Ready(self.as_mut().value())
        } else {
            let waker = ctx.waker().clone();
            std::thread::spawn(move || {
                // We will try to poll every 5 milliseconds.
                // TODO: measure, possibly configure
                std::thread::sleep(Duration::from_millis(5));
                waker.wake();
                return;
            });
            Poll::Pending
        }
    }
}

impl<T> Drop for Fence<T> {
    fn drop(&mut self) {
        unsafe { self.device.destroy_fence(self.handle, None); }
    }
}
