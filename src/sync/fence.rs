use std::future::Future;
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

/// Wrapper around a [`VkFence`](vk::Fence) object. Fences are used for CPU-GPU sync.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Fence<'f> {
    device: Arc<Device>,
    #[derivative(Debug="ignore")]
    first_cleanup_fn: Option<Box<CleanupFnLink<'f>>>,
    pub handle: vk::Fence,
}

pub struct GpuFuture<'f, T> {
    value: Option<T>,
    fence: Fence<'f>,
}


impl<'f> Fence<'f> {
    /// Create a new fence, possibly in the singaled status.
    pub fn new(device: Arc<Device>, signaled: bool) -> Result<Self, vk::Result> {
        let info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: if signaled { vk::FenceCreateFlags::SIGNALED } else { vk::FenceCreateFlags::empty() },
        };
        Ok(Fence {
            device: device.clone(),
            first_cleanup_fn: None,
            handle: unsafe {
                device.create_fence(&info, None)?
            }
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
    pub fn with_cleanup(mut self, f: impl FnOnce() -> () + 'f) -> Self {
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

    pub fn attach_value<T>(self, value: T) -> GpuFuture<'f, T> {
        GpuFuture {
            value: Some(value),
            fence: self,
        }
    }
}

// Note that the future implementation for Fence works by periodically polling the fence.
// This could not be desirable depending on the timeout chosen by the implementation.
impl Future for Fence<'_> {
    type Output = ();

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
            Poll::Ready(())
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

impl<T> Unpin for GpuFuture<'_, T> {}

impl<T> Future for GpuFuture<'_, T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        let fence = Pin::new(&mut self.fence);
        let status = fence.poll(ctx);
        match status {
            Poll::Ready(_) => {
                let value = self.value.take().unwrap();
                Poll::Ready(value)
            }
            Poll::Pending => { Poll::Pending }
        }
    }
}


impl Drop for Fence<'_> {
    fn drop(&mut self) {
        unsafe { self.device.destroy_fence(self.handle, None); }
    }
}