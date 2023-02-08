use std::future::Future;
use std::pin::Pin;
use std::slice;
use std::sync::{Arc};
use std::task::{Context, Poll, Waker};
use std::time::Duration;
use ash::prelude::VkResult;
use ash::vk;
use crate::Device;
use anyhow::Result;

/// Wrapper around a [`VkFence`](vk::Fence) object. Fences are used for CPU-GPU sync.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Fence<'f> {
    #[derivative(Debug="ignore")]
    device: Arc<Device>,
    waker: Option<Waker>,
    #[derivative(Debug="ignore")]
    cleanup_fn: Option<Box<dyn FnMut() -> () + 'f>>,
    pub handle: vk::Fence,
}

pub struct GpuFuture<'f, T> {
    value: Option<T>,
    fence: Fence<'f>,
    waker: Option<Waker>,
}

/// Wrapper around a [`VkSemaphore`](vk::Semaphore) object. Semaphores are used for GPU-GPU sync.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Semaphore {
    #[derivative(Debug="ignore")]
    pub device: Arc<Device>,
    pub handle: vk::Semaphore,
}

impl<'f> Fence<'f> {
    /// Create a new fence, possibly in the singaled status.
    pub fn new(device: Arc<Device>, signaled: bool) -> Result<Self, vk::Result> {
        Ok(Fence {
            device: device.clone(),
            waker: None,
            cleanup_fn: None,
            handle: unsafe {
                device.create_fence(&vk::FenceCreateInfo::builder()
                        .flags(if signaled { vk::FenceCreateFlags::SIGNALED } else { vk::FenceCreateFlags::empty() }),
    None)?
            }
        })
    }

    /// Waits for the fence to be signaled with no timeout.
    pub fn wait(&self) -> VkResult<()> {
        unsafe { self.device.wait_for_fences(slice::from_ref(&self.handle), true, u64::MAX) }
    }

    /// Resets a fence to the unsignaled status.
    pub fn reset(&self) -> VkResult<()> {
        unsafe { self.device.reset_fences(slice::from_ref(&self.handle)) }
    }

    // TODO: Chain cleanup calls?
    pub(crate) fn with_cleanup(mut self, f: impl FnMut() -> () + 'f) -> Self {
        self.cleanup_fn = Some(Box::new(f));
        self
    }

    pub fn attach_value<T>(self, value: T) -> GpuFuture<'f, T> {
        GpuFuture {
            value: Some(value),
            fence: self,
            waker: None,
        }
    }
}

// Note that the future implementation for Fence works by periodically polling the fence.
// This could not be desirable depending on the timeout chosen by the implementation.
impl Future for Fence<'_> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        self.waker = Some(ctx.waker().clone());
        let status = unsafe { self.device.get_fence_status(self.handle).unwrap() };

        if status {
            if let Some(f) = &mut self.cleanup_fn {
                f.call_mut(());
            }
            Poll::Ready(())
        } else {
            let waker = self.waker.clone().unwrap();
            std::thread::spawn(move || {
                    // We will try to poll every millisecond.
                    // TODO: measure, possibly configure
                    std::thread::sleep(Duration::from_millis(1));
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

impl Semaphore {
    pub fn new(device: Arc<Device>) -> Result<Self, vk::Result> {
        Ok(Semaphore {
            device: device.clone(),
            handle: unsafe {
                device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)?
            }
        })
    }
}

impl Drop for Fence<'_> {
    fn drop(&mut self) {
        unsafe { self.device.destroy_fence(self.handle, None); }
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { self.device.destroy_semaphore(self.handle, None); }
    }
}