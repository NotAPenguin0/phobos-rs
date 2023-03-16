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
