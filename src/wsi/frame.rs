//! Contains all the logic responsible for managing presentation and frame synchronization.
//!
//! Every frame should be contained in a call to [`FrameManager::new_frame`], which takes in a closure that is called when the frame is ready to be processed.
//! This also gives you an [`InFlightContext`] object which contains useful data relevant to the current frame's execution context,
//! such as per-frame allocators and a view to the current swapchain image.
//!
//! # Example usage
//!
//! Example code for a main loop using `winit` and `futures::block_on` as the future executor.
//! ```
//! use winit::event_loop::ControlFlow;
//! use winit::event::{Event, WindowEvent};
//! use phobos::prelude::*;
//!
//! let alloc = DefaultAllocator::new(&instance, &device, &physical_device)?;
//! let mut frame = {
//!         let swapchain = Swapchain::new(&instance, device.clone(), &settings, &surface)?;
//!         FrameManager::new(device.clone(), alloc.clone(), &settings, swapchain)?
//!  };
//!
//! event_loop.run(move |event, _, control_flow| {
//!         // Do not render a frame if Exit control flow is specified, to avoid
//!         // sync issues.
//!         if let ControlFlow::ExitWithCode(_) = *control_flow { return; }
//!         *control_flow = ControlFlow::Poll;
//!
//!
//!         // Advance caches to next frame to ensure resources are freed up where possible.
//!         pipeline_cache.next_frame();
//!         descriptor_cache.next_frame();
//!
//!         // Note that we want to handle events after processing our current frame, so that
//!         // requesting an exit doesn't attempt to render another frame, which causes
//!         // sync issues.
//!         match event {
//!             Event::WindowEvent {
//!                 event: WindowEvent::CloseRequested,
//!                 window_id
//!             } if window_id == window.id() => {
//!                 *control_flow = ControlFlow::Exit;
//!                 device.wait_idle().unwrap();
//!             },
//!             Event::MainEventsCleared => {
//!                 window.request_redraw();
//!             },
//!             Event::RedrawRequested(_) => {
//!                 // When a redraw is requested, we'll run our frame logic
//!                 futures::executor::block_on(frame.new_frame(exec.clone(), window, &surface, |mut ifc| {
//!                     // This closure is expected to return a command buffer with this frame's commands.
//!                     // This command buffer should at the very least transition the swapchain image to
//!                     // `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`.
//!                     // This can be done using the render graph API, or with a single command:
//!                     let cmd = exec.on_domain::<domain::Graphics>()?
//!                                   .transition_image(&ifc.swapchain_image,
//!                                         vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
//!                                         vk::ImageLayout::UNDEFINED, vk::ImageLayout::PRESENT_SRC_KHR,
//!                                         vk::AccessFlags::empty(), vk::AccessFlags::empty())
//!                                   .finish()?;
//!             Ok(cmd)
//!         }))?;
//!             }
//!             _ => (),
//!         }
//! });
//! ```

use std::sync::Arc;

use anyhow::Result;
use ash::vk;

use crate::{
    Allocator, CmdBuffer, DefaultAllocator, Device, Error, ExecutionManager, Fence,
    Image, ImageView, Instance, Semaphore, Surface, Swapchain, SurfaceSettings,
};
use crate::pool::{Poolable, Pooled, ResourcePool};
use crate::sync::domain::ExecutionDomain;
use crate::sync::submit_batch::SubmitBatch;
use crate::util::deferred_delete::DeletionQueue;
use crate::wsi::swapchain::SwapchainImage;

use super::window::Window;

/// Information stored for each in-flight frame.
#[derive(Derivative)]
#[derivative(Debug)]
struct PerFrame<A> {
    #[derivative(Debug = "ignore")]
    pub fence: Pooled<Fence<()>>,
    /// Signaled by the GPU when a swapchain image is ready.
    pub image_ready: Arc<Semaphore>,
    /// Signaled by the GPU when all commands for a frame have been processed.
    /// We wait on this before presenting.
    pub gpu_finished: Arc<Semaphore>,
    /// Command buffer that was submitted this frame.
    /// Can be deleted once this frame's data is used again.
    #[derivative(Debug = "ignore")]
    pub command_buffer: Option<Box<dyn CmdBuffer<A>>>,
}

/// Struct that stores the context of a single execution scope, for example a frame or an async task.
/// It is passed to the callback given to [`FrameManager::new_frame()`].
/// All operations specific to a frame require an instance.
/// <br>
/// <br>
/// # Example
/// ```
/// frame.new_frame(&exec, window, &surface, |mut ifc| {
///     // Use ifc here
/// });
///
/// ```
#[derive(Derivative)]
#[derivative(Debug)]
pub struct InFlightContext {
    /// The current frame's swapchain image
    pub swapchain_image: ImageView,
    pub(crate) wait_semaphore: Arc<Semaphore>,
    pub(crate) signal_semaphore: Arc<Semaphore>,
}

/// The number of frames in flight. A frame in-flight is a frame that is rendering on the GPU or scheduled to do so.
/// With two frames in flight, we can prepare a frame on the CPU while one frame is rendering on the GPU.
/// This gives a good amount of parallelization while avoiding input lag.
pub const FRAMES_IN_FLIGHT: usize = 2;

/// Responsible for presentation, frame-frame synchronization and per-frame resources.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct FrameManager<A: Allocator = DefaultAllocator> {
    device: Device,
    per_frame: [PerFrame<A>; FRAMES_IN_FLIGHT],
    current_frame: u32,
    current_image: u32,
    swapchain: Swapchain,
    swapchain_delete: DeletionQueue<Swapchain>,
    pool: ResourcePool<A>,
}

#[derive(Debug, Copy, Clone)]
struct AcquiredImage {
    pub index: u32,
    pub resize_required: bool,
}

impl<A: Allocator> FrameManager<A> {
    fn acquire_image(&mut self) -> Result<AcquiredImage> {
        let frame = &mut self.per_frame[self.current_frame as usize];
        // We do want to call cleanup functions now
        frame.fence.wait()?;
        let result = unsafe {
            self.swapchain.acquire_next_image(
                self.swapchain.handle(),
                u64::MAX,
                frame.image_ready.handle(),
                vk::Fence::null(),
            )
        };

        match result {
            // We ignore the flag for suboptimal swapchain images for now
            Ok((index, _)) => Ok(AcquiredImage {
                index,
                resize_required: false,
            }),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(AcquiredImage {
                index: 0,
                resize_required: true,
            }),
            Err(err) => Err(err.into()),
        }
    }

    fn resize_swapchain(
        &mut self,
        window: &dyn Window,
        surface: &Surface,
    ) -> Result<Swapchain> {
        let mut new_swapchain = Swapchain {
            handle: vk::SwapchainKHR::null(),
            images: vec![],
            format: self.swapchain.format(),
            present_mode: self.swapchain.present_mode(),
            extent: vk::Extent2D {
                width: window.width(),
                height: window.height(),
            },
            functions: self.swapchain.functions.clone(),
        };

        let image_count = self.swapchain.images.len();

        let info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: std::ptr::null(),
            flags: Default::default(),
            surface: unsafe { surface.handle() },
            min_image_count: image_count as u32,
            image_format: self.swapchain.format().format,
            image_color_space: self.swapchain.format().color_space,
            image_extent: *new_swapchain.extent(),
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
            pre_transform: surface.capabilities().current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: self.swapchain.present_mode(),
            clipped: vk::TRUE,
            old_swapchain: unsafe { self.swapchain.handle() },
        };

        new_swapchain.handle = unsafe { self.swapchain.create_swapchain(&info, None)? };

        // Now that the new swapchain is created, we still need to acquire the images again.
        new_swapchain.images =
            unsafe { self.swapchain.get_swapchain_images(new_swapchain.handle)? }
                .iter()
                .map(move |image| -> Result<SwapchainImage> {
                    let image = Image::new_managed(
                        self.device.clone(),
                        *image,
                        new_swapchain.format.format,
                        vk::Extent3D {
                            width: new_swapchain.extent.width,
                            height: new_swapchain.extent.height,
                            depth: 1,
                        },
                        1,
                        1,
                        vk::SampleCountFlags::TYPE_1,
                    );
                    // Create a trivial ImageView.
                    let view = image.whole_view(vk::ImageAspectFlags::COLOR)?;
                    Ok(SwapchainImage {
                        image,
                        view,
                    })
                })
                .collect::<Result<Vec<SwapchainImage>>>()?;

        Ok(new_swapchain)
    }

    /// Submit this frame's commands to be processed. Note that this is the only way a frame's commands
    /// should ever be submitted to a queue. Any other ways to submit commands for this frame should be synchronized properly to this
    /// submission. The reason for this is that [`FrameManager::present`] waits on a semaphore this function's submission
    /// will signal. Any commands for this frame submitted from somewhere else must be synchronized to this submission.
    fn submit<D: ExecutionDomain + 'static>(&mut self, batch: SubmitBatch<D>) -> Result<()> {
        // Finish the submit batch and submit it. We store the fence so we can wait on it later.
        let per_frame = &mut self.per_frame[self.current_frame as usize];
        per_frame.fence = batch.finish()?;
        Ok(())
    }

    /// Present a frame to the swapchain. This is the same as calling
    /// `glfwSwapBuffers()` in OpenGL code.
    fn present(&self, exec: ExecutionManager<A>) -> Result<()> {
        let per_frame = &self.per_frame[self.current_frame as usize];
        let functions = &self.swapchain.functions;
        let queue = exec.get_present_queue();
        if let Some(queue) = queue {
            let gpu_finished = unsafe { per_frame.gpu_finished.handle() };
            let info = vk::PresentInfoKHR {
                s_type: vk::StructureType::PRESENT_INFO_KHR,
                p_next: std::ptr::null(),
                wait_semaphore_count: 1,
                p_wait_semaphores: &gpu_finished,
                swapchain_count: 1,
                p_swapchains: &self.swapchain.handle,
                p_image_indices: &self.current_image,
                p_results: std::ptr::null_mut(),
            };
            let result = unsafe { functions.queue_present(queue.handle(), &info).map(|_| ()) };
            // We will ignore OUT_OF_DATE_KHR here
            match result {
                Ok(_) => Ok(()),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(()),
                Err(e) => Err(e.into()),
            }
        } else {
            Err(Error::NoPresentQueue.into())
        }
    }

    /// Get a reference to the current swapchain image.
    /// This reference is valid as long as the swapchain is not resized.
    #[allow(dead_code)]
    unsafe fn get_swapchain_image(&self) -> Result<ImageView> {
        Ok(self.swapchain.images[self.current_image as usize]
            .view
            .clone())
    }
}

impl<A: Allocator> FrameManager<A> {
    /// Initialize frame manager with per-frame data.
    pub fn new(device: Device, pool: ResourcePool<A>, swapchain: Swapchain) -> Result<Self> {
        Ok(FrameManager {
            device: device.clone(),
            per_frame: (0..FRAMES_IN_FLIGHT)
                .map(|_| -> Result<PerFrame<A>> {
                    Ok(PerFrame {
                        fence: Fence::new(device.clone(), true)?.into_pooled(&pool.fences, ()),
                        image_ready: Arc::new(Semaphore::new(device.clone())?),
                        gpu_finished: Arc::new(Semaphore::new(device.clone())?),
                        command_buffer: None,
                    })
                })
                .collect::<Result<Vec<PerFrame<A>>>>()?
                .try_into()
                .map_err(|_| Error::Uncategorized("Conversion to slice failed"))?,
            current_frame: 0,
            current_image: 0,
            swapchain,
            swapchain_delete: DeletionQueue::<Swapchain>::new((FRAMES_IN_FLIGHT + 2) as u32),
            pool,
        })
    }

    /// Initialize frame manager and create a swapchain.
    pub fn new_with_swapchain(
        instance: &Instance,
        device: Device,
        pool: ResourcePool<A>,
        surface_settings: &SurfaceSettings,
        surface: &Surface,
    ) -> Result<Self> {
        let swapchain = Swapchain::new(instance, device.clone(), surface_settings, surface)?;
        FrameManager::new(device, pool, swapchain)
    }

    /// Obtain a new frame context to run commands in.
    /// This will call the provided callback function to obtain a [`SubmitBatch`](crate::sync::submit_batch::SubmitBatch)
    /// which contains the commands to be submitted for this frame.
    pub async fn new_frame<D, F>(
        &mut self,
        exec: ExecutionManager<A>,
        window: &dyn Window,
        surface: &Surface,
        f: F,
    ) -> Result<()>
    where
        D: ExecutionDomain + 'static,
        F: FnOnce(InFlightContext) -> Result<SubmitBatch<D>>, {
        // Advance deletion queue by one frame
        self.swapchain_delete.next_frame();

        // Increment frame index.
        self.current_frame = (self.current_frame + 1) % self.per_frame.len() as u32;

        let AcquiredImage {
            index,
            resize_required,
        } = self.acquire_image()?;
        self.current_image = index;

        if resize_required {
            let mut new_swapchain = self.resize_swapchain(window, surface)?;
            std::mem::swap(&mut new_swapchain, &mut self.swapchain);
            self.swapchain_delete.push(new_swapchain); // now old swapchain after swapping.

            // Acquire image again. Note that this won't wait on the same fence again is it is never reset.
            let AcquiredImage {
                index,
                ..
            } = self.acquire_image()?;
            self.current_image = index;
        }

        let submission = {
            let per_frame = &mut self.per_frame[self.current_frame as usize];
            // Delete the command buffer used the previous time this frame was allocated.
            if let Some(cmd) = &mut per_frame.command_buffer {
                unsafe { cmd.delete(exec.clone())? }
            }
            per_frame.command_buffer = None;
            let image = self.swapchain.images()[self.current_image as usize]
                .view
                .clone();

            let ifc = InFlightContext {
                swapchain_image: image,
                wait_semaphore: per_frame.image_ready.clone(),
                signal_semaphore: per_frame.gpu_finished.clone(),
            };
            f(ifc)?
        };
        self.submit(submission)?;
        self.present(exec)
    }

    /// Unsafe access to the underlying swapchain.
    /// # Safety
    /// * Any vulkan calls on the `VkSwapchainKHR` handle may put the system in an undefined state.
    pub unsafe fn get_swapchain(&self) -> &Swapchain {
        &self.swapchain
    }
}
