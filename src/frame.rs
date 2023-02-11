//! This module contains all the logic responsible for managing presentation and frame synchronization. Every frame should be contained
//! in a call to [`FrameManager::new_frame`], which takes in a closure that is called when the frame is ready to be processed.
//! This also gives you an [`InFlightContext`] object which contains useful data relevant to the current frame's execution context,
//! such as per-frame allocators and a view to the current swapchain image.
//!
//! # Example usage
//!
//! Example code for a main loop using `winit` and `futures::block_on` as the future executor.
//! ```
//! use phobos as ph;
//! use ash::vk;
//!
//! let mut frame = {
//!         let swapchain = ph::Swapchain::new(&instance, device.clone(), &settings, &surface)?;
//!         ph::FrameManager::new(device.clone(), alloc.clone(), &settings, swapchain)?
//!  };
//!
//! event_loop.run(move |event, _, control_flow| {
//!         // Do not render a frame if Exit control flow is specified, to avoid
//!         // sync issues.
//!         if let ControlFlow::ExitWithCode(_) = *control_flow { return; }
//!         *control_flow = ControlFlow::Poll;
//!
//!         futures::executor::block_on(frame.new_frame(exec.clone(), window, &surface, |mut ifc| {
//!             // This closure is expected to return a command buffer with this frame's commands.
//!             // This command buffer should at the very least transition the swapchain image to
//!             // `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`.
//!             // This can be done using the render graph API, or with a single command:
//!             let cmd = exec.on_domain::<ph::domain::Graphics>()?
//!                           .transition_image(&ifc.swapchain_image,
//!                                 vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
//!                                 vk::ImageLayout::UNDEFINED, vk::ImageLayout::PRESENT_SRC_KHR,
//!                                 vk::AccessFlags::empty(), vk::AccessFlags::empty())
//!                           .finish();
//!             Ok(cmd)
//!         }))?;
//!
//!         // Advance caches to next frame to ensure resources are freed up where possible.
//!         pipeline_cache.lock().unwrap().next_frame();
//!         descriptor_cache.lock().unwrap().next_frame();
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
//!             _ => (),
//!         }
//! });
//! ```

use std::future::Future;
use std::sync::{Arc, Mutex};
use crate::{Device, Swapchain, Error, ExecutionManager, CommandBuffer, CmdBuffer, ImageView, WindowInterface, Surface, Image, SwapchainImage, ScratchAllocator, AppSettings, BufferView};
use crate::sync::*;
use ash::vk;
use gpu_allocator::vulkan::Allocator;
use crate::domain::ExecutionDomain;
use crate::deferred_delete::DeletionQueue;
use anyhow::Result;
/// Information stored for each in-flight frame.
#[derive(Derivative)]
#[derivative(Debug)]
struct PerFrame<'f> {
    pub fence: Arc<Fence<'f>>,
    /// Signaled by the GPU when a swapchain image is ready.
    pub image_ready: Semaphore,
    /// Signaled by the GPU when all commands for a frame have been processed.
    /// We wait on this before presenting.
    pub gpu_finished: Semaphore,
    /// Command buffer that was submitted this frame.
    /// Can be deleted once this frame's data is used again.
    #[derivative(Debug="ignore")]
    pub command_buffer: Option<Box<dyn CmdBuffer>>,
    // Scratch allocators
    pub vertex_allocator: ScratchAllocator,
    pub index_allocator: ScratchAllocator,
    pub uniform_allocator: ScratchAllocator,
    pub storage_allocator: ScratchAllocator
}

/// Information stored for each swapchain image.
#[derive(Debug)]
struct PerImage<'f> {
    /// Fence of the current frame.
    pub fence: Option<Arc<Fence<'f>>>,
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
///
/// Another way to acquire an instance of this struct is through a [`ThreadContext`].
#[derive(Debug)]
pub struct InFlightContext<'f> {
    pub swapchain_image: Option<ImageView>,
    pub swapchain_image_index: Option<usize>,
    pub(crate) vertex_allocator: &'f mut ScratchAllocator,
    pub(crate) index_allocator: &'f mut ScratchAllocator,
    pub(crate) uniform_allocator: &'f mut ScratchAllocator,
    pub(crate) storage_allocator: &'f mut ScratchAllocator
}

/// Responsible for presentation, frame-frame synchronization and per-frame resources.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct FrameManager<'f> {
    device: Arc<Device>,
    per_frame: [PerFrame<'f>; FrameManager::FRAMES_IN_FLIGHT],
    per_image: Vec<PerImage<'f>>,
    current_frame: u32,
    current_image: u32,
    swapchain: Swapchain,
    swapchain_delete: DeletionQueue<Swapchain>,
}

impl FrameManager<'_> {
    /// The number of frames in flight. A frame in-flight is a frame that is rendering on the GPU or scheduled to do so.
    /// With two frames in flight, we can prepare a frame on the CPU while one frame is rendering on the GPU.
    /// This gives a good amount of parallelization while avoiding input lag.
    pub(crate) const FRAMES_IN_FLIGHT: usize = 2;

    /// Initialize frame manager with per-frame data.
    pub fn new<Window>(device: Arc<Device>, allocator: Arc<Mutex<Allocator>>, settings: &AppSettings<Window>, swapchain: Swapchain) -> Result<Self> where Window: WindowInterface {
        let scratch_flags_base = vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
        Ok(FrameManager {
            device: device.clone(),
            per_frame: (0..Self::FRAMES_IN_FLIGHT).into_iter().map(|_| -> Result<PerFrame> {
               Ok(PerFrame {
                   fence: Arc::new(Fence::new(device.clone(), true)?),
                   image_ready: Semaphore::new(device.clone())?,
                   gpu_finished: Semaphore::new(device.clone())?,
                   command_buffer: None,
                   vertex_allocator: ScratchAllocator::new(device.clone(), allocator.clone(), settings.scratch_vbo_size, scratch_flags_base | vk::BufferUsageFlags::VERTEX_BUFFER)?,
                   index_allocator: ScratchAllocator::new(device.clone(), allocator.clone(), settings.scratch_ibo_size, scratch_flags_base | vk::BufferUsageFlags::INDEX_BUFFER)?,
                   uniform_allocator: ScratchAllocator::new(device.clone(), allocator.clone(), settings.scratch_ubo_size, scratch_flags_base | vk::BufferUsageFlags::UNIFORM_BUFFER)?,
                   storage_allocator: ScratchAllocator::new(device.clone(), allocator.clone(), settings.scratch_ssbo_size, scratch_flags_base | vk::BufferUsageFlags::STORAGE_BUFFER)?,
               })
            }).collect::<Result<Vec<PerFrame>>>()?
            .try_into()
            .map_err(|_| Error::Uncategorized("Conversion to slice failed"))?,
            per_image: swapchain.images.iter().map(|_| PerImage { fence: None } ).collect(),
            current_frame: 0,
            current_image: 0,
            swapchain,
            swapchain_delete: DeletionQueue::<Swapchain>::new((Self::FRAMES_IN_FLIGHT + 2) as u32),
        })
    }

    fn acquire_image(&self) -> Result<(u32 /*index*/, bool /*resize required*/)> {
        let functions = &self.swapchain.functions;
        let frame = &self.per_frame[self.current_frame as usize];
        frame.fence.wait()?;
        let result = unsafe {
            functions.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                frame.image_ready.handle,
                vk::Fence::null()
            )
        };

        match result {
            Ok((index, true)) => Ok((index, false)),
            Ok((index, false)) => Ok((index, false)),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok((0, true)),
            Err(err) => Err(anyhow::Error::from(Error::from(err)))
        }
    }

    fn resize_swapchain<Window: WindowInterface>(&mut self, window: &Window, surface: &Surface) -> Result<Swapchain> {
        let mut new_swapchain = Swapchain {
            handle: vk::SwapchainKHR::null(),
            images: vec![],
            format: self.swapchain.format,
            present_mode: self.swapchain.present_mode,
            extent: vk::Extent2D { width: window.width(), height: window.height() },
            functions: self.swapchain.functions.clone()
        };

        let image_count = self.swapchain.images.len();

        new_swapchain.handle = unsafe { self.swapchain.functions.create_swapchain(
            &vk::SwapchainCreateInfoKHR::builder()
                .old_swapchain(self.swapchain.handle)
                .surface(surface.handle)
                .image_format(self.swapchain.format.format)
                .image_color_space(self.swapchain.format.color_space)
                .image_extent(new_swapchain.extent)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .min_image_count(image_count as u32)
                .clipped(true)
                .pre_transform(surface.capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE),
            None)? };

        // Now that the new swapchain is created, we still need to acquire the images again.
        new_swapchain.images = unsafe { self.swapchain.functions.get_swapchain_images(new_swapchain.handle)? }
            .iter()
            .map(move |image| -> Result<SwapchainImage> {
                let image = Image {
                    device: self.device.clone(),
                    handle: *image,
                    format: new_swapchain.format.format,
                    size: vk::Extent3D {
                        width: new_swapchain.extent.width,
                        height: new_swapchain.extent.height,
                        depth: 1
                    },
                    layers: 1,
                    mip_levels: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    // Leave memory at None since this is managed by the swapchain, not our application.
                    memory: None,
                    allocator: None,
                };
                // Create a trivial ImgView.
                let view = image.view(vk::ImageAspectFlags::COLOR)?;
                // Bundle them together into an owning ImageView
                Ok(SwapchainImage {image, view})
            })
            .collect::<Result<Vec<SwapchainImage>>>()?;

        Ok(new_swapchain)
    }

    pub async fn new_frame<Window, D, F>(&mut self, exec: Arc<ExecutionManager>, window: &Window, surface: &Surface, f: F)
                                        -> Result<()>
        where
            Window: WindowInterface,
            D: ExecutionDomain + 'static,
            F: FnOnce(InFlightContext) -> Result<CommandBuffer<D>> {

        // Advance deletion queue by one frame
        self.swapchain_delete.next_frame();

        // Advance per-frame allocator to the next frame
        self.per_frame[self.current_frame as usize].vertex_allocator.reset();
        self.per_frame[self.current_frame as usize].index_allocator.reset();
        self.per_frame[self.current_frame as usize].uniform_allocator.reset();
        self.per_frame[self.current_frame as usize].storage_allocator.reset();

        // Increment frame index.
        self.current_frame = (self.current_frame + 1) % self.per_frame.len() as u32;
        let (index, resize_required) = self.acquire_image()?;
        self.current_image = index;

        if resize_required {
            let mut new_swapchain = self.resize_swapchain(window, surface)?;
            std::mem::swap(&mut new_swapchain, &mut self.swapchain);
            self.swapchain_delete.push(new_swapchain); // now old swapchain after swapping.

            // Acquire image again. Note that this won't wait on the same fence again is it is never reset.
            let (index, _) = self.acquire_image()?;
            self.current_image = index;
        }

        // Wait until this image is absolutely not in use anymore.
        let per_image = &mut self.per_image[self.current_image as usize];
        if let Some(image_fence) = per_image.fence.as_ref() {
            image_fence.wait()? // TODO: Try to use await() here, but for this the fence cannot be inside an Arc
        }

        // Grab the fence for this frame and assign it to the image.
        per_image.fence = Some(self.per_frame[self.current_frame as usize].fence.clone());
        let frame_commands = {
            let mut per_frame = &mut self.per_frame[self.current_frame as usize];
            // Delete the command buffer used the previous time this frame was allocated.
            if let Some(cmd) = &mut per_frame.command_buffer {
                unsafe { cmd.delete(exec.clone())? }
            }
            per_frame.command_buffer = None;
            let image = self.swapchain.images[self.current_image as usize].view.clone();

            let ifc = InFlightContext {
                swapchain_image: Some(image),
                swapchain_image_index: Some(self.current_image as usize),
                vertex_allocator: &mut per_frame.vertex_allocator,
                index_allocator: &mut per_frame.index_allocator,
                uniform_allocator: &mut per_frame.uniform_allocator,
                storage_allocator: &mut per_frame.storage_allocator,
            };
            f(ifc)?
        };
        self.submit(frame_commands, exec.clone())?;
        self.present(exec)
    }

    /// Obtain a new frame context to run commands in.
    pub async fn new_async_frame<Window, D, F, Fut>(&mut self, exec: Arc<ExecutionManager>, window: &Window, surface: &Surface, f: F)
        -> Result<()>
        where
            Window: WindowInterface,
            D: ExecutionDomain + 'static,
            F: FnOnce(InFlightContext) -> Fut,
            Fut: Future<Output = Result<CommandBuffer<D>>> {
        
        // Advance deletion queue by one frame
        self.swapchain_delete.next_frame();

        // Advance per-frame allocator to the next frame
        self.per_frame[self.current_frame as usize].vertex_allocator.reset();
        self.per_frame[self.current_frame as usize].index_allocator.reset();
        self.per_frame[self.current_frame as usize].uniform_allocator.reset();
        self.per_frame[self.current_frame as usize].storage_allocator.reset();
        
        // Increment frame index.
        self.current_frame = (self.current_frame + 1) % self.per_frame.len() as u32;
        let (index, resize_required) = self.acquire_image()?;
        self.current_image = index;

        if resize_required {
            let mut new_swapchain = self.resize_swapchain(window, surface)?;
            std::mem::swap(&mut new_swapchain, &mut self.swapchain);
            self.swapchain_delete.push(new_swapchain); // now old swapchain after swapping.

            // Acquire image again. Note that this won't wait on the same fence again is it is never reset.
            let (index, _) = self.acquire_image()?;
            self.current_image = index;
        }

        // Wait until this image is absolutely not in use anymore.
        let per_image = &mut self.per_image[self.current_image as usize];
        if let Some(image_fence) = per_image.fence.as_ref() {
            image_fence.wait()? // TODO: Try to use await() here, but for this the fence cannot be inside an Arc
        }

        // Grab the fence for this frame and assign it to the image.
        per_image.fence = Some(self.per_frame[self.current_frame as usize].fence.clone());
        let frame_commands = {
            let mut per_frame = &mut self.per_frame[self.current_frame as usize];
            // Delete the command buffer used the previous time this frame was allocated.
            if let Some(cmd) = &mut per_frame.command_buffer {
                unsafe { cmd.delete(exec.clone())? }
            }
            per_frame.command_buffer = None;
            let image = self.swapchain.images[self.current_image as usize].view.clone();

            let ifc = InFlightContext {
                swapchain_image: Some(image),
                swapchain_image_index: Some(self.current_image as usize),
                vertex_allocator: &mut per_frame.vertex_allocator,
                index_allocator: &mut per_frame.index_allocator,
                uniform_allocator: &mut per_frame.uniform_allocator,
                storage_allocator: &mut per_frame.storage_allocator,
            };
            f(ifc).await?
        };
        self.submit(frame_commands, exec.clone())?;
        self.present(exec)
    }

    /// Submit this frame's commands to be processed. Note that this is the only way a frame's commands
    /// should ever be submitted to a queue. Any other ways to submit should be synchronized properly to this
    /// submission. The reason for this is that [`FrameManager::present`] waits on a semaphore this function's submission
    /// will signal. Any commands submitted from somewhere else must be synchronized to this submission.
    /// Note: it's possible this will be enforced through the type system later.
    /// TODO: examine possibilities for this.
    fn submit<D: ExecutionDomain + 'static>(&mut self, cmd: CommandBuffer<D>, exec: Arc<ExecutionManager>) -> Result<()> {
        // Reset frame fence
        let mut per_frame = &mut self.per_frame[self.current_frame as usize];
        per_frame.fence.reset()?;

        let semaphores: Vec<vk::Semaphore> = vec![&per_frame.image_ready]
            .iter()
            .map(|sem| sem.handle)
            .collect();
        let stages = vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        // Grab a copy of the command buffer handle for submission.
        // We do this because we're going to move the actual command buffer into our
        // PerFrame structure to keep it around until we can safely delete it next frame.
        let cmd_handle = cmd.handle.clone();
        per_frame.command_buffer = Some(Box::new(cmd));

        let submit = vk::SubmitInfo::builder()
            .signal_semaphores(std::slice::from_ref(&per_frame.gpu_finished.handle))
            .command_buffers(std::slice::from_ref(&cmd_handle))
            .wait_semaphores(semaphores.as_slice())
            .wait_dst_stage_mask(stages.as_slice())
            .build();

        // Use the command buffer's domain to determine the correct queue to use.
        let queue = exec.get_queue::<D>().ok_or(Error::NoCapableQueue)?;
        unsafe { Ok(queue.submit(std::slice::from_ref(&submit), Some(&per_frame.fence))?) }
    }

    /// Present a frame to the swapchain. This is the same as calling
    /// `glfwSwapBuffers()` in OpenGL code.
    fn present(&self, exec: Arc<ExecutionManager>) -> Result<()> {
        let per_frame = &self.per_frame[self.current_frame as usize];
        let functions = &self.swapchain.functions;
        let queue = exec.get_present_queue();
        if let Some(queue) = queue {
            unsafe {
                let result = functions.queue_present(queue.handle(),
                                        &vk::PresentInfoKHR::builder()
                                            .swapchains(std::slice::from_ref(&self.swapchain.handle))
                                            .wait_semaphores(std::slice::from_ref(&per_frame.gpu_finished.handle))
                                            .image_indices(std::slice::from_ref(&self.current_image)))
                    .map(|_| ());
                // We will ignore OUT_OF_DATE_KHR here
                match result {
                    Ok(_) => Ok(()),
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(()),
                    Err(e) => Err(anyhow::Error::from(Error::from(e)))
                }
            }
        } else { Err(anyhow::Error::from(Error::NoPresentQueue)) }
    }

    /// Get a reference to the current swapchain image.
    /// This reference is valid as long as the swapchain is not resized.
    unsafe fn get_swapchain_image(&self) -> Result<ImageView> {
        Ok(self.swapchain.images[self.current_image as usize].view.clone())
    }

    pub unsafe fn get_swapchain(&self) -> &Swapchain {
        &self.swapchain
    }
}

impl<'f> InFlightContext<'f> {
    pub fn allocate_scratch_vbo(&mut self, size: vk::DeviceSize) -> Result<BufferView> {
        self.vertex_allocator.allocate(size)
    }

    pub fn allocate_scratch_ibo(&mut self, size: vk::DeviceSize) -> Result<BufferView> {
        self.index_allocator.allocate(size)
    }

    pub fn allocate_scratch_ubo(&mut self, size: vk::DeviceSize) -> Result<BufferView> {
        self.uniform_allocator.allocate(size)
    }

    pub fn allocate_scratch_ssbo(&mut self, size: vk::DeviceSize) -> Result<BufferView> {
        self.storage_allocator.allocate(size)
    }
}