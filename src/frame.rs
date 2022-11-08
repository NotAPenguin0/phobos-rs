use std::sync::Arc;
use crate::{Device, Swapchain, Error, ExecutionManager, CommandBuffer, CmdBuffer, ImageView, WindowInterface, Surface, Image};
use crate::sync::*;
use ash::vk;
use crate::domain::ExecutionDomain;
use crate::deferred_delete::DeletionQueue;

/// Information stored for each in-flight frame.
#[derive(Derivative)]
#[derivative(Debug)]
struct PerFrame {
    pub fence: Arc<Fence>,
    /// Signaled by the GPU when a swapchain image is ready.
    pub image_ready: Semaphore,
    /// Signaled by the GPU when all commands for a frame have been processed.
    /// We wait on this before presenting.
    pub gpu_finished: Semaphore,
    /// Command buffer that was submitted this frame.
    /// Can be deleted once this frame's data is used again.
    #[derivative(Debug="ignore")]
    pub command_buffer: Option<Box<dyn CmdBuffer>>
}

/// Information stored for each swapchain image.
#[derive(Debug)]
struct PerImage {
    /// Fence of the current frame.
    pub fence: Option<Arc<Fence>>,
}


/// Struct that stores the context of a single in-flight frame.
/// You can obtain an instance of this from calling [`FrameManager::new_frame()`].
/// All operations specific to a frame require an instance.
/// <br>
/// <br>
/// # Example
/// ```
/// while running {
///     // obtain a Future<InFlightContext>, assumes windowed context was created.
///     let ifc = frame_manager.new_frame();
///     // possibly do some work that does not yet require a frame context.
///     // ...
///     // wait for our resulting frame context now that we really need it.
///     let ifc = futures::executor::block_on(ifc);
/// }
/// ```
#[derive(Debug)]
pub struct InFlightContext {

}

/// Responsible for presentation, frame-frame synchronization and per-frame resources.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct FrameManager {
    device: Arc<Device>,
    per_frame: [PerFrame; FrameManager::FRAMES_IN_FLIGHT],
    per_image: Vec<PerImage>,
    current_frame: u32,
    current_image: u32,
    swapchain: Swapchain,
    swapchain_delete: DeletionQueue<Swapchain>,
}


impl FrameManager {
    /// The number of frames in flight. A frame in-flight is a frame that is rendering on the GPU or scheduled to do so.
    /// With two frames in flight, we can prepare a frame on the CPU while one frame is rendering on the GPU.
    /// This gives a good amount of parallelization while avoiding input lag.
    pub(crate) const FRAMES_IN_FLIGHT: usize = 2;

    /// Initialize frame manager with per-frame data.
    pub fn new(device: Arc<Device>, swapchain: Swapchain) -> Result<Self, Error> {
        Ok(FrameManager {
            device: device.clone(),
            per_frame: (0..Self::FRAMES_IN_FLIGHT).into_iter().map(|_| -> Result<PerFrame, Error> {
               Ok(PerFrame {
                   fence: Arc::new(Fence::new(device.clone(), true)?),
                   image_ready: Semaphore::new(device.clone())?,
                   gpu_finished: Semaphore::new(device.clone())?,
                   command_buffer: None
               })
            }).collect::<Result<Vec<PerFrame>, Error>>()?
            .try_into()
            .map_err(|_| Error::Uncategorized("Conversion to slice failed"))?,
            per_image: swapchain.images.iter().map(|_| PerImage { fence: None } ).collect(),
            current_frame: 0,
            current_image: 0,
            swapchain,
            swapchain_delete: DeletionQueue::<Swapchain>::new((Self::FRAMES_IN_FLIGHT + 2) as u32),
        })
    }

    fn acquire_image(&self) -> Result<(u32 /*index*/, bool /*resize required*/), Error> {
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
            Err(err) => Err(Error::from(err))
        }
    }

    fn resize_swapchain<Window: WindowInterface>(&mut self, window: &Window, surface: &Surface) -> Result<Swapchain, Error> {
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
            .map(move |image| {
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
                    memory: None
                };
                // Create a trivial ImgView.
                let view = image.view(vk::ImageAspectFlags::COLOR);
                // Bundle them together into an owning ImageView
                ImageView::from((image, view))
            })
            .collect();

        Ok(new_swapchain)
    }

    /// This function must be called at the beginning of each frame.
    /// It will return an [`InFlightContext`] object which holds all the information for this current frame.
    /// You can only start doing command recording once the resulting future is awaited.
    /// Note that this function takes a window so it can properly resize the swapchain if a resize
    /// occurs.
    pub async fn new_frame<Window: WindowInterface>(&mut self, exec: &ExecutionManager, window: &Window, surface: &Surface) -> Result<InFlightContext, Error> {
        // Advance deletion queue by one frame
        self.swapchain_delete.next_frame();

        // Increment frame index. We do this here since this is the only mutable function in the frame loop.
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
            image_fence.wait()?;
        }

        // Grab the fence for this frame and assign it to the image.
        per_image.fence = Some(self.per_frame[self.current_frame as usize].fence.clone());
        let mut per_frame = &mut self.per_frame[self.current_frame as usize];
        // Delete the command buffer used the previous time this frame was allocated.
        if let Some(cmd) = &mut per_frame.command_buffer {
            unsafe { cmd.delete(exec)? }
        }
        per_frame.command_buffer = None;

        Ok(InFlightContext {})
    }

    /// Submit this frame's commands to be processed. Note that this is the only way a frame's commands
    /// should ever be submitted to a queue. Any other ways to submit should be synchronized properly to this
    /// submission. The reason for this is that [`FrameManager::present`] waits on a semaphore this function's submission
    /// will signal. Any commands submitted from somewhere else must be synchronized to this submission.
    /// Note: it's possible this will be enforced through the type system later.
    /// TODO: examine possibilities for this.
    pub fn submit<D: ExecutionDomain + 'static>(&mut self, cmd: CommandBuffer<D>, exec: &ExecutionManager) -> Result<(), Error> {
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
        unsafe { Ok(queue.submit(std::slice::from_ref(&submit), &per_frame.fence)?) }
    }

    /// Present a frame to the swapchain. This is the same as calling
    /// `glfwSwapBuffers()` in OpenGL code.
    pub fn present(&self, exec: &ExecutionManager) -> Result<(), Error> {
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
                    Err(e) => Err(Error::from(e))
                }
            }
        } else { Err(Error::NoPresentQueue) }
    }

    /// Get a reference to the current swapchain image.
    /// This reference is valid as long as the swapchain is not resized.
    pub unsafe fn get_swapchain_image(&self, _ifc: &InFlightContext) -> Result<&ImageView, Error> {
        Ok(&self.swapchain.images[self.current_image as usize])
    }
}