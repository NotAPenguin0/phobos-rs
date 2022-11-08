use std::sync::Arc;
use crate::{Device, Swapchain, Error, ExecutionManager, CommandBuffer};
use crate::sync::*;
use ash::vk;
use crate::domain::ExecutionDomain;
use crate::Error::NoCapableQueue;

/// Information stored for each in-flight frame.
#[derive(Debug)]
struct PerFrame {
    pub fence: Arc<Fence>,
    /// Signaled by the GPU when a swapchain image is ready.
    pub image_ready: Semaphore,
    /// Signaled by the GPU when all commands for a frame have been processed.
    /// We wait on this before presenting.
    pub gpu_finished: Semaphore,
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
#[derive(Debug)]
pub struct FrameManager {
    per_frame: [PerFrame; FrameManager::FRAMES_IN_FLIGHT],
    per_image: Vec<PerImage>,
    current_frame: u32,
    current_image: u32,
    swapchain: Swapchain,
}


impl FrameManager {
    /// The number of frames in flight. A frame in-flight is a frame that is rendering on the GPU or scheduled to do so.
    /// With two frames in flight, we can prepare a frame on the CPU while one frame is rendering on the GPU.
    /// This gives a good amount of parallelization while avoiding input lag.
    pub(crate) const FRAMES_IN_FLIGHT: usize = 2;

    /// Initialize frame manager with per-frame data.
    pub fn new(device: Arc<Device>, swapchain: Swapchain) -> Result<Self, Error> {
        Ok(FrameManager {
            per_frame: (0..Self::FRAMES_IN_FLIGHT).into_iter().map(|_| -> Result<PerFrame, Error> {
               Ok(PerFrame {
                   fence: Arc::new(Fence::new(device.clone(), true)?),
                   image_ready: Semaphore::new(device.clone())?,
                   gpu_finished: Semaphore::new(device.clone())?
               })
            }).collect::<Result<Vec<PerFrame>, Error>>()?
            .try_into()
            .map_err(|_| Error::Uncategorized("Conversion to slice failed"))?,
            per_image: swapchain.images.iter().map(|_| PerImage { fence: None } ).collect(),
            current_frame: 0,
            current_image: 0,
            swapchain
        })
    }

    fn acquire_image(&self) -> Result<u32, Error> {
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

        Ok(match result {
            Ok((index, true)) => { /* No resize required */ index },
            Ok((index, false)) => { /* Resize required (vk::Result::SUBOPTIMAL_KHR) */ /*unimplemented!()*/ index },
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => { /* Resize required */  unimplemented!() },
            Err(_) => { /* Some other error occurred */ panic!("Error acquiring image") }
        })
    }

    /// This function must be called at the beginning of each frame.
    /// It will return an [`InFlightContext`] object which holds all the information for this current frame.
    /// You can only start doing command recording once the resulting future is awaited.
    pub async fn new_frame(&mut self) -> Result<InFlightContext, Error> {
        // Increment frame index. We do this here since this is the only mutable function in the frame loop.
        self.current_frame = (self.current_frame + 1) % self.per_frame.len() as u32;
        self.current_image = self.acquire_image()?;

        // Wait until this image is absolutely not in use anymore.
        let per_image = &mut self.per_image[self.current_image as usize];
        if let Some(image_fence) = per_image.fence.as_ref() {
            image_fence.wait().expect("Device lost.");
        }

        per_image.fence = Some(self.per_frame[self.current_frame as usize].fence.clone());

        // TODO: handle resizing here.

        Ok(InFlightContext {})
    }

    /// Submit this frame's commands to be processed. Note that this is the only way a frame's commands
    /// should ever be submitted to a queue. Any other ways to submit should be synchronized properly to this
    /// submission. The reason for this is that [`FrameManager::present`] waits on a semaphore this function's submission
    /// will signal. Any commands submitted from somewhere else must be synchronized to this submission.
    /// Note: it's possible this will be enforced through the type system later.
    /// TODO: examine possibilities for this.
    pub fn submit<D: ExecutionDomain>(&self, cmd: CommandBuffer<D>, exec: &ExecutionManager) -> Result<(), Error> {
        // Reset frame fence
        let per_frame = &self.per_frame[self.current_frame as usize];
        per_frame.fence.reset()?;

        let semaphores: Vec<vk::Semaphore> = vec![&per_frame.image_ready]
            .iter()
            .map(|sem| sem.handle)
            .collect();
        let stages = vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let submit = vk::SubmitInfo::builder()
            .signal_semaphores(std::slice::from_ref(&per_frame.gpu_finished.handle))
            .command_buffers(std::slice::from_ref(&cmd.handle))
            .wait_semaphores(semaphores.as_slice())
            .wait_dst_stage_mask(stages.as_slice())
            .build();

        // Use the command buffer's domain to determine the correct queue to use.
        let queue = exec.get_queue::<D>().ok_or(NoCapableQueue)?;
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
                Ok(functions.queue_present(queue.handle(),
                                        &vk::PresentInfoKHR::builder()
                                            .swapchains(std::slice::from_ref(&self.swapchain.handle))
                                            .wait_semaphores(std::slice::from_ref(&per_frame.gpu_finished.handle))
                                            .image_indices(std::slice::from_ref(&self.current_image)))
                    .map(|_| ())?)
            }
        } else { Err(Error::NoPresentQueue) }
    }
}