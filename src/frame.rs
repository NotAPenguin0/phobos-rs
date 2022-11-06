use std::sync::Arc;
use ash::prelude::VkResult;
use crate::{Device, Context, FrameManager, PerFrame, InFlightContext, Swapchain, PerImage, Queue};
use crate::sync::*;
use ash::vk;

impl FrameManager {
    /// The number of frames in flight. A frame in-flight is a frame that is rendering on the GPU or scheduled to do so.
    /// With two frames in flight, we can prepare a frame on the CPU while one frame is rendering on the GPU.
    /// This gives a good amount of parallelization while avoiding input lag.
    pub(crate) const FRAMES_IN_FLIGHT: usize = 2;

    /// Initialize frame manager with per-frame data.
    pub(crate) fn new(device: Arc<Device>, swapchain: Swapchain) -> FrameManager {
        FrameManager {
            per_frame: (0..Self::FRAMES_IN_FLIGHT).into_iter().map(|_| -> PerFrame {
               PerFrame {
                   fence: Arc::new(Fence::new(device.clone(), true).unwrap()),
                   image_ready: Semaphore::new(device.clone()).unwrap(),
                   gpu_finished: Semaphore::new(device.clone()).unwrap()
               } 
            }).collect::<Vec<PerFrame>>()
            .try_into()
            .unwrap(),
            per_image: swapchain.images.iter().map(|_| PerImage { fence: None } ).collect(),
            current_frame: 0,
            current_image: 0,
            swapchain
        }
    }

    fn acquire_image(&self) -> u32 {
        let ext_functions = self.swapchain.ext_functions.as_ref().unwrap();
        let frame = &self.per_frame[self.current_frame as usize];
        frame.fence.wait().expect("Device lost");
        let result = unsafe {
            ext_functions.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                frame.image_ready.handle,
                vk::Fence::null()
            )
        };

        match result {
            Ok((index, true)) => { /* No resize required */ index },
            Ok((index, false)) => { /* Resize required (vk::Result::SUBOPTIMAL_KHR) */ /*unimplemented!()*/ index },
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => { /* Resize required */  unimplemented!() },
            Err(_) => { /* Some other error occurred */ panic!("Error acquiring image") }
        }
    }

    /// This function must be called at the beginning of each frame.
    /// It will return an [`InFlightContext`] object which holds all the information for this current frame.
    /// You can only start doing command recording once the resulting future is awaited.
    pub async fn new_frame(&mut self) -> InFlightContext {
        self.current_image = self.acquire_image();

        // Wait until this image is absolutely not in use anymore.
        let per_image = &mut self.per_image[self.current_image as usize];
        if let Some(image_fence) = per_image.fence.as_ref() {
            image_fence.wait().expect("Device lost.");
        }

        per_image.fence = Some(self.per_frame[self.current_frame as usize].fence.clone());

        // TODO: handle resizing here.

        InFlightContext {}
    }

    pub fn submit(&self, queue: &Queue) -> VkResult<()> {
        // Reset frame fence
        let per_frame = &self.per_frame[self.current_frame as usize];
        per_frame.fence.reset().expect("Device lost");

        let semaphores: Vec<vk::Semaphore> = vec![&per_frame.image_ready]
            .iter()
            .map(|sem| sem.handle)
            .collect();
        let stages = vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let submit = vk::SubmitInfo::builder()
            .signal_semaphores(std::slice::from_ref(&per_frame.gpu_finished.handle))
            .wait_semaphores(semaphores.as_slice())
            .wait_dst_stage_mask(stages.as_slice())
            .build();
        unsafe { queue.submit(std::slice::from_ref(&submit), &per_frame.fence) }
    }

    pub fn present(&self, queue: &Queue) -> VkResult<()> {
        let per_frame = &self.per_frame[self.current_frame as usize];
        let ext_functions = self.swapchain.ext_functions.as_ref().unwrap();
        unsafe { ext_functions.queue_present(queue.handle,
                     &vk::PresentInfoKHR::builder()
                         .swapchains(std::slice::from_ref(&self.swapchain.handle))
                         .wait_semaphores(std::slice::from_ref(&per_frame.gpu_finished.handle))
                         .image_indices(std::slice::from_ref(&self.current_image)))
            .map(|_| ())
        }
    }
}