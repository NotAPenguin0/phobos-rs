use std::sync::Arc;
use crate::{Device, Context, FrameManager, PerFrame, InFlightContext};
use crate::sync::*;

impl FrameManager {
    /// The number of frames in flight. A frame in-flight is a frame that is rendering on the GPU or scheduled to do so.
    /// With two frames in flight, we can prepare a frame on the CPU while one frame is rendering on the GPU.
    /// This gives a good amount of parallelization while avoiding input lag.
    pub(crate) const FRAMES_IN_FLIGHT: usize = 2;

    /// Initialize frame manager with per-frame data.
    pub(crate) fn new(device: Arc<Device>) -> FrameManager {
        FrameManager {
            per_frame: (0..Self::FRAMES_IN_FLIGHT).into_iter().map(|_| -> PerFrame {
               PerFrame {
                   fence: Fence::new(device.clone(), true).unwrap()
               } 
            }).collect::<Vec<PerFrame>>()
            .try_into()
            .unwrap(),
            current_frame: 0
        }
    }

    /// This function must be called at the beginning of each frame.
    /// It will return an [`InFlightContext`] object which holds all the information for this current frame.
    /// You can only start doing command recording once the resulting future is awaited.
    pub async fn new_frame(&self) -> InFlightContext {
        let frame = &self.per_frame[self.current_frame as usize];
        frame.fence.wait().expect("Device lost");

        

        InFlightContext {}
    }

    pub fn present(&self) {

    }
}