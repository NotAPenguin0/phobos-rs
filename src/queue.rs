use ash::prelude::VkResult;
use ash::vk;
use crate::{Queue, sync};

impl Queue {
    /// Submits a batch of submissions to the queue, and signals the given fence when the
    /// submission is done
    /// <br>
    /// <br>
    /// # Thread safety
    /// This function is **not yet** thread safe! This function is marked as unsafe for now to signal this.
    pub unsafe fn submit(&self, submits: &[vk::SubmitInfo], fence: &sync::Fence) -> VkResult<()> {
        self.device.queue_submit(self.handle, submits, fence.handle)
    }
}