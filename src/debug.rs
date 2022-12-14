use ash::vk;

use crate::{util, VkInstance, Error};

#[derive(Derivative)]
#[derivative(Debug)]
pub struct DebugMessenger {
    handle: vk::DebugUtilsMessengerEXT,
    #[derivative(Debug="ignore")]
    functions: ash::extensions::ext::DebugUtils,
}

impl DebugMessenger {
    /// Creates a new debug messenger. Requires the vulkan validation layers to be enabled to
    /// do anything useful.
    pub fn new(instance: &VkInstance) -> Result<Self, Error> {
        let functions = ash::extensions::ext::DebugUtils::new(&instance.entry, &instance.instance);
        let handle = unsafe {
            functions.create_debug_utils_messenger(
                &vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
                .pfn_user_callback(Some(vk_debug_callback)), None)?
        };
        Ok(DebugMessenger { handle, functions })
    }
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        unsafe { self.functions.destroy_debug_utils_messenger(self.handle, None); }
    }
}

extern "system" fn vk_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void) -> vk::Bool32 {

    let callback_data = unsafe { *p_callback_data };
    let message_id_number = callback_data.message_id_number as i32;
    let message_id_name = unsafe { util::wrap_c_str(callback_data.p_message_id_name) };
    let message = unsafe { util::wrap_c_str(callback_data.p_message) };

    // TODO: switch out logging with log crate: https://docs.rs/log

    println!("[{:?}] [{:?}]: {} ({}): {}",
             severity,
             msg_type,
             message_id_name,
             &message_id_number.to_string(),
             message
    );

    false as vk::Bool32
}