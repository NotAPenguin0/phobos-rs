//! Contains the debug messenger used to log validation layer messages

use std::ops::Deref;

use anyhow::Result;
use ash::vk;

use crate::Instance;
use crate::util::string::wrap_c_str;

/// Vulkan debug messenger, can be passed to certain functions to extend debugging functionality.
///
/// This also dereferences into [`ash::extensions::ext::DebugUtils`], to directly call into the functions
/// of the `VK_EXT_debug_utils` extension.
///
/// Using this requires the Vulkan SDK to be installed, so do not ship production applications with this
/// enabled.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct DebugMessenger {
    handle: vk::DebugUtilsMessengerEXT,
    #[derivative(Debug = "ignore")]
    functions: ash::extensions::ext::DebugUtils,
}

impl DebugMessenger {
    /// Creates a new debug messenger. Requires the vulkan validation layers to be enabled to
    /// do anything useful.
    pub fn new(instance: &Instance) -> Result<Self> {
        // SAFETY: We do not mutate this loader in any way, so the safety contract is satisfied
        let functions =
            ash::extensions::ext::DebugUtils::new(unsafe { instance.loader() }, instance);
        let info = vk::DebugUtilsMessengerCreateInfoEXT {
            s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            p_next: std::ptr::null(),
            flags: Default::default(),
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            pfn_user_callback: Some(vk_debug_callback),
            p_user_data: std::ptr::null::<std::ffi::c_void>() as *mut std::ffi::c_void,
        };
        // SAFETY: Both p_user_data and p_next are allowed to be NULL, sType is correct and there are no other pointers passed in.
        let handle = unsafe { functions.create_debug_utils_messenger(&info, None)? };
        #[cfg(feature = "log-objects")]
        trace!("Created new VkDebugUtilsMessengerEXT {handle:p}");
        Ok(DebugMessenger {
            handle,
            functions,
        })
    }
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkDebugUtilsMessengerEXT {:p}", self.handle);
        unsafe {
            // SAFETY: self is valid, so self.functions and self.handle are valid, non-null objects.
            self.functions
                .destroy_debug_utils_messenger(self.handle, None);
        }
    }
}

impl Deref for DebugMessenger {
    type Target = ash::extensions::ext::DebugUtils;

    fn deref(&self) -> &Self::Target {
        &self.functions
    }
}

extern "system" fn vk_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { *p_callback_data };
    let message_id_number = callback_data.message_id_number;
    let message_id_name = unsafe { wrap_c_str(callback_data.p_message_id_name) };
    let message = unsafe { wrap_c_str(callback_data.p_message) };

    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            trace!(
                "[{:?}]: {} ({}): {}",
                msg_type,
                message_id_name,
                &message_id_number.to_string(),
                message
            );
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            info!(
                "[{:?}]: {} ({}): {}",
                msg_type,
                message_id_name,
                &message_id_number.to_string(),
                message
            );
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            warn!(
                "[{:?}]: {} ({}): {}",
                msg_type,
                message_id_name,
                &message_id_number.to_string(),
                message
            );
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!(
                "[{:?}]: {} ({}): {}",
                msg_type,
                message_id_name,
                &message_id_number.to_string(),
                message
            );
        }
        _ => {
            unimplemented!()
        }
    };

    false as vk::Bool32
}
