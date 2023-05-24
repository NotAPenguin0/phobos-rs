use std::ffi::c_void;
use std::fmt::{Display, Formatter};
use std::mem::MaybeUninit;

use anyhow::Result;
use ash::vk;
use ash::vk::Handle;
use fsr2_sys::{
    FfxDimensions2D, FfxErrorCode, FfxFsr2Context, ffxFsr2ContextCreate, FfxFsr2ContextDescription, ffxFsr2GetInterfaceVK, ffxFsr2GetScratchMemorySizeVK,
    FfxFsr2InitializationFlagBits, FfxFsr2InstanceFunctionPointerTableVk, FfxFsr2Interface, FfxFsr2MsgType, ffxGetDeviceVK, VkDevice,
    VkGetDeviceProcAddrFunc, VkPhysicalDevice,
};
use thiserror::Error;
use widestring::{WideChar as wchar_t, WideCStr};

#[derive(Debug, Error)]
pub struct Fsr2Error {
    pub code: FfxErrorCode,
}

fn check_fsr2_error(code: FfxErrorCode) -> Result<()> {
    if code == FfxErrorCode::Ok {
        Ok(())
    } else if code == FfxErrorCode::Eof {
        Ok(())
    } else {
        Err(Fsr2Error {
            code,
        }
            .into())
    }
}

impl Display for Fsr2Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.code {
            FfxErrorCode::Ok => {
                write!(f, "Ok")
            }
            FfxErrorCode::InvalidPointer => {
                write!(f, "Invalid pointer")
            }
            FfxErrorCode::InvalidAlignment => {
                write!(f, "Invalid alignment")
            }
            FfxErrorCode::InvalidSize => {
                write!(f, "Invalid size")
            }
            FfxErrorCode::Eof => {
                write!(f, "EOF")
            }
            FfxErrorCode::InvalidPath => {
                write!(f, "Invalid path")
            }
            FfxErrorCode::ErrorEof => {
                write!(f, "EOF Error")
            }
            FfxErrorCode::MalformedData => {
                write!(f, "Malformed data")
            }
            FfxErrorCode::OutOfMemory => {
                write!(f, "Out of Memory")
            }
            FfxErrorCode::IncompleteInterface => {
                write!(f, "Incomplete interface")
            }
            FfxErrorCode::InvalidEnum => {
                write!(f, "Invalid enum")
            }
            FfxErrorCode::InvalidArgument => {
                write!(f, "Invalid argument")
            }
            FfxErrorCode::OutOfRange => {
                write!(f, "Out of range")
            }
            FfxErrorCode::NullDevice => {
                write!(f, "Null device")
            }
            FfxErrorCode::BackendApiError => {
                write!(f, "Backend API error")
            }
            FfxErrorCode::InsufficientMemory => {
                write!(f, "Insufficient memory")
            }
        }
    }
}

#[derive(Debug)]
pub struct Fsr2Context {
    pub context: FfxFsr2Context,
    pub backend: FfxFsr2Interface,
    pub backend_scratch_data: Box<[u8]>,
}

extern "system" fn fsr2_message_callback(ty: FfxFsr2MsgType, message: *const wchar_t) {
    let str = unsafe { WideCStr::from_ptr_str(message) };
    match ty {
        FfxFsr2MsgType::Error => {
            error!("FSR2 Error: {}", str.display())
        }
        FfxFsr2MsgType::Warning => {
            warn!("FSR2 Warning: {}", str.display())
        }
    }
}

impl Fsr2Context {
    pub(crate) fn new(instance: &ash::Instance, physical_device: vk::PhysicalDevice, device: vk::Device) -> Result<Self> {
        unsafe {
            // Build a function pointer table with vulkan functions to pass to FSR2
            let functions_1_0 = instance.fp_v1_0();
            let functions_1_1 = instance.fp_v1_1();
            let fp_table = FfxFsr2InstanceFunctionPointerTableVk {
                fp_enumerate_device_extension_properties: std::mem::transmute::<_, _>(functions_1_0.enumerate_device_extension_properties),
                fp_get_device_proc_addr: std::mem::transmute::<_, _>(functions_1_0.get_device_proc_addr),
                fp_get_physical_device_memory_properties: std::mem::transmute::<_, _>(functions_1_0.get_physical_device_memory_properties),
                fp_get_physical_device_properties: std::mem::transmute::<_, _>(functions_1_0.get_physical_device_properties),
                fp_get_physical_device_properties2: std::mem::transmute::<_, _>(functions_1_1.get_physical_device_properties2),
                fp_get_physical_device_features2: std::mem::transmute::<_, _>(functions_1_1.get_physical_device_features2),
            };

            let physical_device = VkPhysicalDevice::from_raw(physical_device.as_raw());
            // First allocate a scratch buffer for backend instance data.
            // SAFETY: We assume a valid VkPhysicalDevice was passed in.
            let scratch_size = ffxFsr2GetScratchMemorySizeVK(physical_device, &fp_table);
            let scratch_data = Box::new_zeroed_slice(scratch_size);

            // SAFETY: We do not care about the contents of this buffer, so we assume it is initialized and let
            // the FSR2 API handle its contents.
            let mut scratch_data = scratch_data.assume_init();

            // Create the backend interface. We create an uninitialized interface struct first and let the API function
            // fill it in.
            let mut interface = MaybeUninit::<FfxFsr2Interface>::uninit();
            let err = ffxFsr2GetInterfaceVK(
                interface.as_mut_ptr(),
                scratch_data.as_mut_ptr() as *mut c_void,
                scratch_size,
                physical_device,
                &fp_table,
            );
            check_fsr2_error(err)?;

            // SAFETY: We just initialized the interface using the FSR2 API call above.
            let interface = interface.assume_init();

            // Now that we have the backend interface we can create the FSR2 context. We use the same strategy to
            // defer initialization to the API as above
            let mut context = MaybeUninit::<FfxFsr2Context>::uninit();

            // Obtain FSR2 device
            let vk_device = VkDevice::from_raw(device.as_raw());
            let device = ffxGetDeviceVK(vk_device);

            let info = FfxFsr2ContextDescription {
                flags: FfxFsr2InitializationFlagBits::ENABLE_DEBUG_CHECKING,
                // TODO: Correct render/display size
                max_render_size: FfxDimensions2D {
                    width: 512,
                    height: 512,
                },
                display_size: FfxDimensions2D {
                    width: 512,
                    height: 512,
                },
                callbacks: interface,
                device,
                fp_message: fsr2_message_callback,
            };

            let err = ffxFsr2ContextCreate(context.as_mut_ptr(), &info);
            check_fsr2_error(err)?;

            let context = context.assume_init();

            info!("Initialized FSR2 context");

            Ok(Self {
                context,
                backend: interface,
                backend_scratch_data: scratch_data,
            })
        }
    }
}

unsafe impl Send for Fsr2Context {}

unsafe impl Sync for Fsr2Context {}
