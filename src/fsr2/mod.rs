use std::ffi::c_void;
use std::fmt::{Display, Formatter};
use std::mem::MaybeUninit;
use std::time::Duration;

use anyhow::Result;
use ash::vk;
use ash::vk::Handle;
use fsr2_sys::{
    FfxDimensions2D, FfxErrorCode, FfxFloatCoords2D, FfxFsr2Context, ffxFsr2ContextCreate, FfxFsr2ContextDescription,
    ffxFsr2ContextDestroy, ffxFsr2ContextDispatch, FfxFsr2DispatchDescription, ffxFsr2GetInterfaceVK, ffxFsr2GetScratchMemorySizeVK, FfxFsr2InitializationFlagBits, FfxFsr2InstanceFunctionPointerTableVk,
    FfxFsr2Interface, FfxFsr2MsgType, ffxGetCommandListVK, ffxGetDeviceVK, ffxGetTextureResourceVK, FfxResource,
    FfxResourceState, VkDevice, VkGetDeviceProcAddrFunc, VkPhysicalDevice,
};
use thiserror::Error;
use widestring::{WideChar as wchar_t, WideCStr};

use crate::{Allocator, ComputeSupport, ImageView, IncompleteCommandBuffer, VirtualResource};
use crate::domain::ExecutionDomain;

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

/// Experimental reactive mask generation parameters
#[derive(Debug, Clone)]
pub struct Fsr2AutoReactiveDescription {
    /// Opaque only color buffer for the current frame, at render resolution
    pub color_opaque_only: Option<ImageView>,
    /// Cutoff value for TC
    pub auto_tc_threshold: f32,
    /// Value to scale the transparency and composition mask
    pub auto_tc_scale: f32,
    /// Value to scale the reactive mask
    pub auto_reactive_scale: f32,
    /// Value to clamp the reactive mask
    pub auto_reactive_max: f32,
}

#[derive(Debug, Clone)]
pub struct Fsr2DispatchResources {
    /// Color buffer for the current frame, at render resolution.
    pub color: ImageView,
    /// Depth buffer for the current frame, at render resolution
    pub depth: ImageView,
    /// Motion vectors for the current frame, at render resolution
    pub motion_vectors: ImageView,
    /// Optional 1x1 texture with the exposure value
    pub exposure: Option<ImageView>,
    /// Optional resource with the alpha value of reactive objects in the scene
    pub reactive: Option<ImageView>,
    /// Optional resource with the alpha value of special objects in the scene
    pub transparency_and_composition: Option<ImageView>,
    /// Output color buffer for the current frame at presentation resolution
    pub output: ImageView,
}

#[derive(Debug, Clone)]
pub struct Fsr2DispatchDescription {
    /// Subpixel jitter offset applied to the camera
    pub jitter_offset: FfxFloatCoords2D,
    /// Scale factor to apply to motion vectors
    pub motion_vector_scale: FfxFloatCoords2D,
    /// Enable additional sharpening
    pub enable_sharpening: bool,
    /// 0..1 value for sharpening, where 0 is no additional sharpness
    pub sharpness: f32,
    /// Delta time between this frame and the previous frame
    pub frametime_delta: Duration,
    /// Pre exposure value, must be > 0.0
    pub pre_exposure: f32,
    /// Indicates the camera has moved discontinuously
    pub reset: bool,
    /// Distance to the near plane of the camera
    pub camera_near: f32,
    /// Distance to the far plane of the camera
    pub camera_far: f32,
    /// Camera angle FOV in the vertical direction, in radians
    pub camera_fov_vertical: f32,
    /// The scale factor to convert view space units to meters
    pub viewspace_to_meters_factor: f32,
    /// Experimental reactive mask generation parameters
    pub auto_reactive: Option<Fsr2AutoReactiveDescription>,
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
                // SAFETY: These are the same functions, but their types are from different crates.
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

            info!(
                "Initialized FSR2 context. FSR2 version: {}.{}.{}",
                fsr2_sys::FFX_FSR2_VERSION_MAJOR,
                fsr2_sys::FFX_FSR2_VERSION_MINOR,
                fsr2_sys::FFX_FSR2_VERSION_PATCH
            );

            Ok(Self {
                context,
                backend: interface,
                backend_scratch_data: scratch_data,
            })
        }
    }

    fn get_image_resource(&mut self, image: &ImageView, state: FfxResourceState) -> FfxResource {
        unsafe {
            let image_raw = fsr2_sys::VkImage::from_raw(image.image().as_raw());
            let view_raw = fsr2_sys::VkImageView::from_raw(image.handle().as_raw());
            ffxGetTextureResourceVK(
                image_raw,
                view_raw,
                image.width(),
                image.height(),
                image.format().as_raw(),
                std::ptr::null(),
                state,
            )
        }
    }

    fn get_optional_image_resource(&mut self, image: &Option<ImageView>, state: FfxResourceState) -> FfxResource {
        image
            .as_ref()
            .map(|image| self.get_image_resource(image, state))
            .unwrap_or_else(|| FfxResource::NULL)
    }

    /// Dispatch FSR2 commands, with no additional synchronization on resources used
    pub(crate) fn dispatch<D: ExecutionDomain + ComputeSupport, A: Allocator>(
        &mut self,
        descr: &Fsr2DispatchDescription,
        resources: &Fsr2DispatchResources,
        cmd: &IncompleteCommandBuffer<D, A>,
    ) -> Result<()> {
        let cmd_raw = unsafe { fsr2_sys::VkCommandBuffer::from_raw(cmd.handle().as_raw()) };
        let cmd_list = unsafe { ffxGetCommandListVK(cmd_raw) };
        if descr.auto_reactive.is_some() {
            warn!("Auto-reactive is currently not supported. Please open an issue if you would like this added.");
        }
        let description = FfxFsr2DispatchDescription {
            command_list: cmd_list,
            color: self.get_image_resource(&resources.color, FfxResourceState::COMPUTE_READ),
            depth: self.get_image_resource(&resources.depth, FfxResourceState::COMPUTE_READ),
            motion_vectors: self.get_image_resource(&resources.motion_vectors, FfxResourceState::COMPUTE_READ),
            exposure: self.get_optional_image_resource(&resources.exposure, FfxResourceState::COMPUTE_READ),
            reactive: self.get_optional_image_resource(&resources.reactive, FfxResourceState::COMPUTE_READ),
            transparency_and_composition: self.get_optional_image_resource(&resources.transparency_and_composition, FfxResourceState::COMPUTE_READ),
            output: self.get_image_resource(&resources.output, FfxResourceState::UNORDERED_ACCESS),
            jitter_offset: descr.jitter_offset,
            motion_vector_scale: descr.motion_vector_scale,
            // Infer render size from color resource size
            render_size: FfxDimensions2D {
                width: resources.color.width(),
                height: resources.color.height(),
            },
            enable_sharpening: descr.enable_sharpening,
            sharpness: descr.sharpness,
            frametime_delta: descr.frametime_delta.as_secs_f32() * 1000.0,
            pre_exposure: descr.pre_exposure,
            reset: descr.reset,
            camera_near: descr.camera_near,
            camera_far: descr.camera_far,
            camera_vertical_fov: descr.camera_fov_vertical,
            viewspace_to_meters_factor: descr.viewspace_to_meters_factor,
            enable_auto_reactive: false,
            color_opaque_only: FfxResource::NULL,
            auto_tc_threshold: 0.0,
            auto_tc_scale: 0.0,
            auto_reactive_scale: 0.0,
            auto_reactive_max: 0.0,
        };

        let err = unsafe { ffxFsr2ContextDispatch(&mut self.context, &description) };
        check_fsr2_error(err)?;
        Ok(())
    }
}

unsafe impl Send for Fsr2Context {}
unsafe impl Sync for Fsr2Context {}

impl Drop for Fsr2Context {
    fn drop(&mut self) {
        unsafe {
            ffxFsr2ContextDestroy(&mut self.context);
        }
    }
}
