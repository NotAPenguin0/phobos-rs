//! Exposes wrappers around the FSR2 library for supersampling, integrated with the phobos library.

use std::ffi::c_void;
use std::fmt::{Display, Formatter};
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};
use std::time::Duration;

use anyhow::Result;
use ash::vk;
use ash::vk::Handle;
use fsr2_sys::{
    FfxErrorCode, FfxFsr2Context, ffxFsr2ContextCreate, FfxFsr2ContextDescription,
    ffxFsr2ContextDestroy, ffxFsr2ContextDispatch, FfxFsr2DispatchDescription,
    ffxFsr2GetInterfaceVK, ffxFsr2GetJitterOffset, ffxFsr2GetJitterPhaseCount, ffxFsr2GetRenderResolutionFromQualityMode,
    ffxFsr2GetScratchMemorySizeVK, FfxFsr2InstanceFunctionPointerTableVk, FfxFsr2Interface, FfxFsr2MsgType,
    ffxGetCommandListVK, ffxGetDeviceVK, ffxGetTextureResourceVK, FfxResource,
    FfxResourceState, VkDevice, VkPhysicalDevice,
};
pub use fsr2_sys::{
    FfxDimensions2D, FfxFloatCoords2D, FfxFsr2InitializationFlagBits, FfxFsr2QualityMode,
};
use thiserror::Error;
use widestring::{WideChar as wchar_t, WideCStr};

use crate::{Allocator, ComputeSupport, DeletionQueue, ImageView, IncompleteCommandBuffer};
use crate::domain::ExecutionDomain;
use crate::pool::{Pool, Poolable, Pooled};

/// FSR2 API error, stores an error code that describes the cause of the error
#[derive(Debug, Error)]
pub struct Fsr2Error {
    /// Error code with the cause of the FSR2 error.
    pub code: FfxErrorCode,
}

/// Check the FSR2 error code for an Ok status, and return Ok(()) in that case,
/// or an error result instead.
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

struct BackendData(pub Box<[u8]>);

impl Deref for BackendData {
    type Target = Box<[u8]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BackendData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Poolable for BackendData {
    type Key = usize;

    fn on_release(&mut self) {}
}

/// Represents the initialized FSR2 context with its backend data.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Fsr2Context {
    context: Box<FfxFsr2Context>,
    backend: FfxFsr2Interface,
    #[derivative(Debug = "ignore")]
    backend_scratch_data: Pooled<BackendData>,
    current_frame: usize,
    #[derivative(Debug = "ignore")]
    fp_table: FfxFsr2InstanceFunctionPointerTableVk,
    #[derivative(Debug = "ignore")]
    scratch_data_pool: Pool<BackendData>,
    create_flags: FfxFsr2InitializationFlagBits,
    display_size: FfxDimensions2D,
    max_render_size: FfxDimensions2D,
    #[derivative(Debug = "ignore")]
    device: VkDevice,
    #[derivative(Debug = "ignore")]
    phys_device: VkPhysicalDevice,
    #[derivative(Debug = "ignore")]
    deferred_backend_delete: DeletionQueue<ReleasedFsr2Context>,
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

/// Stores resources needed to execute an FSR2 dispatch.
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

/// Holds all settings for an FSR2 dispatch.
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

pub(crate) struct Fsr2ContextCreateInfo<'a> {
    pub instance: &'a ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: vk::Device,
    pub flags: FfxFsr2InitializationFlagBits,
    pub max_render_size: FfxDimensions2D,
    pub display_size: FfxDimensions2D,
}

#[allow(dead_code)]
struct ReleasedFsr2Context {
    pub context: Box<FfxFsr2Context>,
    pub backend: FfxFsr2Interface,
    pub backend_data: Pooled<BackendData>,
}

impl Fsr2Context {
    /// Creates the FSR2 context.
    unsafe fn create_context(
        fp_table: FfxFsr2InstanceFunctionPointerTableVk,
        mem_pool: &Pool<BackendData>,
        device: VkDevice,
        phys_device: VkPhysicalDevice,
        flags: FfxFsr2InitializationFlagBits,
        display_size: FfxDimensions2D,
        max_render_size: FfxDimensions2D,
    ) -> Result<(Box<FfxFsr2Context>, FfxFsr2Interface, Pooled<BackendData>)> {
        // First allocate a scratch buffer for backend instance data.
        // SAFETY: We assume a valid VkPhysicalDevice was passed in.
        let scratch_size = ffxFsr2GetScratchMemorySizeVK(phys_device, &fp_table);
        let mut scratch_data = BackendData::new_in_pool(mem_pool, &scratch_size)?;
        let scratch_pointer = scratch_data.as_mut_ptr();

        // Create the backend interface. We create an uninitialized interface struct first and let the API function
        // fill it in.
        let mut interface = MaybeUninit::<FfxFsr2Interface>::uninit();
        let err = ffxFsr2GetInterfaceVK(
            interface.as_mut_ptr(),
            scratch_pointer as *mut c_void,
            scratch_size,
            phys_device,
            &fp_table,
        );
        check_fsr2_error(err)?;

        // SAFETY: We just initialized the interface using the FSR2 API call above.
        let interface = interface.assume_init();

        // Now that we have the backend interface we can create the FSR2 context. We use the same strategy to
        // defer initialization to the API as above
        let mut context = Box::<FfxFsr2Context>::new_uninit();

        // Obtain FSR2 device
        let device = ffxGetDeviceVK(device);

        let info = FfxFsr2ContextDescription {
            flags,
            max_render_size,
            display_size,
            callbacks: interface,
            device,
            fp_message: fsr2_message_callback,
        };

        let err = ffxFsr2ContextCreate(context.as_mut_ptr(), &info);
        check_fsr2_error(err)?;

        let context = context.assume_init();

        Ok((context, interface, scratch_data))
    }

    pub(crate) fn new(info: Fsr2ContextCreateInfo) -> Result<Self> {
        unsafe {
            // Build a function pointer table with vulkan functions to pass to FSR2
            let functions_1_0 = info.instance.fp_v1_0();
            let functions_1_1 = info.instance.fp_v1_1();
            let fp_table = FfxFsr2InstanceFunctionPointerTableVk {
                // SAFETY: These are the same functions, but their types are from different crates.
                fp_enumerate_device_extension_properties: std::mem::transmute::<_, _>(
                    functions_1_0.enumerate_device_extension_properties,
                ),
                fp_get_device_proc_addr: std::mem::transmute::<_, _>(
                    functions_1_0.get_device_proc_addr,
                ),
                fp_get_physical_device_memory_properties: std::mem::transmute::<_, _>(
                    functions_1_0.get_physical_device_memory_properties,
                ),
                fp_get_physical_device_properties: std::mem::transmute::<_, _>(
                    functions_1_0.get_physical_device_properties,
                ),
                fp_get_physical_device_properties2: std::mem::transmute::<_, _>(
                    functions_1_1.get_physical_device_properties2,
                ),
                fp_get_physical_device_features2: std::mem::transmute::<_, _>(
                    functions_1_1.get_physical_device_features2,
                ),
            };

            let phys_device = VkPhysicalDevice::from_raw(info.physical_device.as_raw());
            let device = VkDevice::from_raw(info.device.as_raw());

            let data_pool =
                Pool::new(|size| Ok(BackendData(Box::new_zeroed_slice(*size).assume_init())))?;

            let (context, backend, scratch) = Self::create_context(
                fp_table,
                &data_pool,
                device,
                phys_device,
                info.flags,
                info.display_size,
                info.max_render_size,
            )?;

            info!(
                "Initialized FSR2 context. FSR2 version: {}.{}.{}",
                fsr2_sys::FFX_FSR2_VERSION_MAJOR,
                fsr2_sys::FFX_FSR2_VERSION_MINOR,
                fsr2_sys::FFX_FSR2_VERSION_PATCH
            );

            Ok(Self {
                context,
                backend,
                backend_scratch_data: scratch,
                current_frame: 0,
                fp_table,
                scratch_data_pool: data_pool,
                create_flags: info.flags,
                display_size: info.display_size,
                max_render_size: info.max_render_size,
                device,
                phys_device,
                deferred_backend_delete: DeletionQueue::new(4),
            })
        }
    }

    fn get_image_resource(&mut self, image: &ImageView, state: FfxResourceState) -> FfxResource {
        unsafe {
            let image_raw = fsr2_sys::VkImage::from_raw(image.image().as_raw());
            let view_raw = fsr2_sys::VkImageView::from_raw(image.handle().as_raw());
            ffxGetTextureResourceVK(
                &mut *self.context,
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

    fn get_optional_image_resource(
        &mut self,
        image: &Option<ImageView>,
        state: FfxResourceState,
    ) -> FfxResource {
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
        // Clean up old fsr2 contexts after resizes
        self.deferred_backend_delete.next_frame();
        let cmd_raw = unsafe { fsr2_sys::VkCommandBuffer::from_raw(cmd.handle().as_raw()) };
        let cmd_list = unsafe { ffxGetCommandListVK(cmd_raw) };
        if descr.auto_reactive.is_some() {
            warn!("Auto-reactive is currently not supported. Please open an issue if you would like this added.");
        }
        let description = FfxFsr2DispatchDescription {
            command_list: cmd_list,
            color: self.get_image_resource(&resources.color, FfxResourceState::COMPUTE_READ),
            depth: self.get_image_resource(&resources.depth, FfxResourceState::COMPUTE_READ),
            motion_vectors: self
                .get_image_resource(&resources.motion_vectors, FfxResourceState::COMPUTE_READ),
            exposure: self
                .get_optional_image_resource(&resources.exposure, FfxResourceState::COMPUTE_READ),
            reactive: self
                .get_optional_image_resource(&resources.reactive, FfxResourceState::COMPUTE_READ),
            transparency_and_composition: self.get_optional_image_resource(
                &resources.transparency_and_composition,
                FfxResourceState::COMPUTE_READ,
            ),
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

        let err = unsafe { ffxFsr2ContextDispatch(&mut *self.context, &description) };
        check_fsr2_error(err)?;

        self.current_frame += 1;

        Ok(())
    }

    /// Get the number of jitter phases. Must be supplied with the current render width.
    pub fn jitter_phase_count(&mut self, render_width: u32) -> i32 {
        unsafe { ffxFsr2GetJitterPhaseCount(render_width, self.display_size.width) }
    }

    /// Get the jitter offset values for this frame, given the current render width.
    pub fn jitter_offset(&mut self, render_width: u32) -> Result<(f32, f32)> {
        let phase_count = self.jitter_phase_count(render_width);
        let index = self.current_frame % phase_count as usize;
        let mut jitter_x = 0.0;
        let mut jitter_y = 0.0;
        let error = unsafe {
            ffxFsr2GetJitterOffset(&mut jitter_x, &mut jitter_y, index as i32, phase_count as u32)
        };
        check_fsr2_error(error)?;
        Ok((jitter_x, jitter_y))
    }

    /// Get the recommended render resolution based on a quality setting and the current display resolution.
    pub fn get_render_resolution(
        &mut self,
        quality_mode: FfxFsr2QualityMode,
    ) -> Result<FfxDimensions2D> {
        let mut render_width = 0;
        let mut render_height = 0;
        let err = unsafe {
            ffxFsr2GetRenderResolutionFromQualityMode(
                &mut render_width,
                &mut render_height,
                self.display_size.width,
                self.display_size.height,
                quality_mode,
            )
        };
        check_fsr2_error(err)?;
        Ok(FfxDimensions2D {
            width: render_width,
            height: render_height,
        })
    }

    /// Set the display resolution to a new resolution. After doing this, you should also recreate the relevant
    /// attachments with a new size based on the display size. This new size may not be greater than `max_render_size`.
    /// If `max_render_size` is set to `None`, it is assumed to be equal to `display_size`.
    /// Prefer setting `max_render_size` as low as possible to save memory.
    /// If the new display size and max render size are the same as the old values, this function is a no-op.
    pub fn set_display_resolution(
        &mut self,
        display_size: FfxDimensions2D,
        max_render_size: Option<FfxDimensions2D>,
    ) -> Result<()> {
        // Create new context if something changed
        let max_render_size = max_render_size.unwrap_or(display_size);
        if display_size.width == self.display_size.width
            && display_size.height == self.display_size.height
            && self.max_render_size.width == max_render_size.width
            && self.max_render_size.height == max_render_size.height
        {
            // nothing to do
            return Ok(());
        }
        let (context, backend, scratch) = unsafe {
            Self::create_context(
                self.fp_table,
                &self.scratch_data_pool,
                self.device,
                self.phys_device,
                self.create_flags,
                display_size,
                max_render_size,
            )?
        };
        // Swap out data
        let old_scratch = std::mem::replace(&mut self.backend_scratch_data, scratch);
        let old_context = std::mem::replace(&mut self.context, context);
        let old_backend = std::mem::replace(&mut self.backend, backend);
        // Defer deletion of old context
        self.deferred_backend_delete.push(ReleasedFsr2Context {
            context: old_context,
            backend: old_backend,
            backend_data: old_scratch,
        });
        // Reset context info
        self.display_size = display_size;
        self.max_render_size = max_render_size;
        self.current_frame = 0;
        Ok(())
    }
}

unsafe impl Send for Fsr2Context {}

unsafe impl Sync for Fsr2Context {}

impl Drop for Fsr2Context {
    fn drop(&mut self) {
        unsafe {
            ffxFsr2ContextDestroy(&mut *self.context);
        }
    }
}

impl Drop for ReleasedFsr2Context {
    fn drop(&mut self) {
        unsafe {
            ffxFsr2ContextDestroy(&mut *self.context);
        }
    }
}
