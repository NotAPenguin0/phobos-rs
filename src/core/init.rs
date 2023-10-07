//! Exposes methods to make initialization of the library easier without losing flexibility.

use anyhow::Result;

use crate::{
    Allocator, AppSettings, DebugMessenger, DefaultAllocator, Device, ExecutionManager,
    FrameManager, Instance, PhysicalDevice, Surface, SurfaceSettings,
};
use crate::pool::{ResourcePool, ResourcePoolCreateInfo};

/// Struct that contains all common Phobos resources to be used at initialization
pub type Phobos<A> = (Instance, PhysicalDevice, Option<Surface>, Device, A, ResourcePool<A>, ExecutionManager<A>, Option<FrameManager<A>>, Option<DebugMessenger>);

/// Initialize the context with the default allocator
pub fn initialize(settings: &AppSettings) -> Result<Phobos<DefaultAllocator>> {
    initialize_with_allocator(settings, |instance, physical_device, device| {
        DefaultAllocator::new(instance, device, physical_device)
    })
}

/// Initialize the context with a custom allocator
pub fn initialize_with_allocator<
    A: Allocator + 'static,
    F: FnOnce(&Instance, &PhysicalDevice, &Device) -> Result<A>,
>(
    settings: &AppSettings,
    make_alloc: F,
) -> Result<Phobos<A>> {
    let instance = Instance::new(settings)?;
    

    let mut surface = if let Some(SurfaceSettings { window, .. }) = settings.surface_settings.as_ref() {
        Some(Surface::new(&instance, *window)?)
    } else {
        None
    };

    let physical_device = PhysicalDevice::select(&instance, surface.as_ref(), settings)?;
    
    if let Some(surface) = surface.as_mut() {
        surface.query_details(&physical_device)?;
    }

    let device = Device::new(&instance, &physical_device, settings)?;
    let allocator = make_alloc(&instance, &physical_device, &device)?;
    let pool_info = ResourcePoolCreateInfo {
        device: device.clone(),
        allocator: allocator.clone(),
        scratch_chunk_size: settings.scratch_chunk_size,
    };
    let pool = ResourcePool::new(pool_info)?;
    let exec = ExecutionManager::new(device.clone(), &physical_device, pool.clone())?;

    let frame = if let Some(surface_settings) = settings.surface_settings.as_ref() {
        Some(FrameManager::new_with_swapchain(
            &instance,
            device.clone(),
            pool.clone(),
            surface_settings,
            &surface.as_ref().unwrap(),
        )?)
    } else {
        None
    };

    let debug_messenger = if settings.enable_validation {
        Some(DebugMessenger::new(&instance)?)
    } else {
        None
    };

    Ok((
        instance,
        physical_device,
        surface,
        device,
        allocator,
        pool,
        exec,
        frame,
        debug_messenger,
    ))
}