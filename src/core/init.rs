//! Exposes methods to make initialization of the library easier without losing flexibility.

use std::marker::PhantomData;

use anyhow::Result;

use crate::{
    Allocator, AppSettings, DebugMessenger, DefaultAllocator, Device, ExecutionManager,
    FrameManager, Instance, PhysicalDevice, Surface, WindowInterface,
};
use crate::pool::{ResourcePool, ResourcePoolCreateInfo};

/// ZST implementing initialization without a window
pub struct HeadlessContext;

/// ZST implementing initialization with a window
pub struct WindowedContext<W: WindowInterface> {
    _phantom: PhantomData<W>,
}

/// Trait that helps with initializing the context more easily.
pub trait ContextInit<W: WindowInterface> {
    /// The result type with all created objects.
    type Output<A: Allocator>;

    /// Initialize the context with the default allocator
    fn init(settings: &AppSettings<W>) -> Result<Self::Output<DefaultAllocator>>;

    /// Initialize the context with a custom allocator
    fn init_with_allocator<
        A: Allocator + 'static,
        F: FnOnce(&Instance, &PhysicalDevice, &Device) -> Result<A>,
    >(
        settings: &AppSettings<W>,
        make_alloc: F,
    ) -> Result<Self::Output<A>>;
}

impl<W: WindowInterface> ContextInit<W> for HeadlessContext {
    type Output<A: Allocator> = (
        Instance,
        PhysicalDevice,
        Device,
        A,
        ResourcePool<A>,
        ExecutionManager<A>,
        Option<DebugMessenger>,
    );

    /// Initialize the headless context with the default allocator
    fn init(settings: &AppSettings<W>) -> Result<Self::Output<DefaultAllocator>> {
        Self::init_with_allocator(settings, |instance, physical_device, device| {
            DefaultAllocator::new(instance, device, physical_device)
        })
    }

    /// Initialize the headless context with a custom allocator
    fn init_with_allocator<
        A: Allocator + 'static,
        F: FnOnce(&Instance, &PhysicalDevice, &Device) -> Result<A>,
    >(
        settings: &AppSettings<W>,
        make_alloc: F,
    ) -> Result<Self::Output<A>> {
        let instance = Instance::new(settings)?;
        let physical_device = PhysicalDevice::select(&instance, None, settings)?;
        let device = Device::new(&instance, &physical_device, settings)?;
        let allocator = make_alloc(&instance, &physical_device, &device)?;
        let pool_info = ResourcePoolCreateInfo {
            device: device.clone(),
            allocator: allocator.clone(),
            scratch_chunk_size: settings.scratch_chunk_size,
        };
        let pool = ResourcePool::new(pool_info)?;
        let exec = ExecutionManager::new(device.clone(), &physical_device, pool.clone())?;
        let debug_messenger = if settings.enable_validation {
            Some(DebugMessenger::new(&instance)?)
        } else {
            None
        };

        Ok((instance, physical_device, device, allocator, pool, exec, debug_messenger))
    }
}

impl<W: WindowInterface> ContextInit<W> for WindowedContext<W> {
    /// All created vulkan objects
    type Output<A: Allocator> = (
        Instance,
        PhysicalDevice,
        Surface,
        Device,
        A,
        ResourcePool<A>,
        ExecutionManager<A>,
        FrameManager<A>,
        Option<DebugMessenger>,
    );

    /// Initialize the windowed context with a default allocator
    fn init(settings: &AppSettings<W>) -> Result<Self::Output<DefaultAllocator>> {
        WindowedContext::init_with_allocator(settings, |instance, physical_device, device| {
            DefaultAllocator::new(instance, device, physical_device)
        })
    }

    /// Initialize the windowed context with a custom allocator
    fn init_with_allocator<
        A: Allocator + 'static,
        F: FnOnce(&Instance, &PhysicalDevice, &Device) -> Result<A>,
    >(
        settings: &AppSettings<W>,
        make_alloc: F,
    ) -> Result<Self::Output<A>> {
        let instance = Instance::new(settings)?;
        let (surface, physical_device) =
            { PhysicalDevice::select_with_surface(&instance, settings)? };
        let device = Device::new(&instance, &physical_device, settings)?;
        let allocator = make_alloc(&instance, &physical_device, &device)?;
        let pool_info = ResourcePoolCreateInfo {
            device: device.clone(),
            allocator: allocator.clone(),
            scratch_chunk_size: settings.scratch_chunk_size,
        };
        let pool = ResourcePool::new(pool_info)?;
        let exec = ExecutionManager::new(device.clone(), &physical_device, pool.clone())?;
        let frame = FrameManager::new_with_swapchain(
            &instance,
            device.clone(),
            pool.clone(),
            settings,
            &surface,
        )?;
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
}

/// Initialize all phobos objects with a custom allocator
pub fn initialize_with_allocator<
    W: WindowInterface,
    A: Allocator + 'static,
    F: FnOnce(&Instance, &PhysicalDevice, &Device) -> Result<A>,
>(
    settings: &AppSettings<W>,
    headless: bool,
    make_alloc: F,
) -> Result<(
    Instance,
    PhysicalDevice,
    Option<Surface>,
    Device,
    A,
    ResourcePool<A>,
    ExecutionManager<A>,
    Option<FrameManager<A>>,
    Option<DebugMessenger>,
)> {
    if headless {
        let (instance, physical_device, device, allocator, pool, exec, debug_messenger) =
            HeadlessContext::init_with_allocator(settings, make_alloc)?;
        Ok((
            instance,
            physical_device,
            None,
            device,
            allocator,
            pool,
            exec,
            None,
            debug_messenger,
        ))
    } else {
        let (
            instance,
            physical_device,
            surface,
            device,
            allocator,
            pool,
            exec,
            frame,
            debug_messenger,
        ) = WindowedContext::init_with_allocator(settings, make_alloc)?;
        Ok((
            instance,
            physical_device,
            Some(surface),
            device,
            allocator,
            pool,
            exec,
            Some(frame),
            debug_messenger,
        ))
    }
}

/// Initialize all phobos objects with the default allocator.
pub fn initialize<W: WindowInterface>(
    settings: &AppSettings<W>,
    headless: bool,
) -> Result<(
    Instance,
    PhysicalDevice,
    Option<Surface>,
    Device,
    DefaultAllocator,
    ResourcePool<DefaultAllocator>,
    ExecutionManager<DefaultAllocator>,
    Option<FrameManager<DefaultAllocator>>,
    Option<DebugMessenger>,
)> {
    initialize_with_allocator(settings, headless, |instance, physical_device, device| {
        DefaultAllocator::new(instance, device, physical_device)
    })
}
