use std::marker::PhantomData;

use anyhow::Result;

use crate::{
    Allocator, AppSettings, DebugMessenger, DefaultAllocator, Device, ExecutionManager, FrameManager, PhysicalDevice, Surface, VkInstance, WindowInterface,
};

pub struct HeadlessContext;

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
    fn init_with_allocator<A: Allocator, F: FnOnce(&VkInstance, &PhysicalDevice, &Device) -> Result<A>>(
        settings: &AppSettings<W>,
        make_alloc: F,
    ) -> Result<Self::Output<A>>;
}

impl<W: WindowInterface> ContextInit<W> for HeadlessContext {
    type Output<A: Allocator> = (VkInstance, PhysicalDevice, Device, A, ExecutionManager, Option<DebugMessenger>);

    /// Initialize the headless context with the default allocator
    fn init(settings: &AppSettings<W>) -> Result<Self::Output<DefaultAllocator>> {
        Self::init_with_allocator(settings, |instance, physical_device, device| {
            DefaultAllocator::new(instance, device, physical_device)
        })
    }

    /// Initialize the headless context with a custom allocator
    fn init_with_allocator<A: Allocator, F: FnOnce(&VkInstance, &PhysicalDevice, &Device) -> Result<A>>(
        settings: &AppSettings<W>,
        make_alloc: F,
    ) -> Result<Self::Output<A>> {
        let instance = VkInstance::new(settings)?;
        let physical_device = PhysicalDevice::select(&instance, None, settings)?;
        let device = Device::new(&instance, &physical_device, settings)?;
        let exec = ExecutionManager::new(device.clone(), &physical_device)?;
        let allocator = make_alloc(&instance, &physical_device, &device)?;
        let debug_messenger = if settings.enable_validation {
            Some(DebugMessenger::new(&instance)?)
        } else {
            None
        };

        Ok((instance, physical_device, device, allocator, exec, debug_messenger))
    }
}

impl<W: WindowInterface> ContextInit<W> for WindowedContext<W> {
    type Output<A: Allocator> = (
        VkInstance,
        PhysicalDevice,
        Surface,
        Device,
        A,
        ExecutionManager,
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
    fn init_with_allocator<A: Allocator, F: FnOnce(&VkInstance, &PhysicalDevice, &Device) -> Result<A>>(
        settings: &AppSettings<W>,
        make_alloc: F,
    ) -> Result<Self::Output<A>> {
        let instance = VkInstance::new(settings)?;
        let (surface, physical_device) = { PhysicalDevice::select_with_surface(&instance, settings)? };
        let device = Device::new(&instance, &physical_device, settings)?;
        let allocator = make_alloc(&instance, &physical_device, &device)?;
        let exec = ExecutionManager::new(device.clone(), &physical_device)?;
        let frame = FrameManager::new_with_swapchain(&instance, device.clone(), allocator.clone(), settings, &surface)?;
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
            exec,
            frame,
            debug_messenger,
        ))
    }
}

/// Initialize all phobos objects with a custom allocator
pub fn initialize_with_allocator<W: WindowInterface, A: Allocator, F: FnOnce(&VkInstance, &PhysicalDevice, &Device) -> Result<A>>(
    settings: &AppSettings<W>,
    headless: bool,
    make_alloc: F,
) -> Result<(
    VkInstance,
    PhysicalDevice,
    Option<Surface>,
    Device,
    A,
    ExecutionManager,
    Option<FrameManager<A>>,
    Option<DebugMessenger>,
)> {
    if headless {
        let (instance, physical_device, device, allocator, exec, debug_messenger) = HeadlessContext::init_with_allocator(settings, make_alloc)?;
        Ok((instance, physical_device, None, device, allocator, exec, None, debug_messenger))
    } else {
        let (instance, physical_device, surface, device, allocator, exec, frame, debug_messenger) = WindowedContext::init_with_allocator(settings, make_alloc)?;
        Ok((
            instance,
            physical_device,
            Some(surface),
            device,
            allocator,
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
    VkInstance,
    PhysicalDevice,
    Option<Surface>,
    Device,
    DefaultAllocator,
    ExecutionManager,
    Option<FrameManager<DefaultAllocator>>,
    Option<DebugMessenger>,
)> {
    initialize_with_allocator(settings, headless, |instance, physical_device, device| {
        DefaultAllocator::new(instance, device, physical_device)
    })
}
