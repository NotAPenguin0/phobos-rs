use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle};

#[cfg(feature = "winit")]
use winit;

/// Trait for windows that exposes the content width and height of a window.
pub trait WindowSize {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
}

/// Used as a dummy window interface in case of a headless context. Calling any of the `raw_xxx_handle()` functions on this will result in a panic.
pub struct HeadlessWindowInterface;

unsafe impl HasRawWindowHandle for HeadlessWindowInterface {
    fn raw_window_handle(&self) -> RawWindowHandle {
        panic!("Called raw_window_handle() on headless window context.");
    }
}

unsafe impl HasRawDisplayHandle for HeadlessWindowInterface {
    fn raw_display_handle(&self) -> RawDisplayHandle {
        panic!("Called raw_display_handle() on headless window context.");
    }
}

impl WindowSize for HeadlessWindowInterface {
    fn width(&self) -> u32 {
        panic!("called width() on headless window context.");
    }

    fn height(&self) -> u32 {
        panic!("Called height() on headless window context");
    }
}

#[cfg(feature = "winit")]
impl WindowSize for winit::window::Window {
    fn width(&self) -> u32 {
        self.inner_size().width
    }

    fn height(&self) -> u32 {
        self.inner_size().height
    }
}

/// Parent trait combining all requirements for a window interface. To be a window interface, a type T must implement the following traits:
/// - [`HasRawWindowHandle`](raw_window_handle::HasRawWindowHandle)
/// - [`HasRawDisplayHandle`](raw_window_handle::HasRawDisplayHandle)
/// - [`WindowSize`]
pub trait WindowInterface: HasRawWindowHandle + HasRawDisplayHandle + WindowSize {}
impl<T: HasRawWindowHandle + HasRawDisplayHandle + WindowSize> WindowInterface for T {}
