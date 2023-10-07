//! Utilities for generic window handling

use raw_window_handle::{
    HasRawDisplayHandle, HasRawWindowHandle,
};
#[cfg(feature = "winit")]
use winit;

/// Trait for windows that exposes the content width and height of a window.
pub trait WindowSize {
    /// Get the width of the window
    fn width(&self) -> u32;
    /// Get the height of the window
    fn height(&self) -> u32;
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

/// Generic "window" trait that is applied to all raw_window_handle providers and those who implement window size
pub trait Window: WindowSize + HasRawDisplayHandle + HasRawWindowHandle {}
impl<T: WindowSize + HasRawDisplayHandle + HasRawWindowHandle> Window for T {}