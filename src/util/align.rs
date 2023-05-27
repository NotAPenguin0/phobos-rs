//! Utilities for aligning memory

use std::ops::{Add, Rem, Sub};

/// Align a size to a required alignment. Does not align the base address.
pub fn align<T: Add<T, Output = U> + Sub<T, Output = T> + Rem<T, Output = T> + Copy, U>(
    value: T,
    alignment: T,
) -> U {
    let unaligned_size = value % alignment;
    let padding = alignment - unaligned_size;
    value + padding
}
