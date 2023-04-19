use std::ops::{Add, Rem, Sub};

pub fn align<T: Add<T, Output=U> + Sub<T, Output=T> + Rem<T, Output=T> + Copy, U>(value: T, alignment: T) -> U {
    let unaligned_size = value % alignment;
    let padding = alignment - unaligned_size;
    value + padding
}
