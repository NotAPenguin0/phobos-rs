//! The allocator module exposes a couple interesting parts of the API
//! <br>
//! <br>
//! # Allocator traits
//! These are defined in [`traits`], and can be implemented to supply a custom allocator type to all phobos functions.
//! # Default allocator
//! A default allocator based on the `gpu_allocator` crate is implemented here. Most types that take a generic allocator
//! parameter default to this allocator.
//! # Scratch allocator
//! A linear allocator used for making temporary, short lived allocations. For more information check the [`scratch_allocator`]
//! module documentation.

pub mod traits;
pub mod default_allocator;
pub mod memory_type;
pub mod scratch_allocator;
