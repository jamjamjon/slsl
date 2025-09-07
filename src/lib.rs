mod backend;
mod binary;
mod core;
mod linalg;
mod ops;
mod reduce;
mod slice;
mod unary;

pub use backend::*;
// pub use backend::{global_backend, Backend, OpsTrait};
pub use core::*;
pub use reduce::*;
pub use slice::{elem::*, IntoSliceElem};

pub const MAX_DIM: usize = 8;

/// Get the number of threads to use for parallel operations
///
/// This function respects the RAYON_NUM_THREADS environment variable,
/// falling back to the number of logical CPU cores if not set or invalid.
#[cfg(feature = "rayon")]
pub fn get_num_threads() -> usize {
    use std::str::FromStr;
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}

#[cfg(not(feature = "rayon"))]
pub fn get_num_threads() -> usize {
    1
}
