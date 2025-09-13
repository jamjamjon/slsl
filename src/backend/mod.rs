#[cfg(feature = "accelerate")]
mod accelerate;
#[cfg(not(any(feature = "accelerate", feature = "mkl", feature = "openblas")))]
mod fallback;
#[cfg(feature = "mkl")]
mod mkl;
#[cfg(feature = "openblas")]
mod openblas;

pub(crate) mod r#impl;
pub(crate) mod r#macro;
pub(crate) mod simd;

pub use r#impl::*;
pub use simd::*;

#[cfg(any(feature = "accelerate", feature = "mkl", feature = "openblas"))]
pub(crate) mod cblas_consts {
    pub(crate) const CBLAS_ROW_MAJOR: std::ffi::c_int = 101;
    pub(crate) const CBLAS_NO_TRANS: std::ffi::c_int = 111;
}

/// accelerate > mkl > openblas > fallback
#[cfg(feature = "accelerate")]
pub type Backend = accelerate::AccelerateBackend;

#[cfg(all(not(feature = "accelerate"), feature = "mkl"))]
pub type Backend = mkl::MklBackend;

#[cfg(all(
    not(any(feature = "accelerate", feature = "mkl")),
    feature = "openblas"
))]
pub type Backend = openblas::OpenBlasBackend;

#[cfg(not(any(feature = "accelerate", feature = "mkl", feature = "openblas")))]
pub type Backend = fallback::FallbackBackend;

use std::sync::LazyLock;

#[cfg(feature = "accelerate")]
fn create_backend() -> Backend {
    crate::backend::accelerate::AccelerateBackend
}
#[cfg(all(not(feature = "accelerate"), feature = "mkl"))]
fn create_backend() -> Backend {
    crate::backend::mkl::MklBackend
}
#[cfg(all(
    not(any(feature = "accelerate", feature = "mkl")),
    feature = "openblas"
))]
fn create_backend() -> Backend {
    crate::backend::openblas::OpenBlasBackend
}
#[cfg(not(any(feature = "accelerate", feature = "mkl", feature = "openblas")))]
fn create_backend() -> Backend {
    crate::backend::fallback::FallbackBackend
}

/// Global backend instance
///
/// This provides a singleton backend instance that can be used throughout
/// the application. The backend is initialized lazily on first use.
static GLOBAL_BACKEND: LazyLock<Backend> = LazyLock::new(create_backend);

/// Get a reference to the global backend instance
#[inline(always)]
pub fn global_backend() -> &'static Backend {
    &GLOBAL_BACKEND
}
