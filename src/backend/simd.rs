//! SIMD configuration based on target architecture and runtime detection
//! This provides compile-time and runtime SIMD width detection for optimal performance

use std::sync::LazyLock;

/// Runtime SIMD capabilities detection
#[derive(Debug, Clone, Copy)]
pub(crate) struct SimdCapabilities {
    pub has_avx512: bool,
    pub has_avx2: bool,
    pub has_avx: bool,
    pub has_sse4_1: bool,
    pub has_neon: bool,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities at runtime
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            Self {
                has_avx512: is_x86_feature_detected!("avx512f"),
                has_avx2: is_x86_feature_detected!("avx2"),
                has_avx: is_x86_feature_detected!("avx"),
                has_sse4_1: is_x86_feature_detected!("sse4.1"),
                has_neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                has_avx512: false,
                has_avx2: false,
                has_avx: false,
                has_sse4_1: false,
                has_neon: std::arch::is_aarch64_feature_detected!("neon"),
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
        {
            Self {
                has_avx512: false,
                has_avx2: false,
                has_avx: false,
                has_sse4_1: false,
                has_neon: false,
            }
        }
    }

    /// Get optimal f32 SIMD width based on capabilities
    pub fn optimal_f32_width(&self) -> usize {
        if self.has_avx512 {
            16 // AVX-512 supports 16x f32
        } else if self.has_avx2 || self.has_avx {
            8 // AVX/AVX2 supports 8x f32
        } else if self.has_sse4_1 || self.has_neon {
            4 // SSE4.1 and NEON both support 4x f32
        } else {
            1 // Scalar fallback
        }
    }

    /// Get optimal f64 SIMD width based on capabilities
    pub fn optimal_f64_width(&self) -> usize {
        if self.has_avx512 {
            8 // AVX-512 supports 8x f64
        } else if self.has_avx2 || self.has_avx {
            4 // AVX/AVX2 supports 4x f64
        } else if self.has_sse4_1 || self.has_neon {
            2 // SSE4.1 and NEON both support 2x f64
        } else {
            1 // Scalar fallback
        }
    }
}

/// Global SIMD capabilities instance
static SIMD_CAPS: LazyLock<SimdCapabilities> = LazyLock::new(SimdCapabilities::detect);

/// Get the global SIMD capabilities
pub(crate) fn capabilities() -> &'static SimdCapabilities {
    &SIMD_CAPS
}

/// Helper function to choose optimal SIMD width at runtime
pub(crate) fn choose_f32_simd_width(data_size: usize) -> usize {
    let caps = capabilities();
    let optimal = caps.optimal_f32_width();

    // Choose the largest SIMD width that efficiently processes the data
    if data_size >= optimal && optimal >= 8 {
        8
    } else if data_size >= 4 {
        4
    } else {
        1 // Scalar fallback for small data
    }
}

/// Helper function to choose optimal f64 SIMD width at runtime
pub(crate) fn choose_f64_simd_width(data_size: usize) -> usize {
    let caps = capabilities();
    let optimal = caps.optimal_f64_width();

    // Choose the largest SIMD width that efficiently processes the data
    if data_size >= optimal && optimal >= 4 {
        4
    } else if data_size >= 2 {
        2
    } else {
        1 // Scalar fallback for small data
    }
}
