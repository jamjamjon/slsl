#![allow(dead_code)]
//! SIMD configuration based on target architecture and runtime detection
//! This provides compile-time and runtime SIMD width detection for optimal performance
use wide::{f32x8, f64x4};

// use std::sync::LazyLock;

// /// Runtime SIMD capabilities detection
// #[derive(Debug, Clone, Copy)]
// pub(crate) struct SimdCapabilities {
//     pub has_avx512: bool,
//     pub has_avx2: bool,
//     pub has_avx: bool,
//     pub has_sse4_1: bool,
//     pub has_neon: bool,
// }

// impl SimdCapabilities {
//     /// Detect SIMD capabilities at runtime
//     pub fn detect() -> Self {
//         #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
//         {
//             Self {
//                 has_avx512: is_x86_feature_detected!("avx512f"),
//                 has_avx2: is_x86_feature_detected!("avx2"),
//                 has_avx: is_x86_feature_detected!("avx"),
//                 has_sse4_1: is_x86_feature_detected!("sse4.1"),
//                 has_neon: false,
//             }
//         }
//         #[cfg(target_arch = "aarch64")]
//         {
//             Self {
//                 has_avx512: false,
//                 has_avx2: false,
//                 has_avx: false,
//                 has_sse4_1: false,
//                 has_neon: std::arch::is_aarch64_feature_detected!("neon"),
//             }
//         }
//         #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
//         {
//             Self {
//                 has_avx512: false,
//                 has_avx2: false,
//                 has_avx: false,
//                 has_sse4_1: false,
//                 has_neon: false,
//             }
//         }
//     }

//     /// Get optimal f32 SIMD width based on capabilities
//     pub fn optimal_f32_width(&self) -> usize {
//         if self.has_avx512 {
//             16 // AVX-512 supports 16x f32
//         } else if self.has_avx2 || self.has_avx {
//             8 // AVX/AVX2 supports 8x f32
//         } else if self.has_sse4_1 || self.has_neon {
//             4 // SSE4.1 and NEON both support 4x f32
//         } else {
//             1 // Scalar fallback
//         }
//     }

//     /// Get optimal f64 SIMD width based on capabilities
//     pub fn optimal_f64_width(&self) -> usize {
//         if self.has_avx512 {
//             8 // AVX-512 supports 8x f64
//         } else if self.has_avx2 || self.has_avx {
//             4 // AVX/AVX2 supports 4x f64
//         } else if self.has_sse4_1 || self.has_neon {
//             2 // SSE4.1 and NEON both support 2x f64
//         } else {
//             1 // Scalar fallback
//         }
//     }
// }

// /// Global SIMD capabilities instance
// static SIMD_CAPS: LazyLock<SimdCapabilities> = LazyLock::new(SimdCapabilities::detect);

// /// Get the global SIMD capabilities
// pub(crate) fn capabilities() -> &'static SimdCapabilities {
//     &SIMD_CAPS
// }

// /// Helper function to choose optimal SIMD width at runtime
// pub(crate) fn choose_f32_simd_width(data_size: usize) -> usize {
//     let caps = capabilities();
//     let optimal = caps.optimal_f32_width();

//     // Choose the largest SIMD width that efficiently processes the data
//     if data_size >= optimal && optimal >= 8 {
//         8
//     } else if data_size >= 4 {
//         4
//     } else {
//         1 // Scalar fallback for small data
//     }
// }

// /// Helper function to choose optimal f64 SIMD width at runtime
// pub(crate) fn choose_f64_simd_width(data_size: usize) -> usize {
//     let caps = capabilities();
//     let optimal = caps.optimal_f64_width();

//     // Choose the largest SIMD width that efficiently processes the data
//     if data_size >= optimal && optimal >= 4 {
//         4
//     } else if data_size >= 2 {
//         2
//     } else {
//         1 // Scalar fallback for small data
//     }
// }

pub trait WideSimd: Copy {
    type Element: crate::TensorElement + num_traits::Float;
    type Array: AsRef<[Self::Element]> + AsMut<[Self::Element]> + Copy;
    const LANE: usize;

    const ONE: Self;
    const ZERO: Self;
    const HALF: Self;
    const NEG_INFINITY: Self;
    const INFINITY: Self;

    fn new(arr: Self::Array) -> Self;
    fn from_slice(slice: &[Self::Element]) -> Self;

    /// Load from potentially unaligned memory without intermediate copy
    /// # Safety
    /// slice must have at least LANE elements
    unsafe fn from_slice_unaligned(slice: &[Self::Element]) -> Self {
        debug_assert!(slice.len() >= Self::LANE);
        let ptr = slice.as_ptr() as *const Self::Array;
        Self::new(ptr.read_unaligned())
    }

    fn splat(val: Self::Element) -> Self;
    fn to_array(self) -> Self::Array;
    fn as_array_ref(&self) -> &Self::Array;
    fn as_array_mut(&mut self) -> &mut Self::Array;
    fn abs(self) -> Self;
    fn max(self, rhs: Self) -> Self;
    fn min(self, rhs: Self) -> Self;
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
    fn recip(self) -> Self;
    fn recip_sqrt(self) -> Self;
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn div(self, rhs: Self) -> Self;
    fn sum(self) -> Self::Element
    where
        Self::Element: std::iter::Sum<Self::Element>,
    {
        self.as_array_ref().as_ref().iter().copied().sum()
    }
}

impl WideSimd for f32x8 {
    type Element = f32;
    type Array = [f32; 8];
    const LANE: usize = 8;
    const ONE: Self = Self::ONE;
    const ZERO: Self = Self::ZERO;
    const HALF: Self = Self::HALF;
    const NEG_INFINITY: Self = Self::new([f32::NEG_INFINITY; 8]);
    const INFINITY: Self = Self::new([f32::INFINITY; 8]);

    #[inline(always)]
    fn new(arr: Self::Array) -> Self {
        f32x8::new(arr)
    }

    #[inline(always)]
    fn from_slice(slice: &[f32]) -> Self {
        let mut arr = [0.0f32; 8];
        arr.copy_from_slice(&slice[..8]);
        f32x8::new(arr)
    }

    #[inline(always)]
    fn splat(val: f32) -> Self {
        f32x8::splat(val)
    }

    #[inline(always)]
    fn to_array(self) -> Self::Array {
        *self.as_array_ref()
    }

    #[inline(always)]
    fn as_array_ref(&self) -> &Self::Array {
        self.as_array_ref()
    }

    #[inline(always)]
    fn as_array_mut(&mut self) -> &mut Self::Array {
        self.as_array_mut()
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        f32x8::exp(self)
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        f32x8::sqrt(self)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        f32x8::recip(self)
    }

    #[inline(always)]
    fn recip_sqrt(self) -> Self {
        f32x8::recip_sqrt(self)
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        self / rhs
    }
}

impl WideSimd for f64x4 {
    type Element = f64;
    type Array = [f64; 4];
    const LANE: usize = 4;

    const ONE: Self = Self::ONE;
    const ZERO: Self = Self::ZERO;
    const HALF: Self = Self::HALF;
    const NEG_INFINITY: Self = Self::new([f64::NEG_INFINITY; 4]);
    const INFINITY: Self = Self::new([f64::INFINITY; 4]);

    #[inline(always)]
    fn new(arr: Self::Array) -> Self {
        f64x4::new(arr)
    }

    #[inline(always)]
    fn from_slice(slice: &[f64]) -> Self {
        let mut arr = [0.0f64; 4];
        arr.copy_from_slice(&slice[..4]);
        f64x4::new(arr)
    }

    #[inline(always)]
    fn splat(val: f64) -> Self {
        f64x4::splat(val)
    }

    #[inline(always)]
    fn to_array(self) -> Self::Array {
        *self.as_array_ref()
    }

    #[inline(always)]
    fn as_array_ref(&self) -> &Self::Array {
        self.as_array_ref()
    }

    #[inline(always)]
    fn as_array_mut(&mut self) -> &mut Self::Array {
        self.as_array_mut()
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        f64x4::exp(self)
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        f64x4::sqrt(self)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        Self::splat(1.0) / self
    }

    #[inline(always)]
    fn recip_sqrt(self) -> Self {
        Self::splat(1.0) / f64x4::sqrt(self)
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        self / rhs
    }
}
