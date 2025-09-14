#![allow(clippy::too_many_arguments)]

use crate::backend::simd::WideSimd;
use simsimd::SpatialSimilarity;

/// OpsTrait trait for accelerated tensor operations
///
/// This trait provides a unified interface for various mathematical backends
/// including Accelerate (macOS), Intel MKL, OpenBLAS, and fallback implementations.
pub trait OpsTrait: Send + Sync {
    crate::impl_v_abs!(f32, f64, i8, i16, i32, i64);
    crate::impl_v_abs_half!();
    crate::impl_v_sin!(f32, f64);
    crate::impl_v_sin_half!();

    crate::impl_v_cos!(f32, f64);
    crate::impl_v_cos_half!();

    crate::impl_v_tanh!(f32, f64);
    crate::impl_v_tanh_half!();

    crate::impl_v_exp!(f32, f64);
    crate::impl_v_exp_half!();

    crate::impl_v_log!(f32, f64);
    crate::impl_v_log_half!();

    crate::impl_v_sqrt!(f32, f64);
    crate::impl_v_sqrt_half!();

    crate::impl_v_sqr!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_v_sqr_half!();

    crate::impl_v_tan!(f32, f64);
    crate::impl_v_tan_half!();

    crate::impl_v_recip!(f32, f64);
    crate::impl_v_recip_half!();

    crate::impl_v_floor!(f32, f64);
    crate::impl_v_floor_half!();

    crate::impl_v_ceil!(f32, f64);
    crate::impl_v_ceil_half!();

    crate::impl_v_round!(f32, f64);
    crate::impl_v_round_half!();
    crate::impl_v_neg!(i8, i16, i32, i64, f32, f64);
    crate::impl_v_neg_half!();

    crate::impl_scal!(f32, f64);

    crate::impl_gemm_sd!(f32, f64);
    crate::impl_gemm!(i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_gemm_half!();

    crate::impl_v_add!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_v_add_half!();
    crate::impl_v_sub!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_v_sub_half!();
    crate::impl_v_mul!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_v_mul_half!();
    crate::impl_v_div!(f32, f64);
    crate::impl_v_div_half!();
    crate::impl_v_add_scalar!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_v_add_scalar_half!();
    crate::impl_v_sub_scalar!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_v_sub_scalar_half!();
    crate::impl_v_mul_scalar!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_v_mul_scalar_half!();
    crate::impl_v_div_scalar!(f32, f64);
    crate::impl_v_div_scalar_half!();

    crate::impl_asum_signed!(f32, f64, i8, i16, i32, i64);
    crate::impl_asum_unsigned!(u8, u16, u32, u64);
    crate::impl_asum_half!();

    crate::impl_v_pow!(f32, f64);

    // Sum operations - implemented directly with SIMD optimization
    /// SIMD-optimized sum function for f32 arrays
    /// Uses f32x8 vectors with reduce_add for optimal performance
    #[inline(always)]
    fn sum_f32(&self, x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }

        // Use SIMD optimization for larger arrays
        if x.len() >= 32 {
            use wide::f32x8;

            let chunks = x.chunks_exact(8);
            let remainder = chunks.remainder();

            // Process chunks of 8 elements using SIMD
            let mut sum_vec = f32x8::splat(0.0);
            for chunk in chunks {
                let vec = f32x8::from_slice(chunk);
                sum_vec += vec;
            }

            // Use reduce_add to sum all elements in the vector
            let mut result = sum_vec.reduce_add();

            // Process remaining elements
            for &val in remainder {
                result += val;
            }

            return result;
        }

        // Fallback to standard sum for small arrays
        x.iter().sum()
    }

    /// SIMD-optimized sum function for f64 arrays
    /// Uses f64x4 vectors with reduce_add for optimal performance
    #[inline(always)]
    fn sum_f64(&self, x: &[f64]) -> f64 {
        if x.is_empty() {
            return 0.0;
        }

        // Use SIMD optimization for larger arrays
        if x.len() >= 16 {
            use wide::f64x4;

            let chunks = x.chunks_exact(4);
            let remainder = chunks.remainder();

            // Process chunks of 4 elements using SIMD
            let mut sum_vec = f64x4::splat(0.0);
            for chunk in chunks {
                let vec = f64x4::from_slice(chunk);
                sum_vec += vec;
            }

            // Use reduce_add to sum all elements in the vector
            let mut result = sum_vec.reduce_add();

            // Process remaining elements
            for &val in remainder {
                result += val;
            }

            return result;
        }

        // Fallback to standard sum for small arrays
        x.iter().sum()
    }

    #[inline(always)]
    fn sum_f16(&self, x: &[half::f16]) -> f64 {
        if x.is_empty() {
            return 0.0f64;
        }
        f64::from(x.iter().sum::<half::f16>())
    }

    #[inline(always)]
    fn sum_bf16(&self, x: &[half::bf16]) -> f64 {
        if x.is_empty() {
            return 0.0f64;
        }
        f64::from(x.iter().sum::<half::bf16>())
    }
    crate::impl_sum_int!(
        u8 => u64,
        i8 => i64,
        u16 => u64,
        i16 => i64,
        u32 => u64,
        i32 => i64,
        u64 => u128,
        i64 => i128
    );

    // Mean operations - implemented using macros (sum / n)
    crate::impl_mean_float!(f32, f64);
    crate::impl_mean_int!(i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_mean_half!();

    #[inline(always)]
    fn nrm2_f64(&self, a: &[f64]) -> f64 {
        self.dot_f64(a, a).sqrt()
    }

    #[inline(always)]
    fn nrm2_f32(&self, a: &[f32]) -> f64 {
        self.dot_f32(a, a).sqrt()
    }

    #[inline(always)]
    fn nrm2_f16(&self, a: &[half::f16]) -> f64 {
        self.dot_f16(a, a).sqrt()
    }

    #[inline(always)]
    fn nrm2_bf16(&self, a: &[half::bf16]) -> f64 {
        self.dot_bf16(a, a).sqrt()
    }

    // ReLU activation function - implemented using macros
    crate::impl_relu!(f32, f64);
    crate::impl_relu_int!(i8, i16, i32, i64);
    crate::impl_relu_uint!(u8, u16, u32, u64);
    crate::impl_relu_half!();

    // Vector min/max operations - implemented using macros
    crate::impl_min_v!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_min_v_half!();
    crate::impl_max_v!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_max_v_half!();
    crate::impl_min_vi!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_min_vi_half!();
    crate::impl_max_vi!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_max_vi_half!();

    // Vector min/max index operations - implemented using macros
    crate::impl_min_i!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_min_i_half!();
    crate::impl_max_i!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_max_i_half!();

    // Vector min/max combined operations - implemented using macros
    crate::impl_min_max_v!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_min_max_v_half!();
    crate::impl_min_max_vi!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_min_max_vi_half!();
    crate::impl_min_max_i!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
    crate::impl_min_max_i_half!();

    // dot
    crate::impl_dot!(i16, i32, i64, u8, u16, u32, u64);

    #[inline(always)]
    fn dot_f64(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(
            a.len(),
            b.len(),
            "Input and output slices must have same length"
        );
        f64::dot(a, b).expect("simsimd f64 dot product failed")
    }

    #[inline(always)]
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(
            a.len(),
            b.len(),
            "Input and output slices must have same length"
        );
        f32::dot(a, b).expect("simsimd f32 dot product failed")
    }

    #[inline(always)]
    fn dot_i8(&self, a: &[i8], b: &[i8]) -> f64 {
        assert_eq!(
            a.len(),
            b.len(),
            "Input and output slices must have same length"
        );
        i8::dot(a, b).expect("simsimd i8 dot product failed")
    }

    #[inline(always)]
    fn dot_f16(&self, a: &[half::f16], b: &[half::f16]) -> f64 {
        assert_eq!(
            a.len(),
            b.len(),
            "Input and output slices must have same length"
        );
        let a_simd: &[simsimd::f16] = unsafe { std::mem::transmute(a) };
        let b_simd: &[simsimd::f16] = unsafe { std::mem::transmute(b) };
        simsimd::f16::dot(a_simd, b_simd).expect("simsimd f16 dot product failed")
    }

    fn dot_bf16(&self, a: &[half::bf16], b: &[half::bf16]) -> f64 {
        assert_eq!(
            a.len(),
            b.len(),
            "Input and output slices must have same length"
        );
        let a_simd: &[simsimd::bf16] = unsafe { std::mem::transmute(a) };
        let b_simd: &[simsimd::bf16] = unsafe { std::mem::transmute(b) };
        simsimd::bf16::dot(a_simd, b_simd).expect("simsimd bf16 dot product failed")
    }

    // Sigmoid activation function - most efficient implementation, single pass + vectorized
    #[inline(always)]
    fn sigmoid_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );

        // Method 1: Numerically stable implementation (recommended for production)
        // Use piecewise function to avoid numerical issues
        for (i, &xi) in x.iter().enumerate() {
            if xi >= 0.0 {
                // x >= 0: 1 / (1 + exp(-x))
                // When x is large, exp(-x) ≈ 0, so result ≈ 1
                if xi > 16.0 {
                    out[i] = 1.0;
                } else {
                    let exp_neg_x = (-xi).exp();
                    out[i] = 1.0 / (1.0 + exp_neg_x);
                }
            } else {
                // x < 0: exp(x) / (1 + exp(x))
                // When x is small, exp(x) ≈ 0, so result ≈ 0
                if xi < -16.0 {
                    out[i] = 0.0;
                } else {
                    let exp_x = xi.exp();
                    out[i] = exp_x / (1.0 + exp_x);
                }
            }
        }
    }

    #[inline(always)]
    fn sigmoid_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );

        // Use numerically stable implementation: 1 / (1 + exp(-x))
        // For x >= 0: 1 / (1 + exp(-x))
        // For x < 0:  exp(x) / (1 + exp(x))
        for (i, &xi) in x.iter().enumerate() {
            if xi >= 0.0 {
                // x >= 0: 1 / (1 + exp(-x))
                let exp_neg_x = (-xi).exp();
                out[i] = 1.0 / (1.0 + exp_neg_x);
            } else {
                // x < 0: exp(x) / (1 + exp(x))
                let exp_x = xi.exp();
                out[i] = exp_x / (1.0 + exp_x);
            }
        }
    }

    #[inline(always)]
    fn sigmoid_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );

        // Use numerically stable implementation, avoid temporary vector allocation
        for (i, &xi) in x.iter().enumerate() {
            if xi >= half::f16::ZERO {
                // x >= 0: 1 / (1 + exp(-x))
                let exp_neg_xi = half::f16::from_f32((-xi.to_f32()).exp());
                out[i] = half::f16::ONE / (half::f16::ONE + exp_neg_xi);
            } else {
                // x < 0: exp(x) / (1 + exp(x))
                let exp_xi = half::f16::from_f32(xi.to_f32().exp());
                out[i] = exp_xi / (half::f16::ONE + exp_xi);
            }
        }
    }

    #[inline(always)]
    fn sigmoid_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );

        // Use numerically stable implementation, avoid temporary vector allocation
        for (i, &xi) in x.iter().enumerate() {
            if xi >= half::bf16::ZERO {
                // x >= 0: 1 / (1 + exp(-x))
                let exp_neg_xi = half::bf16::from_f32((-xi.to_f32()).exp());
                out[i] = half::bf16::ONE / (half::bf16::ONE + exp_neg_xi);
            } else {
                // x < 0: exp(x) / (1 + exp(x))
                let exp_xi = half::bf16::from_f32(xi.to_f32().exp());
                out[i] = exp_xi / (half::bf16::ONE + exp_xi);
            }
        }
    }

    // Clamp function - implemented using max and min combination
    #[inline(always)]
    fn clamp_f32(&self, x: &[f32], min: f32, max: f32, out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );

        // Implement clamp: min(max(x, min), max) - single pass
        for (i, xi) in x.iter().enumerate() {
            out[i] = xi.clamp(min, max);
        }
    }

    #[inline(always)]
    fn clamp_f64(&self, x: &[f64], min: f64, max: f64, out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );

        // Implement clamp: min(max(x, min), max) - single pass
        for (i, xi) in x.iter().enumerate() {
            out[i] = xi.clamp(min, max);
        }
    }

    #[inline(always)]
    fn clamp_f16(&self, x: &[half::f16], min: half::f16, max: half::f16, out: &mut [half::f16]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );

        // For f16, convert to f32 for calculation, then convert back to f16 - single pass
        for (i, xi) in x.iter().enumerate() {
            out[i] = xi.clamp(min, max);
        }
    }

    #[inline(always)]
    fn clamp_bf16(
        &self,
        x: &[half::bf16],
        min: half::bf16,
        max: half::bf16,
        out: &mut [half::bf16],
    ) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );

        // For bf16, convert to f32 for calculation, then convert back to bf16 - single pass
        for (i, xi) in x.iter().enumerate() {
            out[i] = xi.clamp(min, max);
        }
    }

    // Clamp methods for integer types
    #[inline(always)]
    fn clamp_i8(&self, x: &[i8], min: i8, max: i8, out: &mut [i8]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        for (i, xi) in x.iter().enumerate() {
            out[i] = (*xi).clamp(min, max);
        }
    }

    #[inline(always)]
    fn clamp_i16(&self, x: &[i16], min: i16, max: i16, out: &mut [i16]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        for (i, xi) in x.iter().enumerate() {
            out[i] = (*xi).clamp(min, max);
        }
    }

    #[inline(always)]
    fn clamp_i32(&self, x: &[i32], min: i32, max: i32, out: &mut [i32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        for (i, xi) in x.iter().enumerate() {
            out[i] = (*xi).clamp(min, max);
        }
    }

    #[inline(always)]
    fn clamp_i64(&self, x: &[i64], min: i64, max: i64, out: &mut [i64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        for (i, xi) in x.iter().enumerate() {
            out[i] = (*xi).clamp(min, max);
        }
    }

    #[inline(always)]
    fn clamp_u8(&self, x: &[u8], min: u8, max: u8, out: &mut [u8]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        for (i, xi) in x.iter().enumerate() {
            out[i] = (*xi).clamp(min, max);
        }
    }

    #[inline(always)]
    fn clamp_u16(&self, x: &[u16], min: u16, max: u16, out: &mut [u16]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        for (i, xi) in x.iter().enumerate() {
            out[i] = (*xi).clamp(min, max);
        }
    }

    #[inline(always)]
    fn clamp_u32(&self, x: &[u32], min: u32, max: u32, out: &mut [u32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        for (i, xi) in x.iter().enumerate() {
            out[i] = (*xi).clamp(min, max);
        }
    }

    #[inline(always)]
    fn clamp_u64(&self, x: &[u64], min: u64, max: u64, out: &mut [u64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        for (i, xi) in x.iter().enumerate() {
            out[i] = (*xi).clamp(min, max);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::global_backend;

    #[test]
    fn test_min_max_operations() {
        // Create a simple test backend (using default implementations)
        let backend = global_backend();

        // Test data
        let f32_data = vec![3.0f32, 1.0f32, 4.0f32, 1.0f32, 5.0f32];
        let i32_data = vec![3i32, 1i32, 4i32, 1i32, 5i32];

        // Test min operations
        assert_eq!(backend.min_v_f32(&f32_data), 1.0f32);
        assert_eq!(backend.min_v_i32(&i32_data), 1i32);

        // Test max operations
        assert_eq!(backend.max_v_f32(&f32_data), 5.0f32);
        assert_eq!(backend.max_v_i32(&i32_data), 5i32);

        // Test min with index
        let (min_val, min_idx) = backend.min_vi_f32(&f32_data);
        assert_eq!(min_val, 1.0f32);
        assert!(min_idx == 1 || min_idx == 3); // 1.0 appears at indices 1 and 3

        // Test max with index
        let (max_val, max_idx) = backend.max_vi_f32(&f32_data);
        assert_eq!(max_val, 5.0f32);
        assert_eq!(max_idx, 4); // 5.0 appears at index 4
    }

    #[test]
    fn test_min_max_all_dtypes() {
        let backend = global_backend();

        // Test all integer types
        let i8_data = vec![3i8, 1i8, 4i8, 1i8, 5i8];
        let i16_data = vec![3i16, 1i16, 4i16, 1i16, 5i16];
        let i32_data = vec![3i32, 1i32, 4i32, 1i32, 5i32];
        let i64_data = vec![3i64, 1i64, 4i64, 1i64, 5i64];
        let u8_data = vec![3u8, 1u8, 4u8, 1u8, 5u8];
        let u16_data = vec![3u16, 1u16, 4u16, 1u16, 5u16];
        let u32_data = vec![3u32, 1u32, 4u32, 1u32, 5u32];
        let u64_data = vec![3u64, 1u64, 4u64, 1u64, 5u64];

        // Test min operations for all integer types
        assert_eq!(backend.min_v_i8(&i8_data), 1i8);
        assert_eq!(backend.min_v_i16(&i16_data), 1i16);
        assert_eq!(backend.min_v_i64(&i64_data), 1i64);
        assert_eq!(backend.min_v_u8(&u8_data), 1u8);
        assert_eq!(backend.min_v_u16(&u16_data), 1u16);
        assert_eq!(backend.min_v_u32(&u32_data), 1u32);
        assert_eq!(backend.min_v_u64(&u64_data), 1u64);

        // Test max operations for all integer types
        assert_eq!(backend.max_v_i8(&i8_data), 5i8);
        assert_eq!(backend.max_v_i16(&i16_data), 5i16);
        assert_eq!(backend.max_v_i64(&i64_data), 5i64);
        assert_eq!(backend.max_v_u8(&u8_data), 5u8);
        assert_eq!(backend.max_v_u16(&u16_data), 5u16);
        assert_eq!(backend.max_v_u32(&u32_data), 5u32);
        assert_eq!(backend.max_v_u64(&u64_data), 5u64);

        // Test min with index for all integer types
        let (min_val, min_idx) = backend.min_vi_i32(&i32_data);
        assert_eq!(min_val, 1i32);
        assert!(min_idx == 1 || min_idx == 3);

        let (max_val, max_idx) = backend.max_vi_i32(&i32_data);
        assert_eq!(max_val, 5i32);
        assert_eq!(max_idx, 4);
    }

    #[test]
    fn test_min_max_floating_point() {
        let backend = global_backend();

        // Test floating point types
        let f64_data = vec![3.0f64, 1.0f64, 4.0f64, 1.0f64, 5.0f64];
        let f16_data = vec![
            half::f16::from_f32(3.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(4.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(5.0),
        ];
        let bf16_data = vec![
            half::bf16::from_f32(3.0),
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(4.0),
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(5.0),
        ];

        // Test min operations for floating point types
        assert_eq!(backend.min_v_f64(&f64_data), 1.0f64);
        assert_eq!(backend.min_v_f16(&f16_data), half::f16::from_f32(1.0));
        assert_eq!(backend.min_v_bf16(&bf16_data), half::bf16::from_f32(1.0));

        // Test max operations for floating point types
        assert_eq!(backend.max_v_f64(&f64_data), 5.0f64);
        assert_eq!(backend.max_v_f16(&f16_data), half::f16::from_f32(5.0));
        assert_eq!(backend.max_v_bf16(&bf16_data), half::bf16::from_f32(5.0));

        // Test min with index for floating point types
        let (min_val, min_idx) = backend.min_vi_f64(&f64_data);
        assert_eq!(min_val, 1.0f64);
        assert!(min_idx == 1 || min_idx == 3);

        let (max_val, max_idx) = backend.max_vi_f64(&f64_data);
        assert_eq!(max_val, 5.0f64);
        assert_eq!(max_idx, 4);
    }

    #[test]
    #[should_panic(expected = "Cannot find minimum of empty vector")]
    fn test_min_empty_vector() {
        let backend = global_backend();
        let empty_data: Vec<f32> = vec![];
        backend.min_v_f32(&empty_data);
    }

    #[test]
    #[should_panic(expected = "Cannot find maximum of empty vector")]
    fn test_max_empty_vector() {
        let backend = global_backend();
        let empty_data: Vec<f32> = vec![];
        backend.max_v_f32(&empty_data);
    }

    #[test]
    fn test_min_max_combined_operations() {
        let backend = global_backend();

        // Test data
        let f32_data = vec![3.0f32, 1.0f32, 4.0f32, 1.0f32, 5.0f32];
        let i32_data = vec![3i32, 1i32, 4i32, 1i32, 5i32];

        // Test min_max_v operations
        let (min_val, max_val) = backend.min_max_v_f32(&f32_data);
        assert_eq!(min_val, 1.0f32);
        assert_eq!(max_val, 5.0f32);

        let (min_val, max_val) = backend.min_max_v_i32(&i32_data);
        assert_eq!(min_val, 1i32);
        assert_eq!(max_val, 5i32);

        // Test min_max_vi operations
        let ((min_val, min_idx), (max_val, max_idx)) = backend.min_max_vi_f32(&f32_data);
        assert_eq!(min_val, 1.0f32);
        assert!(min_idx == 1 || min_idx == 3); // 1.0 appears at indices 1 and 3
        assert_eq!(max_val, 5.0f32);
        assert_eq!(max_idx, 4); // 5.0 appears at index 4

        let ((min_val, min_idx), (max_val, max_idx)) = backend.min_max_vi_i32(&i32_data);
        assert_eq!(min_val, 1i32);
        assert!(min_idx == 1 || min_idx == 3);
        assert_eq!(max_val, 5i32);
        assert_eq!(max_idx, 4);

        // Test min_max_i operations
        let (min_idx, max_idx) = backend.min_max_i_f32(&f32_data);
        assert!(min_idx == 1 || min_idx == 3);
        assert_eq!(max_idx, 4);

        let (min_idx, max_idx) = backend.min_max_i_i32(&i32_data);
        assert!(min_idx == 1 || min_idx == 3);
        assert_eq!(max_idx, 4);
    }

    #[test]
    fn test_min_max_combined_all_dtypes() {
        let backend = global_backend();

        // Test all integer types
        let i8_data = vec![3i8, 1i8, 4i8, 1i8, 5i8];
        let u8_data = vec![3u8, 1u8, 4u8, 1u8, 5u8];

        // Test min_max_v for all integer types
        let (min_val, max_val) = backend.min_max_v_i8(&i8_data);
        assert_eq!(min_val, 1i8);
        assert_eq!(max_val, 5i8);

        let (min_val, max_val) = backend.min_max_v_u8(&u8_data);
        assert_eq!(min_val, 1u8);
        assert_eq!(max_val, 5u8);

        // Test min_max_vi for all integer types
        let ((min_val, min_idx), (max_val, max_idx)) = backend.min_max_vi_i8(&i8_data);
        assert_eq!(min_val, 1i8);
        assert!(min_idx == 1 || min_idx == 3);
        assert_eq!(max_val, 5i8);
        assert_eq!(max_idx, 4);

        // Test min_max_i for all integer types
        let (min_idx, max_idx) = backend.min_max_i_i8(&i8_data);
        assert!(min_idx == 1 || min_idx == 3);
        assert_eq!(max_idx, 4);
    }

    #[test]
    fn test_min_max_combined_floating_point() {
        let backend = global_backend();

        // Test floating point types
        let f64_data = vec![3.0f64, 1.0f64, 4.0f64, 1.0f64, 5.0f64];
        let f16_data = vec![
            half::f16::from_f32(3.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(4.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(5.0),
        ];

        // Test min_max_v for floating point types
        let (min_val, max_val) = backend.min_max_v_f64(&f64_data);
        assert_eq!(min_val, 1.0f64);
        assert_eq!(max_val, 5.0f64);

        let (min_val, max_val) = backend.min_max_v_f16(&f16_data);
        assert_eq!(min_val, half::f16::from_f32(1.0));
        assert_eq!(max_val, half::f16::from_f32(5.0));

        // Test min_max_vi for floating point types
        let ((min_val, min_idx), (max_val, max_idx)) = backend.min_max_vi_f64(&f64_data);
        assert_eq!(min_val, 1.0f64);
        assert!(min_idx == 1 || min_idx == 3);
        assert_eq!(max_val, 5.0f64);
        assert_eq!(max_idx, 4);

        // Test min_max_i for floating point types
        let (min_idx, max_idx) = backend.min_max_i_f64(&f64_data);
        assert!(min_idx == 1 || min_idx == 3);
        assert_eq!(max_idx, 4);
    }

    #[test]
    #[should_panic(expected = "Cannot find min/max of empty vector")]
    fn test_min_max_v_empty_vector() {
        let backend = global_backend();
        let empty_data: Vec<f32> = vec![];
        backend.min_max_v_f32(&empty_data);
    }

    #[test]
    #[should_panic(expected = "Cannot find min/max indices of empty vector")]
    fn test_min_max_i_empty_vector() {
        let backend = global_backend();
        let empty_data: Vec<f32> = vec![];
        backend.min_max_i_f32(&empty_data);
    }

    #[test]
    fn test_min_max_index_operations() {
        let backend = global_backend();

        // Test data
        let f32_data = vec![3.0f32, 1.0f32, 4.0f32, 1.0f32, 5.0f32];
        let i32_data = vec![3i32, 1i32, 4i32, 1i32, 5i32];

        // Test min_i operations
        let min_idx = backend.min_i_f32(&f32_data);
        assert!(min_idx == 1 || min_idx == 3); // 1.0 appears at indices 1 and 3

        let min_idx = backend.min_i_i32(&i32_data);
        assert!(min_idx == 1 || min_idx == 3);

        // Test max_i operations
        let max_idx = backend.max_i_f32(&f32_data);
        assert_eq!(max_idx, 4); // 5.0 appears at index 4

        let max_idx = backend.max_i_i32(&i32_data);
        assert_eq!(max_idx, 4);

        // Test with different data to ensure consistency
        let test_data = vec![10.0f32, 20.0f32, 5.0f32, 15.0f32];
        let min_idx = backend.min_i_f32(&test_data);
        assert_eq!(min_idx, 2); // 5.0 is at index 2

        let max_idx = backend.max_i_f32(&test_data);
        assert_eq!(max_idx, 1); // 20.0 is at index 1
    }

    #[test]
    fn test_min_max_index_all_dtypes() {
        let backend = global_backend();

        // Test all integer types
        let i8_data = vec![3i8, 1i8, 4i8, 1i8, 5i8];
        let u8_data = vec![3u8, 1u8, 4u8, 1u8, 5u8];
        let f16_data = vec![
            half::f16::from_f32(3.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(4.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(5.0),
        ];

        // Test min_i for all types
        let min_idx = backend.min_i_i8(&i8_data);
        assert!(min_idx == 1 || min_idx == 3);

        let min_idx = backend.min_i_u8(&u8_data);
        assert!(min_idx == 1 || min_idx == 3);

        let min_idx = backend.min_i_f16(&f16_data);
        assert!(min_idx == 1 || min_idx == 3);

        // Test max_i for all types
        let max_idx = backend.max_i_i8(&i8_data);
        assert_eq!(max_idx, 4);

        let max_idx = backend.max_i_u8(&u8_data);
        assert_eq!(max_idx, 4);

        let max_idx = backend.max_i_f16(&f16_data);
        assert_eq!(max_idx, 4);
    }

    #[test]
    #[should_panic(expected = "Cannot find minimum index of empty vector")]
    fn test_min_i_empty_vector() {
        let backend = global_backend();
        let empty_data: Vec<f32> = vec![];
        backend.min_i_f32(&empty_data);
    }

    #[test]
    #[should_panic(expected = "Cannot find maximum index of empty vector")]
    fn test_max_i_empty_vector() {
        let backend = global_backend();
        let empty_data: Vec<f32> = vec![];
        backend.max_i_f32(&empty_data);
    }

    #[test]
    fn test_dot_product_operations() {
        // struct TestBackend;
        // impl OpsTrait for TestBackend {}

        // let backend = TestBackend;
        let backend = global_backend();

        // Test f32 dot product
        let a_f32 = vec![1.0f32, 2.0, 3.0];
        let b_f32 = vec![4.0f32, 5.0, 6.0];
        let result_f32 = backend.dot_f32(&a_f32, &b_f32);
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // 32.0
        assert_eq!(result_f32, expected);

        // Test f64 dot product
        let a_f64 = vec![1.0f64, 2.0, 3.0];
        let b_f64 = vec![4.0f64, 5.0, 6.0];
        let result_f64 = backend.dot_f64(&a_f64, &b_f64);
        assert_eq!(result_f64, 32.0);

        // Test integer dot product
        let a_i32 = vec![1i32, 2, 3];
        let b_i32 = vec![4i32, 5, 6];
        let result_i32 = backend.dot_i32(&a_i32, &b_i32);
        assert_eq!(result_i32, 32.0);
    }

    #[test]
    fn test_matrix_multiplication() {
        let backend = global_backend();

        // Test f32 gemm: 2x3 * 3x2 = 2x2
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2 matrix
        let mut c = vec![0.0f32; 4]; // 2x2 result matrix

        unsafe {
            backend.gemm_f32(2, 2, 3, a.as_ptr(), 3, b.as_ptr(), 2, c.as_mut_ptr(), 2);
        }

        // Expected result:
        // [1*7 + 2*9 + 3*11,  1*8 + 2*10 + 3*12]   = [58,  64]
        // [4*7 + 5*9 + 6*11,  4*8 + 5*10 + 6*12]   = [139, 154]
        assert_eq!(c[0], 58.0); // (0,0)
        assert_eq!(c[1], 64.0); // (0,1)
        assert_eq!(c[2], 139.0); // (1,0)
        assert_eq!(c[3], 154.0); // (1,1)
    }

    #[test]
    fn test_mathematical_functions() {
        let backend = global_backend();
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];

        // Test exp
        backend.v_exp_f32(&input, &mut output);
        assert!((output[0] - 1.0f32.exp()).abs() < 1e-6);
        assert!((output[1] - 2.0f32.exp()).abs() < 1e-6);

        // Test sin
        backend.v_sin_f32(&input, &mut output);
        assert!((output[0] - 1.0f32.sin()).abs() < 1e-6);
        assert!((output[1] - 2.0f32.sin()).abs() < 1e-6);

        // Test sqrt
        backend.v_sqrt_f32(&input, &mut output);
        assert!((output[0] - 1.0f32.sqrt()).abs() < 1e-6);
        assert!((output[1] - 2.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_unary_operations() {
        let backend = global_backend();
        let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0f32; 5];

        // Test absolute value
        backend.v_abs_f32(&input, &mut output);
        assert_eq!(output, vec![2.0, 1.0, 0.0, 1.0, 2.0]);

        // Test negation
        backend.v_neg_f32(&input, &mut output);
        assert_eq!(output, vec![2.0, 1.0, 0.0, -1.0, -2.0]);

        // Test square root (with positive values)
        let positive_input = vec![0.0f32, 1.0, 4.0, 9.0, 16.0];
        backend.v_sqrt_f32(&positive_input, &mut output);
        assert_eq!(output, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_relu_activation() {
        let backend = global_backend();
        let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0f32; 5];

        backend.relu_f32(&input, &mut output);
        assert_eq!(output, vec![0.0, 0.0, 0.0, 1.0, 2.0]);

        // Test integer ReLU
        let input_i32 = vec![0i32, 1, -1];
        let mut output_i32 = vec![0i32; 3];
        backend.relu_i32(&input_i32, &mut output_i32);
        assert_eq!(output_i32[0], 0);
        assert_eq!(output_i32[1], 1);
        assert_eq!(output_i32[2], 0);
    }

    #[test]
    fn test_sigmoid_activation() {
        let backend = global_backend();
        let input = vec![0.0f32, 1.0, -1.0];
        let mut output = vec![0.0f32; 3];

        backend.sigmoid_f32(&input, &mut output);

        // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - (1.0 / (1.0 + (-1.0f32).exp()))).abs() < 1e-6);
        assert!((output[2] - (1.0 / (1.0 + (1.0f32).exp()))).abs() < 1e-6);
    }

    #[test]
    fn test_clamp_function() {
        let backend = global_backend();
        let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0f32; 5];
        let min = -0.5f32;
        let max = 1.5f32;

        backend.clamp_f32(&input, min, max, &mut output);

        // clamp to [-0.5, 1.5]
        assert_eq!(output[0], -0.5); // -2.0 -> -0.5
        assert_eq!(output[1], -0.5); // -1.0 -> -0.5
        assert_eq!(output[2], 0.0); // 0.0 -> 0.0
        assert_eq!(output[3], 1.0); // 1.0 -> 1.0
        assert_eq!(output[4], 1.5); // 2.0 -> 1.5
    }

    #[test]
    fn test_asum_operations() {
        let backend = global_backend();

        // Test f32 asum
        let a_f32 = vec![1.0f32, -2.0, 3.0, -4.0];
        let asum_f32 = backend.asum_f32(&a_f32);
        assert_eq!(asum_f32, 10.0); // |1| + |-2| + |3| + |-4| = 1 + 2 + 3 + 4 = 10

        // Test f64 asum
        let a_f64 = vec![1.0f64, -2.0, 3.0, -4.0];
        let asum_f64 = backend.asum_f64(&a_f64);
        assert_eq!(asum_f64, 10.0);

        // Test i32 asum
        let a_i32 = vec![1i32, -2, 3, -4];
        let asum_i32 = backend.asum_i32(&a_i32);
        assert_eq!(asum_i32, 10);

        // Test u32 asum (unsigned, so abs is not needed)
        let a_u32 = vec![1u32, 2, 3, 4];
        let asum_u32 = backend.asum_u32(&a_u32);
        assert_eq!(asum_u32, 10);

        // Test f16 asum
        let a_f16 = vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(-2.0),
            half::f16::from_f32(3.0),
        ];
        let asum_f16 = backend.asum_f16(&a_f16);
        assert!((asum_f16 - 6.0).abs() < 1e-3);

        // Test bf16 asum
        let a_bf16 = vec![
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(-2.0),
            half::bf16::from_f32(3.0),
        ];
        let asum_bf16 = backend.asum_bf16(&a_bf16);
        assert!((asum_bf16 - 6.0).abs() < 1e-3);
    }

    #[test]
    fn test_binary_operations() {
        let backend = global_backend();

        // Test f32 nrm2
        let a_f32 = vec![3.0f32, 4.0];
        let nrm2_f32 = backend.nrm2_f32(&a_f32);
        assert!((nrm2_f32 - 5.0).abs() < 1e-6); // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5

        // Test f64 nrm2
        let a_f64 = vec![3.0f64, 4.0];
        let nrm2_f64 = backend.nrm2_f64(&a_f64);
        assert!((nrm2_f64 - 5.0).abs() < 1e-12);

        // Test f16 nrm2
        let a_f16 = vec![half::f16::from_f32(3.0), half::f16::from_f32(4.0)];
        let nrm2_f16 = backend.nrm2_f16(&a_f16);
        assert!((nrm2_f16 - 5.0).abs() < 1e-3);

        // Test bf16 nrm2
        let a_bf16 = vec![half::bf16::from_f32(3.0), half::bf16::from_f32(4.0)];
        let nrm2_bf16 = backend.nrm2_bf16(&a_bf16);
        assert!((nrm2_bf16 - 5.0).abs() < 1e-3);

        // Test edge case: empty vector
        let empty_f32: Vec<f32> = vec![];
        let nrm2_empty = backend.nrm2_f32(&empty_f32);
        assert_eq!(nrm2_empty, 0.0);

        // Test edge case: single element
        let single_f32 = vec![5.0f32];
        let nrm2_single = backend.nrm2_f32(&single_f32);
        assert_eq!(nrm2_single, 5.0);
    }

    #[test]
    fn test_optimized_nrm2_implementations() {
        let backend = global_backend();

        // Test vectors with known L2 norms
        let test_cases = vec![
            (vec![3.0f32, 4.0], 5.0),            // sqrt(3^2 + 4^2) = 5
            (vec![1.0f32, 1.0, 1.0], 1.7320508), // sqrt(1^2 + 1^2 + 1^2) = sqrt(3) ≈ 1.732
            (vec![5.0f32, 12.0], 13.0),          // sqrt(5^2 + 12^2) = 13 (5-12-13 triangle)
            (vec![0.0f32, 0.0, 0.0], 0.0),       // Zero vector
            (vec![1.0f32], 1.0),                 // Single element
        ];

        for (vector, expected_norm) in test_cases {
            let computed_norm = backend.nrm2_f32(&vector);
            let error = (computed_norm - expected_norm).abs();

            // Allow for floating point precision differences
            assert!(error < 1e-6, "L2 norm calculation error too large: {error}");
        }

        // Test f64 implementation
        let f64_vector = vec![3.0f64, 4.0];
        let f64_norm = backend.nrm2_f64(&f64_vector);
        assert!((f64_norm - 5.0).abs() < 1e-12);

        // Test f16 implementation
        let f16_vector = vec![half::f16::from_f32(3.0), half::f16::from_f32(4.0)];
        let f16_norm = backend.nrm2_f16(&f16_vector);
        assert!((f16_norm - 5.0).abs() < 1e-3);

        // Test bf16 implementation
        let bf16_vector = vec![half::bf16::from_f32(3.0), half::bf16::from_f32(4.0)];
        let bf16_norm = backend.nrm2_bf16(&bf16_vector);
        assert!((bf16_norm - 5.0).abs() < 1e-3);
    }

    #[test]
    fn test_norm_backend_integration() {
        let backend = global_backend();

        // Test L1 norm (norm_l1) - should use backend asum
        let test_vector = vec![3.0f32, -4.0, 5.0, -6.0];
        let l1_norm = backend.asum_f32(&test_vector);
        let expected_l1 = 3.0 + 4.0 + 5.0 + 6.0; // sum of absolute values
        assert!((l1_norm - expected_l1).abs() < 1e-6);

        // Test L2 norm (norm_l2) - should use backend nrm2
        let l2_norm = backend.nrm2_f32(&test_vector);
        let expected_l2 = (3.0f32 * 3.0 + 4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0).sqrt(); // sqrt of sum of squares
        assert!((l2_norm - expected_l2 as f64).abs() < 1e-6);

        // Calculate expected L2 norm with f64 precision for comparison
        let expected_l2_f64 = (3.0f64 * 3.0 + 4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0).sqrt();

        // Test with different data types
        let f64_vector = vec![3.0f64, -4.0, 5.0, -6.0];
        let l1_norm_f64 = backend.asum_f64(&f64_vector);
        let l2_norm_f64 = backend.nrm2_f64(&f64_vector);

        assert!((l1_norm_f64 - expected_l1 as f64).abs() < 1e-12);
        assert!((l2_norm_f64 - expected_l2_f64).abs() < 1e-12);

        // Test with half precision types
        let f16_vector = vec![
            half::f16::from_f32(3.0),
            half::f16::from_f32(-4.0),
            half::f16::from_f32(5.0),
            half::f16::from_f32(-6.0),
        ];
        let l1_norm_f16 = backend.asum_f16(&f16_vector);
        let l2_norm_f16 = backend.nrm2_f16(&f16_vector);

        assert!((l1_norm_f16 - expected_l1).abs() < 1e-3);
        assert!((l2_norm_f16 - expected_l2 as f64).abs() < 1e-3);

        // Test with bf16
        let bf16_vector = vec![
            half::bf16::from_f32(3.0),
            half::bf16::from_f32(-4.0),
            half::bf16::from_f32(5.0),
            half::bf16::from_f32(-6.0),
        ];
        let l1_norm_bf16 = backend.asum_bf16(&bf16_vector);
        let l2_norm_bf16 = backend.nrm2_bf16(&bf16_vector);

        assert!((l1_norm_bf16 - expected_l1).abs() < 1e-3);
        assert!((l2_norm_bf16 - expected_l2 as f64).abs() < 1e-3);

        // Test with integer types for L1 norm
        let i32_vector = vec![3i32, -4, 5, -6];
        let l1_norm_i32 = backend.asum_i32(&i32_vector);
        assert_eq!(l1_norm_i32, 18); // 3 + 4 + 5 + 6

        let u32_vector = vec![3u32, 4, 5, 6];
        let l1_norm_u32 = backend.asum_u32(&u32_vector);
        assert_eq!(l1_norm_u32, 18); // 3 + 4 + 5 + 6 (unsigned, no abs needed)
    }

    #[test]
    fn test_performance_benchmarks() {
        let backend = global_backend();

        // Create test vectors of different sizes
        let small_vec = vec![1.0f32; 100];
        let medium_vec = vec![1.0f32; 1000];
        let large_vec = vec![1.0f32; 10000];

        // Test small vector performance
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = backend.asum_f32(&small_vec);
        }
        let small_time = start.elapsed();

        // Test medium vector performance
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = backend.asum_f32(&medium_vec);
        }
        let medium_time = start.elapsed();

        // Test large vector performance
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = backend.asum_f32(&large_vec);
        }
        let large_time = start.elapsed();

        // Verify correctness
        assert_eq!(backend.asum_f32(&small_vec), 100.0);
        assert_eq!(backend.asum_f32(&medium_vec), 1000.0);
        assert_eq!(backend.asum_f32(&large_vec), 10000.0);

        // Print performance results (optional, for debugging)
        println!("Performance test results:");
        println!("Small vector (100 elements, 1000 iterations): {small_time:?}");
        println!("Medium vector (1000 elements, 100 iterations): {medium_time:?}");
        println!("Large vector (10000 elements, 10 iterations): {large_time:?}");
    }
}
