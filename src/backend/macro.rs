//! Macro definitions for backend operations
//!
//! This module contains macros that generate implementations for all supported
//! data types and operations across different backends.

// ========== BLAS Level 1 Operations ==========

/// Generate dot product implementations for integer types that return larger types to avoid overflow
#[macro_export]
macro_rules! impl_dot {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<dot_ $t>](&self, a: &[$t], b: &[$t]) -> f64 {
                    assert_eq!(a.len(), b.len(), "Vector lengths must match for dot product");
                    a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum()
                }
            }
        )+
    };
}

/// Generate scale operations for all numeric types
#[macro_export]
macro_rules! impl_scal {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<scal_ $t>](&self, a: $t, x: &mut [$t]) {
                    for xi in x.iter_mut() {
                        *xi *= a;
                    }
                }
            }
        )+
    };
}

/// Generate L1 norm (sum of absolute values) operations for signed numeric types
#[macro_export]
macro_rules! impl_asum_signed {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<asum_ $t>](&self, x: &[$t]) -> $t {
                    if x.is_empty() {
                        return 0 as $t;
                    }
                    x.iter().map(|xi| (*xi).abs()).sum()
                }
            }
        )+
    };
}

/// Generate L1 norm (sum of absolute values) operations for unsigned numeric types
#[macro_export]
macro_rules! impl_asum_unsigned {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<asum_ $t>](&self, x: &[$t]) -> $t {
                    if x.is_empty() {
                        return 0 as $t;
                    }
                    // For unsigned integers, the value is already non-negative
                    x.iter().sum()
                }
            }
        )+
    };
}

/// Generate L1 norm (sum of absolute values) operations for half precision types
#[macro_export]
macro_rules! impl_asum_half {
    () => {
        #[inline(always)]
        fn asum_f16(&self, x: &[half::f16]) -> f32 {
            if x.is_empty() {
                return 0.0f32;
            }
            x.iter().map(|xi| xi.to_f32().abs()).sum()
        }

        #[inline(always)]
        fn asum_bf16(&self, x: &[half::bf16]) -> f32 {
            if x.is_empty() {
                return 0.0f32;
            }
            x.iter().map(|xi| xi.to_f32().abs()).sum()
        }
    };
}

// ========== BLAS Level 3 Operations ==========

/// Generate general matrix multiplication for all numeric types
/// Generate optimized GEMM implementations for f32 and f64 using unified gemm library
#[macro_export]
macro_rules! impl_gemm_optimized {
    (f32, f64) => {
        /// Performs matrix multiplication for f32: C = A * B
        /// Uses optimized gemm library implementation with dynamic thread configuration
        #[inline(always)]
        /// # Safety
        ///
        /// The caller must ensure that `a`, `b`, and `c` are valid pointers to arrays of the
        /// correct size, and that `m`, `n`, `k`, `lda`, `ldb`, and `ldc` are valid dimensions
        /// and leading dimensions for the matrices involved in the multiplication.
        /// The matrices must not overlap in a way that would cause data races if accessed
        /// concurrently.
        unsafe fn gemm_f32(
            &self,
            m: usize,
            n: usize,
            k: usize,
            a: *const f32,
            lda: usize,
            b: *const f32,
            ldb: usize,
            c: *mut f32,
            ldc: usize,
        ) {
            // Get thread configuration similar to candle
            #[cfg(feature = "rayon")]
            let parallelism = {
                let num_threads = $crate::get_num_threads();
                if num_threads > 1 {
                    gemm::Parallelism::Rayon(num_threads)
                } else {
                    gemm::Parallelism::None
                }
            };
            #[cfg(not(feature = "rayon"))]
            let parallelism = gemm::Parallelism::None;

            // Use unified gemm library for optimized matrix multiplication
            // Formula: dst := alpha×dst + beta×lhs×rhs
            // For C = A * B, we want: C = 0*C + 1*A*B
            // Note: gemm expects (dst, rhs, lhs) parameter order
            gemm::gemm(
                m,
                n,
                k,
                c,
                ldc as isize,
                1,
                false, // C matrix (dst)
                b,
                ldb as isize,
                1, // B matrix (rhs)
                a,
                lda as isize,
                1, // A matrix (lhs)
                0.0f32,
                1.0f32, // alpha=0, beta=1 for C = A*B
                false,
                false,
                false, // conj_dst, conj_lhs, conj_rhs
                parallelism,
            );
        }

        /// Performs matrix multiplication for f64: C = A * B
        /// Uses optimized gemm library implementation with dynamic thread configuration
        #[inline(always)]
        /// # Safety
        ///
        /// The caller must ensure that `a`, `b`, and `c` are valid pointers to arrays of the
        /// correct size, and that `m`, `n`, `k`, `lda`, `ldb`, and `ldc` are valid dimensions
        /// and leading dimensions for the matrices involved in the multiplication.
        /// The matrices must not overlap in a way that would cause data races if accessed
        /// concurrently.
        unsafe fn gemm_f64(
            &self,
            m: usize,
            n: usize,
            k: usize,
            a: *const f64,
            lda: usize,
            b: *const f64,
            ldb: usize,
            c: *mut f64,
            ldc: usize,
        ) {
            // Get thread configuration similar to candle
            #[cfg(feature = "rayon")]
            let parallelism = {
                let num_threads = $crate::get_num_threads();
                if num_threads > 1 {
                    gemm::Parallelism::Rayon(num_threads)
                } else {
                    gemm::Parallelism::None
                }
            };
            #[cfg(not(feature = "rayon"))]
            let parallelism = gemm::Parallelism::None;

            // Use unified gemm library for optimized matrix multiplication
            // Formula: dst := alpha×dst + beta×lhs×rhs
            // For C = A * B, we want: C = 0*C + 1*A*B
            // Note: gemm expects (dst, rhs, lhs) parameter order
            gemm::gemm(
                m,
                n,
                k,
                c,
                ldc as isize,
                1,
                false, // C matrix (dst)
                b,
                ldb as isize,
                1, // B matrix (rhs)
                a,
                lda as isize,
                1, // A matrix (lhs)
                0.0f64,
                1.0f64, // alpha=0, beta=1 for C = A*B
                false,
                false,
                false, // conj_dst, conj_lhs, conj_rhs
                parallelism,
            );
        }
    };
}

/// Generate GEMM implementations for other numeric types (fallback to simple implementation)
#[macro_export]
macro_rules! impl_gemm {
    ($($t:ty),*) => {
        $(
            paste::paste! {
                /// Performs matrix multiplication C = alpha * A * B + beta * C
                ///
                /// # Safety
                ///
                /// This function is unsafe because it performs raw pointer operations.
                /// The caller must ensure:
                /// - All pointers are valid and point to properly allocated memory
                /// - Matrix dimensions are correct (m, n, k)
                /// - Leading dimensions (lda, ldb, ldc) are valid
                /// - Memory regions do not overlap inappropriately
                #[inline(always)]
                unsafe fn [<gemm_ $t>](
                        &self,
                        m: usize,
                        n: usize,
                        k: usize,
                        a: *const $t,
                        lda: usize,
                        b: *const $t,
                        ldb: usize,
                        c: *mut $t,
                        ldc: usize,
                    ) {
                        // Simple matrix multiplication implementation for non-optimized types
                        for i in 0..m {
                            for j in 0..n {
                                let mut sum = 0 as $t;
                                for l in 0..k {
                                    sum += *a.add(i * lda + l) * *b.add(l * ldb + j);
                                }
                                *c.add(i * ldc + j) = sum;
                            }
                        }
                    }
            }
        )+
    };
}

/// Generate general matrix multiplication for half precision types using unified gemm library
#[macro_export]
macro_rules! impl_gemm_half {
    () => {
        /// Performs matrix multiplication C = A * B for f16 matrices
        /// Uses optimized gemm library implementation with dynamic thread configuration
        ///
        /// # Safety
        ///
        /// This function is unsafe because it performs raw pointer operations.
        /// The caller must ensure:
        /// - All pointers are valid and point to properly allocated memory
        /// - Matrix dimensions are correct (m, n, k)
        /// - Leading dimensions (lda, ldb, ldc) are valid
        /// - Memory regions do not overlap inappropriately
        #[inline(always)]
        /// # Safety
        ///
        /// The caller must ensure that `a`, `b`, and `c` are valid pointers to arrays of the
        /// correct size, and that `m`, `n`, `k`, `lda`, `ldb`, and `ldc` are valid dimensions
        /// and leading dimensions for the matrices involved in the multiplication.
        /// The matrices must not overlap in a way that would cause data races if accessed
        /// concurrently.
        unsafe fn gemm_f16(
            &self,
            m: usize,
            n: usize,
            k: usize,
            a: *const half::f16,
            lda: usize,
            b: *const half::f16,
            ldb: usize,
            c: *mut half::f16,
            ldc: usize,
        ) {
            // Get thread configuration similar to candle
            #[cfg(feature = "rayon")]
            let parallelism = {
                let num_threads = $crate::get_num_threads();
                if num_threads > 1 {
                    gemm::Parallelism::Rayon(num_threads)
                } else {
                    gemm::Parallelism::None
                }
            };
            #[cfg(not(feature = "rayon"))]
            let parallelism = gemm::Parallelism::None;

            // Use unified gemm library for optimized f16 matrix multiplication
            // Formula: dst := alpha×dst + beta×lhs×rhs
            // For C = A * B, we want: C = 0*C + 1*A*B
            // Note: gemm expects (dst, rhs, lhs) parameter order
            gemm::gemm(
                m,
                n,
                k,
                c,
                ldc as isize,
                1,
                false, // C matrix (dst)
                b,
                ldb as isize,
                1, // B matrix (rhs)
                a,
                lda as isize,
                1, // A matrix (lhs)
                gemm::f16::ZERO,
                gemm::f16::ONE, // alpha=0, beta=1 for C = A*B
                false,
                false,
                false, // conj_dst, conj_lhs, conj_rhs
                parallelism,
            );
        }

        /// Performs matrix multiplication C = A * B for bf16 matrices
        ///
        /// # Safety
        ///
        /// This function is unsafe because it performs raw pointer operations.
        /// The caller must ensure:
        /// - All pointers are valid and point to properly allocated memory
        /// - Matrix dimensions are correct (m, n, k)
        /// - Leading dimensions (lda, ldb, ldc) are valid
        /// - Memory regions do not overlap inappropriately
        #[inline(always)]
        unsafe fn gemm_bf16(
            &self,
            m: usize,
            n: usize,
            k: usize,
            a: *const half::bf16,
            lda: usize,
            b: *const half::bf16,
            ldb: usize,
            c: *mut half::bf16,
            ldc: usize,
        ) {
            // Simple matrix multiplication implementation for bf16
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        let a_val = (*a.add(i * lda + l)).to_f32();
                        let b_val = (*b.add(l * ldb + j)).to_f32();
                        sum += a_val * b_val;
                    }
                    *c.add(i * ldc + j) = half::bf16::from_f32(sum);
                }
            }
        }
    };
}

// ========== Vectorized Math Functions ==========

/// Generate vectorized exponential function for all numeric types
#[macro_export]
macro_rules! impl_v_exp {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<v_exp_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = (*xi ).exp() ;
                    }
                }
            }
        )+
    };
}

/// Generate vectorized exponential function for f16 and bf16
#[macro_export]
macro_rules! impl_v_exp_half {
    () => {
        #[inline(always)]
        fn v_exp_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32().exp());
            }
        }

        #[inline(always)]
        fn v_exp_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32().exp());
            }
        }
    };
}

/// Generate vectorized sine function for all numeric types
#[macro_export]
macro_rules! impl_v_sin {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<v_sin_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = (*xi).sin() ;
                    }
                }
            }
        )+
    };
}

/// Generate vectorized sine for f16 and bf16
#[macro_export]
macro_rules! impl_v_sin_half {
    () => {
        #[inline(always)]
        fn v_sin_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32().sin());
            }
        }

        #[inline(always)]
        fn v_sin_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32().sin());
            }
        }
    };
}

/// Generate vectorized cosine function for all numeric types
#[macro_export]
macro_rules! impl_v_cos {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<v_cos_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = (*xi  ).cos() ;
                    }
                }
            }
        )+
    };
}

/// Generate vectorized cosine function for f16 and bf16
#[macro_export]
macro_rules! impl_v_cos_half {
    () => {
        #[inline(always)]
        fn v_cos_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32().cos());
            }
        }

        #[inline(always)]
        fn v_cos_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32().cos());
            }
        }
    };
}

/// Generate vectorized hyperbolic tangent for all numeric types
#[macro_export]
macro_rules! impl_v_tanh {
    ($($t:ty),+) => {
        $(
            paste::paste! {
    #[inline(always)]
        fn [<v_tanh_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = (*xi ).tanh();
                    }
                }
            }
        )+
    };
}

/// Generate vectorized hyperbolic tangent for f16 and bf16
#[macro_export]
macro_rules! impl_v_tanh_half {
    () => {
        fn v_tanh_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32().tanh());
            }
        }

        fn v_tanh_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32().tanh());
            }
        }
    };
}

/// Generate vectorized natural logarithm for all numeric types
#[macro_export]
macro_rules! impl_v_log {
    ($($t:ty),+) => {
        $(
            paste::paste! {
    #[inline(always)]
    fn [<v_log_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = (*xi  ).ln();
                    }
                }
            }
        )+
    };
}

/// Generate vectorized natural logarithm for f16 and bf16
#[macro_export]
macro_rules! impl_v_log_half {
    () => {
        #[inline(always)]
        fn v_log_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32().ln());
            }
        }

        #[inline(always)]
        fn v_log_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32().ln());
            }
        }
    };
}

/// Generate vectorized square root for all numeric types
#[macro_export]
macro_rules! impl_v_sqrt {
    ($($t:ty),+) => {
        $(
            paste::paste! {
    #[inline(always)]
    fn [<v_sqrt_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = (*xi ).sqrt() ;
                    }
                }
            }
        )+
    };
}

/// Generate vectorized square root for f16 and bf16
#[macro_export]
macro_rules! impl_v_sqrt_half {
    () => {
        #[inline(always)]
        fn v_sqrt_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32().sqrt());
            }
        }

        #[inline(always)]
        fn v_sqrt_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32().sqrt());
            }
        }
    };
}

/// Generate vectorized element-wise addition for all numeric types
#[macro_export]
macro_rules! impl_v_add {
    ($($t:ty),+) => {
        $(
            paste::paste! {
    #[inline(always)]
    fn [<v_add_ $t>](&self, a: &[$t], b: &[$t], out: &mut [$t]) {
                    assert_eq!(a.len(), b.len(), "Input slices must have same length");
                    assert_eq!(
                        a.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for ((o, ai), bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                        *o = *ai + *bi;
                    }
                }
            }
        )+
    };
}

/// Generate vectorized element-wise addition for half precision types
#[macro_export]
macro_rules! impl_v_add_half {
    () => {
        #[inline(always)]
        fn v_add_f16(&self, a: &[half::f16], b: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(a.len(), b.len(), "Input slices must have same length");
            assert_eq!(
                a.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for ((o, ai), bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                *o = half::f16::from_f32(ai.to_f32() + bi.to_f32());
            }
        }

        #[inline(always)]
        fn v_add_bf16(&self, a: &[half::bf16], b: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(a.len(), b.len(), "Input slices must have same length");
            assert_eq!(
                a.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for ((o, ai), bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                *o = half::bf16::from_f32(ai.to_f32() + bi.to_f32());
            }
        }
    };
}

/// Generate vectorized element-wise subtraction for all numeric types
#[macro_export]
macro_rules! impl_v_sub {
    ($($t:ty),+) => {
        $(
            paste::paste! {
    #[inline(always)]
    fn [<v_sub_ $t>](&self, a: &[$t], b: &[$t], out: &mut [$t]) {
                    assert_eq!(a.len(), b.len(), "Input slices must have same length");
                    assert_eq!(
                        a.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for ((o, ai), bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                        *o = *ai - *bi;
                    }
                }
            }
        )+
    };
}

/// Generate vectorized element-wise subtraction for half precision types
#[macro_export]
macro_rules! impl_v_sub_half {
    () => {
        #[inline(always)]
        fn v_sub_f16(&self, a: &[half::f16], b: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(a.len(), b.len(), "Input slices must have same length");
            assert_eq!(
                a.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for ((o, ai), bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                *o = half::f16::from_f32(ai.to_f32() - bi.to_f32());
            }
        }

        #[inline(always)]
        fn v_sub_bf16(&self, a: &[half::bf16], b: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(a.len(), b.len(), "Input slices must have same length");
            assert_eq!(
                a.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for ((o, ai), bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                *o = half::bf16::from_f32(ai.to_f32() - bi.to_f32());
            }
        }
    };
}

/// Generate vectorized element-wise multiplication for all numeric types
#[macro_export]
macro_rules! impl_v_mul {
    ($($t:ty),+) => {
        $(
            paste::paste! {
    #[inline(always)]
    fn [<v_mul_ $t>](&self, a: &[$t], b: &[$t], out: &mut [$t]) {
                    assert_eq!(a.len(), b.len(), "Input slices must have same length");
                    assert_eq!(
                        a.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for ((o, ai), bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                        *o = *ai * *bi;
                    }
                }
            }
        )+
    };
}

/// Generate vectorized element-wise multiplication for half precision types
#[macro_export]
macro_rules! impl_v_mul_half {
    () => {
        #[inline(always)]
        fn v_mul_f16(&self, a: &[half::f16], b: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(a.len(), b.len(), "Input slices must have same length");
            assert_eq!(
                a.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for ((o, ai), bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                *o = half::f16::from_f32(ai.to_f32() * bi.to_f32());
            }
        }

        #[inline(always)]
        fn v_mul_bf16(&self, a: &[half::bf16], b: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(a.len(), b.len(), "Input slices must have same length");
            assert_eq!(
                a.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for ((o, ai), bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                *o = half::bf16::from_f32(ai.to_f32() * bi.to_f32());
            }
        }
    };
}

/// Generate vectorized element-wise division for all numeric types
#[macro_export]
macro_rules! impl_v_div {
    ($($t:ty),+) => {
        $(
            paste::paste! {
    #[inline(always)]
    fn [<v_div_ $t>](&self, a: &[$t], b: &[$t], out: &mut [$t]) {
                    assert_eq!(a.len(), b.len(), "Input slices must have same length");
                    assert_eq!(
                        a.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for ((o, ai), bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                        *o = *ai / *bi;
                    }
                }
            }
        )+
    };
}

/// Generate vectorized element-wise division for half precision types
#[macro_export]
macro_rules! impl_v_div_half {
    () => {
        #[inline(always)]
        fn v_div_f16(&self, a: &[half::f16], b: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(a.len(), b.len(), "Input slices must have same length");
            assert_eq!(
                a.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for ((o, ai), bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                *o = half::f16::from_f32(ai.to_f32() / bi.to_f32());
            }
        }

        #[inline(always)]
        fn v_div_bf16(&self, a: &[half::bf16], b: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(a.len(), b.len(), "Input slices must have same length");
            assert_eq!(
                a.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for ((o, ai), bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                *o = half::bf16::from_f32(ai.to_f32() / bi.to_f32());
            }
        }
    };
}

/// Generate scalar division operations for all numeric types
#[macro_export]
macro_rules! impl_v_div_scalar {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<v_div_scalar_ $t>](&self, x: &[$t], scalar: $t, out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = *xi / scalar;
                    }
                }
            }
        )+
    };
}

/// Generate scalar division operations for half precision types
#[macro_export]
macro_rules! impl_v_div_scalar_half {
    () => {
        #[inline(always)]
        fn v_div_scalar_f16(&self, x: &[half::f16], scalar: half::f16, out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32() / scalar.to_f32());
            }
        }

        #[inline(always)]
        fn v_div_scalar_bf16(&self, x: &[half::bf16], scalar: half::bf16, out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32() / scalar.to_f32());
            }
        }
    };
}

/// Generate vectorized tangent function for all numeric types
#[macro_export]
macro_rules! impl_v_tan {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<v_tan_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = (*xi ).tan() ;
                    }
                }
            }
        )+
    };
}

/// Generate vectorized tangent function for f16 and bf16
#[macro_export]
macro_rules! impl_v_tan_half {
    () => {
        #[inline(always)]
        fn v_tan_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32().tan());
            }
        }

        #[inline(always)]
        fn v_tan_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32().tan());
            }
        }
    };
}

/// Generate vectorized reciprocal function for all numeric types
#[macro_export]
macro_rules! impl_v_recip {
    ($($t:ty),+) => {
        $(
            paste::paste! {
    #[inline(always)]
    fn [<v_recip_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = (1.0 / (*xi ));
                    }
                }
            }
        )+
    };
}

/// Generate vectorized reciprocal function for f16 and bf16
#[macro_export]
macro_rules! impl_v_recip_half {
    () => {
        #[inline(always)]
        fn v_recip_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(1.0 / xi.to_f32());
            }
        }

        #[inline(always)]
        fn v_recip_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(1.0 / xi.to_f32());
            }
        }
    };
}

/// Generate vectorized floor function for all numeric types
#[macro_export]
macro_rules! impl_v_floor {
    ($($t:ty),+) => {
        $(
            paste::paste! {
    #[inline(always)]
    fn [<v_floor_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = (*xi ).floor() ;
                    }
                }
            }
        )+
    };
}

/// Generate vectorized floor function for f16 and bf16
#[macro_export]
macro_rules! impl_v_floor_half {
    () => {
        #[inline(always)]
        fn v_floor_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32().floor());
            }
        }

        #[inline(always)]
        fn v_floor_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32().floor());
            }
        }
    };
}

/// Generate vectorized ceiling function for all numeric types
#[macro_export]
macro_rules! impl_v_ceil {
    ($($t:ty),+) => {
        $(
            paste::paste! {
    #[inline(always)]
    fn [<v_ceil_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = (*xi  ).ceil();
                    }
                }
            }
        )+
    };
}

/// Generate vectorized ceiling function for f16 and bf16
#[macro_export]
macro_rules! impl_v_ceil_half {
    () => {
        #[inline(always)]
        fn v_ceil_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32().ceil());
            }
        }

        #[inline(always)]
        fn v_ceil_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32().ceil());
            }
        }
    };
}

/// Generate vectorized round function for all numeric types
#[macro_export]
macro_rules! impl_v_round {
    ($($t:ty),+) => {
        $(
            paste::paste! {
    #[inline(always)]
    fn [<v_round_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = (*xi ).round() ;
                    }
                }
            }
        )+
    };
}

/// Generate vectorized round function for f16 and bf16
#[macro_export]
macro_rules! impl_v_round_half {
    () => {
        #[inline(always)]
        fn v_round_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32().round());
            }
        }

        #[inline(always)]
        fn v_round_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32().round());
            }
        }
    };
}

/// Generate vectorized absolute value for all numeric types
#[macro_export]
macro_rules! impl_v_abs {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<v_abs_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = (*xi).abs();
                    }
                }
            }
        )+
    };
}

/// Generate vectorized absolute value for f16 and bf16
#[macro_export]
macro_rules! impl_v_abs_half {
    () => {
        #[inline(always)]
        fn v_abs_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32().abs());
            }
        }

        #[inline(always)]
        fn v_abs_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32().abs());
            }
        }
    };
}

/// Generate vectorized negation for signed numeric types
#[macro_export]
macro_rules! impl_v_neg {
    ($($t:ty),+) => {
        $(
            paste::paste! {
    #[inline(always)]
    fn [<v_neg_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = -(*xi);
                    }
                }
            }
        )+
    };
}

/// Generate vectorized negation for f16 and bf16
#[macro_export]
macro_rules! impl_v_neg_half {
    () => {
        #[inline(always)]
        fn v_neg_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(-xi.to_f32());
            }
        }

        #[inline(always)]
        fn v_neg_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(-xi.to_f32());
            }
        }
    };
}

/// Generate vectorized power function for all numeric types
#[macro_export]
macro_rules! impl_v_pow {
    ($($t:ty),+) => {
        $(
            paste::paste! {
    #[inline(always)]
    fn [<v_pow_ $t>](&self, a: &[$t], b: &[$t], out: &mut [$t]) {
                    assert_eq!(a.len(), b.len(), "Input slices must have same length");
                    assert_eq!(
                        a.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for ((o, ai), bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                        *o = ((*ai ).powf(*bi )) ;
                    }
                }
            }
        )+
    };
}

// TODO: rename
/// Generate vectorized ReLU operation for floating point types
#[macro_export]
macro_rules! impl_relu {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<relu_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = (*xi).max(0.0);
                    }
                }
            }
        )+
    };
}

// TODO: rename
/// Generate vectorized ReLU operation for integer types
#[macro_export]
macro_rules! impl_relu_int {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<relu_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = (*xi).max(0);
                    }
                }
            }
        )+
    };
}

// TODO: rename
/// Generate vectorized ReLU operation for unsigned integer types
#[macro_export]
macro_rules! impl_relu_uint {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<relu_ $t>](&self, x: &[$t], out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = *xi;
                    }
                }
            }
        )+
    };
}

// TODO: rename
/// Generate vectorized ReLU operation for half precision types
#[macro_export]
macro_rules! impl_relu_half {
    () => {
        #[inline(always)]
        fn relu_f16(&self, x: &[half::f16], out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32().max(0.0));
            }
        }

        #[inline(always)]
        fn relu_bf16(&self, x: &[half::bf16], out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32().max(0.0));
            }
        }
    };
}

/// Generate sum operations for floating point types
#[macro_export]
macro_rules! impl_sum_float {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<sum_ $t>](&self, x: &[$t]) -> $t {
                    if x.is_empty() {
                        return 0 as $t;
                    }
                    x.iter().sum()
                }
            }
        )+
    };
}

/// Generate sum operations for integer types
#[macro_export]
macro_rules! impl_sum_int {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<sum_ $t>](&self, x: &[$t]) -> f64 {
                    if x.is_empty() {
                        return 0.0f64;
                    }
                    // Convert to f64 to avoid overflow
                    x.iter().map(|&val| val as f64).sum()
                }
            }
        )+
    };
}

/// Generate sum operations for all numeric types
#[macro_export]
macro_rules! impl_sum {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<sum_ $t>](&self, x: &[$t]) -> $t {
                    if x.is_empty() {
                        return 0 as $t;
                    }
                    x.iter().sum()
                }
            }
        )+
    };
}

/// Generate sum operations for half precision types
#[macro_export]
macro_rules! impl_sum_half {
    () => {
        #[inline(always)]
        fn sum_f16(&self, x: &[half::f16]) -> f64 {
            if x.is_empty() {
                return 0.0f64;
            }
            x.iter().map(|xi| xi.to_f64()).sum()
        }

        #[inline(always)]
        fn sum_bf16(&self, x: &[half::bf16]) -> f64 {
            if x.is_empty() {
                return 0.0f64;
            }
            x.iter().map(|xi| xi.to_f64()).sum()
        }
    };
}

/// Generate mean operations for floating point types
#[macro_export]
macro_rules! impl_mean_float {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<mean_ $t>](&self, x: &[$t]) -> $t {
                    if x.is_empty() {
                        return 0 as $t;
                    }
                    let sum = self.[<sum_ $t>](x);
                    sum / (x.len() as $t)
                }
            }
        )+
    };
}

/// Generate mean operations for half precision types
#[macro_export]
macro_rules! impl_mean_half {
    () => {
        #[inline(always)]
        fn mean_f16(&self, x: &[half::f16]) -> f64 {
            if x.is_empty() {
                return 0.0f64;
            }
            let sum = self.sum_f16(x);
            sum / (x.len() as f64)
        }

        #[inline(always)]
        fn mean_bf16(&self, x: &[half::bf16]) -> f64 {
            if x.is_empty() {
                return 0.0f64;
            }
            let sum = self.sum_bf16(x);
            sum / (x.len() as f64)
        }
    };
}

/// Generate mean operations for integer types
#[macro_export]
macro_rules! impl_mean_int {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<mean_ $t>](&self, x: &[$t]) -> f64 {
                    if x.is_empty() {
                        return 0.0f64;
                    }
                    let sum = self.[<sum_ $t>](x);
                    sum / (x.len() as f64)
                }
            }
        )+
    };
}

/// Generate scalar addition operations for all numeric types
#[macro_export]
macro_rules! impl_v_add_scalar {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<v_add_scalar_ $t>](&self, x: &[$t], scalar: $t, out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = *xi + scalar;
                    }
                }
            }
        )+
    };
}

/// Generate scalar addition operations for half precision types
#[macro_export]
macro_rules! impl_v_add_scalar_half {
    () => {
        #[inline(always)]
        fn v_add_scalar_f16(&self, x: &[half::f16], scalar: half::f16, out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32() + scalar.to_f32());
            }
        }

        #[inline(always)]
        fn v_add_scalar_bf16(&self, x: &[half::bf16], scalar: half::bf16, out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32() + scalar.to_f32());
            }
        }
    };
}

/// Generate scalar subtraction operations for all numeric types
#[macro_export]
macro_rules! impl_v_sub_scalar {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<v_sub_scalar_ $t>](&self, x: &[$t], scalar: $t, out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = *xi - scalar;
                    }
                }
            }
        )+
    };
}

/// Generate scalar subtraction operations for half precision types
#[macro_export]
macro_rules! impl_v_sub_scalar_half {
    () => {
        #[inline(always)]
        fn v_sub_scalar_f16(&self, x: &[half::f16], scalar: half::f16, out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32() - scalar.to_f32());
            }
        }

        #[inline(always)]
        fn v_sub_scalar_bf16(&self, x: &[half::bf16], scalar: half::bf16, out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32() - scalar.to_f32());
            }
        }
    };
}

/// Generate scalar multiplication operations for all numeric types
#[macro_export]
macro_rules! impl_v_mul_scalar {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<v_mul_scalar_ $t>](&self, x: &[$t], scalar: $t, out: &mut [$t]) {
                    assert_eq!(
                        x.len(),
                        out.len(),
                        "Input and output slices must have same length"
                    );
                    for (o, xi) in out.iter_mut().zip(x.iter()) {
                        *o = *xi * scalar;
                    }
                }
            }
        )+
    };
}

/// Generate scalar multiplication operations for half precision types
#[macro_export]
macro_rules! impl_v_mul_scalar_half {
    () => {
        #[inline(always)]
        fn v_mul_scalar_f16(&self, x: &[half::f16], scalar: half::f16, out: &mut [half::f16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::f16::from_f32(xi.to_f32() * scalar.to_f32());
            }
        }

        #[inline(always)]
        fn v_mul_scalar_bf16(&self, x: &[half::bf16], scalar: half::bf16, out: &mut [half::bf16]) {
            assert_eq!(
                x.len(),
                out.len(),
                "Input and output slices must have same length"
            );
            for (o, xi) in out.iter_mut().zip(x.iter()) {
                *o = half::bf16::from_f32(xi.to_f32() * scalar.to_f32());
            }
        }
    };
}

/// Generate vectorized maximum value for all numeric types
#[macro_export]
macro_rules! impl_max_v {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<max_v_ $t>](&self, x: &[$t]) -> $t {
                    if x.is_empty() {
                        panic!("Cannot find maximum of empty vector");
                    }
                    x.iter().fold(x[0], |a, &b| a.max(b))
                }
            }
        )+
    };
}

/// Generate vectorized minimum value for all numeric types
#[macro_export]
macro_rules! impl_min_v {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<min_v_ $t>](&self, x: &[$t]) -> $t {
                    if x.is_empty() {
                        panic!("Cannot find minimum of empty vector");
                    }
                    x.iter().fold(x[0], |a, &b| a.min(b))
                }
            }
        )+
    };
}

/// Generate vectorized maximum value with index for all numeric types
#[macro_export]
macro_rules! impl_max_vi {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<max_vi_ $t>](&self, x: &[$t]) -> ($t, u64) {
                    if x.is_empty() {
                        panic!("Cannot find maximum of empty vector");
                    }
                    x.iter()
                        .enumerate()
                        .fold((x[0], 0), |(max_val, max_idx), (i, &val)| {
                            if val > max_val {
                                (val, i as u64)
                            } else {
                                (max_val, max_idx)
                            }
                        })
                }
            }
        )+
    };
}

/// Generate vectorized minimum value with index for all numeric types
#[macro_export]
macro_rules! impl_min_vi {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<min_vi_ $t>](&self, x: &[$t]) -> ($t, u64) {
                    if x.is_empty() {
                        panic!("Cannot find minimum of empty vector");
                    }
                    x.iter()
                        .enumerate()
                        .fold((x[0], 0), |(min_val, min_idx), (i, &val)| {
                            if val < min_val {
                                (val, i as u64)
                            } else {
                                (min_val, min_idx)
                            }
                        })
                }
            }
        )+
    };
}

/// Generate vectorized maximum value for half precision types
#[macro_export]
macro_rules! impl_max_v_half {
    () => {
        #[inline(always)]
        fn max_v_f16(&self, x: &[half::f16]) -> half::f16 {
            if x.is_empty() {
                panic!("Cannot find maximum of empty vector");
            }
            x.iter().fold(x[0], |a, &b| a.max(b))
        }

        #[inline(always)]
        fn max_v_bf16(&self, x: &[half::bf16]) -> half::bf16 {
            if x.is_empty() {
                panic!("Cannot find maximum of empty vector");
            }
            x.iter().fold(x[0], |a, &b| a.max(b))
        }
    };
}

/// Generate vectorized minimum value for half precision types
#[macro_export]
macro_rules! impl_min_v_half {
    () => {
        #[inline(always)]
        fn min_v_f16(&self, x: &[half::f16]) -> half::f16 {
            if x.is_empty() {
                panic!("Cannot find minimum of empty vector");
            }
            x.iter().fold(x[0], |a, &b| a.min(b))
        }

        #[inline(always)]
        fn min_v_bf16(&self, x: &[half::bf16]) -> half::bf16 {
            if x.is_empty() {
                panic!("Cannot find minimum of empty vector");
            }
            x.iter().fold(x[0], |a, &b| a.min(b))
        }
    };
}

/// Generate vectorized maximum value with index for half precision types
#[macro_export]
macro_rules! impl_max_vi_half {
    () => {
        #[inline(always)]
        fn max_vi_f16(&self, x: &[half::f16]) -> (half::f16, u64) {
            if x.is_empty() {
                panic!("Cannot find maximum of empty vector");
            }
            x.iter()
                .enumerate()
                .fold((x[0], 0), |(max_val, max_idx), (i, &val)| {
                    if val > max_val {
                        (val, i as u64)
                    } else {
                        (max_val, max_idx)
                    }
                })
        }

        #[inline(always)]
        fn max_vi_bf16(&self, x: &[half::bf16]) -> (half::bf16, u64) {
            if x.is_empty() {
                panic!("Cannot find maximum of empty vector");
            }
            x.iter()
                .enumerate()
                .fold((x[0], 0), |(max_val, max_idx), (i, &val)| {
                    if val > max_val {
                        (val, i as u64)
                    } else {
                        (max_val, max_idx)
                    }
                })
        }
    };
}

/// Generate vectorized minimum value with index for half precision types
#[macro_export]
macro_rules! impl_min_vi_half {
    () => {
        #[inline(always)]
        fn min_vi_f16(&self, x: &[half::f16]) -> (half::f16, u64) {
            if x.is_empty() {
                panic!("Cannot find minimum of empty vector");
            }
            x.iter()
                .enumerate()
                .fold((x[0], 0), |(min_val, min_idx), (i, &val)| {
                    if val < min_val {
                        (val, i as u64)
                    } else {
                        (min_val, min_idx)
                    }
                })
        }

        #[inline(always)]
        fn min_vi_bf16(&self, x: &[half::bf16]) -> (half::bf16, u64) {
            if x.is_empty() {
                panic!("Cannot find minimum of empty vector");
            }
            x.iter()
                .enumerate()
                .fold((x[0], 0), |(min_val, min_idx), (i, &val)| {
                    if val < min_val {
                        (val, i as u64)
                    } else {
                        (min_val, min_idx)
                    }
                })
        }
    };
}

/// Generate vectorized min and max values for all numeric types
#[macro_export]
macro_rules! impl_min_max_v {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<min_max_v_ $t>](&self, x: &[$t]) -> ($t, $t) {
                    if x.is_empty() {
                        panic!("Cannot find min/max of empty vector");
                    }
                    x.iter().fold((x[0], x[0]), |(min_val, max_val), &val| {
                        (min_val.min(val), max_val.max(val))
                    })
                }
            }
        )+
    };
}

/// Generate vectorized min and max values with indices for all numeric types
#[macro_export]
macro_rules! impl_min_max_vi {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<min_max_vi_ $t>](&self, x: &[$t]) -> (($t, u64), ($t, u64)) {
                    if x.is_empty() {
                        panic!("Cannot find min/max of empty vector");
                    }
                    let mut min_val = x[0];
                    let mut min_idx = 0;
                    let mut max_val = x[0];
                    let mut max_idx = 0;

                    for (i, &val) in x.iter().enumerate() {
                        if val < min_val {
                            min_val = val;
                            min_idx = i;
                        }
                        if val > max_val {
                            max_val = val;
                            max_idx = i;
                        }
                    }

                    ((min_val, min_idx as u64), (max_val, max_idx as u64))
                }
            }
        )+
    };
}

/// Generate vectorized min and max indices for all numeric types
#[macro_export]
macro_rules! impl_min_max_i {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<min_max_i_ $t>](&self, x: &[$t]) -> (u64, u64) {
                    if x.is_empty() {
                        panic!("Cannot find min/max indices of empty vector");
                    }
                    x.iter()
                        .enumerate()
                        .fold((0, 0), |(min_idx, max_idx), (i, &val)| {
                            let new_min_idx = if val < x[min_idx as usize] { i as u64 } else { min_idx };
                            let new_max_idx = if val > x[max_idx as usize] { i as u64 } else { max_idx };
                            (new_min_idx, new_max_idx)
                        })
                }
            }
        )+
    };
}

/// Generate vectorized min and max values for half precision types
#[macro_export]
macro_rules! impl_min_max_v_half {
    () => {
        #[inline(always)]
        fn min_max_v_f16(&self, x: &[half::f16]) -> (half::f16, half::f16) {
            if x.is_empty() {
                panic!("Cannot find min/max of empty vector");
            }
            x.iter().fold((x[0], x[0]), |(min_val, max_val), &val| {
                (min_val.min(val), max_val.max(val))
            })
        }

        #[inline(always)]
        fn min_max_v_bf16(&self, x: &[half::bf16]) -> (half::bf16, half::bf16) {
            if x.is_empty() {
                panic!("Cannot find min/max of empty vector");
            }
            x.iter().fold((x[0], x[0]), |(min_val, max_val), &val| {
                (min_val.min(val), max_val.max(val))
            })
        }
    };
}

/// Generate vectorized min and max values with indices for half precision types
#[macro_export]
macro_rules! impl_min_max_vi_half {
    () => {
        #[inline(always)]
        fn min_max_vi_f16(&self, x: &[half::f16]) -> ((half::f16, u64), (half::f16, u64)) {
            if x.is_empty() {
                panic!("Cannot find min/max of empty vector");
            }
            let mut min_val = x[0];
            let mut min_idx = 0;
            let mut max_val = x[0];
            let mut max_idx = 0;

            for (i, &val) in x.iter().enumerate() {
                if val < min_val {
                    min_val = val;
                    min_idx = i;
                }
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            ((min_val, min_idx as u64), (max_val, max_idx as u64))
        }

        #[inline(always)]
        fn min_max_vi_bf16(&self, x: &[half::bf16]) -> ((half::bf16, u64), (half::bf16, u64)) {
            if x.is_empty() {
                panic!("Cannot find min/max of empty vector");
            }
            let mut min_val = x[0];
            let mut min_idx = 0;
            let mut max_val = x[0];
            let mut max_idx = 0;

            for (i, &val) in x.iter().enumerate() {
                if val < min_val {
                    min_val = val;
                    min_idx = i;
                }
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            ((min_val, min_idx as u64), (max_val, max_idx as u64))
        }
    };
}

/// Generate vectorized min and max indices for half precision types
#[macro_export]
macro_rules! impl_min_max_i_half {
    () => {
        #[inline(always)]
        fn min_max_i_f16(&self, x: &[half::f16]) -> (u64, u64) {
            if x.is_empty() {
                panic!("Cannot find min/max indices of empty vector");
            }
            x.iter()
                .enumerate()
                .fold((0, 0), |(min_idx, max_idx), (i, &val)| {
                    let new_min_idx = if val < x[min_idx as usize] {
                        i as u64
                    } else {
                        min_idx
                    };
                    let new_max_idx = if val > x[max_idx as usize] {
                        i as u64
                    } else {
                        max_idx
                    };
                    (new_min_idx, new_max_idx)
                })
        }

        #[inline(always)]
        fn min_max_i_bf16(&self, x: &[half::bf16]) -> (u64, u64) {
            if x.is_empty() {
                panic!("Cannot find min/max indices of empty vector");
            }
            x.iter()
                .enumerate()
                .fold((0, 0), |(min_idx, max_idx), (i, &val)| {
                    let new_min_idx = if val < x[min_idx as usize] {
                        i as u64
                    } else {
                        min_idx
                    };
                    let new_max_idx = if val > x[max_idx as usize] {
                        i as u64
                    } else {
                        max_idx
                    };
                    (new_min_idx, new_max_idx)
                })
        }
    };
}

/// Generate vectorized minimum index for all numeric types
#[macro_export]
macro_rules! impl_min_i {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<min_i_ $t>](&self, x: &[$t]) -> u64 {
                    if x.is_empty() {
                        panic!("Cannot find minimum index of empty vector");
                    }
                    x.iter()
                        .enumerate()
                        .fold((0, x[0]), |(min_idx, min_val), (i, &val)| {
                            if val < min_val {
                                (i as u64, val)
                            } else {
                                (min_idx, min_val)
                            }
                        })
                        .0
                }
            }
        )+
    };
}

/// Generate vectorized maximum index for all numeric types
#[macro_export]
macro_rules! impl_max_i {
    ($($t:ty),+) => {
        $(
            paste::paste! {
                #[inline(always)]
                fn [<max_i_ $t>](&self, x: &[$t]) -> u64 {
                    if x.is_empty() {
                        panic!("Cannot find maximum index of empty vector");
                    }
                    x.iter()
                        .enumerate()
                        .fold((0, x[0]), |(max_idx, max_val), (i, &val)| {
                            if val > max_val {
                                (i as u64, val)
                            } else {
                                (max_idx, max_val)
                            }
                        })
                        .0
                }
            }
        )+
    };
}

/// Generate vectorized minimum index for half precision types
#[macro_export]
macro_rules! impl_min_i_half {
    () => {
        #[inline(always)]
        fn min_i_f16(&self, x: &[half::f16]) -> u64 {
            if x.is_empty() {
                panic!("Cannot find minimum index of empty vector");
            }
            x.iter()
                .enumerate()
                .fold((0, x[0]), |(min_idx, min_val), (i, &val)| {
                    if val < min_val {
                        (i as u64, val)
                    } else {
                        (min_idx, min_val)
                    }
                })
                .0
        }

        #[inline(always)]
        fn min_i_bf16(&self, x: &[half::bf16]) -> u64 {
            if x.is_empty() {
                panic!("Cannot find minimum index of empty vector");
            }
            x.iter()
                .enumerate()
                .fold((0, x[0]), |(min_idx, min_val), (i, &val)| {
                    if val < min_val {
                        (i as u64, val)
                    } else {
                        (min_idx, min_val)
                    }
                })
                .0
        }
    };
}

/// Generate vectorized maximum index for half precision types
#[macro_export]
macro_rules! impl_max_i_half {
    () => {
        #[inline(always)]
        fn max_i_f16(&self, x: &[half::f16]) -> u64 {
            if x.is_empty() {
                panic!("Cannot find maximum index of empty vector");
            }
            x.iter()
                .enumerate()
                .fold((0, x[0]), |(max_idx, max_val), (i, &val)| {
                    if val > max_val {
                        (i as u64, val)
                    } else {
                        (max_idx, max_val)
                    }
                })
                .0
        }

        #[inline(always)]
        fn max_i_bf16(&self, x: &[half::bf16]) -> u64 {
            if x.is_empty() {
                panic!("Cannot find maximum index of empty vector");
            }
            x.iter()
                .enumerate()
                .fold((0, x[0]), |(max_idx, max_val), (i, &val)| {
                    if val > max_val {
                        (i as u64, val)
                    } else {
                        (max_idx, max_val)
                    }
                })
                .0
        }
    };
}
