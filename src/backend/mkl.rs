use std::os::raw::{c_double, c_float, c_int};

use crate::{OpsTrait, UninitVec};

use super::cblas_consts::{CBLAS_NO_TRANS, CBLAS_ROW_MAJOR};

#[link(name = "mkl_rt")]
extern "C" {
    // BLAS Level 1
    fn cblas_sdot(
        n: c_int,
        x: *const c_float,
        incx: c_int,
        y: *const c_float,
        incy: c_int,
    ) -> c_float;

    fn cblas_ddot(
        n: c_int,
        x: *const c_double,
        incx: c_int,
        y: *const c_double,
        incy: c_int,
    ) -> c_double;

    fn cblas_sasum(n: c_int, x: *const c_float, incx: c_int) -> c_float;
    fn cblas_dasum(n: c_int, x: *const c_double, incx: c_int) -> c_double;

    #[allow(dead_code)]
    fn cblas_snrm2(n: c_int, x: *const c_float, incx: c_int) -> c_float;
    #[allow(dead_code)]
    fn cblas_dnrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;

    fn cblas_sscal(n: c_int, alpha: c_float, x: *mut c_float, incx: c_int);
    fn cblas_dscal(n: c_int, alpha: c_double, x: *mut c_double, incx: c_int);

    // BLAS Level 1 - AXPY operations (y = alpha * x + y)
    fn cblas_saxpy(
        n: c_int,
        alpha: c_float,
        x: *const c_float,
        incx: c_int,
        y: *mut c_float,
        incy: c_int,
    );
    fn cblas_daxpy(
        n: c_int,
        alpha: c_double,
        x: *const c_double,
        incx: c_int,
        y: *mut c_double,
        incy: c_int,
    );

    // BLAS Level 3
    fn cblas_sgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
    );

    fn cblas_dgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        b: *const c_double,
        ldb: c_int,
        beta: c_double,
        c: *mut c_double,
        ldc: c_int,
    );

    // Vector Math Library functions
    fn vsExp(n: c_int, a: *const c_float, y: *mut c_float);
    fn vdExp(n: c_int, a: *const c_double, y: *mut c_double);

    fn vsSin(n: c_int, a: *const c_float, y: *mut c_float);
    fn vdSin(n: c_int, a: *const c_double, y: *mut c_double);

    fn vsCos(n: c_int, a: *const c_float, y: *mut c_float);
    fn vdCos(n: c_int, a: *const c_double, y: *mut c_double);

    fn vsTanh(n: c_int, a: *const c_float, y: *mut c_float);
    fn vdTanh(n: c_int, a: *const c_double, y: *mut c_double);

    fn vsLn(n: c_int, a: *const c_float, y: *mut c_float);
    fn vdLn(n: c_int, a: *const c_double, y: *mut c_double);

    fn vsSqrt(n: c_int, a: *const c_float, y: *mut c_float);
    fn vdSqrt(n: c_int, a: *const c_double, y: *mut c_double);

    fn vsSqr(n: c_int, a: *const c_float, y: *mut c_float);
    fn vdSqr(n: c_int, a: *const c_double, y: *mut c_double);

    fn vsAdd(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
    fn vdAdd(n: c_int, a: *const c_double, b: *const c_double, y: *mut c_double);

    fn vsMul(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
    fn vdMul(n: c_int, a: *const c_double, b: *const c_double, y: *mut c_double);

    // Additional VML functions for extended operations
    fn vsSub(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
    fn vdSub(n: c_int, a: *const c_double, b: *const c_double, y: *mut c_double);

    fn vsDiv(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
    fn vdDiv(n: c_int, a: *const c_double, b: *const c_double, y: *mut c_double);

    fn vsTan(n: c_int, a: *const c_float, y: *mut c_float);
    fn vdTan(n: c_int, a: *const c_double, y: *mut c_double);

    fn vsInv(n: c_int, a: *const c_float, y: *mut c_float);
    fn vdInv(n: c_int, a: *const c_double, y: *mut c_double);

    fn vsFloor(n: c_int, a: *const c_float, y: *mut c_float);
    fn vdFloor(n: c_int, a: *const c_double, y: *mut c_double);

    fn vsCeil(n: c_int, a: *const c_float, y: *mut c_float);
    fn vdCeil(n: c_int, a: *const c_double, y: *mut c_double);

    fn vsRound(n: c_int, a: *const c_float, y: *mut c_float);
    fn vdRound(n: c_int, a: *const c_double, y: *mut c_double);

    fn vsAbs(n: c_int, a: *const c_float, y: *mut c_float);
    fn vdAbs(n: c_int, a: *const c_double, y: *mut c_double);

    fn vsNeg(n: c_int, a: *const c_float, y: *mut c_float);
    fn vdNeg(n: c_int, a: *const c_double, y: *mut c_double);

    fn vsPow(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
    fn vdPow(n: c_int, a: *const c_double, b: *const c_double, y: *mut c_double);

    // VML maximum and minimum functions, element-wise
    fn vsFmax(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
    fn vdFmax(n: c_int, a: *const c_double, b: *const c_double, y: *mut c_double);
    fn vsFmin(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
    fn vdFmin(n: c_int, a: *const c_double, b: *const c_double, y: *mut c_double);

    // Vector L2 norm functions (more optimized than cblas_nrm2)
    fn vdNorm(n: c_int, x: *const c_double, incx: c_int, norm: *mut c_double);
    fn vsNorm(n: c_int, x: *const c_float, incx: c_int, norm: *mut c_float);

}

/// MKL backend using Intel Math Kernel Library
///
/// This backend provides optimized implementations using Intel MKL BLAS and VML functions.
/// It leverages the highly optimized routines for maximum performance on Intel hardware.
#[derive(Debug)]
pub struct MklBackend;

impl OpsTrait for MklBackend {
    #[inline(always)]
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(
            a.len(),
            b.len(),
            "Vector lengths must match for dot product"
        );
        unsafe { cblas_sdot(a.len() as c_int, a.as_ptr(), 1, b.as_ptr(), 1) as f64 }
    }

    #[inline(always)]
    fn dot_f64(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(
            a.len(),
            b.len(),
            "Vector lengths must match for dot product"
        );
        unsafe { cblas_ddot(a.len() as c_int, a.as_ptr(), 1, b.as_ptr(), 1) }
    }

    #[inline(always)]
    fn asum_f32(&self, x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0f32;
        }
        unsafe { cblas_sasum(x.len() as c_int, x.as_ptr(), 1) }
    }

    #[inline(always)]
    fn asum_f64(&self, x: &[f64]) -> f64 {
        if x.is_empty() {
            return 0.0f64;
        }
        unsafe { cblas_dasum(x.len() as c_int, x.as_ptr(), 1) }
    }

    // // Sum operations using sdot with incY=0 and y=1.0
    // #[inline(always)]
    // fn sum_f32(&self, x: &[f32]) -> f32 {
    //     if x.is_empty() {
    //         return 0.0f32;
    //     }
    //     // Use sdot with incY=0 and y=1.0 to compute sum
    //     // This is equivalent to dot(x, ones) but more efficient
    //     let y = 1.0f32;
    //     unsafe { cblas_sdot(x.len() as c_int, x.as_ptr(), 1, &y, 0) }
    // }

    // #[inline(always)]
    // fn sum_f64(&self, x: &[f64]) -> f64 {
    //     if x.is_empty() {
    //         return 0.0f64;
    //     }
    //     // Use ddot with incY=0 and y=1.0 to compute sum
    //     let y = 1.0f64;
    //     unsafe { cblas_ddot(x.len() as c_int, x.as_ptr(), 1, &y, 0) }
    // }

    #[inline(always)]
    fn scal_f32(&self, a: f32, x: &mut [f32]) {
        unsafe { cblas_sscal(x.len() as c_int, a, x.as_mut_ptr(), 1) }
    }

    #[inline(always)]
    fn scal_f64(&self, a: f64, x: &mut [f64]) {
        unsafe { cblas_dscal(x.len() as c_int, a, x.as_mut_ptr(), 1) }
    }

    #[inline(always)]
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
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            1.0f32,
            a,
            lda as c_int,
            b,
            ldb as c_int,
            0.0f32,
            c,
            ldc as c_int,
        );
    }

    #[inline(always)]
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
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            1.0f64,
            a,
            lda as c_int,
            b,
            ldb as c_int,
            0.0f64,
            c,
            ldc as c_int,
        );
    }

    #[inline(always)]
    fn v_exp_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsExp(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_exp_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdExp(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_sin_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsSin(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_sin_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdSin(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_cos_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsCos(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_cos_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdCos(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_tanh_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsTanh(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_tanh_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdTanh(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_log_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsLn(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_log_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdLn(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_sqrt_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsSqrt(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_sqrt_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdSqrt(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_sqr_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsSqr(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_sqr_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdSqr(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsAdd(a.len() as c_int, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_add_f64(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdAdd(a.len() as c_int, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsMul(a.len() as c_int, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_mul_f64(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdMul(a.len() as c_int, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_sub_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsSub(a.len() as c_int, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_sub_f64(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdSub(a.len() as c_int, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_div_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsDiv(a.len() as c_int, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_div_f64(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdDiv(a.len() as c_int, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_tan_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsTan(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_tan_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdTan(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_recip_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsInv(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_recip_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdInv(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_floor_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsFloor(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_floor_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdFloor(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_ceil_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsCeil(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_ceil_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdCeil(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_round_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsRound(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_round_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdRound(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_abs_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsAbs(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_abs_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdAbs(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_neg_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsNeg(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_neg_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdNeg(x.len() as c_int, x.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_pow_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vsPow(a.len() as c_int, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn v_pow_f64(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vdPow(a.len() as c_int, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) }
    }

    #[inline(always)]
    fn relu_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let zero = UninitVec::<f32>::new(x.len()).full(0.0f32);
        unsafe {
            vsFmax(
                x.len() as c_int,
                x.as_ptr(),
                zero.as_ptr(),
                out.as_mut_ptr(),
            )
        }
    }

    #[inline(always)]
    fn relu_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let zero = UninitVec::<f64>::new(x.len()).full(0.0f64);
        unsafe {
            vdFmax(
                x.len() as c_int,
                x.as_ptr(),
                zero.as_ptr(),
                out.as_mut_ptr(),
            )
        }
    }

    // TODO
    #[inline(always)]
    fn clamp_f32(&self, x: &[f32], min: f32, max: f32, out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let mut temp = vec![0.0f32; x.len()];
        let min_vec = UninitVec::<f32>::new(x.len()).full(min);
        unsafe {
            vsFmax(
                x.len() as c_int,
                x.as_ptr(),
                min_vec.as_ptr(),
                temp.as_mut_ptr(),
            )
        }

        let max_vec = UninitVec::<f32>::new(x.len()).full(max);
        unsafe {
            vsFmin(
                x.len() as c_int,
                temp.as_ptr(),
                max_vec.as_ptr(),
                out.as_mut_ptr(),
            )
        }
    }

    #[inline(always)]
    fn clamp_f64(&self, x: &[f64], min: f64, max: f64, out: &mut [f64]) {
        // TODO
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let mut temp = vec![0.0f64; x.len()];
        let min_vec = UninitVec::<f64>::new(x.len()).full(min);
        unsafe {
            vdFmax(
                x.len() as c_int,
                x.as_ptr(),
                min_vec.as_ptr(),
                temp.as_mut_ptr(),
            )
        }

        let max_vec = UninitVec::<f64>::new(x.len()).full(max);
        unsafe {
            vdFmin(
                x.len() as c_int,
                temp.as_ptr(),
                max_vec.as_ptr(),
                out.as_mut_ptr(),
            )
        }
    }

    #[inline(always)]
    fn v_add_scalar_f32(&self, x: &[f32], scalar: f32, out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        out.copy_from_slice(x);
        unsafe { cblas_saxpy(x.len() as c_int, scalar, x.as_ptr(), 1, out.as_mut_ptr(), 1) }
    }

    #[inline(always)]
    fn v_add_scalar_f64(&self, x: &[f64], scalar: f64, out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        out.copy_from_slice(x);
        unsafe { cblas_daxpy(x.len() as c_int, scalar, x.as_ptr(), 1, out.as_mut_ptr(), 1) }
    }

    #[inline(always)]
    fn nrm2_f32(&self, x: &[f32]) -> f64 {
        if x.is_empty() {
            return 0.0f64;
        }
        // Use vsNorm for better performance on Intel hardware
        let mut result = 0.0f32;
        unsafe { vsNorm(x.len() as c_int, x.as_ptr(), 1, &mut result) }
        result as f64
    }

    #[inline(always)]
    fn nrm2_f64(&self, x: &[f64]) -> f64 {
        if x.is_empty() {
            return 0.0f64;
        }
        // Use vdNorm for better performance on Intel hardware
        let mut result = 0.0f64;
        unsafe { vdNorm(x.len() as c_int, x.as_ptr(), 1, &mut result) }
        result
    }
}
