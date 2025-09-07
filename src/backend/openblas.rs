use crate::OpsTrait;
use std::os::raw::{c_double, c_float, c_int};

use super::cblas_consts::{CBLAS_NO_TRANS, CBLAS_ROW_MAJOR};

#[link(name = "openblas")]
extern "C" {
    // BLAS Level 1
    fn cblas_sscal(n: c_int, alpha: c_float, x: *mut c_float, incx: c_int);
    fn cblas_dscal(n: c_int, alpha: c_double, x: *mut c_double, incx: c_int);
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
    fn cblas_snrm2(n: c_int, x: *const c_float, incx: c_int) -> c_float;
    fn cblas_dnrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;

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
}

/// OpenBLAS backend for high-performance linear algebra
///
/// This backend uses OpenBLAS for BLAS operations and falls back to
/// parallel implementations using rayon for vectorized math functions.
#[derive(Debug)]
pub struct OpenBlasBackend;

impl OpsTrait for OpenBlasBackend {
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

    // Sum operations using sdot with incY=0 and y=1.0
    #[inline(always)]
    fn sum_f32(&self, x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0f32;
        }
        // Use sdot with incY=0 and y=1.0 to compute sum
        // This is equivalent to dot(x, ones) but more efficient
        let y = 1.0f32;
        unsafe { cblas_sdot(x.len() as c_int, x.as_ptr(), 1, &y, 0) }
    }

    #[inline(always)]
    fn sum_f64(&self, x: &[f64]) -> f64 {
        if x.is_empty() {
            return 0.0f64;
        }
        // Use ddot with incY=0 and y=1.0 to compute sum
        let y = 1.0f64;
        unsafe { cblas_ddot(x.len() as c_int, x.as_ptr(), 1, &y, 0) }
    }

    #[inline(always)]
    fn nrm2_f32(&self, x: &[f32]) -> f64 {
        if x.is_empty() {
            return 0.0f64;
        }
        unsafe { cblas_snrm2(x.len() as c_int, x.as_ptr(), 1) as f64 }
    }

    #[inline(always)]
    fn nrm2_f64(&self, x: &[f64]) -> f64 {
        if x.is_empty() {
            return 0.0f64;
        }
        unsafe { cblas_dnrm2(x.len() as c_int, x.as_ptr(), 1) }
    }

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

    // Rewrite scalar add implementation, optimize with cblas_axpy
    #[inline(always)]
    fn v_add_scalar_f32(&self, x: &[f32], scalar: f32, out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        // Use cblas_saxpy for vector-scalar addition operation
        // First copy x to out, then perform y = alpha * x + y
        out.copy_from_slice(x);
        unsafe { cblas_saxpy(x.len() as c_int, scalar, x.as_ptr(), 1, out.as_mut_ptr(), 1) }
    }

    #[inline(always)]
    fn v_add_scalar_f64(&self, x: &[f64], scalar: f64, out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length",
        );
        // Use cblas_daxpy for vector-scalar addition operation
        // First copy x to out, then perform y = alpha * x + y
        out.copy_from_slice(x);
        unsafe { cblas_daxpy(x.len() as c_int, scalar, x.as_ptr(), 1, out.as_mut_ptr(), 1) }
    }
}
