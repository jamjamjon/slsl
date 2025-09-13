use crate::{OpsTrait, UninitVec};
use std::os::raw::{c_double, c_float, c_int};

use super::cblas_consts::{CBLAS_NO_TRANS, CBLAS_ROW_MAJOR};

// Accelerate framework bindings for macOS
#[link(name = "Accelerate", kind = "framework")]
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
    #[allow(dead_code)]
    fn cblas_sasum(n: c_int, x: *const c_float, incx: c_int) -> c_float;
    #[allow(dead_code)]
    fn cblas_dasum(n: c_int, x: *const c_double, incx: c_int) -> c_double;

    // L2 norm functions
    #[allow(dead_code)]
    fn cblas_snrm2(n: c_int, x: *const c_float, incx: c_int) -> c_float;
    #[allow(dead_code)]
    fn cblas_dnrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;

    fn cblas_sscal(n: c_int, alpha: c_float, x: *mut c_float, incx: c_int);
    fn cblas_dscal(n: c_int, alpha: c_double, x: *mut c_double, incx: c_int);

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

    // vDSP vectorized functions
    fn vvexpf(y: *mut c_float, x: *const c_float, n: *const c_int);
    fn vvexp(y: *mut c_double, x: *const c_double, n: *const c_int);
    fn vvsinf(y: *mut c_float, x: *const c_float, n: *const c_int);
    fn vvsin(y: *mut c_double, x: *const c_double, n: *const c_int);
    fn vvcosf(y: *mut c_float, x: *const c_float, n: *const c_int);
    fn vvcos(y: *mut c_double, x: *const c_double, n: *const c_int);
    fn vvtanhf(y: *mut c_float, x: *const c_float, n: *const c_int);
    fn vvtanh(y: *mut c_double, x: *const c_double, n: *const c_int);
    fn vvlogf(y: *mut c_float, x: *const c_float, n: *const c_int);
    fn vvlog(y: *mut c_double, x: *const c_double, n: *const c_int);
    fn vvsqrtf(y: *mut c_float, x: *const c_float, n: *const c_int);
    fn vvsqrt(y: *mut c_double, x: *const c_double, n: *const c_int);

    // vDSP square functions
    fn vDSP_vsq(x: *const c_float, ix: c_int, y: *mut c_float, iy: c_int, n: c_int);
    fn vDSP_vsqD(x: *const c_double, ix: c_int, y: *mut c_double, iy: c_int, n: c_int);

    // vDSP basic operations
    fn vDSP_vneg(a: *const c_float, ia: c_int, c: *mut c_float, ic: c_int, n: c_int);
    fn vDSP_vnegD(a: *const c_double, ia: c_int, c: *mut c_double, ic: c_int, n: c_int);

    // vDSP sum of absolute values (L1 norm)
    fn vDSP_sasum(x: *const c_float, incx: c_int, result: *mut c_float, n: c_int);
    fn vDSP_dasum(x: *const c_double, incx: c_int, result: *mut c_double, n: c_int);

    fn vDSP_vadd(
        a: *const c_float,
        ia: c_int,
        b: *const c_float,
        ib: c_int,
        c: *mut c_float,
        ic: c_int,
        n: c_int,
    );
    fn vDSP_vaddD(
        a: *const c_double,
        ia: c_int,
        b: *const c_double,
        ib: c_int,
        c: *mut c_double,
        ic: c_int,
        n: c_int,
    );
    fn vDSP_vmul(
        a: *const c_float,
        ia: c_int,
        b: *const c_float,
        ib: c_int,
        c: *mut c_float,
        ic: c_int,
        n: c_int,
    );
    fn vDSP_vmulD(
        a: *const c_double,
        ia: c_int,
        b: *const c_double,
        ib: c_int,
        c: *mut c_double,
        ic: c_int,
        n: c_int,
    );

    // Additional vDSP operations
    fn vDSP_vsub(
        a: *const c_float,
        ia: c_int,
        b: *const c_float,
        ib: c_int,
        c: *mut c_float,
        ic: c_int,
        n: c_int,
    );
    fn vDSP_vsubD(
        a: *const c_double,
        ia: c_int,
        b: *const c_double,
        ib: c_int,
        c: *mut c_double,
        ic: c_int,
        n: c_int,
    );
    fn vDSP_vdiv(
        a: *const c_float,
        ia: c_int,
        b: *const c_float,
        ib: c_int,
        c: *mut c_float,
        ic: c_int,
        n: c_int,
    );
    fn vDSP_vdivD(
        a: *const c_double,
        ia: c_int,
        b: *const c_double,
        ib: c_int,
        c: *mut c_double,
        ic: c_int,
        n: c_int,
    );

    // Additional vDSP math functions
    fn vvtanf(y: *mut c_float, x: *const c_float, n: *const c_int);
    fn vvtan(y: *mut c_double, x: *const c_double, n: *const c_int);
    fn vvrecf(y: *mut c_float, x: *const c_float, n: *const c_int);
    fn vvrec(y: *mut c_double, x: *const c_double, n: *const c_int);
    fn vvfloorf(y: *mut c_float, x: *const c_float, n: *const c_int);
    fn vvfloor(y: *mut c_double, x: *const c_double, n: *const c_int);
    fn vvceilf(y: *mut c_float, x: *const c_float, n: *const c_int);
    fn vvceil(y: *mut c_double, x: *const c_double, n: *const c_int);
    fn vvpowf(y: *mut c_float, x: *const c_float, e: *const c_float, n: *const c_int);
    fn vvpow(y: *mut c_double, x: *const c_double, e: *const c_double, n: *const c_int);

    // Additional vDSP operations for scalar multiplication
    #[allow(dead_code)]
    fn vDSP_vsmul(
        x: *const c_float,
        ix: c_int,
        s: *const c_float,
        y: *mut c_float,
        iy: c_int,
        n: c_int,
    );
    #[allow(dead_code)]
    fn vDSP_vsmulD(
        x: *const c_double,
        ix: c_int,
        s: *const c_double,
        y: *mut c_double,
        iy: c_int,
        n: c_int,
    );

    // vDSP scalar addition functions
    fn vDSP_vsadd(
        x: *const c_float,
        ix: c_int,
        s: *const c_float,
        y: *mut c_float,
        iy: c_int,
        n: c_int,
    );
    fn vDSP_vsaddD(
        x: *const c_double,
        ix: c_int,
        s: *const c_double,
        y: *mut c_double,
        iy: c_int,
        n: c_int,
    );

    // vDSP absolute value functions
    fn vDSP_vabs(x: *const c_float, ix: c_int, y: *mut c_float, iy: c_int, n: c_int);
    fn vDSP_vabsD(x: *const c_double, ix: c_int, y: *mut c_double, iy: c_int, n: c_int);

    // vDSP maximum and minimum functions
    fn vDSP_vmax(
        x: *const c_float,
        ix: c_int,
        y: *const c_float,
        iy: c_int,
        z: *mut c_float,
        iz: c_int,
        n: c_int,
    );
    fn vDSP_vmaxD(
        x: *const c_double,
        ix: c_int,
        y: *const c_double,
        iy: c_int,
        z: *mut c_double,
        iz: c_int,
        n: c_int,
    );
    #[allow(dead_code)]
    fn vDSP_vmin(
        x: *const c_float,
        ix: c_int,
        y: *const c_float,
        iy: c_int,
        z: *mut c_float,
        iz: c_int,
        n: c_int,
    );
    #[allow(dead_code)]
    fn vDSP_vminD(
        x: *const c_double,
        ix: c_int,
        y: *const c_double,
        iy: c_int,
        z: *mut c_double,
        iz: c_int,
        n: c_int,
    );

    // vDSP clip function
    fn vDSP_vclip(
        x: *const c_float,
        ix: c_int,
        min: *const c_float,
        max: *const c_float,
        y: *mut c_float,
        iy: c_int,
        n: c_int,
    );
    fn vDSP_vclipD(
        x: *const c_double,
        ix: c_int,
        min: *const c_double,
        max: *const c_double,
        y: *mut c_double,
        iy: c_int,
        n: c_int,
    );

    // vDSP vector sum of squares (for L2 norm calculation)
    fn vDSP_svesq(x: *const c_float, ix: c_int, result: *mut c_float, n: c_int);
    fn vDSP_dvesq(x: *const c_double, ix: c_int, result: *mut c_double, n: c_int);

    // vDSP vector sum operations
    fn vDSP_sve(x: *const c_float, ix: c_int, result: *mut c_float, n: c_int);
    fn vDSP_sveD(x: *const c_double, ix: c_int, result: *mut c_double, n: c_int);

    // vDSP vector mean operations
    fn vDSP_meanv(x: *const c_float, ix: c_int, result: *mut c_float, n: c_int);
    fn vDSP_meanvD(x: *const c_double, ix: c_int, result: *mut c_double, n: c_int);

    // vDSP min/max functions
    fn vDSP_maxv(x: *const c_float, ix: c_int, result: *mut c_float, n: c_int);
    fn vDSP_maxvD(x: *const c_double, ix: c_int, result: *mut c_double, n: c_int);
    fn vDSP_minv(x: *const c_float, ix: c_int, result: *mut c_float, n: c_int);
    fn vDSP_minvD(x: *const c_double, ix: c_int, result: *mut c_double, n: c_int);

    // vDSP min/max with index functions
    fn vDSP_maxvi(x: *const c_float, ix: c_int, result: *mut c_float, index: *mut c_int, n: c_int);
    fn vDSP_maxviD(
        x: *const c_double,
        ix: c_int,
        result: *mut c_double,
        index: *mut c_int,
        n: c_int,
    );
    fn vDSP_minvi(x: *const c_float, ix: c_int, result: *mut c_float, index: *mut c_int, n: c_int);
    fn vDSP_minviD(
        x: *const c_double,
        ix: c_int,
        result: *mut c_double,
        index: *mut c_int,
        n: c_int,
    );
}

/// Accelerate backend using Apple's Accelerate framework
///
/// This backend provides highly optimized implementations using Apple's Accelerate framework,
/// which includes BLAS, LAPACK, and vDSP (vectorized digital signal processing) functions.
#[derive(Debug)]
pub struct AccelerateBackend;

impl OpsTrait for AccelerateBackend {
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
        // Use vDSP_sasum for better performance on Apple hardware
        let mut result = 0.0f32;
        unsafe { vDSP_sasum(x.as_ptr(), 1, &mut result, x.len() as c_int) }
        result
    }

    #[inline(always)]
    fn asum_f64(&self, x: &[f64]) -> f64 {
        if x.is_empty() {
            return 0.0f64;
        }
        // Use vDSP_dasum for better performance on Apple hardware
        let mut result = 0.0f64;
        unsafe { vDSP_dasum(x.as_ptr(), 1, &mut result, x.len() as c_int) }
        result
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
    fn nrm2_f32(&self, x: &[f32]) -> f64 {
        if x.is_empty() {
            return 0.0f64;
        }
        // Use vDSP_svesq for better performance on Apple hardware
        let mut sum_squares = 0.0f32;
        unsafe { vDSP_svesq(x.as_ptr(), 1, &mut sum_squares, x.len() as c_int) }
        sum_squares.sqrt() as f64
    }

    #[inline(always)]
    fn nrm2_f64(&self, x: &[f64]) -> f64 {
        if x.is_empty() {
            return 0.0f64;
        }
        // Use vDSP_dvesq for better performance on Apple hardware
        let mut sum_squares = 0.0f64;
        unsafe { vDSP_dvesq(x.as_ptr(), 1, &mut sum_squares, x.len() as c_int) }
        sum_squares.sqrt()
    }

    // Sum operations using vDSP_sve and vDSP_sveD
    #[inline(always)]
    fn sum_f32(&self, x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0f32;
        }
        let mut result = 0.0f32;
        unsafe { vDSP_sve(x.as_ptr(), 1, &mut result, x.len() as c_int) }
        result
    }

    #[inline(always)]
    fn sum_f64(&self, x: &[f64]) -> f64 {
        if x.is_empty() {
            return 0.0f64;
        }
        let mut result = 0.0f64;
        unsafe { vDSP_sveD(x.as_ptr(), 1, &mut result, x.len() as c_int) }
        result
    }

    // #[inline(always)]
    // fn sum_bf16(&self, x: &[half::bf16]) -> f64 {
    //     if x.is_empty() {
    //         return 0.0f64;
    //     }
    //     x.iter().map(|xi| xi.to_f64()).sum()
    // }

    // Mean operations using vDSP_meanv and vDSP_meanvD
    #[inline(always)]
    fn mean_f32(&self, x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0f32;
        }
        let mut result = 0.0f32;
        unsafe { vDSP_meanv(x.as_ptr(), 1, &mut result, x.len() as c_int) }
        result
    }

    #[inline(always)]
    fn mean_f64(&self, x: &[f64]) -> f64 {
        if x.is_empty() {
            return 0.0f64;
        }
        let mut result = 0.0f64;
        unsafe { vDSP_meanvD(x.as_ptr(), 1, &mut result, x.len() as c_int) }
        result
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
        let n = x.len() as c_int;
        unsafe { vvexpf(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_exp_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvexp(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_sin_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvsinf(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_sin_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvsin(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_cos_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvcosf(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_cos_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvcos(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_tanh_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvtanhf(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_tanh_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvtanh(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_log_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvlogf(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_log_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvlog(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_sqrt_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvsqrtf(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_sqrt_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvsqrt(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_sqr_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vDSP_vsq(x.as_ptr(), 1, out.as_mut_ptr(), 1, x.len() as c_int) }
    }

    #[inline(always)]
    fn v_sqr_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vDSP_vsqD(x.as_ptr(), 1, out.as_mut_ptr(), 1, x.len() as c_int) }
    }

    #[inline(always)]
    fn v_add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe {
            vDSP_vadd(
                a.as_ptr(),
                1,
                b.as_ptr(),
                1,
                out.as_mut_ptr(),
                1,
                a.len() as c_int,
            )
        }
    }

    #[inline(always)]
    fn v_add_f64(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe {
            vDSP_vaddD(
                a.as_ptr(),
                1,
                b.as_ptr(),
                1,
                out.as_mut_ptr(),
                1,
                a.len() as c_int,
            )
        }
    }

    #[inline(always)]
    fn v_mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe {
            vDSP_vmul(
                a.as_ptr(),
                1,
                b.as_ptr(),
                1,
                out.as_mut_ptr(),
                1,
                a.len() as c_int,
            )
        }
    }

    #[inline(always)]
    fn v_mul_f64(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe {
            vDSP_vmulD(
                a.as_ptr(),
                1,
                b.as_ptr(),
                1,
                out.as_mut_ptr(),
                1,
                a.len() as c_int,
            )
        }
    }

    #[inline(always)]
    fn v_sub_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe {
            // vDSP_vsub(B, A, C) computes C = A - B, so we pass b, a to get a - b
            vDSP_vsub(
                b.as_ptr(),
                1,
                a.as_ptr(),
                1,
                out.as_mut_ptr(),
                1,
                a.len() as c_int,
            )
        }
    }

    #[inline(always)]
    fn v_sub_f64(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe {
            // vDSP_vsubD(B, A, C) computes C = A - B, so we pass b, a to get a - b
            vDSP_vsubD(
                b.as_ptr(),
                1,
                a.as_ptr(),
                1,
                out.as_mut_ptr(),
                1,
                a.len() as c_int,
            )
        }
    }

    #[inline(always)]
    fn v_div_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe {
            vDSP_vdiv(
                b.as_ptr(),
                1,
                a.as_ptr(),
                1,
                out.as_mut_ptr(),
                1,
                a.len() as c_int,
            )
        }
    }

    #[inline(always)]
    fn v_div_f64(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe {
            vDSP_vdivD(
                b.as_ptr(),
                1,
                a.as_ptr(),
                1,
                out.as_mut_ptr(),
                1,
                a.len() as c_int,
            )
        }
    }

    #[inline(always)]
    fn v_tan_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvtanf(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_tan_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvtan(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_recip_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvrecf(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_recip_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvrec(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_floor_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvfloorf(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_floor_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvfloor(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_ceil_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvceilf(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_ceil_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vvceil(out.as_mut_ptr(), x.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_neg_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vDSP_vneg(x.as_ptr(), 1, out.as_mut_ptr(), 1, x.len() as c_int) }
    }

    #[inline(always)]
    fn v_neg_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe { vDSP_vnegD(x.as_ptr(), 1, out.as_mut_ptr(), 1, x.len() as c_int) }
    }

    #[inline(always)]
    fn v_pow_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = a.len() as c_int;
        unsafe { vvpowf(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_pow_f64(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        assert_eq!(a.len(), b.len(), "Input slices must have same length");
        assert_eq!(
            a.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = a.len() as c_int;
        unsafe { vvpow(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), &n) }
    }

    #[inline(always)]
    fn v_abs_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vDSP_vabs(x.as_ptr(), 1, out.as_mut_ptr(), 1, n) }
    }

    #[inline(always)]
    fn v_abs_f64(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let n = x.len() as c_int;
        unsafe { vDSP_vabsD(x.as_ptr(), 1, out.as_mut_ptr(), 1, n) }
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
            vDSP_vmax(
                x.as_ptr(),
                1,
                zero.as_ptr(),
                1,
                out.as_mut_ptr(),
                1,
                x.len() as c_int,
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
            vDSP_vmaxD(
                x.as_ptr(),
                1,
                zero.as_ptr(),
                1,
                out.as_mut_ptr(),
                1,
                x.len() as c_int,
            )
        }
    }

    #[inline(always)]
    fn clamp_f32(&self, x: &[f32], min: f32, max: f32, out: &mut [f32]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let min_vec = UninitVec::<f32>::new(x.len()).full(min);
        let max_vec = UninitVec::<f32>::new(x.len()).full(max);
        unsafe {
            vDSP_vclip(
                x.as_ptr(),
                1,
                min_vec.as_ptr(),
                max_vec.as_ptr(),
                out.as_mut_ptr(),
                1,
                x.len() as c_int,
            )
        }
    }

    #[inline(always)]
    fn clamp_f64(&self, x: &[f64], min: f64, max: f64, out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        let min_vec = UninitVec::<f64>::new(x.len()).full(min);
        let max_vec = UninitVec::<f64>::new(x.len()).full(max);
        unsafe {
            vDSP_vclipD(
                x.as_ptr(),
                1,
                min_vec.as_ptr(),
                max_vec.as_ptr(),
                out.as_mut_ptr(),
                1,
                x.len() as c_int,
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
        unsafe {
            vDSP_vsadd(
                x.as_ptr(),
                1,
                &scalar,
                out.as_mut_ptr(),
                1,
                x.len() as c_int,
            )
        }
    }

    #[inline(always)]
    fn v_add_scalar_f64(&self, x: &[f64], scalar: f64, out: &mut [f64]) {
        assert_eq!(
            x.len(),
            out.len(),
            "Input and output slices must have same length"
        );
        unsafe {
            vDSP_vsaddD(
                x.as_ptr(),
                1,
                &scalar,
                out.as_mut_ptr(),
                1,
                x.len() as c_int,
            )
        }
    }

    // Vector min/max operations using vDSP
    #[inline(always)]
    fn min_v_f32(&self, x: &[f32]) -> f32 {
        if x.is_empty() {
            panic!("Cannot find minimum of empty vector");
        }
        let mut result = 0.0f32;
        unsafe { vDSP_minv(x.as_ptr(), 1, &mut result, x.len() as c_int) }
        result
    }

    #[inline(always)]
    fn min_v_f64(&self, x: &[f64]) -> f64 {
        if x.is_empty() {
            panic!("Cannot find minimum of empty vector");
        }
        let mut result = 0.0f64;
        unsafe { vDSP_minvD(x.as_ptr(), 1, &mut result, x.len() as c_int) }
        result
    }

    #[inline(always)]
    fn max_v_f32(&self, x: &[f32]) -> f32 {
        if x.is_empty() {
            panic!("Cannot find maximum of empty vector");
        }
        let mut result = 0.0f32;
        unsafe { vDSP_maxv(x.as_ptr(), 1, &mut result, x.len() as c_int) }
        result
    }

    #[inline(always)]
    fn max_v_f64(&self, x: &[f64]) -> f64 {
        if x.is_empty() {
            panic!("Cannot find maximum of empty vector");
        }
        let mut result = 0.0f64;
        unsafe { vDSP_maxvD(x.as_ptr(), 1, &mut result, x.len() as c_int) }
        result
    }

    #[inline(always)]
    fn min_vi_f32(&self, x: &[f32]) -> (f32, u64) {
        if x.is_empty() {
            panic!("Cannot find minimum of empty vector");
        }
        let mut result = 0.0f32;
        let mut index = 0;
        unsafe { vDSP_minvi(x.as_ptr(), 1, &mut result, &mut index, x.len() as c_int) }
        (result, index as u64)
    }

    #[inline(always)]
    fn min_vi_f64(&self, x: &[f64]) -> (f64, u64) {
        if x.is_empty() {
            panic!("Cannot find minimum of empty vector");
        }
        let mut result = 0.0f64;
        let mut index = 0;
        unsafe { vDSP_minviD(x.as_ptr(), 1, &mut result, &mut index, x.len() as c_int) }
        (result, index as u64)
    }

    #[inline(always)]
    fn max_vi_f32(&self, x: &[f32]) -> (f32, u64) {
        if x.is_empty() {
            panic!("Cannot find maximum of empty vector");
        }
        let mut result = 0.0f32;
        let mut index = 0;
        unsafe { vDSP_maxvi(x.as_ptr(), 1, &mut result, &mut index, x.len() as c_int) }
        (result, index as u64)
    }

    #[inline(always)]
    fn max_vi_f64(&self, x: &[f64]) -> (f64, u64) {
        if x.is_empty() {
            panic!("Cannot find maximum of empty vector");
        }
        let mut result = 0.0f64;
        let mut index = 0;
        unsafe { vDSP_maxviD(x.as_ptr(), 1, &mut result, &mut index, x.len() as c_int) }
        (result, index as u64)
    }

    // Index-only min/max operations using optimized vDSP
    #[inline(always)]
    fn min_i_f32(&self, x: &[f32]) -> u64 {
        if x.is_empty() {
            panic!("Cannot find minimum index of empty vector");
        }
        let mut result = 0.0f32;
        let mut index = 0;
        unsafe { vDSP_minvi(x.as_ptr(), 1, &mut result, &mut index, x.len() as c_int) }
        index as u64
    }

    #[inline(always)]
    fn min_i_f64(&self, x: &[f64]) -> u64 {
        if x.is_empty() {
            panic!("Cannot find minimum index of empty vector");
        }
        let mut result = 0.0f64;
        let mut index = 0;
        unsafe { vDSP_minviD(x.as_ptr(), 1, &mut result, &mut index, x.len() as c_int) }
        index as u64
    }

    #[inline(always)]
    fn max_i_f32(&self, x: &[f32]) -> u64 {
        if x.is_empty() {
            panic!("Cannot find maximum index of empty vector");
        }
        let mut result = 0.0f32;
        let mut index = 0;
        unsafe { vDSP_maxvi(x.as_ptr(), 1, &mut result, &mut index, x.len() as c_int) }
        index as u64
    }

    #[inline(always)]
    fn max_i_f64(&self, x: &[f64]) -> u64 {
        if x.is_empty() {
            panic!("Cannot find maximum index of empty vector");
        }
        let mut result = 0.0f64;
        let mut index = 0;
        unsafe { vDSP_maxviD(x.as_ptr(), 1, &mut result, &mut index, x.len() as c_int) }
        index as u64
    }

    // Combined min/max operations using optimized vDSP
    #[inline(always)]
    fn min_max_v_f32(&self, x: &[f32]) -> (f32, f32) {
        if x.is_empty() {
            panic!("Cannot find min/max of empty vector");
        }
        let mut min_result = 0.0f32;
        let mut max_result = 0.0f32;
        unsafe {
            vDSP_minv(x.as_ptr(), 1, &mut min_result, x.len() as c_int);
            vDSP_maxv(x.as_ptr(), 1, &mut max_result, x.len() as c_int);
        }
        (min_result, max_result)
    }

    #[inline(always)]
    fn min_max_v_f64(&self, x: &[f64]) -> (f64, f64) {
        if x.is_empty() {
            panic!("Cannot find min/max of empty vector");
        }
        let mut min_result = 0.0f64;
        let mut max_result = 0.0f64;
        unsafe {
            vDSP_minvD(x.as_ptr(), 1, &mut min_result, x.len() as c_int);
            vDSP_maxvD(x.as_ptr(), 1, &mut max_result, x.len() as c_int);
        }
        (min_result, max_result)
    }

    #[inline(always)]
    fn min_max_vi_f32(&self, x: &[f32]) -> ((f32, u64), (f32, u64)) {
        if x.is_empty() {
            panic!("Cannot find min/max of empty vector");
        }
        let mut min_result = 0.0f32;
        let mut min_index = 0;
        let mut max_result = 0.0f32;
        let mut max_index = 0;
        unsafe {
            vDSP_minvi(
                x.as_ptr(),
                1,
                &mut min_result,
                &mut min_index,
                x.len() as c_int,
            );
            vDSP_maxvi(
                x.as_ptr(),
                1,
                &mut max_result,
                &mut max_index,
                x.len() as c_int,
            );
        }
        (
            (min_result, min_index as u64),
            (max_result, max_index as u64),
        )
    }

    #[inline(always)]
    fn min_max_vi_f64(&self, x: &[f64]) -> ((f64, u64), (f64, u64)) {
        if x.is_empty() {
            panic!("Cannot find min/max of empty vector");
        }
        let mut min_result = 0.0f64;
        let mut min_index = 0;
        let mut max_result = 0.0f64;
        let mut max_index = 0;
        unsafe {
            vDSP_minviD(
                x.as_ptr(),
                1,
                &mut min_result,
                &mut min_index,
                x.len() as c_int,
            );
            vDSP_maxviD(
                x.as_ptr(),
                1,
                &mut max_result,
                &mut max_index,
                x.len() as c_int,
            );
        }
        (
            (min_result, min_index as u64),
            (max_result, max_index as u64),
        )
    }

    #[inline(always)]
    fn min_max_i_f32(&self, x: &[f32]) -> (u64, u64) {
        if x.is_empty() {
            panic!("Cannot find min/max indices of empty vector");
        }
        let mut min_result = 0.0f32;
        let mut min_index = 0;
        let mut max_result = 0.0f32;
        let mut max_index = 0;
        unsafe {
            vDSP_minvi(
                x.as_ptr(),
                1,
                &mut min_result,
                &mut min_index,
                x.len() as c_int,
            );
            vDSP_maxvi(
                x.as_ptr(),
                1,
                &mut max_result,
                &mut max_index,
                x.len() as c_int,
            );
        }
        (min_index as u64, max_index as u64)
    }

    #[inline(always)]
    fn min_max_i_f64(&self, x: &[f64]) -> (u64, u64) {
        if x.is_empty() {
            panic!("Cannot find min/max indices of empty vector");
        }
        let mut min_result = 0.0f64;
        let mut min_index = 0;
        let mut max_result = 0.0f64;
        let mut max_index = 0;
        unsafe {
            vDSP_minviD(
                x.as_ptr(),
                1,
                &mut min_result,
                &mut min_index,
                x.len() as c_int,
            );
            vDSP_maxviD(
                x.as_ptr(),
                1,
                &mut max_result,
                &mut max_index,
                x.len() as c_int,
            );
        }
        (min_index as u64, max_index as u64)
    }
}
