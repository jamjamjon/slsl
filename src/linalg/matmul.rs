use anyhow::Result;

use crate::{global_backend, DType, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    /// Compute matrix multiplication between two 2D tensors
    ///
    /// # Arguments
    /// * `other` - The other tensor to compute matrix multiplication with
    ///
    /// # Returns
    /// * `Result<Tensor>` - The matrix multiplication result
    ///
    /// # Errors
    /// * Returns error if tensors are not 2D
    /// * Returns error if matrix dimensions are incompatible
    /// * Returns error if dtypes don't match
    /// * Returns error if dtype is not supported
    pub fn matmul(&self, other: &Self) -> Result<Tensor> {
        if self.rank() != 2 || other.rank() != 2 {
            anyhow::bail!("matmul() currently only supports 2D tensors");
        }

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        if k != other.shape[0] {
            anyhow::bail!(
                "Matrix dimensions incompatible for multiplication: ({}, {}) × ({}, {})",
                m,
                k,
                other.shape[0],
                n
            );
        }

        if self.dtype != other.dtype {
            anyhow::bail!("Both tensors must have the same dtype for matrix multiplication");
        }

        // Check if both tensors are contiguous for optimized path
        if self.is_contiguous() && other.is_contiguous() {
            self.matmul_contiguous(other)
        } else {
            self.matmul_non_contiguous(other)
        }
    }

    /// Optimized matrix multiplication for contiguous tensors
    #[inline(always)]
    fn matmul_contiguous(&self, other: &Self) -> Result<Tensor> {
        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];
        let backend = global_backend();

        match self.dtype {
            DType::Fp32 => {
                let a_data = self.as_slice::<f32>()?;
                let b_data = other.as_slice::<f32>()?;
                let y = UninitVec::<f32>::new(m * n).init_with(|y| unsafe {
                    backend.gemm_f32(
                        m,
                        n,
                        k,
                        a_data.as_ptr(),
                        k,
                        b_data.as_ptr(),
                        n,
                        y.as_mut_ptr(),
                        n,
                    );
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Fp64 => {
                let a_data = self.as_slice::<f64>()?;
                let b_data = other.as_slice::<f64>()?;
                let y = UninitVec::<f64>::new(m * n).init_with(|y| unsafe {
                    backend.gemm_f64(
                        m,
                        n,
                        k,
                        a_data.as_ptr(),
                        k,
                        b_data.as_ptr(),
                        n,
                        y.as_mut_ptr(),
                        n,
                    );
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Int8 => {
                let a_data = self.as_slice::<i8>()?;
                let b_data = other.as_slice::<i8>()?;
                let y = UninitVec::<i8>::new(m * n).init_with(|y| unsafe {
                    backend.gemm_i8(
                        m,
                        n,
                        k,
                        a_data.as_ptr(),
                        k,
                        b_data.as_ptr(),
                        n,
                        y.as_mut_ptr(),
                        n,
                    );
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Int16 => {
                let a_data = self.as_slice::<i16>()?;
                let b_data = other.as_slice::<i16>()?;
                let y = UninitVec::<i16>::new(m * n).init_with(|y| unsafe {
                    backend.gemm_i16(
                        m,
                        n,
                        k,
                        a_data.as_ptr(),
                        k,
                        b_data.as_ptr(),
                        n,
                        y.as_mut_ptr(),
                        n,
                    );
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Int32 => {
                let a_data = self.as_slice::<i32>()?;
                let b_data = other.as_slice::<i32>()?;
                let y = UninitVec::<i32>::new(m * n).init_with(|y| unsafe {
                    backend.gemm_i32(
                        m,
                        n,
                        k,
                        a_data.as_ptr(),
                        k,
                        b_data.as_ptr(),
                        n,
                        y.as_mut_ptr(),
                        n,
                    );
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Int64 => {
                let a_data = self.as_slice::<i64>()?;
                let b_data = other.as_slice::<i64>()?;
                let y = UninitVec::<i64>::new(m * n).init_with(|y| unsafe {
                    backend.gemm_i64(
                        m,
                        n,
                        k,
                        a_data.as_ptr(),
                        k,
                        b_data.as_ptr(),
                        n,
                        y.as_mut_ptr(),
                        n,
                    );
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Uint8 => {
                let a_data = self.as_slice::<u8>()?;
                let b_data = other.as_slice::<u8>()?;
                let y = UninitVec::<u8>::new(m * n).init_with(|y| unsafe {
                    backend.gemm_u8(
                        m,
                        n,
                        k,
                        a_data.as_ptr(),
                        k,
                        b_data.as_ptr(),
                        n,
                        y.as_mut_ptr(),
                        n,
                    );
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Uint16 => {
                let a_data = self.as_slice::<u16>()?;
                let b_data = other.as_slice::<u16>()?;
                let y = UninitVec::<u16>::new(m * n).init_with(|y| unsafe {
                    backend.gemm_u16(
                        m,
                        n,
                        k,
                        a_data.as_ptr(),
                        k,
                        b_data.as_ptr(),
                        n,
                        y.as_mut_ptr(),
                        n,
                    );
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Uint32 => {
                let a_data = self.as_slice::<u32>()?;
                let b_data = other.as_slice::<u32>()?;
                let y = UninitVec::<u32>::new(m * n).init_with(|y| unsafe {
                    backend.gemm_u32(
                        m,
                        n,
                        k,
                        a_data.as_ptr(),
                        k,
                        b_data.as_ptr(),
                        n,
                        y.as_mut_ptr(),
                        n,
                    );
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Uint64 => {
                let a_data = self.as_slice::<u64>()?;
                let b_data = other.as_slice::<u64>()?;
                let y = UninitVec::<u64>::new(m * n).init_with(|y| unsafe {
                    backend.gemm_u64(
                        m,
                        n,
                        k,
                        a_data.as_ptr(),
                        k,
                        b_data.as_ptr(),
                        n,
                        y.as_mut_ptr(),
                        n,
                    );
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Fp16 => {
                let a_data = self.as_slice::<half::f16>()?;
                let b_data = other.as_slice::<half::f16>()?;
                let y = UninitVec::<half::f16>::new(m * n).init_with(|y| unsafe {
                    backend.gemm_f16(
                        m,
                        n,
                        k,
                        a_data.as_ptr(),
                        k,
                        b_data.as_ptr(),
                        n,
                        y.as_mut_ptr(),
                        n,
                    );
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Bf16 => {
                let a_data = self.as_slice::<half::bf16>()?;
                let b_data = other.as_slice::<half::bf16>()?;
                let y = UninitVec::<half::bf16>::new(m * n).init_with(|y| unsafe {
                    backend.gemm_bf16(
                        m,
                        n,
                        k,
                        a_data.as_ptr(),
                        k,
                        b_data.as_ptr(),
                        n,
                        y.as_mut_ptr(),
                        n,
                    );
                });
                Tensor::from_vec(y, [m, n])
            }
            _ => anyhow::bail!("Unsupported dtype for matmul operation: {:?}", self.dtype),
        }
    }

    /// Matrix multiplication for non-contiguous tensors
    #[inline(always)]
    fn matmul_non_contiguous(&self, other: &Self) -> Result<Tensor> {
        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        match self.dtype {
            DType::Fp32 => {
                // TODO: optimize
                let y = UninitVec::<f32>::new(m * n).init_with(|y| {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0f32;
                            for l in 0..k {
                                let a_val = self.at::<f32>([i, l]);
                                let b_val = other.at::<f32>([l, j]);
                                sum += a_val * b_val;
                            }
                            y[i * n + j] = sum;
                        }
                    }
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Fp64 => {
                let y = UninitVec::<f64>::new(m * n).init_with(|y| {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0f64;
                            for l in 0..k {
                                let a_val = self.at::<f64>([i, l]);
                                let b_val = other.at::<f64>([l, j]);
                                sum += a_val * b_val;
                            }
                            y[i * n + j] = sum;
                        }
                    }
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Int8 => {
                let y = UninitVec::<i8>::new(m * n).init_with(|y| {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0i8;
                            for l in 0..k {
                                let a_val = self.at::<i8>([i, l]);
                                let b_val = other.at::<i8>([l, j]);
                                sum += a_val * b_val;
                            }
                            y[i * n + j] = sum;
                        }
                    }
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Int16 => {
                let y = UninitVec::<i16>::new(m * n).init_with(|y| {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0i16;
                            for l in 0..k {
                                let a_val = self.at::<i16>([i, l]);
                                let b_val = other.at::<i16>([l, j]);
                                sum += a_val * b_val;
                            }
                            y[i * n + j] = sum;
                        }
                    }
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Int32 => {
                let y = UninitVec::<i32>::new(m * n).init_with(|y| {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0i32;
                            for l in 0..k {
                                let a_val = self.at::<i32>([i, l]);
                                let b_val = other.at::<i32>([l, j]);
                                sum += a_val * b_val;
                            }
                            y[i * n + j] = sum;
                        }
                    }
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Int64 => {
                let y = UninitVec::<i64>::new(m * n).init_with(|y| {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0i64;
                            for l in 0..k {
                                let a_val = self.at::<i64>([i, l]);
                                let b_val = other.at::<i64>([l, j]);
                                sum += a_val * b_val;
                            }
                            y[i * n + j] = sum;
                        }
                    }
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Uint8 => {
                let y = UninitVec::<u8>::new(m * n).init_with(|y| {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0u8;
                            for l in 0..k {
                                let a_val = self.at::<u8>([i, l]);
                                let b_val = other.at::<u8>([l, j]);
                                sum += a_val * b_val;
                            }
                            y[i * n + j] = sum;
                        }
                    }
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Uint16 => {
                let y = UninitVec::<u16>::new(m * n).init_with(|y| {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0u16;
                            for l in 0..k {
                                let a_val = self.at::<u16>([i, l]);
                                let b_val = other.at::<u16>([l, j]);
                                sum += a_val * b_val;
                            }
                            y[i * n + j] = sum;
                        }
                    }
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Uint32 => {
                let y = UninitVec::<u32>::new(m * n).init_with(|y| {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0u32;
                            for l in 0..k {
                                let a_val = self.at::<u32>([i, l]);
                                let b_val = other.at::<u32>([l, j]);
                                sum += a_val * b_val;
                            }
                            y[i * n + j] = sum;
                        }
                    }
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Uint64 => {
                let y = UninitVec::<u64>::new(m * n).init_with(|y| {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0u64;
                            for l in 0..k {
                                let a_val = self.at::<u64>([i, l]);
                                let b_val = other.at::<u64>([l, j]);
                                sum += a_val * b_val;
                            }
                            y[i * n + j] = sum;
                        }
                    }
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Fp16 => {
                let y = UninitVec::<half::f16>::new(m * n).init_with(|y| {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = half::f16::ZERO;
                            for l in 0..k {
                                let a_val = self.at::<half::f16>([i, l]);
                                let b_val = other.at::<half::f16>([l, j]);
                                sum = half::f16::from_f32(
                                    sum.to_f32() + a_val.to_f32() * b_val.to_f32(),
                                );
                            }
                            y[i * n + j] = sum;
                        }
                    }
                });
                Tensor::from_vec(y, [m, n])
            }
            DType::Bf16 => {
                let y = UninitVec::<half::bf16>::new(m * n).init_with(|y| {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = half::bf16::ZERO;
                            for l in 0..k {
                                let a_val = self.at::<half::bf16>([i, l]);
                                let b_val = other.at::<half::bf16>([l, j]);
                                sum = half::bf16::from_f32(
                                    sum.to_f32() + a_val.to_f32() * b_val.to_f32(),
                                );
                            }
                            y[i * n + j] = sum;
                        }
                    }
                });
                Tensor::from_vec(y, [m, n])
            }
            _ => anyhow::bail!("Unsupported dtype for matmul operation: {:?}", self.dtype),
        }
    }
}
