use crate::UninitVec;
use crate::{DType, Dim, StorageTrait, Tensor, TensorBase};
use anyhow::Result;
use wide::{f32x4, f32x8, f64x2, f64x4};

impl<S: StorageTrait> TensorBase<S> {
    /// Computes the softmax function along the specified dimension.
    ///
    /// The softmax function is defined as:
    /// softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    ///
    /// This implementation uses the numerically stable version:
    /// softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    ///
    /// This optimized implementation avoids unnecessary dtype conversions and
    /// broadcast operations for better performance.
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to compute the softmax
    ///
    /// # Returns
    /// A new tensor with the same shape and dtype as the input, containing the softmax values
    ///
    /// # Examples
    /// ```
    /// use slsl::Tensor;
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3]).unwrap();
    /// let result = x.softmax(0).unwrap();
    /// ```
    pub fn softmax<D: Dim>(&self, dim: D) -> Result<Tensor> {
        let dim_idx = dim.to_dim(self.rank())?;

        if self.dims()[dim_idx] == 1 {
            // If the dimension size is 1, softmax is just a tensor of ones in original dtype
            match self.dtype() {
                DType::Fp32 => return Tensor::ones::<f32>(self.dims()),
                DType::Fp64 => return Tensor::ones::<f64>(self.dims()),
                DType::Fp16 => return Tensor::ones::<half::f16>(self.dims()),
                DType::Bf16 => return Tensor::ones::<half::bf16>(self.dims()),
                _ => {
                    return Err(anyhow::anyhow!(
                        "softmax only supports floating-point types, got {:?}",
                        self.dtype()
                    ))
                }
            }
        }

        // Use native dtype for computation to avoid unnecessary conversions
        match self.dtype() {
            DType::Fp32 => self.softmax_impl::<f32>(dim_idx),
            DType::Fp64 => self.softmax_impl::<f64>(dim_idx),
            DType::Fp16 => self.softmax_impl::<half::f16>(dim_idx),
            DType::Bf16 => self.softmax_impl::<half::bf16>(dim_idx),
            _ => anyhow::bail!(
                "softmax only supports floating-point types, got {:?}",
                self.dtype()
            ),
        }
    }

    /// Internal optimized implementation of softmax with fused operations
    fn softmax_impl<T>(&self, dim_idx: usize) -> Result<Tensor>
    where
        T: num_traits::Float + Copy + Send + Sync + 'static + crate::TensorElement,
    {
        // Fast path for 1D tensors (most common case)
        if self.rank() == 1 && dim_idx == 0 {
            return self.softmax_1d::<T>();
        }

        // Fast path for contiguous tensors with softmax along the last dimension
        if self.is_contiguous() && dim_idx == self.rank() - 1 {
            let data = self.as_slice::<T>()?;
            let dims = self.dims();
            let last_dim_size = dims[dims.len() - 1];
            let batch_size = self.numel() / last_dim_size;
            return self.softmax_contiguous_last_dim::<T>(data, batch_size, last_dim_size);
        }

        // Fallback to general implementation
        self.softmax_general::<T>(dim_idx)
    }

    /// Generic SIMD implementation dispatcher
    fn softmax_contiguous_last_dim<T>(
        &self,
        data: &[T],
        batch_size: usize,
        last_dim_size: usize,
    ) -> Result<Tensor>
    where
        T: Copy + 'static + crate::TensorElement + num_traits::Float,
    {
        match T::DTYPE {
            crate::DType::Fp32 => {
                let simd_width = crate::backend::simd::choose_f32_simd_width(last_dim_size);
                match simd_width {
                    8 => {
                        // Use f32x8 SIMD
                        let result_data = self.softmax_contiguous_simd_f32::<T, 8>(
                            data,
                            batch_size,
                            last_dim_size,
                        );

                        Tensor::from_vec(result_data, self.dims())
                    }
                    4 => {
                        // Use f32x4 SIMD
                        let result_data = self.softmax_contiguous_simd_f32::<T, 4>(
                            data,
                            batch_size,
                            last_dim_size,
                        );
                        Tensor::from_vec(result_data, self.dims())
                    }
                    _ => {
                        // Scalar fallback
                        let result_data =
                            self.softmax_contiguous_scalar(data, batch_size, last_dim_size);
                        Tensor::from_vec(result_data, self.dims())
                    }
                }
            }
            crate::DType::Fp64 => {
                let simd_width = crate::backend::simd::choose_f64_simd_width(last_dim_size);
                match simd_width {
                    4 => {
                        // Use f64x4 SIMD
                        let result_data = self.softmax_contiguous_simd_f64::<T, 4>(
                            data,
                            batch_size,
                            last_dim_size,
                        );
                        Tensor::from_vec(result_data, self.dims())
                    }
                    2 => {
                        // Use f64x2 SIMD
                        let result_data = self.softmax_contiguous_simd_f64::<T, 2>(
                            data,
                            batch_size,
                            last_dim_size,
                        );
                        Tensor::from_vec(result_data, self.dims())
                    }
                    _ => {
                        // Scalar fallback
                        let result_data =
                            self.softmax_contiguous_scalar(data, batch_size, last_dim_size);
                        Tensor::from_vec(result_data, self.dims())
                    }
                }
            }
            _ => {
                // Fallback to scalar implementation for other types (f16, bf16, etc.)
                let result_data = self.softmax_contiguous_scalar(data, batch_size, last_dim_size);
                Tensor::from_vec(result_data, self.dims())
            }
        }
    }

    /// Scalar fallback implementation for contiguous softmax
    fn softmax_contiguous_scalar<T>(
        &self,
        data: &[T],
        batch_size: usize,
        last_dim_size: usize,
    ) -> Vec<T>
    where
        T: num_traits::Float + Copy + Send + Sync + 'static + crate::TensorElement,
    {
        UninitVec::new(self.numel()).init_with(|result_slice: &mut [T]| {
            for batch_idx in 0..batch_size {
                let start_idx = batch_idx * last_dim_size;
                let batch_slice = &data[start_idx..start_idx + last_dim_size];
                let result_batch = &mut result_slice[start_idx..start_idx + last_dim_size];

                // Find max for numerical stability
                let mut max_val = T::neg_infinity();
                for &val in batch_slice {
                    if val > max_val {
                        max_val = val;
                    }
                }

                // Compute exp and sum
                let mut exp_sum = <T as num_traits::Zero>::zero();
                for (i, &val) in batch_slice.iter().enumerate() {
                    let exp_val = (val - max_val).exp();
                    result_batch[i] = exp_val;
                    exp_sum = exp_sum + exp_val;
                }

                // Normalize
                let inv_sum = <T as num_traits::One>::one() / exp_sum;
                for result_val in result_batch.iter_mut() {
                    *result_val = *result_val * inv_sum;
                }
            }
        })
    }

    /// Generic SIMD implementation for f32 types
    fn softmax_contiguous_simd_f32<T, const WIDTH: usize>(
        &self,
        data: &[T],
        batch_size: usize,
        last_dim_size: usize,
    ) -> Vec<T>
    where
        T: Copy + 'static + crate::TensorElement,
    {
        let input_f32: &[f32] = unsafe { std::mem::transmute(data) };

        UninitVec::new(self.numel()).init_with(|result_slice: &mut [T]| {
            let result_f32: &mut [f32] = unsafe { std::mem::transmute(result_slice) };

            let chunks_per_batch = last_dim_size / WIDTH;

            for batch_idx in 0..batch_size {
                let batch_start = batch_idx * last_dim_size;
                let batch_end = batch_start + last_dim_size;

                // Find maximum using SIMD
                let mut max_val = f32::NEG_INFINITY;

                // Process SIMD chunks
                if WIDTH == 4 {
                    let mut max_vec = f32x4::splat(f32::NEG_INFINITY);
                    for chunk_idx in 0..chunks_per_batch {
                        let chunk_start = batch_start + chunk_idx * WIDTH;
                        let mut chunk_array = [0.0f32; 4];
                        chunk_array.copy_from_slice(&input_f32[chunk_start..chunk_start + WIDTH]);
                        let chunk = f32x4::from(chunk_array);
                        max_vec = max_vec.max(chunk);
                    }
                    let max_array = max_vec.to_array();
                    max_val = max_array.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                } else if WIDTH == 8 {
                    let mut max_vec = f32x8::splat(f32::NEG_INFINITY);
                    for chunk_idx in 0..chunks_per_batch {
                        let chunk_start = batch_start + chunk_idx * WIDTH;
                        let mut chunk_array = [0.0f32; 8];
                        chunk_array.copy_from_slice(&input_f32[chunk_start..chunk_start + WIDTH]);
                        let chunk = f32x8::from(chunk_array);
                        max_vec = max_vec.max(chunk);
                    }
                    let max_array = max_vec.to_array();
                    max_val = max_array.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                }

                // Handle remainder
                for &val in input_f32
                    .iter()
                    .take(batch_end)
                    .skip(batch_start + chunks_per_batch * WIDTH)
                {
                    max_val = max_val.max(val);
                }

                // Compute exp and sum using SIMD
                let mut sum = 0.0f32;

                if WIDTH == 4 {
                    let max_vec = f32x4::splat(max_val);
                    let mut sum_vec = f32x4::splat(0.0);

                    for chunk_idx in 0..chunks_per_batch {
                        let chunk_start = batch_start + chunk_idx * WIDTH;
                        let mut chunk_array = [0.0f32; 4];
                        chunk_array.copy_from_slice(&input_f32[chunk_start..chunk_start + WIDTH]);
                        let chunk = f32x4::from(chunk_array);
                        let exp_chunk = (chunk - max_vec).exp();
                        let exp_array = exp_chunk.to_array();
                        result_f32[chunk_start..chunk_start + WIDTH].copy_from_slice(&exp_array);
                        sum_vec += exp_chunk;
                    }
                    sum = sum_vec.to_array().iter().sum::<f32>();
                } else if WIDTH == 8 {
                    let max_vec = f32x8::splat(max_val);
                    let mut sum_vec = f32x8::splat(0.0);

                    for chunk_idx in 0..chunks_per_batch {
                        let chunk_start = batch_start + chunk_idx * WIDTH;
                        let mut chunk_array = [0.0f32; 8];
                        chunk_array.copy_from_slice(&input_f32[chunk_start..chunk_start + WIDTH]);
                        let chunk = f32x8::from(chunk_array);
                        let exp_chunk = (chunk - max_vec).exp();
                        let exp_array = exp_chunk.to_array();
                        result_f32[chunk_start..chunk_start + WIDTH].copy_from_slice(&exp_array);
                        sum_vec += exp_chunk;
                    }
                    sum = sum_vec.to_array().iter().sum::<f32>();
                }

                // Handle remainder
                for (idx, &val) in input_f32
                    .iter()
                    .enumerate()
                    .take(batch_end)
                    .skip(batch_start + chunks_per_batch * WIDTH)
                {
                    let exp_val = (val - max_val).exp();
                    result_f32[idx] = exp_val;
                    sum += exp_val;
                }

                // Normalize using SIMD
                let inv_sum = 1.0 / sum;

                if WIDTH == 4 {
                    let inv_sum_vec = f32x4::splat(inv_sum);
                    for chunk_idx in 0..chunks_per_batch {
                        let chunk_start = batch_start + chunk_idx * WIDTH;
                        let mut chunk_array = [0.0f32; 4];
                        chunk_array.copy_from_slice(&result_f32[chunk_start..chunk_start + WIDTH]);
                        let chunk = f32x4::from(chunk_array);
                        let normalized = chunk * inv_sum_vec;
                        let norm_array = normalized.to_array();
                        result_f32[chunk_start..chunk_start + WIDTH].copy_from_slice(&norm_array);
                    }
                } else if WIDTH == 8 {
                    let inv_sum_vec = f32x8::splat(inv_sum);
                    for chunk_idx in 0..chunks_per_batch {
                        let chunk_start = batch_start + chunk_idx * WIDTH;
                        let mut chunk_array = [0.0f32; 8];
                        chunk_array.copy_from_slice(&result_f32[chunk_start..chunk_start + WIDTH]);
                        let chunk = f32x8::from(chunk_array);
                        let normalized = chunk * inv_sum_vec;
                        let norm_array = normalized.to_array();
                        result_f32[chunk_start..chunk_start + WIDTH].copy_from_slice(&norm_array);
                    }
                }

                for val in result_f32
                    .iter_mut()
                    .take(batch_end)
                    .skip(batch_start + chunks_per_batch * WIDTH)
                {
                    *val *= inv_sum;
                }
            }
        })
    }

    /// Generic SIMD implementation for f64 types
    fn softmax_contiguous_simd_f64<T, const WIDTH: usize>(
        &self,
        data: &[T],
        batch_size: usize,
        last_dim_size: usize,
    ) -> Vec<T>
    where
        T: Copy + 'static + crate::TensorElement,
    {
        let input_f64: &[f64] = unsafe { std::mem::transmute(data) };

        UninitVec::new(self.numel()).init_with(|result_slice: &mut [T]| {
            let result_f64: &mut [f64] = unsafe { std::mem::transmute(result_slice) };

            let chunks_per_batch = last_dim_size / WIDTH;

            for batch_idx in 0..batch_size {
                let batch_start = batch_idx * last_dim_size;
                let batch_end = batch_start + last_dim_size;

                // Find maximum using SIMD
                let mut max_val = f64::NEG_INFINITY;

                // Process SIMD chunks
                if WIDTH == 2 {
                    let mut max_vec = f64x2::splat(f64::NEG_INFINITY);
                    for chunk_idx in 0..chunks_per_batch {
                        let chunk_start = batch_start + chunk_idx * WIDTH;
                        let mut chunk_array = [0.0f64; 2];
                        chunk_array.copy_from_slice(&input_f64[chunk_start..chunk_start + WIDTH]);
                        let chunk = f64x2::from(chunk_array);
                        max_vec = max_vec.max(chunk);
                    }
                    let max_array = max_vec.to_array();
                    max_val = max_array.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                } else if WIDTH == 4 {
                    let mut max_vec = f64x4::splat(f64::NEG_INFINITY);
                    for chunk_idx in 0..chunks_per_batch {
                        let chunk_start = batch_start + chunk_idx * WIDTH;
                        let mut chunk_array = [0.0f64; 4];
                        chunk_array.copy_from_slice(&input_f64[chunk_start..chunk_start + WIDTH]);
                        let chunk = f64x4::from(chunk_array);
                        max_vec = max_vec.max(chunk);
                    }
                    let max_array = max_vec.to_array();
                    max_val = max_array.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                }

                // Handle remainder
                for &val in input_f64
                    .iter()
                    .take(batch_end)
                    .skip(batch_start + chunks_per_batch * WIDTH)
                {
                    max_val = max_val.max(val);
                }

                // Compute exp and sum using SIMD
                let mut sum = 0.0f64;

                if WIDTH == 2 {
                    let max_vec = f64x2::splat(max_val);
                    let mut sum_vec = f64x2::splat(0.0);

                    for chunk_idx in 0..chunks_per_batch {
                        let chunk_start = batch_start + chunk_idx * WIDTH;
                        let mut chunk_array = [0.0f64; 2];
                        chunk_array.copy_from_slice(&input_f64[chunk_start..chunk_start + WIDTH]);
                        let chunk = f64x2::from(chunk_array);
                        let exp_chunk = (chunk - max_vec).exp();
                        let exp_array = exp_chunk.to_array();
                        result_f64[chunk_start..chunk_start + WIDTH].copy_from_slice(&exp_array);
                        sum_vec += exp_chunk;
                    }
                    sum = sum_vec.to_array().iter().sum::<f64>();
                } else if WIDTH == 4 {
                    let max_vec = f64x4::splat(max_val);
                    let mut sum_vec = f64x4::splat(0.0);

                    for chunk_idx in 0..chunks_per_batch {
                        let chunk_start = batch_start + chunk_idx * WIDTH;
                        let mut chunk_array = [0.0f64; 4];
                        chunk_array.copy_from_slice(&input_f64[chunk_start..chunk_start + WIDTH]);
                        let chunk = f64x4::from(chunk_array);
                        let exp_chunk = (chunk - max_vec).exp();
                        let exp_array = exp_chunk.to_array();
                        result_f64[chunk_start..chunk_start + WIDTH].copy_from_slice(&exp_array);
                        sum_vec += exp_chunk;
                    }
                    sum = sum_vec.to_array().iter().sum::<f64>();
                }

                // Handle remainder
                for (idx, &val) in input_f64
                    .iter()
                    .enumerate()
                    .take(batch_end)
                    .skip(batch_start + chunks_per_batch * WIDTH)
                {
                    let exp_val = (val - max_val).exp();
                    result_f64[idx] = exp_val;
                    sum += exp_val;
                }

                // Normalize using SIMD
                let inv_sum = 1.0 / sum;

                if WIDTH == 2 {
                    let inv_sum_vec = f64x2::splat(inv_sum);
                    for chunk_idx in 0..chunks_per_batch {
                        let chunk_start = batch_start + chunk_idx * WIDTH;
                        let mut chunk_array = [0.0f64; 2];
                        chunk_array.copy_from_slice(&result_f64[chunk_start..chunk_start + WIDTH]);
                        let chunk = f64x2::from(chunk_array);
                        let normalized = chunk * inv_sum_vec;
                        let norm_array = normalized.to_array();
                        result_f64[chunk_start..chunk_start + WIDTH].copy_from_slice(&norm_array);
                    }
                } else if WIDTH == 4 {
                    let inv_sum_vec = f64x4::splat(inv_sum);
                    for chunk_idx in 0..chunks_per_batch {
                        let chunk_start = batch_start + chunk_idx * WIDTH;
                        let mut chunk_array = [0.0f64; 4];
                        chunk_array.copy_from_slice(&result_f64[chunk_start..chunk_start + WIDTH]);
                        let chunk = f64x4::from(chunk_array);
                        let normalized = chunk * inv_sum_vec;
                        let norm_array = normalized.to_array();
                        result_f64[chunk_start..chunk_start + WIDTH].copy_from_slice(&norm_array);
                    }
                }

                for val in result_f64
                    .iter_mut()
                    .take(batch_end)
                    .skip(batch_start + chunks_per_batch * WIDTH)
                {
                    *val *= inv_sum;
                }
            }
        })
    }

    /// 1D softmax
    /// This is the most common case and deserves special optimization
    fn softmax_1d<T>(&self) -> Result<Tensor>
    where
        T: num_traits::Float + Copy + Send + Sync + 'static + crate::TensorElement,
    {
        let data = self.as_slice::<T>()?;
        let len = data.len();

        // Use SIMD for supported types with sufficient size
        match T::DTYPE {
            crate::DType::Fp32 => {
                if len >= 8 {
                    return self.softmax_1d_simd_f32::<T, 8>(data, len);
                } else if len >= 4 {
                    return self.softmax_1d_simd_f32::<T, 4>(data, len);
                }
            }
            crate::DType::Fp64 => {
                if len >= 4 {
                    return self.softmax_1d_simd_f64::<T, 4>(data, len);
                } else if len >= 2 {
                    return self.softmax_1d_simd_f64::<T, 2>(data, len);
                }
            }
            _ => {}
        }

        // Use UninitVec for zero-cost memory allocation
        Tensor::from_vec(
            UninitVec::new(len).init_with(|result_slice: &mut [T]| {
                // Step 1: Find max for numerical stability
                let mut max_val = T::neg_infinity();
                for &val in data {
                    if val > max_val {
                        max_val = val;
                    }
                }

                // Step 2: Compute exp(x - max) and sum in single pass
                let mut exp_sum = <T as num_traits::Zero>::zero();
                for (i, &val) in data.iter().enumerate() {
                    let exp_val = (val - max_val).exp();
                    result_slice[i] = exp_val;
                    exp_sum = exp_sum + exp_val;
                }

                // Step 3: Normalize (pre-compute inverse to avoid division in loop)
                let inv_sum = <T as num_traits::One>::one() / exp_sum;
                for result_val in result_slice.iter_mut() {
                    *result_val = *result_val * inv_sum;
                }
            }),
            self.dims(),
        )
    }

    /// Generic 1D SIMD implementation for f32 types
    #[inline(always)]
    fn softmax_1d_simd_f32<T, const WIDTH: usize>(&self, data: &[T], len: usize) -> Result<Tensor>
    where
        T: Copy + 'static + crate::TensorElement,
    {
        let input_f32: &[f32] = unsafe { std::mem::transmute(data) };

        let result_data = UninitVec::new(len).init_with(|result_slice: &mut [T]| {
            let result_f32: &mut [f32] = unsafe { std::mem::transmute(result_slice) };

            let chunks = len / WIDTH;

            // Pass 1: Find maximum value using SIMD
            let mut max_val = f32::NEG_INFINITY;

            if WIDTH == 4 {
                let mut max_vec = f32x4::splat(f32::NEG_INFINITY);
                for chunk_idx in 0..chunks {
                    let start = chunk_idx * WIDTH;
                    let mut chunk_array = [0.0f32; 4];
                    chunk_array.copy_from_slice(&input_f32[start..start + WIDTH]);
                    let chunk = f32x4::from(chunk_array);
                    max_vec = max_vec.max(chunk);
                }
                let max_array = max_vec.to_array();
                max_val = max_array.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            } else if WIDTH == 8 {
                let mut max_vec = f32x8::splat(f32::NEG_INFINITY);
                for chunk_idx in 0..chunks {
                    let start = chunk_idx * WIDTH;
                    let mut chunk_array = [0.0f32; 8];
                    chunk_array.copy_from_slice(&input_f32[start..start + WIDTH]);
                    let chunk = f32x8::from(chunk_array);
                    max_vec = max_vec.max(chunk);
                }
                let max_array = max_vec.to_array();
                max_val = max_array.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            }

            // Handle remainder
            for &val in input_f32.iter().take(len).skip(chunks * WIDTH) {
                max_val = max_val.max(val);
            }

            // Pass 2: Compute exp and sum
            let mut sum = 0.0f32;

            if WIDTH == 4 {
                let max_vec = f32x4::splat(max_val);
                let mut sum_vec = f32x4::splat(0.0);
                for chunk_idx in 0..chunks {
                    let start = chunk_idx * WIDTH;
                    let mut chunk_array = [0.0f32; 4];
                    chunk_array.copy_from_slice(&input_f32[start..start + WIDTH]);
                    let chunk = f32x4::from(chunk_array);
                    let exp_chunk = (chunk - max_vec).exp();
                    let exp_array = exp_chunk.to_array();
                    result_f32[start..start + WIDTH].copy_from_slice(&exp_array);
                    sum_vec += exp_chunk;
                }
                sum = sum_vec.to_array().iter().sum::<f32>();
            } else if WIDTH == 8 {
                let max_vec = f32x8::splat(max_val);
                let mut sum_vec = f32x8::splat(0.0);
                for chunk_idx in 0..chunks {
                    let start = chunk_idx * WIDTH;
                    let mut chunk_array = [0.0f32; 8];
                    chunk_array.copy_from_slice(&input_f32[start..start + WIDTH]);
                    let chunk = f32x8::from(chunk_array);
                    let exp_chunk = (chunk - max_vec).exp();
                    let exp_array = exp_chunk.to_array();
                    result_f32[start..start + WIDTH].copy_from_slice(&exp_array);
                    sum_vec += exp_chunk;
                }
                sum = sum_vec.to_array().iter().sum::<f32>();
            }

            // Handle remainder
            for (idx, &val) in input_f32.iter().enumerate().take(len).skip(chunks * WIDTH) {
                let exp_val = (val - max_val).exp();
                result_f32[idx] = exp_val;
                sum += exp_val;
            }

            // Pass 3: Normalize
            let inv_sum = 1.0 / sum;

            if WIDTH == 4 {
                let inv_sum_vec = f32x4::splat(inv_sum);
                for chunk_idx in 0..chunks {
                    let start = chunk_idx * WIDTH;
                    let mut chunk_array = [0.0f32; 4];
                    chunk_array.copy_from_slice(&result_f32[start..start + WIDTH]);
                    let chunk = f32x4::from(chunk_array);
                    let normalized = chunk * inv_sum_vec;
                    let norm_array = normalized.to_array();
                    result_f32[start..start + WIDTH].copy_from_slice(&norm_array);
                }
            } else if WIDTH == 8 {
                let inv_sum_vec = f32x8::splat(inv_sum);
                for chunk_idx in 0..chunks {
                    let start = chunk_idx * WIDTH;
                    let mut chunk_array = [0.0f32; 8];
                    chunk_array.copy_from_slice(&result_f32[start..start + WIDTH]);
                    let chunk = f32x8::from(chunk_array);
                    let normalized = chunk * inv_sum_vec;
                    let norm_array = normalized.to_array();
                    result_f32[start..start + WIDTH].copy_from_slice(&norm_array);
                }
            }

            for val in result_f32.iter_mut().take(len).skip(chunks * WIDTH) {
                *val *= inv_sum;
            }
        });

        Tensor::from_vec(result_data, self.dims())
    }

    /// Generic 1D SIMD implementation for f64 types
    #[inline(always)]
    fn softmax_1d_simd_f64<T, const WIDTH: usize>(&self, data: &[T], len: usize) -> Result<Tensor>
    where
        T: Copy + 'static + crate::TensorElement,
    {
        let input_f64: &[f64] = unsafe { std::mem::transmute(data) };

        let result_data = UninitVec::new(len).init_with(|result_slice: &mut [T]| {
            let result_f64: &mut [f64] = unsafe { std::mem::transmute(result_slice) };

            let chunks = len / WIDTH;

            // Pass 1: Find maximum value using SIMD
            let mut max_val = f64::NEG_INFINITY;

            if WIDTH == 2 {
                let mut max_vec = f64x2::splat(f64::NEG_INFINITY);
                for chunk_idx in 0..chunks {
                    let start = chunk_idx * WIDTH;
                    let mut chunk_array = [0.0f64; 2];
                    chunk_array.copy_from_slice(&input_f64[start..start + WIDTH]);
                    let chunk = f64x2::from(chunk_array);
                    max_vec = max_vec.max(chunk);
                }
                let max_array = max_vec.to_array();
                max_val = max_array.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            } else if WIDTH == 4 {
                let mut max_vec = f64x4::splat(f64::NEG_INFINITY);
                for chunk_idx in 0..chunks {
                    let start = chunk_idx * WIDTH;
                    let mut chunk_array = [0.0f64; 4];
                    chunk_array.copy_from_slice(&input_f64[start..start + WIDTH]);
                    let chunk = f64x4::from(chunk_array);
                    max_vec = max_vec.max(chunk);
                }
                let max_array = max_vec.to_array();
                max_val = max_array.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            }

            // Handle remainder
            for &val in input_f64.iter().take(len).skip(chunks * WIDTH) {
                max_val = max_val.max(val);
            }

            // Pass 2: Compute exp and sum
            let mut sum = 0.0f64;

            if WIDTH == 2 {
                let max_vec = f64x2::splat(max_val);
                let mut sum_vec = f64x2::splat(0.0);
                for chunk_idx in 0..chunks {
                    let start = chunk_idx * WIDTH;
                    let mut chunk_array = [0.0f64; 2];
                    chunk_array.copy_from_slice(&input_f64[start..start + WIDTH]);
                    let chunk = f64x2::from(chunk_array);
                    let exp_chunk = (chunk - max_vec).exp();
                    let exp_array = exp_chunk.to_array();
                    result_f64[start..start + WIDTH].copy_from_slice(&exp_array);
                    sum_vec += exp_chunk;
                }
                sum = sum_vec.to_array().iter().sum::<f64>();
            } else if WIDTH == 4 {
                let max_vec = f64x4::splat(max_val);
                let mut sum_vec = f64x4::splat(0.0);
                for chunk_idx in 0..chunks {
                    let start = chunk_idx * WIDTH;
                    let mut chunk_array = [0.0f64; 4];
                    chunk_array.copy_from_slice(&input_f64[start..start + WIDTH]);
                    let chunk = f64x4::from(chunk_array);
                    let exp_chunk = (chunk - max_vec).exp();
                    let exp_array = exp_chunk.to_array();
                    result_f64[start..start + WIDTH].copy_from_slice(&exp_array);
                    sum_vec += exp_chunk;
                }
                sum = sum_vec.to_array().iter().sum::<f64>();
            }

            // Handle remainder
            for (idx, &val) in input_f64.iter().enumerate().take(len).skip(chunks * WIDTH) {
                let exp_val = (val - max_val).exp();
                result_f64[idx] = exp_val;
                sum += exp_val;
            }

            // Pass 3: Normalize
            let inv_sum = 1.0 / sum;

            if WIDTH == 2 {
                let inv_sum_vec = f64x2::splat(inv_sum);
                for chunk_idx in 0..chunks {
                    let start = chunk_idx * WIDTH;
                    let mut chunk_array = [0.0f64; 2];
                    chunk_array.copy_from_slice(&result_f64[start..start + WIDTH]);
                    let chunk = f64x2::from(chunk_array);
                    let normalized = chunk * inv_sum_vec;
                    let norm_array = normalized.to_array();
                    result_f64[start..start + WIDTH].copy_from_slice(&norm_array);
                }
            } else if WIDTH == 4 {
                let inv_sum_vec = f64x4::splat(inv_sum);
                for chunk_idx in 0..chunks {
                    let start = chunk_idx * WIDTH;
                    let mut chunk_array = [0.0f64; 4];
                    chunk_array.copy_from_slice(&result_f64[start..start + WIDTH]);
                    let chunk = f64x4::from(chunk_array);
                    let normalized = chunk * inv_sum_vec;
                    let norm_array = normalized.to_array();
                    result_f64[start..start + WIDTH].copy_from_slice(&norm_array);
                }
            }

            for val in result_f64.iter_mut().take(len).skip(chunks * WIDTH) {
                *val *= inv_sum;
            }
        });

        Tensor::from_vec(result_data, self.dims())
    }

    /// Fully fused softmax implementation - no intermediate tensors
    /// Optimized version with reduced memory allocations and better cache efficiency
    fn softmax_general<T>(&self, dim_idx: usize) -> Result<Tensor>
    where
        T: num_traits::Float + Copy + Send + Sync + 'static + crate::TensorElement,
    {
        let shape = self.shape();
        let numel = shape.numel();

        // Calculate dimensions for efficient iteration
        let outer_size: usize = shape.as_slice()[..dim_idx].iter().product();
        let inner_size: usize = shape.as_slice()[dim_idx + 1..].iter().product();
        let reduce_size = shape.as_slice()[dim_idx];

        let result = UninitVec::<T>::new(numel).init_with(|result_data| {
            if self.is_contiguous() {
                let input_data = self.as_slice::<T>().unwrap();
                // Optimized path for contiguous tensors - eliminate temporary vector allocation
                for outer_idx in 0..outer_size {
                    for inner_idx in 0..inner_size {
                        let base_idx = outer_idx * reduce_size * inner_size + inner_idx;

                        // Pass 1: Find maximum value for numerical stability
                        let mut max_val = <T as num_traits::Float>::neg_infinity();
                        for reduce_idx in 0..reduce_size {
                            let src_idx = base_idx + reduce_idx * inner_size;
                            let val = input_data[src_idx];
                            if val > max_val {
                                max_val = val;
                            }
                        }

                        // Pass 2: Compute exp(x - max) and sum, store directly in result
                        let mut sum = <T as num_traits::Zero>::zero();
                        for reduce_idx in 0..reduce_size {
                            let src_idx = base_idx + reduce_idx * inner_size;
                            let exp_val = (input_data[src_idx] - max_val).exp();
                            result_data[src_idx] = exp_val;
                            sum = sum + exp_val;
                        }

                        // Pass 3: Normalize by sum (pre-compute inverse)
                        let inv_sum = <T as num_traits::One>::one() / sum;
                        for reduce_idx in 0..reduce_size {
                            let dst_idx = base_idx + reduce_idx * inner_size;
                            result_data[dst_idx] = result_data[dst_idx] * inv_sum;
                        }
                    }
                }
            } else {
                // Optimized path for non-contiguous tensors - cache index calculations
                for outer_idx in 0..outer_size {
                    for inner_idx in 0..inner_size {
                        // Pre-compute base indices to avoid repeated calculations
                        let mut base_indices = vec![0; shape.len()];

                        // Compute outer indices once
                        let mut temp_outer = outer_idx;
                        for i in (0..dim_idx).rev() {
                            base_indices[i] = temp_outer % shape[i];
                            temp_outer /= shape[i];
                        }

                        // Compute inner indices once
                        let mut temp_inner = inner_idx;
                        for i in (dim_idx + 1..shape.len()).rev() {
                            base_indices[i] = temp_inner % shape[i];
                            temp_inner /= shape[i];
                        }

                        // Pass 1: Find maximum value
                        let mut max_val = <T as num_traits::Float>::neg_infinity();
                        for reduce_idx in 0..reduce_size {
                            base_indices[dim_idx] = reduce_idx;
                            let val: T = self.at(&*base_indices);
                            if val > max_val {
                                max_val = val;
                            }
                        }

                        // Pass 2: Compute exp and sum, store directly in result
                        let mut sum = <T as num_traits::Zero>::zero();
                        for reduce_idx in 0..reduce_size {
                            base_indices[dim_idx] = reduce_idx;
                            let val: T = self.at(&*base_indices);
                            let exp_val = (val - max_val).exp();

                            let result_idx = outer_idx * reduce_size * inner_size
                                + reduce_idx * inner_size
                                + inner_idx;
                            result_data[result_idx] = exp_val;
                            sum = sum + exp_val;
                        }

                        // Pass 3: Normalize (pre-compute inverse)
                        let inv_sum = <T as num_traits::One>::one() / sum;
                        for reduce_idx in 0..reduce_size {
                            let result_idx = outer_idx * reduce_size * inner_size
                                + reduce_idx * inner_size
                                + inner_idx;
                            result_data[result_idx] = result_data[result_idx] * inv_sum;
                        }
                    }
                }
            }
        });

        Tensor::from_vec(result, self.dims())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_softmax_1d_basic() -> Result<()> {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3])?;
        let result = x.softmax(0)?;

        // Expected values computed manually
        let expected = [0.09003057, 0.24472847, 0.66524096];
        let result_data = result.to_flat_vec::<f32>()?;

        for (i, (actual, expected)) in result_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Index {i}: expected {expected}, got {actual}"
            );
        }

        // Check that sum is approximately 1
        let sum: f32 = result_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum should be 1.0, got {sum}");

        Ok(())
    }

    #[test]
    fn test_softmax_2d_dim0() -> Result<()> {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])?;
        let result = x.softmax(0)?;

        // Each column should sum to 1
        let result_data = result.to_flat_vec::<f32>()?;

        // Check column sums
        for col in 0..3 {
            let col_sum = result_data[col] + result_data[col + 3];
            assert!(
                (col_sum - 1.0).abs() < 1e-6,
                "Column {col} should sum to 1.0, got {col_sum}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_softmax_2d_dim1() -> Result<()> {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])?;
        let result = x.softmax(1)?;

        // Each row should sum to 1
        let result_data = result.to_flat_vec::<f32>()?;

        // Check row sums
        for row in 0..2 {
            let row_sum: f32 = (0..3).map(|col| result_data[row * 3 + col]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Row {row} should sum to 1.0, got {row_sum}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_softmax_different_dtypes() -> Result<()> {
        // Test f32
        let x_f32 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3])?;
        let result_f32 = x_f32.softmax(0)?;
        assert_eq!(result_f32.dtype(), DType::Fp32);

        // Test f64
        let x_f64 = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], [3])?;
        let result_f64 = x_f64.softmax(0)?;
        assert_eq!(result_f64.dtype(), DType::Fp64);

        Ok(())
    }

    #[test]
    fn test_softmax_single_element() -> Result<()> {
        let x = Tensor::from_vec(vec![5.0f32], [1])?;
        let result = x.softmax(0)?;
        let result_data = result.to_flat_vec::<f32>()?;

        assert!(
            (result_data[0] - 1.0).abs() < 1e-6,
            "Single element softmax should be 1.0, got {}",
            result_data[0]
        );

        Ok(())
    }

    #[test]
    fn test_softmax_numerical_stability() -> Result<()> {
        // Test with large values that could cause overflow without numerical stability
        let x = Tensor::from_vec(vec![1000.0f32, 1001.0, 1002.0], [3])?;
        let result = x.softmax(0)?;

        let result_data = result.to_flat_vec::<f32>()?;

        // Should not contain NaN or infinity
        for (i, &val) in result_data.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Result at index {i} should be finite, got {val}"
            );
            assert!(
                val >= 0.0,
                "Result at index {i} should be non-negative, got {val}"
            );
        }

        // Sum should be 1
        let sum: f32 = result_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum should be 1.0, got {sum}");

        Ok(())
    }

    #[test]
    fn test_softmax_zero_values() -> Result<()> {
        let x = Tensor::from_vec(vec![0.0f32, 0.0, 0.0], [3])?;
        let result = x.softmax(0)?;

        let result_data = result.to_flat_vec::<f32>()?;

        // All values should be equal (1/3)
        let expected = 1.0 / 3.0;
        for (i, &val) in result_data.iter().enumerate() {
            assert!(
                (val - expected).abs() < 1e-6,
                "Index {i}: expected {expected}, got {val}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_softmax_negative_values() -> Result<()> {
        let x = Tensor::from_vec(vec![-1.0f32, -2.0, -3.0], [3])?;
        let result = x.softmax(0)?;

        let result_data = result.to_flat_vec::<f32>()?;

        // All values should be positive and sum to 1
        for (i, &val) in result_data.iter().enumerate() {
            assert!(
                val > 0.0,
                "Value at index {i} should be positive, got {val}"
            );
        }

        let sum: f32 = result_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum should be 1.0, got {sum}");

        Ok(())
    }

    #[test]
    fn test_softmax_3d_tensor() -> Result<()> {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])?;
        let result = x.softmax(2)?;

        let result_data = result.to_flat_vec::<f32>()?;

        // Check that each slice along the last dimension sums to 1
        for i in 0..4 {
            let slice_sum = result_data[i * 2] + result_data[i * 2 + 1];
            assert!(
                (slice_sum - 1.0).abs() < 1e-6,
                "Slice {i} should sum to 1.0, got {slice_sum}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_softmax_invalid_dtype() {
        let x = Tensor::from_vec(vec![1i32, 2, 3], [3]).unwrap();
        let result = x.softmax(0);

        assert!(result.is_err());
    }

    #[test]
    fn test_softmax_preserves_shape() -> Result<()> {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])?;
        let result = x.softmax(1)?;

        assert_eq!(result.dims(), x.dims());
        assert_eq!(result.rank(), x.rank());

        Ok(())
    }
}
