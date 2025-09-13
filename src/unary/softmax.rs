use anyhow::Result;
use num_traits::Float;
use wide::{f32x8, f64x4};

use crate::{DType, Dim, StorageTrait, Tensor, TensorBase, TensorElement, UninitVec, WideSimd};

impl<S: StorageTrait> TensorBase<S> {
    /// Computes the softmax function along the specified dimension
    ///
    /// The softmax function is defined as:
    /// softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    ///
    /// This implementation uses the numerically stable version:
    /// softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
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
    /// let result = x.softmax(-1).unwrap();
    /// ```
    pub fn softmax<D: Dim>(&self, dim: D) -> Result<Tensor> {
        let dim_idx = dim.to_dim(self.rank())?;

        // If the dimension size is 1, softmax is just a tensor of ones in original dtype
        if self.dims()[dim_idx] == 1 {
            match self.dtype() {
                DType::Fp32 => return Tensor::ones::<f32>(self.dims()),
                DType::Fp64 => return Tensor::ones::<f64>(self.dims()),
                DType::Fp16 => return Tensor::ones::<half::f16>(self.dims()),
                DType::Bf16 => return Tensor::ones::<half::bf16>(self.dims()),
                _ => {
                    anyhow::bail!(
                        "softmax only supports floating-point types, got {:?}",
                        self.dtype()
                    )
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

    /// Internal implementation of softmax
    fn softmax_impl<T>(&self, dim_idx: usize) -> Result<Tensor>
    where
        T: num_traits::Float + Copy + Send + Sync + 'static + TensorElement,
    {
        // For 1D tensors (most common case)
        if self.rank() == 1 && dim_idx == 0 {
            return self.softmax_1d::<T>();
        }

        // For contiguous tensors with softmax along the last dimension
        if self.is_contiguous() && dim_idx == self.rank() - 1 {
            let dims = self.dims();
            let last_dim_size = dims[dims.len() - 1];
            let batch_size = self.numel() / last_dim_size;
            return self.softmax_contiguous_last_dim::<T>(batch_size, last_dim_size);
        }

        // Fallback to general implementation
        self.softmax_general::<T>(dim_idx)
    }

    /// Softmax implementation for contiguous tensors with softmax along the last dimension
    fn softmax_contiguous_last_dim<T>(
        &self,
        batch_size: usize,
        last_dim_size: usize,
    ) -> Result<Tensor>
    where
        T: Copy + 'static + TensorElement + num_traits::Float,
    {
        match T::DTYPE {
            crate::DType::Fp32 => {
                // Always use f32x8 - wide library will handle degradation automatically
                let f32_data = self.as_slice::<f32>()?;
                let result_data =
                    self.softmax_contiguous_simd::<f32x8>(f32_data, batch_size, last_dim_size);
                Tensor::from_vec(result_data, self.dims())
            }
            crate::DType::Fp64 => {
                let f64_data = self.as_slice::<f64>()?;
                let result_data =
                    self.softmax_contiguous_simd::<f64x4>(f64_data, batch_size, last_dim_size);
                Tensor::from_vec(result_data, self.dims())
            }
            crate::DType::Fp16 => {
                let f16_data = self.as_slice::<half::f16>()?;
                let f32_data: Vec<f32> = f16_data.iter().map(|&x| x.to_f32()).collect();
                let f32_data =
                    self.softmax_contiguous_simd::<f32x8>(&f32_data, batch_size, last_dim_size);
                let f16_result: Vec<half::f16> =
                    f32_data.iter().map(|&x| half::f16::from_f32(x)).collect();
                Tensor::from_vec(f16_result, self.dims())
            }
            crate::DType::Bf16 => {
                let bf16_data = self.as_slice::<half::bf16>()?;
                let len = bf16_data.len();
                let f32_data: Vec<f32> = bf16_data.iter().map(|&x| x.to_f32()).collect();
                let f32_data =
                    self.softmax_contiguous_simd::<f32x8>(&f32_data, batch_size, last_dim_size);
                let bf16_result = UninitVec::new(len).init_with(|result_slice| {
                    for (i, &f32_val) in f32_data.iter().enumerate() {
                        result_slice[i] = half::bf16::from_f32(f32_val);
                    }
                });
                Tensor::from_vec(bf16_result, self.dims())
            }
            _ => {
                anyhow::bail!(
                    "softmax only supports floating-point types, got {:?}",
                    T::DTYPE
                )
            }
        }
    }

    /// 1D softmax
    fn softmax_1d<T>(&self) -> Result<Tensor>
    where
        T: num_traits::Float + Copy + Send + Sync + 'static + TensorElement,
    {
        match T::DTYPE {
            crate::DType::Fp32 => {
                let f32_data = self.as_slice::<f32>()?;
                let result_vec = Self::softmax_1d_simd::<f32x8>(f32_data);
                Tensor::from_vec(result_vec, self.dims())
            }
            crate::DType::Fp64 => {
                let f64_data = self.as_slice::<f64>()?;
                let result_vec = Self::softmax_1d_simd::<f64x4>(f64_data);
                Tensor::from_vec(result_vec, self.dims())
            }
            crate::DType::Fp16 => {
                let f16_data = self.as_slice::<half::f16>()?;
                let f32_data: Vec<f32> = f16_data.iter().map(|&x| x.to_f32()).collect();
                let f32_data = Self::softmax_1d_simd::<f32x8>(&f32_data);
                let f16_result: Vec<half::f16> =
                    f32_data.iter().map(|&x| half::f16::from_f32(x)).collect();
                Tensor::from_vec(f16_result, self.dims())
            }
            crate::DType::Bf16 => {
                let bf16_data = self.as_slice::<half::bf16>()?;
                let f32_data: Vec<f32> = bf16_data.iter().map(|&x| x.to_f32()).collect();
                let f32_data = Self::softmax_1d_simd::<f32x8>(&f32_data);
                let bf16_result: Vec<half::bf16> =
                    f32_data.iter().map(|&x| half::bf16::from_f32(x)).collect();
                Tensor::from_vec(bf16_result, self.dims())
            }
            _ => {
                anyhow::bail!("Unsupported dtype for softmax: {:?}", T::DTYPE);
            }
        }
    }

    #[inline(always)]
    fn softmax_1d_simd<Simd>(data: &[Simd::Element]) -> Vec<Simd::Element>
    where
        Simd: WideSimd,
        Simd::Element: std::iter::Sum<Simd::Element>,
    {
        const SMALL_ARRAY_THRESHOLD: usize = 16;

        let len = data.len();
        if len == 0 {
            return Vec::new();
        }

        UninitVec::new(len).init_with(|result_slice: &mut [Simd::Element]| {
            if len <= SMALL_ARRAY_THRESHOLD {
                let max_val = data
                    .iter()
                    .fold(Simd::Element::neg_infinity(), |acc, &x| acc.max(x));
                let mut sum = Simd::Element::ZERO;

                // Combined exp computation and sum accumulation
                for (i, &val) in data.iter().enumerate() {
                    let exp_val = (val - max_val).exp();
                    result_slice[i] = exp_val;
                    sum = sum + exp_val;
                }

                // Normalize
                let inv_sum = Simd::Element::ONE / sum;
                for val in result_slice.iter_mut() {
                    *val = *val * inv_sum;
                }
                return;
            }

            // Pass 1: Find maximum value using SIMD
            let mut max_vec = Simd::NEG_INFINITY;
            let chunks = len / Simd::LANE;
            for chunk_idx in 0..chunks {
                let start = chunk_idx * Simd::LANE;
                let chunk = unsafe { Simd::from_slice_unaligned(&data[start..]) };
                max_vec = max_vec.max(chunk);
            }

            let mut max_val = max_vec
                .as_array_ref()
                .as_ref()
                .iter()
                .fold(Simd::Element::neg_infinity(), |a, &b| a.max(b));

            // Process remainder elements for max finding
            for &val in &data[chunks * Simd::LANE..] {
                max_val = max_val.max(val);
            }

            // Pass 2: Compute exp and sum
            let max_vec = Simd::splat(max_val);
            let mut sum_vec = Simd::ZERO;
            for chunk_idx in 0..chunks {
                let start = chunk_idx * Simd::LANE;
                let chunk = unsafe { Simd::from_slice_unaligned(&data[start..]) };
                let exp_chunk = (chunk.sub(max_vec)).exp();
                let exp_array = exp_chunk.as_array_ref();
                result_slice[start..start + Simd::LANE].copy_from_slice(exp_array.as_ref());
                sum_vec = sum_vec.add(exp_chunk);
            }
            let mut sum = sum_vec.sum();
            for i in chunks * Simd::LANE..len {
                let exp_val = (data[i] - max_val).exp();
                result_slice[i] = exp_val;
                sum = sum + exp_val;
            }

            // Pass 3: Normalize
            let inv_sum = Simd::Element::ONE / sum;
            let inv_sum_vec = Simd::splat(inv_sum);
            for chunk_idx in 0..chunks {
                let start = chunk_idx * Simd::LANE;
                let chunk = unsafe { Simd::from_slice_unaligned(&result_slice[start..]) };
                let normalized = chunk.mul(inv_sum_vec);
                let norm_array = normalized.as_array_ref();
                result_slice[start..start + Simd::LANE].copy_from_slice(norm_array.as_ref());
            }
            for val in result_slice[chunks * Simd::LANE..].iter_mut() {
                *val = *val * inv_sum;
            }
        })
    }

    #[inline(always)]
    fn softmax_contiguous_simd<Simd>(
        &self,
        data: &[Simd::Element],
        batch_size: usize,
        last_dim_size: usize,
    ) -> Vec<Simd::Element>
    where
        Simd: WideSimd,
        Simd::Element: std::iter::Sum<Simd::Element>,
    {
        UninitVec::new(self.numel()).init_with(|result_slice: &mut [Simd::Element]| {
            let chunks_per_batch = last_dim_size / Simd::LANE;
            for batch_idx in 0..batch_size {
                let batch_start = batch_idx * last_dim_size;
                let batch_end = batch_start + last_dim_size;

                // Find maximum using SIMD
                let mut max_vec = Simd::NEG_INFINITY;
                for chunk_idx in 0..chunks_per_batch {
                    let chunk_start = batch_start + chunk_idx * Simd::LANE;
                    let chunk = unsafe { Simd::from_slice_unaligned(&data[chunk_start..]) };
                    max_vec = max_vec.max(chunk);
                }
                let max_array = max_vec.as_array_ref();
                let mut max_val = max_array
                    .as_ref()
                    .iter()
                    .fold(Simd::Element::neg_infinity(), |a, &b| a.max(b));

                // Handle remainder
                for &val in data
                    .iter()
                    .take(batch_end)
                    .skip(batch_start + chunks_per_batch * Simd::LANE)
                {
                    max_val = max_val.max(val);
                }

                // Compute exp and sum using SIMD
                let max_vec = Simd::splat(max_val);
                let mut sum_vec = Simd::ZERO;

                for chunk_idx in 0..chunks_per_batch {
                    let chunk_start = batch_start + chunk_idx * Simd::LANE;
                    let chunk = unsafe { Simd::from_slice_unaligned(&data[chunk_start..]) };
                    let exp_chunk = (chunk.sub(max_vec)).exp();
                    let exp_array = exp_chunk.as_array_ref();
                    result_slice[chunk_start..chunk_start + Simd::LANE]
                        .copy_from_slice(exp_array.as_ref());
                    sum_vec = sum_vec.add(exp_chunk);
                }
                let mut sum = sum_vec.sum();

                // Handle remainder
                for (idx, &val) in data
                    .iter()
                    .enumerate()
                    .take(batch_end)
                    .skip(batch_start + chunks_per_batch * Simd::LANE)
                {
                    let exp_val = (val - max_val).exp();
                    result_slice[idx] = exp_val;
                    sum = sum + exp_val;
                }

                // Normalize using SIMD
                let inv_sum = Simd::Element::ONE / sum;
                let inv_sum_vec = Simd::splat(inv_sum);

                for chunk_idx in 0..chunks_per_batch {
                    let chunk_start = batch_start + chunk_idx * Simd::LANE;
                    let chunk = unsafe { Simd::from_slice_unaligned(&result_slice[chunk_start..]) };
                    let normalized = chunk.mul(inv_sum_vec);
                    let norm_array = normalized.as_array_ref();
                    result_slice[chunk_start..chunk_start + Simd::LANE]
                        .copy_from_slice(norm_array.as_ref());
                }
                for val in result_slice
                    .iter_mut()
                    .take(batch_end)
                    .skip(batch_start + chunks_per_batch * Simd::LANE)
                {
                    *val = *val * inv_sum;
                }
            }
        })
    }

    fn softmax_general<T>(&self, dim_idx: usize) -> Result<Tensor>
    where
        T: num_traits::Float + Copy + Send + Sync + 'static + TensorElement,
    {
        let shape = self.shape();
        let numel = shape.numel();
        let outer_size: usize = shape.as_slice()[..dim_idx].iter().product();
        let inner_size: usize = shape.as_slice()[dim_idx + 1..].iter().product();
        let reduce_size = shape.as_slice()[dim_idx];

        let result = UninitVec::<T>::new(numel).init_with(|result_data| {
            if self.is_contiguous() {
                let input_data = self.as_slice::<T>().unwrap();
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
                // Use stack-allocated array for small dimensions, heap for large ones
                const MAX_STACK_DIMS: usize = 8;
                let use_stack = shape.len() <= MAX_STACK_DIMS;
                if use_stack {
                    // Stack-allocated path for common small-dimensional tensors
                    let mut base_indices = [0usize; MAX_STACK_DIMS];

                    for outer_idx in 0..outer_size {
                        for inner_idx in 0..inner_size {
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

                            // Pre-compute result base index to avoid repeated calculation
                            let result_base = outer_idx * reduce_size * inner_size + inner_idx;

                            // Pass 1: Find maximum value
                            let mut max_val = <T as num_traits::Float>::neg_infinity();
                            for reduce_idx in 0..reduce_size {
                                base_indices[dim_idx] = reduce_idx;
                                let val: T = self.at(&base_indices[..shape.len()]);
                                if val > max_val {
                                    max_val = val;
                                }
                            }

                            // Pass 2: Compute exp and sum, store directly in result
                            let mut sum = <T as num_traits::Zero>::zero();
                            for reduce_idx in 0..reduce_size {
                                base_indices[dim_idx] = reduce_idx;
                                let val: T = self.at(&base_indices[..shape.len()]);
                                let exp_val = (val - max_val).exp();

                                let result_idx = result_base + reduce_idx * inner_size;
                                result_data[result_idx] = exp_val;
                                sum = sum + exp_val;
                            }

                            // Pass 3: Normalize (pre-compute inverse)
                            let inv_sum = <T as num_traits::One>::one() / sum;
                            for reduce_idx in 0..reduce_size {
                                let result_idx = result_base + reduce_idx * inner_size;
                                result_data[result_idx] = result_data[result_idx] * inv_sum;
                            }
                        }
                    }
                } else {
                    // Heap-allocated path for high-dimensional tensors
                    let mut base_indices = vec![0; shape.len()];
                    for outer_idx in 0..outer_size {
                        for inner_idx in 0..inner_size {
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

                            // Pre-compute result base index to avoid repeated calculation
                            let result_base = outer_idx * reduce_size * inner_size + inner_idx;

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

                                let result_idx = result_base + reduce_idx * inner_size;
                                result_data[result_idx] = exp_val;
                                sum = sum + exp_val;
                            }

                            // Pass 3: Normalize (pre-compute inverse)
                            let inv_sum = <T as num_traits::One>::one() / sum;
                            for reduce_idx in 0..reduce_size {
                                let result_idx = result_base + reduce_idx * inner_size;
                                result_data[result_idx] = result_data[result_idx] * inv_sum;
                            }
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
