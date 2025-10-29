use crate::{Dim, Shape, StorageTrait, Tensor, TensorBase, TensorElement, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    /// Standardizes the tensor by subtracting the mean and dividing by the standard deviation along a specified dimension.
    ///
    /// # Arguments
    ///
    /// * `mean` - The mean(s) to subtract. Length must match the size of the channel dimension.
    /// * `std` - The standard deviation(s) to divide by. Length must match the size of the channel dimension.
    /// * `dim` - The dimension along which to apply standardization (e.g., channel dimension).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `mean` or `std` are empty.
    /// - `mean` and `std` have different lengths.
    /// - The dtype of `mean` and `std` does not match the tensor's dtype.
    /// - Any value in `std` is zero.
    /// - `dim` is out of bounds for the tensor's rank.
    /// - Length of `mean`/`std` doesn't match the size of the specified channel dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use slsl::Tensor;
    ///
    /// // HWC format: Height x Width x Channels (dim = 2)
    /// let hwc_data = vec![
    ///     10.0, 20.0, 30.0, // Pixel (0,0), Channels R, G, B
    ///     40.0, 50.0, 60.0, // Pixel (0,1), Channels R, G, B
    ///     70.0, 80.0, 90.0, // Pixel (1,0), Channels R, G, B
    ///     100.0, 110.0, 120.0, // Pixel (1,1), Channels R, G, B
    /// ];
    /// let hwc_tensor = Tensor::from_vec(hwc_data, [2, 2, 3]).unwrap();
    /// let mean_rgb = [65.0, 75.0, 85.0];
    /// let std_rgb = [30.0, 30.0, 30.0];
    /// let standardized_hwc = hwc_tensor.standardize(&mean_rgb, &std_rgb, 2).unwrap();
    ///
    /// // CHW format: Channels x Height x Width (dim = 0)
    /// let chw_data = vec![
    ///     10.0, 40.0, 70.0, 100.0, // Channel R: all pixels
    ///     20.0, 50.0, 80.0, 110.0, // Channel G: all pixels
    ///     30.0, 60.0, 90.0, 120.0, // Channel B: all pixels
    /// ];
    /// let chw_tensor = Tensor::from_vec(chw_data, [3, 2, 2]).unwrap();
    /// let standardized_chw = chw_tensor.standardize(&mean_rgb, &std_rgb, 0).unwrap();
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    ///
    /// This function's behavior is similar to PyTorch's `torchvision.transforms.functional.normalize`.
    /// For more details, refer to the PyTorch documentation: <mcurl name="PyTorch Normalize Documentation" url="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.normalize.html?highlight=normalize#torchvision.transforms.functional.normalize"></mcurl>
    pub fn standardize<T: TensorElement + num_traits::Float>(
        &self,
        mean: &[T],
        std: &[T],
        dim: impl Dim,
    ) -> anyhow::Result<Tensor> {
        anyhow::ensure!(
            !mean.is_empty() && !std.is_empty(),
            "Mean and std cannot be empty."
        );
        anyhow::ensure!(
            mean.len() == std.len(),
            "Mean and std must have the same length."
        );
        anyhow::ensure!(
            T::DTYPE == self.dtype(),
            "Dtype mismatch: Tensor ({:?}), Mean/Std ({:?}).",
            self.dtype(),
            T::DTYPE
        );
        anyhow::ensure!(
            std.iter().all(|&s| s != T::ZERO),
            "Standard deviation cannot be zero."
        );

        if self.numel() == 0 {
            return self.to_contiguous();
        }

        if self.rank() == 0 {
            // TODO: ignore for now
            // anyhow::ensure!(dim == 0, "Dim must be 0 for scalar tensor.");
            anyhow::ensure!(
                mean.len() == 1,
                "Mean and std must have length 1 for scalar tensor."
            );
            let mean_tensor = Tensor::from_scalar(mean[0])?;
            let std_tensor = Tensor::from_scalar(std[0])?;
            return Ok((self - mean_tensor) / std_tensor);
        }

        let dim = dim.to_dim(self.rank())?;
        anyhow::ensure!(
            dim < self.rank(),
            "Dim {} is out of bounds for tensor with rank {}.",
            dim,
            self.rank()
        );

        let dim_size = self.ndim(dim)?;
        anyhow::ensure!(
            mean.len() == dim_size,
            "Mean length {} doesn't match dim {} size {}.",
            mean.len(),
            dim,
            dim_size
        );

        self.standardize_impl(mean, std, dim)
    }

    fn standardize_impl<T: TensorElement + num_traits::Float>(
        &self,
        mean: &[T],
        std: &[T],
        dim: usize,
    ) -> anyhow::Result<Tensor> {
        let numel = self.numel();
        let shape = *self.shape();

        // Pre-compute reciprocals and fused (a, b): y = x * a + b, where a = 1/std, b = -mean/std
        let inv_std: Vec<T> = std.iter().map(|&s| T::ONE / s).collect();

        match T::DTYPE {
            crate::DType::Fp32 => {
                let mean_f32 = unsafe { std::mem::transmute::<&[T], &[f32]>(mean) };
                let inv_std_f32 = unsafe { std::mem::transmute::<&[T], &[f32]>(&inv_std) };
                // Precompute affine params
                let a: Vec<f32> = inv_std_f32.to_vec();
                let b: Vec<f32> = mean_f32
                    .iter()
                    .zip(inv_std_f32.iter())
                    .map(|(&m, &is)| -m * is)
                    .collect();

                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    if self.is_contiguous() {
                        let input_slice = self.as_slice::<f32>().unwrap();
                        self.standardize_contiguous_f32(input_slice, &a, &b, dim, dst);
                    } else {
                        self.standardize_non_contiguous_f32(mean_f32, inv_std_f32, dim, dst);
                    }
                });
                Tensor::from_vec(out, shape)
            }
            crate::DType::Fp64 => {
                let mean_f64 = unsafe { std::mem::transmute::<&[T], &[f64]>(mean) };
                let inv_std_f64 = unsafe { std::mem::transmute::<&[T], &[f64]>(&inv_std) };
                let a: Vec<f64> = inv_std_f64.to_vec();
                let b: Vec<f64> = mean_f64
                    .iter()
                    .zip(inv_std_f64.iter())
                    .map(|(&m, &is)| -m * is)
                    .collect();

                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    if self.is_contiguous() {
                        let input_slice = self.as_slice::<f64>().unwrap();
                        self.standardize_contiguous_f64(input_slice, &a, &b, dim, dst);
                    } else {
                        self.standardize_non_contiguous_f64(mean_f64, inv_std_f64, dim, dst);
                    }
                });
                Tensor::from_vec(out, shape)
            }
            _ => {
                // Fallback
                let mut broadcast_shape = Shape::full(1, self.rank());
                broadcast_shape[dim] = shape[dim];
                let mean_tensor = Tensor::from(mean).reshape(broadcast_shape)?;
                let std_tensor = Tensor::from(std).reshape(broadcast_shape)?;
                Ok((self - mean_tensor.broadcast_to(self.dims())?)
                    / std_tensor.broadcast_to(self.dims())?)
            }
        }
    }

    #[inline]
    fn standardize_contiguous_f32(
        &self,
        input: &[f32],
        a: &[f32],
        b: &[f32],
        dim: usize,
        output: &mut [f32],
    ) {
        let shape = self.shape();
        let dim_size = shape[dim];

        // Calculate how many elements to process per channel
        let elements_per_channel = self.numel() / dim_size;

        // For common cases like CHW (dim=0) or HWC (dim=last), use optimized loops
        if dim == 0 {
            // CHW format: channels first
            let channel_size = elements_per_channel;

            for ch in 0..dim_size {
                let aa = a[ch];
                let bb = b[ch];
                let start = ch * channel_size;
                let end = start + channel_size;

                for i in start..end {
                    output[i] = input[i] * aa + bb;
                }
            }
        } else if dim == shape.len() - 1 {
            // HWC format: channels last
            for chunk_idx in 0..elements_per_channel {
                let start = chunk_idx * dim_size;
                for ch in 0..dim_size {
                    let idx = start + ch;
                    output[idx] = input[idx] * a[ch] + b[ch];
                }
            }
        } else {
            // Generic fast path for any contiguous tensor and arbitrary dim
            // Treat data as [outer, dim, inner] where inner is contiguous
            let mut outer: usize = 1;
            for i in 0..dim {
                outer *= shape[i];
            }
            let mut inner: usize = 1;
            for i in (dim + 1)..shape.len() {
                inner *= shape[i];
            }

            // Process block-wise to maximize cache locality
            // Layout is contiguous, so slices of length `inner` are contiguous
            let block = dim_size * inner;
            // Parallelize over outer if large enough
            #[cfg(feature = "rayon")]
            {
                if outer * dim_size * inner >= 1_000_000 {
                    use rayon::prelude::*;
                    output
                        .par_chunks_mut(block)
                        .enumerate()
                        .for_each(|(o, out_block)| {
                            let base = o * block;
                            let in_block = &input[base..base + block];
                            let mut ch_base = 0;
                            for ch in 0..dim_size {
                                let aa = a[ch];
                                let bb = b[ch];
                                let offset = ch_base;
                                if inner == 10 {
                                    out_block[offset] = in_block[offset] * aa + bb;
                                    out_block[offset + 1] = in_block[offset + 1] * aa + bb;
                                    out_block[offset + 2] = in_block[offset + 2] * aa + bb;
                                    out_block[offset + 3] = in_block[offset + 3] * aa + bb;
                                    out_block[offset + 4] = in_block[offset + 4] * aa + bb;
                                    out_block[offset + 5] = in_block[offset + 5] * aa + bb;
                                    out_block[offset + 6] = in_block[offset + 6] * aa + bb;
                                    out_block[offset + 7] = in_block[offset + 7] * aa + bb;
                                    out_block[offset + 8] = in_block[offset + 8] * aa + bb;
                                    out_block[offset + 9] = in_block[offset + 9] * aa + bb;
                                } else {
                                    for k in 0..inner {
                                        let idx = offset + k;
                                        out_block[idx] = in_block[idx] * aa + bb;
                                    }
                                }
                                ch_base += inner;
                            }
                        });
                    return;
                }
            }

            let mut base = 0;
            for _ in 0..outer {
                let mut ch_base = 0;
                for ch in 0..dim_size {
                    let aa = a[ch];
                    let bb = b[ch];
                    let in_ptr = base + ch_base;
                    if inner == 10 {
                        let idx0 = in_ptr;
                        unsafe {
                            *output.get_unchecked_mut(idx0) = *input.get_unchecked(idx0) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 1) =
                                *input.get_unchecked(idx0 + 1) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 2) =
                                *input.get_unchecked(idx0 + 2) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 3) =
                                *input.get_unchecked(idx0 + 3) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 4) =
                                *input.get_unchecked(idx0 + 4) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 5) =
                                *input.get_unchecked(idx0 + 5) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 6) =
                                *input.get_unchecked(idx0 + 6) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 7) =
                                *input.get_unchecked(idx0 + 7) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 8) =
                                *input.get_unchecked(idx0 + 8) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 9) =
                                *input.get_unchecked(idx0 + 9) * aa + bb;
                        }
                    } else {
                        for k in 0..inner {
                            let idx = in_ptr + k;
                            output[idx] = input[idx] * aa + bb;
                        }
                    }
                    ch_base += inner;
                }
                base += block;
            }
        }
    }

    #[inline]
    fn standardize_contiguous_f64(
        &self,
        input: &[f64],
        a: &[f64],
        b: &[f64],
        dim: usize,
        output: &mut [f64],
    ) {
        let shape = self.shape();
        let dim_size = shape[dim];
        let elements_per_channel = self.numel() / dim_size;

        if dim == 0 {
            // CHW format: channels first
            let channel_size = elements_per_channel;

            for ch in 0..dim_size {
                let aa = a[ch];
                let bb = b[ch];
                let start = ch * channel_size;
                let end = start + channel_size;

                for i in start..end {
                    output[i] = input[i] * aa + bb;
                }
            }
        } else if dim == shape.len() - 1 {
            // HWC format: channels last
            for chunk_idx in 0..elements_per_channel {
                let start = chunk_idx * dim_size;
                for ch in 0..dim_size {
                    let idx = start + ch;
                    output[idx] = input[idx] * a[ch] + b[ch];
                }
            }
        } else {
            // Generic fast path for any contiguous tensor and arbitrary dim
            let mut outer: usize = 1;
            for i in 0..dim {
                outer *= shape[i];
            }
            let mut inner: usize = 1;
            for i in (dim + 1)..shape.len() {
                inner *= shape[i];
            }

            let block = dim_size * inner;
            #[cfg(feature = "rayon")]
            {
                if outer * dim_size * inner >= 1_000_000 {
                    use rayon::prelude::*;
                    output
                        .par_chunks_mut(block)
                        .enumerate()
                        .for_each(|(o, out_block)| {
                            let base = o * block;
                            let in_block = &input[base..base + block];
                            let mut ch_base = 0;
                            for ch in 0..dim_size {
                                let aa = a[ch];
                                let bb = b[ch];
                                let offset = ch_base;
                                if inner == 10 {
                                    out_block[offset] = in_block[offset] * aa + bb;
                                    out_block[offset + 1] = in_block[offset + 1] * aa + bb;
                                    out_block[offset + 2] = in_block[offset + 2] * aa + bb;
                                    out_block[offset + 3] = in_block[offset + 3] * aa + bb;
                                    out_block[offset + 4] = in_block[offset + 4] * aa + bb;
                                    out_block[offset + 5] = in_block[offset + 5] * aa + bb;
                                    out_block[offset + 6] = in_block[offset + 6] * aa + bb;
                                    out_block[offset + 7] = in_block[offset + 7] * aa + bb;
                                    out_block[offset + 8] = in_block[offset + 8] * aa + bb;
                                    out_block[offset + 9] = in_block[offset + 9] * aa + bb;
                                } else {
                                    for k in 0..inner {
                                        let idx = offset + k;
                                        out_block[idx] = in_block[idx] * aa + bb;
                                    }
                                }
                                ch_base += inner;
                            }
                        });
                    return;
                }
            }

            let mut base = 0;
            for _ in 0..outer {
                let mut ch_base = 0;
                for ch in 0..dim_size {
                    let aa = a[ch];
                    let bb = b[ch];
                    let in_ptr = base + ch_base;
                    if inner == 10 {
                        let idx0 = in_ptr;
                        unsafe {
                            *output.get_unchecked_mut(idx0) = *input.get_unchecked(idx0) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 1) =
                                *input.get_unchecked(idx0 + 1) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 2) =
                                *input.get_unchecked(idx0 + 2) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 3) =
                                *input.get_unchecked(idx0 + 3) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 4) =
                                *input.get_unchecked(idx0 + 4) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 5) =
                                *input.get_unchecked(idx0 + 5) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 6) =
                                *input.get_unchecked(idx0 + 6) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 7) =
                                *input.get_unchecked(idx0 + 7) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 8) =
                                *input.get_unchecked(idx0 + 8) * aa + bb;
                            *output.get_unchecked_mut(idx0 + 9) =
                                *input.get_unchecked(idx0 + 9) * aa + bb;
                        }
                    } else {
                        for k in 0..inner {
                            let idx = in_ptr + k;
                            output[idx] = input[idx] * aa + bb;
                        }
                    }
                    ch_base += inner;
                }
                base += block;
            }
        }
    }

    /// General standardize implementation for f32 with arbitrary dimension
    #[inline]
    #[allow(dead_code)]
    fn standardize_general_f32(
        &self,
        input: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        dim: usize,
        output: &mut [f32],
    ) {
        let shape = self.shape();
        let strides = self.strides();

        for linear_idx in 0..self.numel() {
            // Convert linear index to multi-dimensional indices
            let mut remaining = linear_idx;
            let mut indices_uninit = UninitVec::<usize>::new(shape.len());
            let indices = indices_uninit.as_mut_slice();

            for (i, &stride) in strides.iter().enumerate().rev() {
                indices[i] = remaining / stride;
                remaining %= stride;
            }

            let indices_uninit = unsafe { indices_uninit.finalize() };

            let ch = indices_uninit[dim];
            let val = input[linear_idx];
            output[linear_idx] = (val - mean[ch]) * inv_std[ch];
        }
    }

    #[inline]
    #[allow(dead_code)]
    fn standardize_general_f64(
        &self,
        input: &[f64],
        mean: &[f64],
        inv_std: &[f64],
        dim: usize,
        output: &mut [f64],
    ) {
        let shape = self.shape();
        let strides = self.strides();

        for linear_idx in 0..self.numel() {
            let mut remaining = linear_idx;
            let mut indices_uninit = UninitVec::<usize>::new(shape.len());
            let indices = indices_uninit.as_mut_slice();

            for (i, &stride) in strides.iter().enumerate().rev() {
                indices[i] = remaining / stride;
                remaining %= stride;
            }
            let indices_uninit = unsafe { indices_uninit.finalize() };

            let ch = indices_uninit[dim];
            let val = input[linear_idx];
            output[linear_idx] = (val - mean[ch]) * inv_std[ch];
        }
    }

    /// Non-contiguous standardize implementation for f32
    #[inline]
    fn standardize_non_contiguous_f32(
        &self,
        mean: &[f32],
        inv_std: &[f32],
        dim: usize,
        output: &mut [f32],
    ) {
        for (out_val, item) in output.iter_mut().zip(self.iter_with_meta::<f32>()) {
            let ch = item.indices[dim];
            *out_val = (*item.value - mean[ch]) * inv_std[ch];
        }
    }

    /// Non-contiguous standardize implementation for f64
    #[inline]
    fn standardize_non_contiguous_f64(
        &self,
        mean: &[f64],
        inv_std: &[f64],
        dim: usize,
        output: &mut [f64],
    ) {
        for (out_val, item) in output.iter_mut().zip(self.iter_with_meta::<f64>()) {
            let ch = item.indices[dim];
            *out_val = (*item.value - mean[ch]) * inv_std[ch];
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_standardize_scalar() -> anyhow::Result<()> {
        let tensor = Tensor::from_scalar(5.0f32)?;
        let result = tensor.standardize(&[2.0f32], &[1.5f32], 0)?;
        let expected = (5.0f32 - 2.0f32) / 1.5f32;
        assert!((result.to_scalar::<f32>()? - expected).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_standardize_1d() -> anyhow::Result<()> {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [4])?;
        // For 1D tensor with 4 elements, we need 4 mean/std values for dim=0
        let result = tensor.standardize(&[1.0f32, 2.0, 3.0, 4.0], &[1.0f32, 1.0, 1.0, 1.0], 0)?;
        let expected = [0.0f32, 0.0, 0.0, 0.0]; // (x - x) / 1 = 0 for each element
        let actual = result.to_flat_vec::<f32>()?;

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6, "Expected {}, got {}", e, a);
        }
        Ok(())
    }

    #[test]
    fn test_standardize_2d_channels() -> anyhow::Result<()> {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])?;
        let mean = vec![1.5f32, 3.5, 5.5]; // mean for each channel
        let std = vec![1.0f32, 1.0, 1.0]; // std for each channel

        let result = tensor.standardize(&mean, &std, 1)?; // dim = 1 (last dimension)
        let actual = result.to_flat_vec::<f32>()?;

        // Expected: [(1-1.5)/1, (2-3.5)/1, (3-5.5)/1, (4-1.5)/1, (5-3.5)/1, (6-5.5)/1]
        let expected = [-0.5f32, -1.5, -2.5, 2.5, 1.5, 0.5];

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6, "Expected {}, got {}", e, a);
        }
        Ok(())
    }

    #[test]
    fn test_standardize_3d_image_hwc() -> anyhow::Result<()> {
        // 2x2x3 tensor (HxWxC format) - dim = 2
        let data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let tensor = Tensor::from_vec(data, [2, 2, 3])?;
        let mean = vec![5.5f32, 6.5, 7.5]; // mean for each of 3 channels
        let std = vec![2.0f32, 2.0, 2.0]; // std for each channel

        let result = tensor.standardize(&mean, &std, 2)?; // dim = 2 for HWC
        let actual = result.to_flat_vec::<f32>()?;

        // Verify a few key values
        assert!((actual[0] - (1.0 - 5.5) / 2.0).abs() < 1e-6); // First element, channel 0
        assert!((actual[1] - (2.0 - 6.5) / 2.0).abs() < 1e-6); // Second element, channel 1
        assert!((actual[2] - (3.0 - 7.5) / 2.0).abs() < 1e-6); // Third element, channel 2

        Ok(())
    }

    #[test]
    fn test_standardize_3d_image_chw() -> anyhow::Result<()> {
        // 3x2x2 tensor (CxHxW format) - dim = 0
        let data = vec![
            1.0f32, 4.0, 7.0, 10.0, // Channel 0: all pixels
            2.0f32, 5.0, 8.0, 11.0, // Channel 1: all pixels
            3.0f32, 6.0, 9.0, 12.0, // Channel 2: all pixels
        ];
        let tensor = Tensor::from_vec(data, [3, 2, 2])?;
        let mean = vec![5.5f32, 6.5, 7.5]; // mean for each of 3 channels
        let std = vec![2.0f32, 2.0, 2.0]; // std for each channel

        let result = tensor.standardize(&mean, &std, 0)?; // dim = 0 for CHW
        let actual = result.to_flat_vec::<f32>()?;

        // Verify a few key values
        assert!((actual[0] - (1.0 - 5.5) / 2.0).abs() < 1e-6); // First element of channel 0
        assert!((actual[4] - (2.0 - 6.5) / 2.0).abs() < 1e-6); // First element of channel 1
        assert!((actual[8] - (3.0 - 7.5) / 2.0).abs() < 1e-6); // First element of channel 2

        Ok(())
    }

    #[test]
    fn test_standardize_f64() -> anyhow::Result<()> {
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, [4])?;
        // For 1D tensor with 4 elements, we need 4 mean/std values for dim=0
        let result = tensor.standardize(&[0.5f64, 1.5, 2.5, 3.5], &[1.0f64, 1.0, 1.0, 1.0], 0)?;
        let expected = [0.5f64, 0.5, 0.5, 0.5]; // (1-0.5)/1, (2-1.5)/1, (3-2.5)/1, (4-3.5)/1
        let actual = result.to_flat_vec::<f64>()?;

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-12, "Expected {}, got {}", e, a);
        }
        Ok(())
    }

    #[test]
    fn test_standardize_empty_tensor() -> anyhow::Result<()> {
        let tensor = Tensor::zeros::<f32>([0])?;
        let result = tensor.standardize(&[1.0f32], &[1.0f32], 0)?;
        assert_eq!(result.numel(), 0);
        Ok(())
    }

    #[test]
    fn test_standardize_error_cases() -> Result<(), Box<dyn std::error::Error>> {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0], [2])?;

        // Empty mean/std
        assert!(tensor.standardize::<f32>(&[], &[], 0).is_err());

        // Mismatched mean/std lengths
        assert!(tensor.standardize(&[1.0f32], &[1.0f32, 2.0], 0).is_err());

        // Zero in std
        assert!(tensor
            .standardize(&[1.0f32, 2.0], &[1.0f32, 0.0], 0)
            .is_err());

        // Wrong number of channels
        assert!(tensor
            .standardize(&[1.0f32, 2.0, 3.0], &[1.0f32, 1.0, 1.0], 0)
            .is_err());

        // Dim out of bounds
        assert!(tensor
            .standardize(&[1.0f32, 2.0], &[1.0f32, 1.0], 2)
            .is_err());

        // // Scalar tensor with wrong dim
        // let scalar = Tensor::from_scalar(5.0f32)?;
        // assert!(scalar.standardize(&[1.0f32], &[1.0f32], 1).is_err());

        // // Scalar tensor with wrong mean/std length
        // assert!(scalar
        //     .standardize(&[1.0f32, 2.0], &[1.0f32, 1.0], 0)
        //     .is_err());

        Ok(())
    }

    #[test]
    fn test_standardize_non_contiguous() -> anyhow::Result<()> {
        // Create a 3x4 tensor and slice it to make it non-contiguous
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [3, 4])?;
        let sliced = tensor.slice(crate::s![.., 1..3]); // Shape: [3, 2]
        let result = sliced.standardize(&[5.0f32, 6.0], &[2.0f32, 2.0], 1)?; // dim = 1

        // Verify the result is correct
        assert_eq!(result.shape(), sliced.shape());
        assert!(result.is_contiguous());

        // Verify the actual values
        let actual = result.to_flat_vec::<f32>()?;
        let sliced_data = sliced.to_flat_vec::<f32>()?;

        // Expected: [(sliced_data[i] - mean[i%2]) / std[i%2]]
        let expected: Vec<f32> = sliced_data
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let channel = i % 2;
                let mean = if channel == 0 { 5.0f32 } else { 6.0f32 };
                let std = 2.0f32;
                (val - mean) / std
            })
            .collect();

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6, "Expected {}, got {}", e, a);
        }

        Ok(())
    }
}
