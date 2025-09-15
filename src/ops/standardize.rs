use crate::{Dim, Shape, StorageTrait, Tensor, TensorBase, TensorElement};

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

        let mut broadcast_shape = Shape::full(1, self.rank());
        broadcast_shape[dim] = dim_size;
        let mean = Tensor::from(mean).reshape(broadcast_shape)?;
        let std = Tensor::from(std).reshape(broadcast_shape)?;

        Ok((self - mean.broadcast_to(self.dims())?) / std.broadcast_to(self.dims())?)
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
