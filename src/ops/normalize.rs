use crate::{StorageTrait, Tensor, TensorBase, TensorElement};

impl<S: StorageTrait> TensorBase<S> {
    /// Normalizes the tensor to the range [0, 1] using min-max normalization.
    ///
    /// The formula used is: `x_normalized = (x - min) / (max - min)`
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value in the original range
    /// * `max` - The maximum value in the original range
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The dtype of `min` and `max` does not match the tensor's dtype.
    /// - `min` equals `max` (division by zero).
    /// - `min` is greater than `max`.
    ///
    /// # Examples
    ///
    /// ```
    /// use slsl::Tensor;
    ///
    /// // Normalize a 3D image tensor from [0, 255] to [0, 1]
    /// let image_data = vec![
    ///     0.0, 128.0, 255.0,   // Pixel 1, Channel R, G, B
    ///     64.0, 192.0, 32.0,   // Pixel 2, Channel R, G, B
    /// ];
    /// let tensor = Tensor::from_vec(image_data, [2, 3]).unwrap();
    /// let normalized = tensor.normalize(0., 255.).unwrap();
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn normalize<T: TensorElement + num_traits::Float + std::fmt::Debug>(
        &self,
        min: T,
        max: T,
    ) -> anyhow::Result<Tensor> {
        if T::DTYPE != self.dtype() {
            anyhow::bail!(
                "Dtype mismatch: Tensor ({:?}), Min/Max ({:?}).",
                self.dtype(),
                T::DTYPE
            );
        }
        if min >= max {
            anyhow::bail!(
                "Min value ({:?}) must be less than max value ({:?}).",
                min,
                max
            );
        }
        if self.numel() == 0 {
            return self.to_contiguous();
        }

        let range = max - min;

        if self.rank() == 0 {
            let min_tensor = Tensor::from_scalar(min)?;
            let range_tensor = Tensor::from_scalar(range)?;
            return Ok((self - min_tensor) / range_tensor);
        }

        let shape = self.dims();
        let min_scalar = Tensor::from_scalar(min)?;
        let range_scalar = Tensor::from_scalar(range)?;
        let min_tensor = min_scalar.broadcast_to(shape)?;
        let range_tensor = range_scalar.broadcast_to(shape)?;
        Ok((self - min_tensor) / range_tensor)
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_normalize_scalar() -> anyhow::Result<()> {
        let tensor = Tensor::from_scalar(5.0f32)?;
        let result = tensor.normalize(0.0f32, 10.0f32)?;
        let expected = 0.5f32; // (5 - 0) / (10 - 0) = 0.5
        assert!((result.to_scalar::<f32>()? - expected).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_normalize_1d() -> anyhow::Result<()> {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [4])?;
        let result = tensor.normalize(1.0f32, 4.0f32)?;
        let expected = [0.0f32, 1.0 / 3.0, 2.0 / 3.0, 1.0]; // (x - 1) / (4 - 1)
        let actual = result.to_flat_vec::<f32>()?;

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6, "Expected {}, got {}", e, a);
        }
        Ok(())
    }

    #[test]
    fn test_normalize_2d() -> anyhow::Result<()> {
        let tensor = Tensor::from_vec(vec![0.0f32, 50.0, 100.0, 150.0, 200.0, 255.0], [2, 3])?;
        let result = tensor.normalize(0.0f32, 255.0f32)?;
        let expected = [
            0.0f32,
            50.0 / 255.0,
            100.0 / 255.0,
            150.0 / 255.0,
            200.0 / 255.0,
            1.0,
        ];
        let actual = result.to_flat_vec::<f32>()?;

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6, "Expected {}, got {}", e, a);
        }
        Ok(())
    }

    #[test]
    fn test_normalize_3d_image() -> anyhow::Result<()> {
        // 2x2x3 tensor (HxWxC format) with values from 0 to 255
        let data = vec![
            0.0f32, 85.0, 170.0, // Pixel 1
            42.5, 127.5, 212.5, // Pixel 2
            85.0, 170.0, 255.0, // Pixel 3
            127.5, 212.5, 42.5, // Pixel 4
        ];
        let tensor = Tensor::from_vec(data, [2, 2, 3])?;
        let result = tensor.normalize(0.0f32, 255.0f32)?;
        let actual = result.to_flat_vec::<f32>()?;

        // Verify a few key values
        assert!((actual[0] - 0.0).abs() < 1e-6); // 0/255 = 0
        assert!((actual[2] - (170.0 / 255.0)).abs() < 1e-6); // 170/255
        assert!((actual[8] - 1.0).abs() < 1e-6); // 255/255 = 1 (index 8 contains 255.0)

        Ok(())
    }

    #[test]
    fn test_normalize_f64() -> anyhow::Result<()> {
        let data = vec![-10.0f64, -5.0, 0.0, 5.0, 10.0];
        let tensor = Tensor::from_vec(data, [5])?;
        let result = tensor.normalize(-10.0f64, 10.0f64)?;
        let expected = [0.0f64, 0.25, 0.5, 0.75, 1.0]; // (x + 10) / 20
        let actual = result.to_flat_vec::<f64>()?;

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-12, "Expected {}, got {}", e, a);
        }
        Ok(())
    }

    #[test]
    fn test_normalize_empty_tensor() -> anyhow::Result<()> {
        let tensor = Tensor::zeros::<f32>([0])?;
        let result = tensor.normalize(0.0f32, 1.0f32)?;
        assert_eq!(result.numel(), 0);
        Ok(())
    }

    #[test]
    fn test_normalize_error_cases() -> Result<(), Box<dyn std::error::Error>> {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0], [2])?;

        // Min equals max
        assert!(tensor.normalize(5.0f32, 5.0f32).is_err());

        // Min greater than max
        assert!(tensor.normalize(10.0f32, 5.0f32).is_err());

        // Wrong dtype (trying to normalize f32 tensor with f64 values)
        // This would be caught at compile time, so we test with different tensor
        let _tensor_f64 = Tensor::from_vec(vec![1.0f64, 2.0], [2])?;
        assert!(tensor.normalize(1.0f64, 2.0f64).is_err()); // f32 tensor with f64 params

        Ok(())
    }

    #[test]
    fn test_normalize_non_contiguous() -> anyhow::Result<()> {
        // Create a 3x4 tensor and slice it to make it non-contiguous
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [3, 4])?;
        let sliced = tensor.slice(crate::s![.., 1..3]); // Shape: [3, 2]
        let result = sliced.normalize(0.0f32, 11.0f32)?;

        // Verify the result is correct
        assert_eq!(result.shape(), sliced.shape());
        assert!(result.is_contiguous());

        // Verify the actual values
        let actual = result.to_flat_vec::<f32>()?;
        let sliced_data = sliced.to_flat_vec::<f32>()?;

        // Expected: (sliced_data[i] - 0) / (11 - 0)
        let expected: Vec<f32> = sliced_data.iter().map(|&val| val / 11.0f32).collect();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6, "Expected {}, got {}", e, a);
        }

        Ok(())
    }
}
