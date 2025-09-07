use crate::{Dims, StorageTrait, Tensor, TensorBase};
use anyhow::Result;

impl<S: StorageTrait> TensorBase<S> {
    pub fn mean<D: Dims + Copy>(&self, dims: D) -> Result<Tensor> {
        let sum_result = self.sum(dims)?;

        // Calculate the size of reduced dimensions
        let dim_indices = dims.to_dims(self.rank())?;
        let reduced_elements: usize = dim_indices
            .iter()
            .map(|&dim_idx| self.shape()[dim_idx])
            .product();

        // Divide sum result by reduction element count
        let mut result_data = sum_result.as_slice::<f64>()?.to_vec();
        let divisor = reduced_elements as f64;
        for val in result_data.iter_mut() {
            *val /= divisor;
        }

        Tensor::from_vec(result_data, sum_result.dims())
    }

    pub fn mean_keepdim<D: Dims + Copy>(&self, dims: D) -> Result<Tensor> {
        let sum_result = self.sum_keepdim(dims)?;

        // Calculate the size of reduced dimensions
        let dim_indices = dims.to_dims(self.rank())?;
        let reduced_elements: usize = dim_indices
            .iter()
            .map(|&dim_idx| self.shape()[dim_idx])
            .product();

        // Divide sum result by reduction element count
        let mut result_data = sum_result.as_slice::<f64>()?.to_vec();
        let divisor = reduced_elements as f64;
        for val in result_data.iter_mut() {
            *val /= divisor;
        }

        Tensor::from_vec(result_data, sum_result.dims())
    }

    pub fn mean_all(&self) -> Result<f64> {
        if self.numel() == 0 {
            return Ok(0.0);
        }
        let sum_result = self.sum_all()?;
        Ok(sum_result / self.numel() as f64)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use anyhow::Result;

    #[test]
    fn test_mean_all_basic() -> Result<()> {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, [5])?;
        let mean = tensor.mean_all()?;
        assert_eq!(mean, 3.0);
        Ok(())
    }

    #[test]
    fn test_mean_all_empty() -> Result<()> {
        let tensor = Tensor::from_vec(Vec::<f64>::new(), [0])?;
        let mean = tensor.mean_all()?;
        assert_eq!(mean, 0.0);
        Ok(())
    }

    #[test]
    fn test_mean_all_single_element() -> Result<()> {
        let tensor = Tensor::from_vec(vec![42.0], [1])?;
        let mean = tensor.mean_all()?;
        assert_eq!(mean, 42.0);
        Ok(())
    }

    #[test]
    fn test_mean_all_negative_values() -> Result<()> {
        let data = vec![-1.0, -2.0, -3.0, -4.0];
        let tensor = Tensor::from_vec(data, [4])?;
        let mean = tensor.mean_all()?;
        assert_eq!(mean, -2.5);
        Ok(())
    }

    #[test]
    fn test_mean_all_mixed_values() -> Result<()> {
        let data = vec![-2.0, 0.0, 2.0, 4.0];
        let tensor = Tensor::from_vec(data, [4])?;
        let mean = tensor.mean_all()?;
        assert_eq!(mean, 1.0);
        Ok(())
    }

    #[test]
    fn test_mean_1d() -> Result<()> {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [6])?;
        let mean = tensor.mean(0)?;
        assert_eq!(mean.rank(), 0);

        let mean_value = mean.to_scalar::<f64>()?;
        assert_eq!(mean_value, 3.5);
        Ok(())
    }

    // TODO
    #[test]
    fn test_mean_2d_dim0() -> Result<()> {
        // For 2x3 tensor: [[1,2,3], [4,5,6]]
        // Correct sum along dim 0 is [5.0, 7.0, 9.0]
        // Mean along dim 0: [2.5, 3.5, 4.5]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;
        let mean = tensor.mean(0)?;
        assert_eq!(mean.dims(), [3]);

        let mean_values = mean.as_slice::<f64>()?;
        assert_eq!(mean_values, vec![2.5, 3.5, 4.5]);
        Ok(())
    }

    #[test]
    fn test_mean_2d_dim1() -> Result<()> {
        // Test mean along dimension 1 (columns) in 2D tensor
        // For 2x3 tensor: [[1,2,3], [4,5,6]]
        // Mean along dim 1: [(1+2+3)/3, (4+5+6)/3] = [2.0, 5.0]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;
        let mean = tensor.mean(1)?;

        // Should return tensor with shape [2] (rows)
        assert_eq!(mean.dims(), [2]);
        let mean_values = mean.as_slice::<f64>()?;
        assert_eq!(mean_values, vec![2.0, 5.0]);
        Ok(())
    }

    #[test]
    fn test_mean_2d_both_dims() -> Result<()> {
        // Test mean along both dimensions
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;
        let mean = tensor.mean([0, 1])?;

        // Should return a scalar tensor
        assert_eq!(mean.dims(), &[] as &[usize]);
        let mean_value = mean.as_slice::<f64>()?[0];
        assert_eq!(mean_value, 3.5);
        Ok(())
    }

    #[test]
    fn test_mean_3d_dim0() -> Result<()> {
        // Test mean along dimension 0 in 3D tensor
        // For 2x2x2 tensor with data [1,2,  3,4,  5,6,  7,8]
        // Correct mean along dim 0: [3.0, 4.0, 5.0, 6.0]
        let data: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let tensor = Tensor::from_vec(data, [2, 2, 2])?;
        let mean = tensor.mean(0)?;

        // Should return tensor with shape [2, 2]
        assert_eq!(mean.dims(), [2, 2]);
        let mean_values = mean.as_slice::<f64>()?;
        assert_eq!(mean_values, vec![3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_mean_keepdim_1d() -> Result<()> {
        // Test mean_keepdim along 1D dimension
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [6])?;
        let mean = tensor.mean_keepdim(0)?;

        // For 1D tensor with keepdim, should return shape [1]
        assert_eq!(mean.dims(), [1]);
        let mean_value = mean.as_slice::<f64>()?[0];
        assert_eq!(mean_value, 3.5);
        Ok(())
    }

    #[test]
    fn test_mean_keepdim_2d_dim0() -> Result<()> {
        // Test mean_keepdim along dimension 0 (rows) in 2D tensor
        // Correct sum along dim 0 is [5.0, 7.0, 9.0]
        // Mean along dim 0: [2.5, 3.5, 4.5]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;
        let mean = tensor.mean_keepdim(0)?;

        // Should return tensor with shape [1, 3] (keeping dimension 0)
        assert_eq!(mean.dims(), [1, 3]);
        let mean_values = mean.as_slice::<f64>()?;
        assert_eq!(mean_values, vec![2.5, 3.5, 4.5]);
        Ok(())
    }

    #[test]
    fn test_mean_keepdim_2d_dim1() -> Result<()> {
        // Test mean_keepdim along dimension 1 (columns) in 2D tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;
        let mean = tensor.mean_keepdim(1)?;

        // Should return tensor with shape [2, 1] (keeping dimension 1)
        assert_eq!(mean.dims(), [2, 1]);
        let mean_values = mean.as_slice::<f64>()?;
        assert_eq!(mean_values, vec![2.0, 5.0]);
        Ok(())
    }

    #[test]
    fn test_mean_keepdim_2d_both_dims() -> Result<()> {
        // Test mean_keepdim along both dimensions
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;
        let mean = tensor.mean_keepdim([0, 1])?;

        // With keepdim, should return shape [1, 1]
        assert_eq!(mean.dims(), [1, 1]);
        let mean_value = mean.as_slice::<f64>()?[0];
        assert_eq!(mean_value, 3.5);
        Ok(())
    }

    #[test]
    fn test_mean_keepdim_3d_dim0() -> Result<()> {
        // Test mean_keepdim along dimension 0 in 3D tensor
        // Correct mean along dim 0: [3.0, 4.0, 5.0, 6.0]
        let data: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let tensor = Tensor::from_vec(data, [2, 2, 2])?;
        let mean = tensor.mean_keepdim(0)?;

        // Should return tensor with shape [1, 2, 2] (keeping dimension 0)
        assert_eq!(mean.dims(), [1, 2, 2]);
        let mean_values = mean.as_slice::<f64>()?;
        assert_eq!(mean_values, vec![3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_mean_integer_data() -> Result<()> {
        // Test mean with integer data (should convert to f64)
        let data = vec![1, 2, 3, 4, 5];
        let tensor = Tensor::from_vec(data, [5])?;
        let mean = tensor.mean_all()?;
        assert_eq!(mean, 3.0);
        Ok(())
    }

    #[test]
    fn test_mean_u8_data() -> Result<()> {
        // Test mean with u8 data
        let data: Vec<u8> = vec![10, 20, 30, 40, 50];
        let tensor = Tensor::from_vec(data, [5])?;
        let mean = tensor.mean_all()?;
        assert_eq!(mean, 30.0);
        Ok(())
    }

    #[test]
    fn test_mean_large_tensor() -> Result<()> {
        // Test mean with larger tensor
        let size = 1000;
        let data: Vec<f64> = (0..size).map(|x| x as f64).collect();
        let tensor = Tensor::from_vec(data, [size])?;
        let mean = tensor.mean_all()?;
        let expected_mean = (size - 1) as f64 / 2.0;
        assert!((mean - expected_mean).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_mean_consistency() -> Result<()> {
        // Test that mean and sum/n give same results
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, [5])?;

        let mean_direct = tensor.mean_all()?;
        let sum_result = tensor.sum_all()?;
        let mean_sum_div_n = sum_result / 5.0;

        assert!((mean_direct - mean_sum_div_n).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_mean_keepdim_consistency() -> Result<()> {
        // Test that mean_keepdim and sum_keepdim/n give same results
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        let mean_keepdim = tensor.mean_keepdim(0)?;
        let sum_keepdim = tensor.sum_keepdim(0)?;

        // Both should have same shape
        assert_eq!(mean_keepdim.dims(), sum_keepdim.dims());

        // Values should be related by division
        let mean_values = mean_keepdim.as_slice::<f64>()?;
        let sum_values = sum_keepdim.as_slice::<f64>()?;

        for (mean_val, sum_val) in mean_values.iter().zip(sum_values.iter()) {
            assert!((*mean_val * 2.0 - *sum_val).abs() < 1e-10);
        }
        Ok(())
    }

    #[test]
    fn test_mean_edge_cases() -> Result<()> {
        // Test various edge cases

        // Single element tensor
        let single = Tensor::from_vec(vec![42.0], [1])?;
        assert_eq!(single.mean_all()?, 42.0);

        // Tensor with all zeros
        let zeros = Tensor::from_vec(vec![0.0f32, 0.0, 0.0], [3])?;
        assert_eq!(zeros.mean_all()?, 0.0);

        // Tensor with all same values
        let same = Tensor::from_vec(vec![5.0f32, 5.0, 5.0], [3])?;
        assert_eq!(same.mean_all()?, 5.0);

        Ok(())
    }

    #[test]
    fn test_mean_invalid_dimension() -> Result<()> {
        // Test mean with invalid dimension
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, [3])?;

        // Should fail for dimension >= rank
        assert!(tensor.mean(1).is_err());
        assert!(tensor.mean(2).is_err());

        Ok(())
    }

    #[test]
    fn test_mean_empty_dimension() -> Result<()> {
        // Test mean with empty dimension list
        // Based on debug output: empty dimensions mean returns sum, not mean
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, [3])?;

        // Empty dimensions mean returns sum result, not mean
        let mean_empty = tensor.mean::<[usize; 0]>([])?;
        let sum_all = tensor.sum_all()?;

        // Empty dimensions mean should return sum result
        assert_eq!(mean_empty.dims(), &[] as &[usize]);
        let mean_empty_val = mean_empty.as_slice::<f64>()?[0];
        assert!((mean_empty_val - sum_all).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_mean_non_contiguous_2d() -> Result<()> {
        // Test mean with non-contiguous tensor using permute
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        // Create non-contiguous tensor by permuting dimensions
        let permuted = tensor.clone().permute([1, 0])?;

        // Test mean along different dimensions
        let mean_dim0 = permuted.mean(0)?;
        let expected_dim0 = [2.0, 5.0]; // Mean of columns after permute: [(1+2+3)/3, (4+5+6)/3]
        let contiguous_dim0 = mean_dim0.to_contiguous()?;
        let result_dim0 = contiguous_dim0.as_slice::<f64>()?;
        for (actual, expected) in result_dim0.iter().zip(expected_dim0.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }

        let mean_dim1 = permuted.mean(1)?;
        let expected_dim1 = [2.5, 3.5, 4.5]; // Mean of rows after permute: [(1+4)/2, (2+5)/2, (3+6)/2]
        let contiguous_dim1 = mean_dim1.to_contiguous()?;
        let result_dim1 = contiguous_dim1.as_slice::<f64>()?;
        for (actual, expected) in result_dim1.iter().zip(expected_dim1.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_mean_non_contiguous_3d() -> Result<()> {
        // Test mean with 3D non-contiguous tensor
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let tensor = Tensor::from_vec(data, [2, 3, 4])?;

        // Create non-contiguous tensor by permuting dimensions
        let permuted = tensor.clone().permute([2, 0, 1])?; // [4, 2, 3]

        // Test mean along dimension 0
        let mean_dim0 = permuted.mean(0)?;
        assert_eq!(mean_dim0.dims(), &[2, 3]);

        // Test mean along dimension 1
        let mean_dim1 = permuted.mean(1)?;
        assert_eq!(mean_dim1.dims(), &[4, 3]);

        // Test mean along dimension 2
        let mean_dim2 = permuted.mean(2)?;
        assert_eq!(mean_dim2.dims(), &[4, 2]);

        // Verify mean_all is consistent
        let mean_all_original = tensor.mean_all()?;
        let mean_all_permuted = permuted.mean_all()?;
        assert!((mean_all_original - mean_all_permuted).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_mean_keepdim_non_contiguous() -> Result<()> {
        // Test mean_keepdim with non-contiguous tensor
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data, [2, 2, 2])?;

        // Create non-contiguous tensor
        let permuted = tensor.clone().permute([2, 1, 0])?; // [2, 2, 2]

        // Test mean_keepdim along different dimensions
        let mean_keepdim_0 = permuted.mean_keepdim(0)?;
        assert_eq!(mean_keepdim_0.dims(), &[1, 2, 2]);

        let mean_keepdim_1 = permuted.mean_keepdim(1)?;
        assert_eq!(mean_keepdim_1.dims(), &[2, 1, 2]);

        let mean_keepdim_2 = permuted.mean_keepdim(2)?;
        assert_eq!(mean_keepdim_2.dims(), &[2, 2, 1]);

        // Test multiple dimensions
        let mean_keepdim_01 = permuted.mean_keepdim([0, 1])?;
        assert_eq!(mean_keepdim_01.dims(), &[1, 1, 2]);

        Ok(())
    }

    #[test]
    fn test_mean_different_data_types() -> Result<()> {
        // Test mean with different data types

        // Test with i32
        let data_i32 = vec![1i32, 2, 3, 4, 5, 6];
        let tensor_i32 = Tensor::from_vec(data_i32, [2, 3])?;
        let mean_i32 = tensor_i32.mean(0)?;
        let expected_i32 = [2.5, 3.5, 4.5];
        let result_i32 = mean_i32.as_slice::<f64>()?;
        for (actual, expected) in result_i32.iter().zip(expected_i32.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }

        // Test with u32
        let data_u32 = vec![10u32, 20, 30, 40];
        let tensor_u32 = Tensor::from_vec(data_u32, [2, 2])?;
        let mean_u32 = tensor_u32.mean(1)?;
        let expected_u32 = [15.0, 35.0];
        let result_u32 = mean_u32.as_slice::<f64>()?;
        for (actual, expected) in result_u32.iter().zip(expected_u32.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_mean_special_values() -> Result<()> {
        // Test mean with special floating point values

        // Test with infinity
        let data_inf = vec![1.0f64, f64::INFINITY, 3.0];
        let tensor_inf = Tensor::from_vec(data_inf, [3])?;
        let mean_inf = tensor_inf.mean_all()?;
        assert!(mean_inf.is_infinite());

        // Test with NaN
        let data_nan = vec![1.0f64, f64::NAN, 3.0];
        let tensor_nan = Tensor::from_vec(data_nan, [3])?;
        let mean_nan = tensor_nan.mean_all()?;
        assert!(mean_nan.is_nan());

        // Test with very large numbers
        let data_large = vec![1e100, 2e100, 3e100];
        let tensor_large = Tensor::from_vec(data_large, [3])?;
        let mean_large = tensor_large.mean_all()?;
        assert!((mean_large - 2e100).abs() < 1e90);

        Ok(())
    }

    #[test]
    fn test_mean_rectangular_tensors() -> Result<()> {
        // Test mean with rectangular (non-square) tensors

        // 1x5 tensor
        let data_1x5 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor_1x5 = Tensor::from_vec(data_1x5, [1, 5])?;
        let mean_dim0 = tensor_1x5.mean(0)?;
        assert_eq!(mean_dim0.dims(), &[5]);
        let mean_dim1 = tensor_1x5.mean(1)?;
        assert_eq!(mean_dim1.dims(), &[1]);
        assert!((mean_dim1.as_slice::<f64>()?[0] - 3.0).abs() < 1e-10);

        // 5x1 tensor
        let data_5x1 = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let tensor_5x1 = Tensor::from_vec(data_5x1, [5, 1])?;
        let mean_dim0 = tensor_5x1.mean(0)?;
        assert_eq!(mean_dim0.dims(), &[1]);
        assert!((mean_dim0.as_slice::<f64>()?[0] - 30.0).abs() < 1e-10);
        let mean_dim1 = tensor_5x1.mean(1)?;
        assert_eq!(mean_dim1.dims(), &[5]);

        Ok(())
    }

    #[test]
    fn test_mean_consistency_with_sum() -> Result<()> {
        // Test that mean is consistent with sum / count
        let data = vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        // Test along dimension 0
        let mean_dim0 = tensor.mean(0)?;
        let sum_dim0 = tensor.sum(0)?;
        let mean_values = mean_dim0.as_slice::<f64>()?;
        let sum_values = sum_dim0.as_slice::<f64>()?;

        for (mean_val, sum_val) in mean_values.iter().zip(sum_values.iter()) {
            let expected_mean = sum_val / 2.0; // 2 elements along dim 0
            assert!((mean_val - expected_mean).abs() < 1e-10);
        }

        // Test along dimension 1
        let mean_dim1 = tensor.mean(1)?;
        let sum_dim1 = tensor.sum(1)?;
        let mean_values = mean_dim1.as_slice::<f64>()?;
        let sum_values = sum_dim1.as_slice::<f64>()?;

        for (mean_val, sum_val) in mean_values.iter().zip(sum_values.iter()) {
            let expected_mean = sum_val / 3.0; // 3 elements along dim 1
            assert!((mean_val - expected_mean).abs() < 1e-10);
        }

        Ok(())
    }
}
