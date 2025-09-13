use anyhow::Result;

use crate::{Dim, StorageTrait, Tensor, TensorBase};

impl<S: StorageTrait> TensorBase<S> {
    /// Compute various norms of a tensor, similar to PyTorch's `torch.linalg.norm`.
    /// Supports L1, L2, and Lp norms for floating-point tensors.
    ///
    /// * `ord` - Order of the norm. Supported values:
    ///   - `None` (default): L2 norm (Euclidean norm)
    ///   - `Some(1.0)`: L1 norm (Manhattan norm) - uses backend asum
    ///   - `Some(2.0)`: L2 norm (Euclidean norm) - uses backend nrm2
    #[inline(always)]
    pub fn norm<D: Dim>(&self, dim: D, ord: f32) -> Result<Tensor> {
        match ord {
            1.0 => self.norm_l1(dim),
            2.0 => self.norm_l2(dim),
            _ => anyhow::bail!("Invalid norm order: {}", ord),
        }
    }

    /// Compute various norms of a tensor, similar to PyTorch's `torch.linalg.norm`,
    /// but keeping the specified dimension(s).
    #[inline(always)]
    pub fn norm_keepdim<D: Dim>(&self, dim: D, ord: f32) -> Result<Tensor> {
        match ord {
            1.0 => self.norm1_keepdim(dim),
            2.0 => self.norm2_keepdim(dim),
            _ => anyhow::bail!("Invalid norm order: {}", ord),
        }
    }

    /// Compute L1 norm (sum of absolute values) along the specified dimension
    #[inline(always)]
    pub fn norm_l1<D: Dim>(&self, dim: D) -> Result<Tensor> {
        self.abs()?.sum(dim)
    }

    /// Compute L1 norm (sum of absolute values) along the specified dimension, keeping dimensions
    #[inline(always)]
    pub fn norm1_keepdim<D: Dim>(&self, dim: D) -> Result<Tensor> {
        self.abs()?.sum_keepdim(dim)
    }

    /// Compute L2 norm (Euclidean norm) along the specified dimension
    #[inline(always)]
    pub fn norm_l2<D: Dim>(&self, dim: D) -> Result<Tensor> {
        if !self.dtype.is_float() {
            anyhow::bail!(
                "norm_l2 only supports floating-point types, got {:?}",
                self.dtype
            );
        }
        self.sqr()?.sum(dim)?.sqrt()
    }

    /// Compute L2 norm (Euclidean norm) along the specified dimension, keeping dimensions
    #[inline(always)]
    pub fn norm2_keepdim<D: Dim>(&self, dim: D) -> Result<Tensor> {
        if !self.dtype.is_float() {
            anyhow::bail!(
                "norm_l2 only supports floating-point types, got {:?}",
                self.dtype
            );
        }
        self.sqr()?.sum_keepdim(dim)?.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;
    use anyhow::Result;

    #[test]
    fn test_norm1_1d_f32() -> Result<()> {
        let data = vec![3.0f32, -4.0];
        let tensor = Tensor::from_vec(data, [2])?;
        let result = tensor.norm_l1(0)?;

        let expected = 7.0; // |3| + |-4| = 3 + 4 = 7
        let result_val = result.to_scalar::<f64>()?;
        assert!((result_val - expected).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_norm1_1d_f64() -> Result<()> {
        let data = vec![3.0f64, -4.0];
        let tensor = Tensor::from_vec(data, [2])?;
        let result = tensor.norm_l1(0)?;

        let expected = 7.0;
        let result_val = result.to_scalar::<f64>()?;
        assert!((result_val - expected).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_norm1_2d_dim0() -> Result<()> {
        // [[1, -2], [-3, 4]]
        let data = vec![1.0f32, -2.0, -3.0, 4.0];
        let tensor = Tensor::from_vec(data, [2, 2])?;
        let result = tensor.norm_l1(0)?;

        // norm along dim 0: [|1|+|-3|, |-2|+|4|] = [1+3, 2+4] = [4, 6]
        let result_vec = result.to_vec::<f64>()?;
        let expected = [4.0f64, 6.0f64];

        for (actual, expected) in result_vec.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_norm1_2d_dim1() -> Result<()> {
        // [[1, -2], [-3, 4]]
        let data = vec![1.0f32, -2.0, -3.0, 4.0];
        let tensor = Tensor::from_vec(data, [2, 2])?;
        let result = tensor.norm_l1(1)?;

        // norm along dim 1: [|1|+|-2|, |-3|+|4|] = [1+2, 3+4] = [3, 7]
        let result_vec = result.to_vec::<f64>()?;
        let expected = [3.0f64, 7.0f64];

        for (actual, expected) in result_vec.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_norm1_keepdim() -> Result<()> {
        let data = vec![3.0f32, -4.0];
        let tensor = Tensor::from_vec(data, [2])?;
        let result = tensor.norm1_keepdim(0)?;

        assert_eq!(result.dims(), &[1]);
        let expected = 7.0;
        let result_vec = result.to_vec::<f64>()?;
        assert!((result_vec[0] - expected).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_norm1_3d() -> Result<()> {
        // 2x2x2 tensor
        let data = vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let tensor = Tensor::from_vec(data, [2, 2, 2])?;
        let result = tensor.norm_l1(0)?;

        assert_eq!(result.dims(), &[2, 2]);
        assert_eq!(result.numel(), 4);
        Ok(())
    }

    #[test]
    fn test_norm1_zero_vector() -> Result<()> {
        let data = vec![0.0f32, 0.0, 0.0];
        let tensor = Tensor::from_vec(data, [3])?;
        let result = tensor.norm_l1(0)?;

        let result_val = result.to_scalar::<f64>()?;
        assert!((result_val - 0.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_norm1_single_element() -> Result<()> {
        let data = vec![-5.0f32];
        let tensor = Tensor::from_vec(data, [1])?;
        let result = tensor.norm_l1(0)?;

        let result_val = result.to_scalar::<f64>()?;
        assert!((result_val - 5.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_norm1_mixed_signs() -> Result<()> {
        let data = vec![-1.0f32, 2.0, -3.0, 4.0, -5.0];
        let tensor = Tensor::from_vec(data, [5])?;
        let result = tensor.norm_l1(0)?;

        let expected = 15.0; // |−1| + |2| + |−3| + |4| + |−5| = 1 + 2 + 3 + 4 + 5 = 15
        let result_val = result.to_scalar::<f64>()?;
        assert!((result_val - expected).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_norm2_1d_f32() -> Result<()> {
        let data = vec![3.0f32, 4.0];
        let tensor = Tensor::from_vec(data, [2])?;
        let result = tensor.norm_l2(0)?;

        let expected = 5.0; // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
        let result_val = result.to_scalar::<f64>()?;
        assert!((result_val - expected).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_norm2_1d_f64() -> Result<()> {
        let data = vec![3.0f64, 4.0];
        let tensor = Tensor::from_vec(data, [2])?;
        let result = tensor.norm_l2(0)?;

        let expected = 5.0;
        let result_val = result.to_scalar::<f64>()?;
        assert!((result_val - expected).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_norm2_2d_dim0() -> Result<()> {
        // [[1, 2], [3, 4]]
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, [2, 2])?;
        let result = tensor.norm_l2(0)?;

        // norm along dim 0: [sqrt(1^2+3^2), sqrt(2^2+4^2)] = [sqrt(10), sqrt(20)]
        let result_vec = result.to_vec::<f64>()?;
        let expected = [10.0f64.sqrt(), 20.0f64.sqrt()];

        for (actual, expected) in result_vec.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_norm2_2d_dim1() -> Result<()> {
        // [[1, 2], [3, 4]]
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, [2, 2])?;
        let result = tensor.norm_l2(1)?;

        // norm along dim 1: [sqrt(1^2+2^2), sqrt(3^2+4^2)] = [sqrt(5), sqrt(25)] = [sqrt(5), 5]
        let result_vec = result.to_vec::<f64>()?;
        let expected = [5.0f64.sqrt(), 5.0f64];

        for (actual, expected) in result_vec.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_norm2_keepdim() -> Result<()> {
        let data = vec![3.0f32, 4.0];
        let tensor = Tensor::from_vec(data, [2])?;
        let result = tensor.norm2_keepdim(0)?;

        assert_eq!(result.dims(), &[1]);
        let expected = 5.0;
        let result_vec = result.to_vec::<f64>()?;
        assert!((result_vec[0] - expected).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_norm2_3d() -> Result<()> {
        // 2x2x2 tensor
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data, [2, 2, 2])?;
        let result = tensor.norm_l2(0)?;

        assert_eq!(result.dims(), &[2, 2]);
        assert_eq!(result.numel(), 4);
        Ok(())
    }

    #[test]
    fn test_norm2_zero_vector() -> Result<()> {
        let data = vec![0.0f32, 0.0, 0.0];
        let tensor = Tensor::from_vec(data, [3])?;
        let result = tensor.norm_l2(0)?;

        let result_val = result.to_scalar::<f64>()?;
        assert!((result_val - 0.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_norm2_single_element() -> Result<()> {
        let data = vec![5.0f32];
        let tensor = Tensor::from_vec(data, [1])?;
        let result = tensor.norm_l2(0)?;

        let result_val = result.to_scalar::<f64>()?;
        assert!((result_val - 5.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_norm2_non_float_error() -> Result<()> {
        let data = vec![1i32, 2, 3];
        let tensor = Tensor::from_vec(data, [3])?;
        let result = tensor.norm_l2(0);

        assert!(result.is_err());
        Ok(())
    }
}
