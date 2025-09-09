use anyhow::Result;

use crate::{global_backend, DType, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    #[inline(always)]
    pub fn sigmoid(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            self.sigmoid_contiguous()
        } else {
            self.sigmoid_non_contiguous()
        }
    }

    #[inline(always)]
    fn sigmoid_contiguous(&self) -> Result<Tensor> {
        let numel = self.shape.numel();
        let backend = global_backend();

        match self.dtype {
            DType::Fp32 => {
                let input_data = self.as_slice::<f32>()?;
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    backend.sigmoid_f32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let input_data = self.as_slice::<f64>()?;
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    backend.sigmoid_f64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let input_data = self.as_slice::<half::f16>()?;
                let out = UninitVec::<half::f16>::new(numel).init_with(|dst| {
                    backend.sigmoid_f16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let input_data = self.as_slice::<half::bf16>()?;
                let out = UninitVec::<half::bf16>::new(numel).init_with(|dst| {
                    backend.sigmoid_bf16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!(
                "Sigmoid operation not supported for dtype: {:?}",
                self.dtype
            ),
        }
    }

    #[inline(always)]
    fn sigmoid_non_contiguous(&self) -> Result<Tensor> {
        match self.dtype {
            DType::Fp32 => self.map_non_contiguous::<f32>(|x| 1.0 / (1.0 + (-x).exp())),
            DType::Fp64 => self.map_non_contiguous::<f64>(|x| 1.0 / (1.0 + (-x).exp())),
            DType::Fp16 => self.map_non_contiguous::<half::f16>(|x| {
                let result = 1.0 / (1.0 + (-x.to_f32()).exp());
                half::f16::from_f32(result)
            }),
            DType::Bf16 => self.map_non_contiguous::<half::bf16>(|x| {
                let result = 1.0 / (1.0 + (-x.to_f32()).exp());
                half::bf16::from_f32(result)
            }),
            _ => anyhow::bail!(
                "Sigmoid operation not supported for dtype: {:?}",
                self.dtype
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s;

    #[test]
    fn test_sigmoid_f32() -> Result<()> {
        let tensor = Tensor::from_vec(vec![0.0f32, 1.0, -1.0, 2.0], [2, 2])?;
        let sigmoid_output = tensor.sigmoid()?;

        // sigmoid(0) = 0.5
        assert!((sigmoid_output.at::<f32>([0, 0]) - 0.5).abs() < 1e-6);
        // sigmoid(1) ≈ 0.7310586
        assert!((sigmoid_output.at::<f32>([0, 1]) - 0.7310586).abs() < 1e-6);
        // sigmoid(-1) ≈ 0.2689414
        assert!((sigmoid_output.at::<f32>([1, 0]) - 0.2689414).abs() < 1e-6);
        // sigmoid(2) ≈ 0.8807971
        assert!((sigmoid_output.at::<f32>([1, 1]) - 0.8807971).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_sigmoid_f64() -> Result<()> {
        let tensor = Tensor::from_vec(vec![0.0f64, 1.0, -1.0], [3])?;
        let sigmoid_output = tensor.sigmoid()?;

        // sigmoid(0) = 0.5
        assert!((sigmoid_output.at::<f64>([0]) - 0.5).abs() < 1e-10);
        // sigmoid(1) ≈ 0.7310586
        assert!((sigmoid_output.at::<f64>([1]) - 0.7310586).abs() < 1e-7);
        // sigmoid(-1) ≈ 0.2689414
        assert!((sigmoid_output.at::<f64>([2]) - 0.2689414).abs() < 1e-7);

        Ok(())
    }

    #[test]
    fn test_sigmoid_f16() -> Result<()> {
        let tensor = Tensor::from_vec(
            vec![half::f16::from_f32(0.0), half::f16::from_f32(1.0)],
            [2],
        )?;
        let sigmoid_output = tensor.sigmoid()?;

        // sigmoid(0) = 0.5
        assert!((sigmoid_output.at::<half::f16>([0]).to_f32() - 0.5).abs() < 1e-3);
        // sigmoid(1) ≈ 0.7310586
        assert!((sigmoid_output.at::<half::f16>([1]).to_f32() - 0.7310586).abs() < 1e-3);

        Ok(())
    }

    #[test]
    fn test_sigmoid_bf16() -> Result<()> {
        let tensor = Tensor::from_vec(
            vec![half::bf16::from_f32(0.0), half::bf16::from_f32(1.0)],
            [2],
        )?;
        let sigmoid_output = tensor.sigmoid()?;

        // sigmoid(0) = 0.5
        assert!((sigmoid_output.at::<half::bf16>([0]).to_f32() - 0.5).abs() < 1e-3);
        // sigmoid(1) ≈ 0.7310586
        assert!((sigmoid_output.at::<half::bf16>([1]).to_f32() - 0.7310586).abs() < 1e-3);

        Ok(())
    }

    #[test]
    fn test_sigmoid_non_contiguous() -> Result<()> {
        let tensor = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, -1.0, -2.0, 3.0], [2, 3])?;
        // Create a non-contiguous view by selecting the second row
        let sliced = tensor.slice(s![1]); // Select the second row
        let sigmoid_output = sliced.sigmoid()?;

        // The second row is [-1.0, -2.0, 3.0]
        // sigmoid(-1) ≈ 0.2689414
        assert!((sigmoid_output.at::<f32>([0]) - 0.2689414).abs() < 1e-6);
        // sigmoid(-2) ≈ 0.1192029
        assert!((sigmoid_output.at::<f32>([1]) - 0.1192029).abs() < 1e-6);
        // sigmoid(3) ≈ 0.9525741
        assert!((sigmoid_output.at::<f32>([2]) - 0.9525741).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_sigmoid_empty() -> Result<()> {
        let tensor = Tensor::from_vec(vec![0.0f32; 0], [0])?;
        let sigmoid_output = tensor.sigmoid()?;

        assert_eq!(sigmoid_output.numel(), 0);
        assert_eq!(sigmoid_output.dims(), [0]);

        Ok(())
    }

    #[test]
    fn test_sigmoid_extreme_values() -> Result<()> {
        let tensor = Tensor::from_vec(vec![100.0f32, -100.0], [2])?;
        let sigmoid_output = tensor.sigmoid()?;

        // sigmoid(100) ≈ 1.0
        assert!((sigmoid_output.at::<f32>([0]) - 1.0).abs() < 1e-6);
        // sigmoid(-100) ≈ 0.0
        assert!(sigmoid_output.at::<f32>([1]) < 1e-6);

        Ok(())
    }
}
