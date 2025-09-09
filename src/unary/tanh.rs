use anyhow::Result;

use crate::{global_backend, DType, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    #[inline(always)]
    pub fn tanh(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            self.tanh_contiguous()
        } else {
            self.tanh_non_contiguous()
        }
    }

    #[inline(always)]
    fn tanh_contiguous(&self) -> Result<Tensor> {
        let numel = self.shape.numel();
        let backend = global_backend();

        match self.dtype {
            DType::Fp32 => {
                let input_data = self.as_slice::<f32>()?;
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    backend.v_tanh_f32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let input_data = self.as_slice::<f64>()?;
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    backend.v_tanh_f64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let input_data = self.as_slice::<half::f16>()?;
                let out = UninitVec::<half::f16>::new(numel).init_with(|dst| {
                    backend.v_tanh_f16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let input_data = self.as_slice::<half::bf16>()?;
                let out = UninitVec::<half::bf16>::new(numel).init_with(|dst| {
                    backend.v_tanh_bf16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Unsupported dtype for tanh operation: {:?}", self.dtype),
        }
    }

    #[inline(always)]
    fn tanh_non_contiguous(&self) -> Result<Tensor> {
        match self.dtype {
            DType::Fp32 => self.map_non_contiguous::<f32>(|x| x.tanh()),
            DType::Fp64 => self.map_non_contiguous::<f64>(|x| x.tanh()),
            DType::Fp16 => {
                self.map_non_contiguous::<half::f16>(|x| half::f16::from_f32(x.to_f32().tanh()))
            }
            DType::Bf16 => {
                self.map_non_contiguous::<half::bf16>(|x| half::bf16::from_f32(x.to_f32().tanh()))
            }
            _ => anyhow::bail!("Unsupported dtype for tanh operation: {:?}", self.dtype),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_tanh_f32() -> Result<()> {
        let data = vec![0.0f32, 1.0, -1.0, 2.0];
        let tensor = Tensor::from_vec(data, [4])?;
        let result = tensor.tanh()?;

        let expected = [0.0, 0.761_594_2, -0.761_594_2, 0.964_027_6]; // tanh(0) = 0, tanh(1) ≈ 0.762, tanh(-1) ≈ -0.762, tanh(2) ≈ 0.964
        let result_vec = result.to_flat_vec::<f32>()?;

        for (i, (actual, expected)) in result_vec.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Index {i}: expected {expected}, got {actual}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_tanh_f64() -> Result<()> {
        let data = vec![0.0f64, 1.0, -1.0, 2.0];
        let tensor = Tensor::from_vec(data, [4])?;
        let result = tensor.tanh()?;

        let expected = [
            0.0,
            0.7615941559557649,
            -0.7615941559557649,
            0.9640275800758169,
        ];
        let result_vec = result.to_flat_vec::<f64>()?;

        for (i, (actual, expected)) in result_vec.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-12,
                "Index {i}: expected {expected}, got {actual}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_tanh_2d() -> Result<()> {
        let data = vec![0.0f32, 1.0, -1.0, 2.0];
        let tensor = Tensor::from_vec(data, [2, 2])?;
        let result = tensor.tanh()?;

        let expected = [0.0, 0.761_594_2, -0.761_594_2, 0.964_027_6];
        let result_vec = result.to_flat_vec::<f32>()?;

        for (i, (actual, expected)) in result_vec.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Index {i}: expected {expected}, got {actual}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_tanh_non_contiguous() -> Result<()> {
        let data = vec![0.0f32, 1.0, -1.0, 2.0];
        let tensor = Tensor::from_vec(data, [2, 2])?;

        // Directly test non-contiguous path by calling tanh_non_contiguous method
        let result = tensor.tanh_non_contiguous()?;

        // Check the result
        let expected = [0.0, 0.761_594_2, -0.761_594_2, 0.964_027_6]; // tanh(0) = 0, tanh(1) ≈ 0.762, tanh(-1) ≈ -0.762, tanh(2) ≈ 0.964
        let result_vec = result.to_flat_vec::<f32>()?;

        for (i, (actual, expected)) in result_vec.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Index {i}: expected {expected}, got {actual}"
            );
        }
        Ok(())
    }
}
