use anyhow::Result;

use crate::{global_backend, DType, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    #[inline(always)]
    pub fn recip(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            self.recip_contiguous()
        } else {
            self.recip_non_contiguous()
        }
    }

    #[inline(always)]
    fn recip_contiguous(&self) -> Result<Tensor> {
        let numel = self.shape.numel();
        let backend = global_backend();

        match self.dtype {
            DType::Fp32 => {
                let input_data = self.as_slice::<f32>()?;
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    backend.v_recip_f32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let input_data = self.as_slice::<f64>()?;
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    backend.v_recip_f64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let input_data = self.as_slice::<half::f16>()?;
                let out = UninitVec::<half::f16>::new(numel).init_with(|dst| {
                    backend.v_recip_f16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let input_data = self.as_slice::<half::bf16>()?;
                let out = UninitVec::<half::bf16>::new(numel).init_with(|dst| {
                    backend.v_recip_bf16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Unsupported dtype for recip operation: {:?}", self.dtype),
        }
    }

    #[inline(always)]
    fn recip_non_contiguous(&self) -> Result<Tensor> {
        match self.dtype {
            DType::Fp32 => self.map_non_contiguous::<f32>(|x| 1.0 / x),
            DType::Fp64 => self.map_non_contiguous::<f64>(|x| 1.0 / x),
            DType::Fp16 => {
                self.map_non_contiguous::<half::f16>(|x| half::f16::from_f32(1.0 / x.to_f32()))
            }
            DType::Bf16 => {
                self.map_non_contiguous::<half::bf16>(|x| half::bf16::from_f32(1.0 / x.to_f32()))
            }
            _ => anyhow::bail!("Unsupported dtype for recip operation: {:?}", self.dtype),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_recip_f32() -> Result<()> {
        let data = vec![1.0f32, 2.0, 4.0, 0.5];
        let tensor = Tensor::from_vec(data, [4])?;
        let result = tensor.recip()?;

        let expected = [1.0, 0.5, 0.25, 2.0]; // 1/1 = 1, 1/2 = 0.5, 1/4 = 0.25, 1/0.5 = 2
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
    fn test_recip_f64() -> Result<()> {
        let data = vec![1.0f64, 2.0, 4.0, 0.5];
        let tensor = Tensor::from_vec(data, [4])?;
        let result = tensor.recip()?;

        let expected = [1.0, 0.5, 0.25, 2.0];
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
    fn test_recip_2d() -> Result<()> {
        let data = vec![1.0f32, 2.0, 4.0, 0.5];
        let tensor = Tensor::from_vec(data, [2, 2])?;
        let result = tensor.recip()?;

        let expected = [1.0, 0.5, 0.25, 2.0];
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
    fn test_recip_non_contiguous() -> Result<()> {
        let data = vec![1.0f32, 2.0, 4.0, 0.5];
        let tensor = Tensor::from_vec(data, [2, 2])?;

        // Test non-contiguous path directly by calling recip_non_contiguous method
        let result = tensor.recip_non_contiguous()?;

        // Check results
        let expected = [1.0, 0.5, 0.25, 2.0]; // 1/1 = 1, 1/2 = 0.5, 1/4 = 0.25, 1/0.5 = 2
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
