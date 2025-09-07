use anyhow::Result;

use crate::{
    backend::{global_backend, r#impl::OpsTrait},
    DType, StorageTrait, Tensor, TensorBase, UninitVec,
};

impl<S: StorageTrait> TensorBase<S> {
    #[inline(always)]
    pub fn cos(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            self.cos_contiguous()
        } else {
            self.cos_non_contiguous()
        }
    }

    #[inline(always)]
    fn cos_contiguous(&self) -> Result<Tensor> {
        let numel = self.shape.numel();
        let backend = global_backend();

        match self.dtype {
            DType::Fp32 => {
                let input_data = self.as_slice::<f32>()?;
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    backend.v_cos_f32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let input_data = self.as_slice::<f64>()?;
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    backend.v_cos_f64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let input_data = self.as_slice::<half::f16>()?;
                let out = UninitVec::<half::f16>::new(numel).init_with(|dst| {
                    backend.v_cos_f16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let input_data = self.as_slice::<half::bf16>()?;
                let out = UninitVec::<half::bf16>::new(numel).init_with(|dst| {
                    backend.v_cos_bf16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Unsupported dtype for cos operation: {:?}", self.dtype),
        }
    }

    #[inline(always)]
    fn cos_non_contiguous(&self) -> Result<Tensor> {
        // numel is no longer needed

        match self.dtype {
            DType::Fp32 => self.map_non_contiguous::<f32>(|x| x.cos()),
            DType::Fp64 => self.map_non_contiguous::<f64>(|x| x.cos()),
            DType::Fp16 => {
                self.map_non_contiguous::<half::f16>(|x| half::f16::from_f32(x.to_f32().cos()))
            }
            DType::Bf16 => {
                self.map_non_contiguous::<half::bf16>(|x| half::bf16::from_f32(x.to_f32().cos()))
            }
            _ => anyhow::bail!("Unsupported dtype for cos operation: {:?}", self.dtype),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_cos_f32() -> Result<()> {
        let data = vec![0.0f32, std::f32::consts::PI / 2.0, std::f32::consts::PI];
        let tensor = Tensor::from_vec(data, [3])?;
        let result = tensor.cos()?;

        let expected = [1.0, 0.0, -1.0]; // cos(0) = 1, cos(π/2) = 0, cos(π) = -1
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
    fn test_cos_f64() -> Result<()> {
        let data = vec![0.0f64, std::f64::consts::PI / 2.0, std::f64::consts::PI];
        let tensor = Tensor::from_vec(data, [3])?;
        let result = tensor.cos()?;

        let expected = [1.0, 0.0, -1.0];
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
    fn test_cos_2d() -> Result<()> {
        let data = vec![
            0.0f32,
            std::f32::consts::PI / 2.0,
            std::f32::consts::PI,
            0.0,
        ];
        let tensor = Tensor::from_vec(data, [2, 2])?;
        let result = tensor.cos()?;

        let expected = [1.0, 0.0, -1.0, 1.0];
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
    fn test_cos_non_contiguous() -> Result<()> {
        let data = vec![
            0.0f32,
            std::f32::consts::PI / 2.0,
            std::f32::consts::PI,
            0.0,
        ];
        let tensor = Tensor::from_vec(data, [2, 2])?;

        // Test non-contiguous path directly by calling cos_non_contiguous method
        let result = tensor.cos_non_contiguous()?;

        // Check results
        let expected = [1.0, 0.0, -1.0, 1.0]; // cos(0) = 1, cos(π/2) = 0, cos(π) = -1, cos(0) = 1
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
