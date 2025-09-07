use anyhow::Result;

use crate::{
    backend::{global_backend, r#impl::OpsTrait},
    DType, StorageTrait, Tensor, TensorBase, UninitVec,
};

impl<S: StorageTrait> TensorBase<S> {
    #[inline(always)]
    pub fn tan(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            self.tan_contiguous()
        } else {
            self.tan_non_contiguous()
        }
    }

    #[inline(always)]
    fn tan_contiguous(&self) -> Result<Tensor> {
        let numel = self.shape.numel();
        let backend = global_backend();

        match self.dtype {
            DType::Fp32 => {
                let input_data = self.as_slice::<f32>()?;
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    backend.v_tan_f32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let input_data = self.as_slice::<f64>()?;
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    backend.v_tan_f64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let input_data = self.as_slice::<half::f16>()?;
                let out = UninitVec::<half::f16>::new(numel).init_with(|dst| {
                    backend.v_tan_f16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let input_data = self.as_slice::<half::bf16>()?;
                let out = UninitVec::<half::bf16>::new(numel).init_with(|dst| {
                    backend.v_tan_bf16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Unsupported dtype for tan operation: {:?}", self.dtype),
        }
    }

    #[inline(always)]
    fn tan_non_contiguous(&self) -> Result<Tensor> {
        match self.dtype {
            DType::Fp32 => self.map_non_contiguous::<f32>(|x| x.tan()),
            DType::Fp64 => self.map_non_contiguous::<f64>(|x| x.tan()),
            DType::Fp16 => {
                self.map_non_contiguous::<half::f16>(|x| half::f16::from_f32(x.to_f32().tan()))
            }
            DType::Bf16 => {
                self.map_non_contiguous::<half::bf16>(|x| half::bf16::from_f32(x.to_f32().tan()))
            }
            _ => anyhow::bail!("Unsupported dtype for tan operation: {:?}", self.dtype),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_tan_f32() -> Result<()> {
        let data = vec![
            0.0f32,
            std::f32::consts::PI / 4.0,
            std::f32::consts::PI / 6.0,
        ];
        let tensor = Tensor::from_vec(data, [3])?;
        let result = tensor.tan()?;

        let expected = [0.0, 1.0, 0.577_350_26]; // tan(0) = 0, tan(π/4) = 1, tan(π/6) ≈ 0.577
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
    fn test_tan_f64() -> Result<()> {
        let data = vec![
            0.0f64,
            std::f64::consts::PI / 4.0,
            std::f64::consts::PI / 6.0,
        ];
        let tensor = Tensor::from_vec(data, [3])?;
        let result = tensor.tan()?;

        let expected = [0.0, 1.0, 0.5773502691896258];
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
    fn test_tan_2d() -> Result<()> {
        let data = vec![
            0.0f32,
            std::f32::consts::PI / 4.0,
            std::f32::consts::PI / 6.0,
            0.0,
        ];
        let tensor = Tensor::from_vec(data, [2, 2])?;
        let result = tensor.tan()?;

        let expected = [0.0, 1.0, 0.577_350_26, 0.0];
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
    fn test_tan_non_contiguous() -> Result<()> {
        let data = vec![
            0.0f32,
            std::f32::consts::PI / 4.0,
            std::f32::consts::PI / 6.0,
            0.0,
        ];
        let tensor = Tensor::from_vec(data, [2, 2])?;

        // Test non-contiguous path directly by calling tan_non_contiguous method
        let result = tensor.tan_non_contiguous()?;

        // Check results
        let expected = [0.0, 1.0, 0.577_350_26, 0.0]; // tan(0) = 0, tan(π/4) = 1, tan(π/6) ≈ 0.577, tan(0) = 0
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
