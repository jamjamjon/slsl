use anyhow::Result;

use crate::{global_backend, DType, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    #[inline(always)]
    pub fn floor(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            self.floor_contiguous()
        } else {
            self.floor_non_contiguous()
        }
    }

    #[inline(always)]
    fn floor_contiguous(&self) -> Result<Tensor> {
        let numel = self.shape.numel();
        let backend = global_backend();

        match self.dtype {
            DType::Fp32 => {
                let input_data = self.as_slice::<f32>()?;
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    backend.v_floor_f32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let input_data = self.as_slice::<f64>()?;
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    backend.v_floor_f64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let input_data = self.as_slice::<half::f16>()?;
                let out = UninitVec::<half::f16>::new(numel).init_with(|dst| {
                    backend.v_floor_f16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let input_data = self.as_slice::<half::bf16>()?;
                let out = UninitVec::<half::bf16>::new(numel).init_with(|dst| {
                    backend.v_floor_bf16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Unsupported dtype for floor operation: {:?}", self.dtype),
        }
    }

    #[inline(always)]
    fn floor_non_contiguous(&self) -> Result<Tensor> {
        match self.dtype {
            DType::Fp32 => self.map_non_contiguous::<f32>(|x| x.floor()),
            DType::Fp64 => self.map_non_contiguous::<f64>(|x| x.floor()),
            DType::Fp16 => {
                self.map_non_contiguous::<half::f16>(|x| half::f16::from_f32(x.to_f32().floor()))
            }
            DType::Bf16 => {
                self.map_non_contiguous::<half::bf16>(|x| half::bf16::from_f32(x.to_f32().floor()))
            }
            _ => anyhow::bail!("Unsupported dtype for floor operation: {:?}", self.dtype),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_floor_f32() -> Result<()> {
        let data = vec![1.7f32, -1.7, 2.0, -2.0, 0.0];
        let tensor = Tensor::from_vec(data, [5])?;
        let result = tensor.floor()?;

        let expected = [1.0, -2.0, 2.0, -2.0, 0.0]; // floor(1.7) = 1, floor(-1.7) = -2, floor(2.0) = 2, floor(-2.0) = -2, floor(0.0) = 0
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
    fn test_floor_f64() -> Result<()> {
        let data = vec![1.7f64, -1.7, 2.0, -2.0, 0.0];
        let tensor = Tensor::from_vec(data, [5])?;
        let result = tensor.floor()?;

        let expected = [1.0, -2.0, 2.0, -2.0, 0.0];
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
    fn test_floor_2d() -> Result<()> {
        let data = vec![1.7f32, -1.7, 2.0, -2.0];
        let tensor = Tensor::from_vec(data, [2, 2])?;
        let result = tensor.floor()?;

        let expected = [1.0, -2.0, 2.0, -2.0];
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
    fn test_floor_non_contiguous() -> Result<()> {
        let data = vec![1.7f32, -1.7, 2.0, -2.0];
        let tensor = Tensor::from_vec(data, [2, 2])?;

        // Test non-contiguous path directly by calling floor_non_contiguous method
        let result = tensor.floor_non_contiguous()?;

        // Check results
        let expected = [1.0, -2.0, 2.0, -2.0]; // floor(1.7) = 1, floor(-1.7) = -2, floor(2.0) = 2, floor(-2.0) = -2
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
