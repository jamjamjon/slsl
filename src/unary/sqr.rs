use anyhow::Result;

use crate::{global_backend, DType, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    #[inline(always)]
    pub fn sqr(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            self.sqr_contiguous()
        } else {
            self.sqr_non_contiguous()
        }
    }

    #[inline(always)]
    fn sqr_contiguous(&self) -> Result<Tensor> {
        let numel = self.shape.numel();
        let backend = global_backend();

        match self.dtype {
            DType::Fp32 => {
                let input_data = self.as_slice::<f32>()?;
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    backend.v_sqr_f32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let input_data = self.as_slice::<f64>()?;
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    backend.v_sqr_f64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let input_data = self.as_slice::<half::f16>()?;
                let out = UninitVec::<half::f16>::new(numel).init_with(|dst| {
                    backend.v_sqr_f16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let input_data = self.as_slice::<half::bf16>()?;
                let out = UninitVec::<half::bf16>::new(numel).init_with(|dst| {
                    backend.v_sqr_bf16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int8 => {
                let input_data = self.as_slice::<i8>()?;
                let out = UninitVec::<i8>::new(numel).init_with(|dst| {
                    backend.v_sqr_i8(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int16 => {
                let input_data = self.as_slice::<i16>()?;
                let out = UninitVec::<i16>::new(numel).init_with(|dst| {
                    backend.v_sqr_i16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int32 => {
                let input_data = self.as_slice::<i32>()?;
                let out = UninitVec::<i32>::new(numel).init_with(|dst| {
                    backend.v_sqr_i32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int64 => {
                let input_data = self.as_slice::<i64>()?;
                let out = UninitVec::<i64>::new(numel).init_with(|dst| {
                    backend.v_sqr_i64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint8 => {
                let input_data = self.as_slice::<u8>()?;
                let out = UninitVec::<u8>::new(numel).init_with(|dst| {
                    backend.v_sqr_u8(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint16 => {
                let input_data = self.as_slice::<u16>()?;
                let out = UninitVec::<u16>::new(numel).init_with(|dst| {
                    backend.v_sqr_u16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint32 => {
                let input_data = self.as_slice::<u32>()?;
                let out = UninitVec::<u32>::new(numel).init_with(|dst| {
                    backend.v_sqr_u32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint64 => {
                let input_data = self.as_slice::<u64>()?;
                let out = UninitVec::<u64>::new(numel).init_with(|dst| {
                    backend.v_sqr_u64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Unsupported dtype for sqr operation: {:?}", self.dtype),
        }
    }

    #[inline(always)]
    fn sqr_non_contiguous(&self) -> Result<Tensor> {
        match self.dtype {
            DType::Fp32 => self.map_non_contiguous::<f32>(|x| x * x),
            DType::Fp64 => self.map_non_contiguous::<f64>(|x| x * x),
            DType::Fp16 => self.map_non_contiguous::<half::f16>(|x| {
                let val = x.to_f32();
                half::f16::from_f32(val * val)
            }),
            DType::Bf16 => self.map_non_contiguous::<half::bf16>(|x| {
                let val = x.to_f32();
                half::bf16::from_f32(val * val)
            }),
            DType::Int8 => self.map_non_contiguous::<i8>(|x| x * x),
            DType::Int16 => self.map_non_contiguous::<i16>(|x| x * x),
            DType::Int32 => self.map_non_contiguous::<i32>(|x| x * x),
            DType::Int64 => self.map_non_contiguous::<i64>(|x| x * x),
            DType::Uint8 => self.map_non_contiguous::<u8>(|x| x * x),
            DType::Uint16 => self.map_non_contiguous::<u16>(|x| x * x),
            DType::Uint32 => self.map_non_contiguous::<u32>(|x| x * x),
            DType::Uint64 => self.map_non_contiguous::<u64>(|x| x * x),
            _ => anyhow::bail!("Unsupported dtype for sqr operation: {:?}", self.dtype),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_sqr_f32() -> Result<()> {
        let data = vec![0.0f32, 1.0, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, [4])?;
        let result = tensor.sqr()?;

        let expected = [0.0, 1.0, 4.0, 9.0]; // sqr(0) = 0, sqr(1) = 1, sqr(2) = 4, sqr(3) = 9
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
    fn test_sqr_f64() -> Result<()> {
        let data = vec![0.0f64, 1.0, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, [4])?;
        let result = tensor.sqr()?;

        let expected = [0.0, 1.0, 4.0, 9.0];
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
    fn test_sqr_i32() -> Result<()> {
        let data = vec![0i32, 1, 2, 3, -2, -3];
        let tensor = Tensor::from_vec(data, [6])?;
        let result = tensor.sqr()?;

        let expected = [0, 1, 4, 9, 4, 9]; // sqr(-2) = 4, sqr(-3) = 9
        let result_vec = result.to_flat_vec::<i32>()?;

        for (i, (actual, expected)) in result_vec.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                *actual, *expected,
                "Index {i}: expected {expected}, got {actual}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_sqr_2d() -> Result<()> {
        let data = vec![0.0f32, 1.0, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, [2, 2])?;
        let result = tensor.sqr()?;

        let expected = [0.0, 1.0, 4.0, 9.0];
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
    fn test_sqr_non_contiguous() -> Result<()> {
        let data = vec![0.0f32, 1.0, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, [2, 2])?;

        // Test non-contiguous path directly by calling sqr_non_contiguous method
        let result = tensor.sqr_non_contiguous()?;

        // Check results
        let expected = [0.0, 1.0, 4.0, 9.0]; // sqr(0) = 0, sqr(1) = 1, sqr(2) = 4, sqr(3) = 9
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
