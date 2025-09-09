use anyhow::Result;

use crate::{global_backend, DType, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    #[inline(always)]
    pub fn relu(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            self.relu_contiguous()
        } else {
            self.relu_non_contiguous()
        }
    }

    #[inline(always)]
    fn relu_contiguous(&self) -> Result<Tensor> {
        let numel = self.shape.numel();
        let backend = global_backend();

        match self.dtype {
            DType::Bool | DType::Uint8 | DType::Uint16 | DType::Uint32 | DType::Uint64 => {
                self.clone_or_copy()
            }
            DType::Fp32 => {
                let input_data = self.as_slice::<f32>()?;
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    backend.relu_f32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let input_data = self.as_slice::<f64>()?;
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    backend.relu_f64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int8 => {
                let input_data = self.as_slice::<i8>()?;
                let out = UninitVec::<i8>::new(numel).init_with(|dst| {
                    backend.relu_i8(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int16 => {
                let input_data = self.as_slice::<i16>()?;
                let out = UninitVec::<i16>::new(numel).init_with(|dst| {
                    backend.relu_i16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int32 => {
                let input_data = self.as_slice::<i32>()?;
                let out = UninitVec::<i32>::new(numel).init_with(|dst| {
                    backend.relu_i32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int64 => {
                let input_data = self.as_slice::<i64>()?;
                let out = UninitVec::<i64>::new(numel).init_with(|dst| {
                    backend.relu_i64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let input_data = self.as_slice::<half::f16>()?;
                let out = UninitVec::<half::f16>::new(numel).init_with(|dst| {
                    backend.relu_f16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let input_data = self.as_slice::<half::bf16>()?;
                let out = UninitVec::<half::bf16>::new(numel).init_with(|dst| {
                    backend.relu_bf16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Unsupported dtype for relu operation: {:?}", self.dtype),
        }
    }

    #[inline(always)]
    fn relu_non_contiguous(&self) -> Result<Tensor> {
        match self.dtype {
            DType::Bool | DType::Uint8 | DType::Uint16 | DType::Uint32 | DType::Uint64 => {
                self.clone_or_copy()
            }
            DType::Fp32 => self.map_non_contiguous::<f32>(|x| x.max(0.0)),
            DType::Fp64 => self.map_non_contiguous::<f64>(|x| x.max(0.0)),
            DType::Int8 => self.map_non_contiguous::<i8>(|x| *x.max(&0)),
            DType::Int16 => self.map_non_contiguous::<i16>(|x| *x.max(&0)),
            DType::Int32 => self.map_non_contiguous::<i32>(|x| *x.max(&0)),
            DType::Int64 => self.map_non_contiguous::<i64>(|x| *x.max(&0)),
            DType::Fp16 => {
                self.map_non_contiguous::<half::f16>(|x| half::f16::from_f32(x.to_f32().max(0.0)))
            }
            DType::Bf16 => {
                self.map_non_contiguous::<half::bf16>(|x| half::bf16::from_f32(x.to_f32().max(0.0)))
            }
            _ => anyhow::bail!("Unsupported dtype for relu operation: {:?}", self.dtype),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s;

    #[test]
    fn test_relu_f32() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, -4.0], [2, 2])?;
        let relu_output = tensor.relu()?;

        assert_eq!(relu_output.at::<f32>([0, 0]), 1.0);
        assert_eq!(relu_output.at::<f32>([0, 1]), 0.0);
        assert_eq!(relu_output.at::<f32>([1, 0]), 3.0);
        assert_eq!(relu_output.at::<f32>([1, 1]), 0.0);

        Ok(())
    }

    #[test]
    fn test_relu_f64() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0f64, -2.0, 3.0], [3])?;
        let relu_output = tensor.relu()?;

        assert_eq!(relu_output.at::<f64>([0]), 1.0);
        assert_eq!(relu_output.at::<f64>([1]), 0.0);
        assert_eq!(relu_output.at::<f64>([2]), 3.0);

        Ok(())
    }

    #[test]
    fn test_relu_i32() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1i32, -2, 3, -4], [2, 2])?;
        let relu_output = tensor.relu()?;

        assert_eq!(relu_output.at::<i32>([0, 0]), 1);
        assert_eq!(relu_output.at::<i32>([0, 1]), 0);
        assert_eq!(relu_output.at::<i32>([1, 0]), 3);
        assert_eq!(relu_output.at::<i32>([1, 1]), 0);

        Ok(())
    }

    #[test]
    fn test_relu_u8() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1u8, 2, 3, 4], [2, 2])?;
        let relu_output = tensor.relu()?;

        // Unsigned types should return original values for ReLU
        assert_eq!(relu_output.at::<u8>([0, 0]), 1);
        assert_eq!(relu_output.at::<u8>([0, 1]), 2);
        assert_eq!(relu_output.at::<u8>([1, 0]), 3);
        assert_eq!(relu_output.at::<u8>([1, 1]), 4);

        Ok(())
    }

    #[test]
    fn test_relu_f16() -> Result<()> {
        let tensor = Tensor::from_vec(
            vec![half::f16::from_f32(1.0), half::f16::from_f32(-2.0)],
            [2],
        )?;
        let relu_output = tensor.relu()?;

        assert_eq!(relu_output.at::<half::f16>([0]), half::f16::from_f32(1.0));
        assert_eq!(relu_output.at::<half::f16>([1]), half::f16::from_f32(0.0));

        Ok(())
    }

    #[test]
    fn test_relu_bf16() -> Result<()> {
        let tensor = Tensor::from_vec(
            vec![half::bf16::from_f32(1.0), half::bf16::from_f32(-2.0)],
            [2],
        )?;
        let relu_output = tensor.relu()?;

        assert_eq!(relu_output.at::<half::bf16>([0]), half::bf16::from_f32(1.0));
        assert_eq!(relu_output.at::<half::bf16>([1]), half::bf16::from_f32(0.0));

        Ok(())
    }

    #[test]
    fn test_relu_non_contiguous() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, -4.0, 5.0, 6.0], [2, 3])?;
        // Create a non-contiguous view by selecting the second row
        let sliced = tensor.slice(s![1]); // Select the second row
        let relu_output = sliced.relu()?;

        assert_eq!(relu_output.at::<f32>([0]), 0.0); // -4.0 -> 0.0
        assert_eq!(relu_output.at::<f32>([1]), 5.0); // 5.0 -> 5.0
        assert_eq!(relu_output.at::<f32>([2]), 6.0); // 6.0 -> 6.0

        Ok(())
    }

    #[test]
    fn test_relu_empty() -> Result<()> {
        let tensor = Tensor::from_vec(vec![0u8; 0], [0])?;
        let relu_output = tensor.relu()?;

        assert_eq!(relu_output.numel(), 0);
        assert_eq!(relu_output.dims(), [0]);

        Ok(())
    }
}
