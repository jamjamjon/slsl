use anyhow::Result;

use crate::{global_backend, DType, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    #[inline(always)]
    pub fn neg(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            self.neg_contiguous()
        } else {
            self.neg_non_contiguous()
        }
    }

    #[inline(always)]
    fn neg_contiguous(&self) -> Result<Tensor> {
        let numel = self.shape.numel();
        let backend = global_backend();

        match self.dtype {
            DType::Bool | DType::Uint8 | DType::Uint16 | DType::Uint32 | DType::Uint64 => {
                self.clone_or_copy()
            }
            DType::Fp32 => {
                let input_data = self.as_slice::<f32>()?;
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    backend.v_neg_f32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let input_data = self.as_slice::<f64>()?;
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    backend.v_neg_f64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int8 => {
                let input_data = self.as_slice::<i8>()?;
                let out = UninitVec::<i8>::new(numel).init_with(|dst| {
                    backend.v_neg_i8(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int16 => {
                let input_data = self.as_slice::<i16>()?;
                let out = UninitVec::<i16>::new(numel).init_with(|dst| {
                    backend.v_neg_i16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int32 => {
                let input_data = self.as_slice::<i32>()?;
                let out = UninitVec::<i32>::new(numel).init_with(|dst| {
                    backend.v_neg_i32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int64 => {
                let input_data = self.as_slice::<i64>()?;
                let out = UninitVec::<i64>::new(numel).init_with(|dst| {
                    backend.v_neg_i64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let input_data = self.as_slice::<half::f16>()?;
                let out = UninitVec::<half::f16>::new(numel).init_with(|dst| {
                    backend.v_neg_f16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let input_data = self.as_slice::<half::bf16>()?;
                let out = UninitVec::<half::bf16>::new(numel).init_with(|dst| {
                    backend.v_neg_bf16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Unsupported dtype for neg operation: {:?}", self.dtype),
        }
    }

    #[inline(always)]
    fn neg_non_contiguous(&self) -> Result<Tensor> {
        match self.dtype {
            DType::Bool | DType::Uint8 | DType::Uint16 | DType::Uint32 | DType::Uint64 => {
                self.clone_or_copy()
            }
            DType::Fp32 => self.map_non_contiguous::<f32>(|x| -x),
            DType::Fp64 => self.map_non_contiguous::<f64>(|x| -x),
            DType::Int8 => self.map_non_contiguous::<i8>(|x| -x),
            DType::Int16 => self.map_non_contiguous::<i16>(|x| -x),
            DType::Int32 => self.map_non_contiguous::<i32>(|x| -x),
            DType::Int64 => self.map_non_contiguous::<i64>(|x| -x),
            DType::Fp16 => {
                self.map_non_contiguous::<half::f16>(|x| half::f16::from_f32(-x.to_f32()))
            }
            DType::Bf16 => {
                self.map_non_contiguous::<half::bf16>(|x| half::bf16::from_f32(-x.to_f32()))
            }
            _ => anyhow::bail!("Unsupported dtype for neg operation: {:?}", self.dtype),
        }
    }
}

impl<S: StorageTrait> std::ops::Neg for &TensorBase<S> {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        TensorBase::neg(self).expect("Negation failed")
    }
}

impl<S: StorageTrait> std::ops::Neg for TensorBase<S> {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        TensorBase::neg(&self).expect("Negation failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s;

    #[test]
    fn test_neg_f32() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, -4.0], [2, 2])?;
        let negated = tensor.neg()?;

        assert_eq!(negated.at::<f32>([0, 0]), -1.0);
        assert_eq!(negated.at::<f32>([0, 1]), 2.0);
        assert_eq!(negated.at::<f32>([1, 0]), -3.0);
        assert_eq!(negated.at::<f32>([1, 1]), 4.0);

        Ok(())
    }

    #[test]
    fn test_neg_f64() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0f64, -2.0, 3.0], [3])?;
        let negated = tensor.neg()?;

        assert_eq!(negated.at::<f64>([0]), -1.0);
        assert_eq!(negated.at::<f64>([1]), 2.0);
        assert_eq!(negated.at::<f64>([2]), -3.0);

        Ok(())
    }

    #[test]
    fn test_neg_i32() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1i32, -2, 3, -4], [2, 2])?;
        let negated = tensor.neg()?;

        assert_eq!(negated.at::<i32>([0, 0]), -1);
        assert_eq!(negated.at::<i32>([0, 1]), 2);
        assert_eq!(negated.at::<i32>([1, 0]), -3);
        assert_eq!(negated.at::<i32>([1, 1]), 4);

        Ok(())
    }

    #[test]
    fn test_neg_u8() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1u8, 2, 3, 4], [2, 2])?;
        let negated = tensor.neg()?;

        // Unsigned types should return original values when negated
        assert_eq!(negated.at::<u8>([0, 0]), 1);
        assert_eq!(negated.at::<u8>([0, 1]), 2);
        assert_eq!(negated.at::<u8>([1, 0]), 3);
        assert_eq!(negated.at::<u8>([1, 1]), 4);

        Ok(())
    }

    #[test]
    fn test_neg_f16() -> Result<()> {
        let tensor = Tensor::from_vec(
            vec![half::f16::from_f32(1.0), half::f16::from_f32(-2.0)],
            [2],
        )?;
        let negated = tensor.neg()?;

        assert_eq!(negated.at::<half::f16>([0]), half::f16::from_f32(-1.0));
        assert_eq!(negated.at::<half::f16>([1]), half::f16::from_f32(2.0));

        Ok(())
    }

    #[test]
    fn test_neg_bf16() -> Result<()> {
        let tensor = Tensor::from_vec(
            vec![half::bf16::from_f32(1.0), half::bf16::from_f32(-2.0)],
            [2],
        )?;
        let negated = tensor.neg()?;

        assert_eq!(negated.at::<half::bf16>([0]), half::bf16::from_f32(-1.0));
        assert_eq!(negated.at::<half::bf16>([1]), half::bf16::from_f32(2.0));

        Ok(())
    }

    #[test]
    fn test_neg_non_contiguous() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])?;
        // Create a non-contiguous view by selecting the second row
        let sliced = tensor.slice(s![1]); // Select the second row
        let negated = sliced.neg()?;

        assert_eq!(negated.at::<f32>([0]), -4.0);
        assert_eq!(negated.at::<f32>([1]), -5.0);
        assert_eq!(negated.at::<f32>([2]), -6.0);

        Ok(())
    }

    #[test]
    fn test_neg_empty() -> Result<()> {
        let tensor = Tensor::from_vec(vec![0u8; 0], [0])?;
        let negated = tensor.neg()?;

        assert_eq!(negated.numel(), 0);
        assert_eq!(negated.dims(), [0]);

        Ok(())
    }
}
