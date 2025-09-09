use anyhow::Result;

use crate::{global_backend, DType, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    #[inline(always)]
    pub fn abs(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            self.abs_contiguous()
        } else {
            self.abs_non_contiguous()
        }
    }

    #[inline(always)]
    fn abs_contiguous(&self) -> Result<Tensor> {
        let numel = self.shape.numel();
        let backend = global_backend();

        match self.dtype {
            DType::Fp32 => {
                let input_data = self.as_slice::<f32>()?;
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    backend.v_abs_f32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bool | DType::Uint8 | DType::Uint16 | DType::Uint32 | DType::Uint64 => {
                self.clone_or_copy()
            }
            DType::Fp64 => {
                let input_data = self.as_slice::<f64>()?;
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    backend.v_abs_f64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int8 => {
                let input_data = self.as_slice::<i8>()?;
                let out = UninitVec::<i8>::new(numel).init_with(|dst| {
                    backend.v_abs_i8(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int16 => {
                let input_data = self.as_slice::<i16>()?;
                let out = UninitVec::<i16>::new(numel).init_with(|dst| {
                    backend.v_abs_i16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int32 => {
                let input_data = self.as_slice::<i32>()?;
                let out = UninitVec::<i32>::new(numel).init_with(|dst| {
                    backend.v_abs_i32(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int64 => {
                let input_data = self.as_slice::<i64>()?;
                let out = UninitVec::<i64>::new(numel).init_with(|dst| {
                    backend.v_abs_i64(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let input_data = self.as_slice::<half::f16>()?;
                let out = UninitVec::<half::f16>::new(numel).init_with(|dst| {
                    backend.v_abs_f16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let input_data = self.as_slice::<half::bf16>()?;
                let out = UninitVec::<half::bf16>::new(numel).init_with(|dst| {
                    backend.v_abs_bf16(input_data, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Unsupported dtype for abs operation: {:?}", self.dtype),
        }
    }

    #[inline(always)]
    fn abs_non_contiguous(&self) -> Result<Tensor> {
        match self.dtype {
            DType::Fp32 => self.map_non_contiguous::<f32>(|x| x.abs()),
            DType::Bool | DType::Uint8 | DType::Uint16 | DType::Uint32 | DType::Uint64 => {
                self.clone_or_copy()
            }
            DType::Int8 => self.map_non_contiguous::<i8>(|x| x.abs()),
            DType::Int16 => self.map_non_contiguous::<i16>(|x| x.abs()),
            DType::Int32 => self.map_non_contiguous::<i32>(|x| x.abs()),
            DType::Int64 => self.map_non_contiguous::<i64>(|x| x.abs()),
            DType::Fp64 => self.map_non_contiguous::<f64>(|x| x.abs()),
            DType::Fp16 => {
                self.map_non_contiguous::<half::f16>(|x| half::f16::from_f32(x.to_f32().abs()))
            }
            DType::Bf16 => {
                self.map_non_contiguous::<half::bf16>(|x| half::bf16::from_f32(x.to_f32().abs()))
            }
            _ => anyhow::bail!("Unsupported dtype for abs operation: {:?}", self.dtype),
        }
    }
}
