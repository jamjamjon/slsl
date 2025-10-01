use anyhow::Result;
use half::{bf16, f16};

use crate::{
    global_backend, reduce_shape_stride, DType, Dim, OpsTrait, Shape, StorageTrait, Stride, Tensor,
    TensorBase, TensorElement, UninitVec,
};

impl<S: StorageTrait> TensorBase<S> {
    pub fn max_all<T: TensorElement + std::cmp::PartialOrd>(&self) -> Result<T> {
        if self.numel() == 0 {
            anyhow::bail!("Cannot find max of empty tensor");
        }

        if self.is_contiguous() {
            let backend = global_backend();

            match self.dtype() {
                DType::Fp32 => {
                    let data = self.as_slice::<f32>()?;
                    let max_val = backend.max_v_f32(data);
                    Ok(T::from_f32(max_val))
                }
                DType::Fp64 => {
                    let data = self.as_slice::<f64>()?;
                    let max_val = backend.max_v_f64(data);
                    Ok(T::from_f64(max_val))
                }
                DType::Bf16 => {
                    let data = self.as_slice::<bf16>()?;
                    let max_val = backend.max_v_bf16(data);
                    Ok(T::from_bf16(max_val))
                }
                DType::Fp16 => {
                    let data = self.as_slice::<f16>()?;
                    let max_val = backend.max_v_f16(data);
                    Ok(T::from_f16(max_val))
                }
                DType::Int8 => {
                    let data = self.as_slice::<i8>()?;
                    let max_val = backend.max_v_i8(data);
                    Ok(T::from_i8(max_val))
                }
                DType::Int16 => {
                    let data = self.as_slice::<i16>()?;
                    let max_val = backend.max_v_i16(data);
                    Ok(T::from_i16(max_val))
                }
                DType::Int32 => {
                    let data = self.as_slice::<i32>()?;
                    let max_val = backend.max_v_i32(data);
                    Ok(T::from_i32(max_val))
                }
                DType::Int64 => {
                    let data = self.as_slice::<i64>()?;
                    let max_val = backend.max_v_i64(data);
                    Ok(T::from_i64(max_val))
                }
                DType::Uint8 => {
                    let data = self.as_slice::<u8>()?;
                    let max_val = backend.max_v_u8(data);
                    Ok(T::from_u8(max_val))
                }
                DType::Uint16 => {
                    let data = self.as_slice::<u16>()?;
                    let max_val = backend.max_v_u16(data);
                    Ok(T::from_u16(max_val))
                }
                DType::Uint32 => {
                    let data = self.as_slice::<u32>()?;
                    let max_val = backend.max_v_u32(data);
                    Ok(T::from_u32(max_val))
                }
                DType::Uint64 => {
                    let data = self.as_slice::<u64>()?;
                    let max_val = backend.max_v_u64(data);
                    Ok(T::from_u64(max_val))
                }
                _ => anyhow::bail!("Max not supported for dtype {:?}", self.dtype()),
            }
        } else {
            let max_val = match self.dtype() {
                DType::Fp32 => self
                    .iter_with_meta::<f32>()
                    .map(|it| *it.value as f64)
                    .fold(f64::NEG_INFINITY, f64::max),
                DType::Fp64 => self
                    .iter_with_meta::<f64>()
                    .map(|it| *it.value)
                    .fold(f64::NEG_INFINITY, f64::max),
                DType::Fp16 => self
                    .iter_with_meta::<f16>()
                    .map(|it| f64::from(*it.value))
                    .fold(f64::NEG_INFINITY, f64::max),
                DType::Bf16 => self
                    .iter_with_meta::<bf16>()
                    .map(|it| f64::from(*it.value))
                    .fold(f64::NEG_INFINITY, f64::max),
                DType::Int8 => self
                    .iter_with_meta::<i8>()
                    .map(|it| *it.value as f64)
                    .fold(f64::NEG_INFINITY, f64::max),
                DType::Int16 => self
                    .iter_with_meta::<i16>()
                    .map(|it| *it.value as f64)
                    .fold(f64::NEG_INFINITY, f64::max),
                DType::Int32 => self
                    .iter_with_meta::<i32>()
                    .map(|it| *it.value as f64)
                    .fold(f64::NEG_INFINITY, f64::max),
                DType::Int64 => self
                    .iter_with_meta::<i64>()
                    .map(|it| *it.value as f64)
                    .fold(f64::NEG_INFINITY, f64::max),
                DType::Uint8 => self
                    .iter_with_meta::<u8>()
                    .map(|it| *it.value as f64)
                    .fold(f64::NEG_INFINITY, f64::max),
                DType::Uint16 => self
                    .iter_with_meta::<u16>()
                    .map(|it| *it.value as f64)
                    .fold(f64::NEG_INFINITY, f64::max),
                DType::Uint32 => self
                    .iter_with_meta::<u32>()
                    .map(|it| *it.value as f64)
                    .fold(f64::NEG_INFINITY, f64::max),
                DType::Uint64 => self
                    .iter_with_meta::<u64>()
                    .map(|it| *it.value as f64)
                    .fold(f64::NEG_INFINITY, f64::max),
                _ => unreachable!(),
            };

            Ok(T::from_f64(max_val))
        }
    }

    #[inline(always)]
    pub fn max<D: Dim + Clone>(&self, dim: D) -> Result<Tensor> {
        self.max_impl(dim, false)
    }

    #[inline(always)]
    pub fn max_keepdim<D: Dim + Clone>(&self, dim: D) -> Result<Tensor> {
        self.max_impl(dim, true)
    }

    #[inline(always)]
    pub fn max_impl<D: Dim + Clone>(&self, dim: D, keepdim: bool) -> Result<Tensor> {
        let dim_index = dim.to_dim(self.rank())?;
        if self.numel() == 0 {
            anyhow::bail!("Cannot find max of empty tensor");
        }
        let (new_shape, new_strides) = reduce_shape_stride(self.shape, &[dim_index], keepdim);
        if self.is_contiguous() && self.can_reduce_over_last_dims(&[dim_index]) {
            self.max_contiguous(dim_index, new_shape)
        } else {
            self.max_non_contiguous(dim_index, new_shape, new_strides)
        }
    }

    #[inline(always)]
    fn max_contiguous(&self, dim_index: usize, new_shape: Shape) -> Result<Tensor> {
        let backend = global_backend();
        let shape = self.shape();
        let reduce_size = shape[dim_index];
        let output_size = self.numel() / reduce_size;
        let stride = reduce_size;

        match self.dtype() {
            DType::Fp32 => {
                let data = self.as_slice::<f32>()?;
                let output = UninitVec::<f32>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.max_v_f32(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Fp64 => {
                let data = self.as_slice::<f64>()?;
                let output = UninitVec::<f64>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.max_v_f64(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Fp16 => {
                let data = self.as_slice::<f16>()?;
                let output = UninitVec::<f16>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.max_v_f16(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Bf16 => {
                let data = self.as_slice::<bf16>()?;
                let output = UninitVec::<bf16>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.max_v_bf16(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Int8 => {
                let data = self.as_slice::<i8>()?;
                let output = UninitVec::<i8>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.max_v_i8(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Int16 => {
                let data = self.as_slice::<i16>()?;
                let output = UninitVec::<i16>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.max_v_i16(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Int32 => {
                let data = self.as_slice::<i32>()?;
                let output = UninitVec::<i32>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.max_v_i32(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Int64 => {
                let data = self.as_slice::<i64>()?;
                let output = UninitVec::<i64>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.max_v_i64(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Uint8 => {
                let data = self.as_slice::<u8>()?;
                let output = UninitVec::<u8>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.max_v_u8(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Uint16 => {
                let data = self.as_slice::<u16>()?;
                let output = UninitVec::<u16>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.max_v_u16(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Uint32 => {
                let data = self.as_slice::<u32>()?;
                let output = UninitVec::<u32>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.max_v_u32(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Uint64 => {
                let data = self.as_slice::<u64>()?;
                let output = UninitVec::<u64>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.max_v_u64(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            _ => anyhow::bail!("Max not supported for dtype {:?}", self.dtype()),
        }
    }

    #[inline(always)]
    fn max_non_contiguous(
        &self,
        dim_index: usize,
        new_shape: Shape,
        _new_strides: Stride,
    ) -> Result<Tensor> {
        let result_size = new_shape.iter().product();
        let keepdim = new_shape.len() == self.rank();
        match self.dtype() {
            DType::Fp32 => {
                let mut result_data = vec![f32::NEG_INFINITY; result_size];
                let mut result_indices = vec![0; new_shape.len()];

                for item in self.iter_with_meta::<f32>() {
                    let i = item.indices;
                    let val = *item.value;
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                result_indices[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            result_indices[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut result_linear_idx = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        result_linear_idx += result_indices[j] * stride;
                        stride *= new_shape[j];
                    }

                    result_data[result_linear_idx] = result_data[result_linear_idx].max(val);
                }
                Tensor::from_vec(result_data, new_shape)
            }
            DType::Fp64 => {
                let mut result_data = vec![f64::NEG_INFINITY; result_size];
                let mut result_indices = vec![0; new_shape.len()];

                for item in self.iter_with_meta::<f64>() {
                    let i = item.indices;
                    let val = *item.value;
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                result_indices[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            result_indices[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut result_linear_idx = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        result_linear_idx += result_indices[j] * stride;
                        stride *= new_shape[j];
                    }

                    result_data[result_linear_idx] = result_data[result_linear_idx].max(val);
                }
                Tensor::from_vec(result_data, new_shape)
            }
            DType::Fp16 => {
                let mut result_data = vec![f16::from_f32(f32::NEG_INFINITY); result_size];
                let mut result_indices = vec![0; new_shape.len()];

                for item in self.iter_with_meta::<f16>() {
                    let i = item.indices;
                    let val = *item.value;
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                result_indices[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            result_indices[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut result_linear_idx = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        result_linear_idx += result_indices[j] * stride;
                        stride *= new_shape[j];
                    }

                    result_data[result_linear_idx] = result_data[result_linear_idx].max(val);
                }
                Tensor::from_vec(result_data, new_shape)
            }
            DType::Bf16 => {
                let mut result_data = vec![bf16::from_f32(f32::NEG_INFINITY); result_size];
                let mut result_indices = vec![0; new_shape.len()];

                for item in self.iter_with_meta::<bf16>() {
                    let i = item.indices;
                    let val = *item.value;
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                result_indices[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            result_indices[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut result_linear_idx = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        result_linear_idx += result_indices[j] * stride;
                        stride *= new_shape[j];
                    }

                    result_data[result_linear_idx] = result_data[result_linear_idx].max(val);
                }
                Tensor::from_vec(result_data, new_shape)
            }
            DType::Int8 => {
                let mut result_data = vec![i8::MIN; result_size];
                let mut result_indices = vec![0; new_shape.len()];

                for item in self.iter_with_meta::<i8>() {
                    let i = item.indices;
                    let val = *item.value;
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                result_indices[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            result_indices[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut result_linear_idx = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        result_linear_idx += result_indices[j] * stride;
                        stride *= new_shape[j];
                    }

                    result_data[result_linear_idx] = result_data[result_linear_idx].max(val);
                }
                Tensor::from_vec(result_data, new_shape)
            }
            DType::Int16 => {
                let mut result_data = vec![i16::MIN; result_size];
                let mut result_indices = vec![0; new_shape.len()];

                for item in self.iter_with_meta::<i16>() {
                    let i = item.indices;
                    let val = *item.value;
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                result_indices[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            result_indices[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut result_linear_idx = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        result_linear_idx += result_indices[j] * stride;
                        stride *= new_shape[j];
                    }

                    result_data[result_linear_idx] = result_data[result_linear_idx].max(val);
                }
                Tensor::from_vec(result_data, new_shape)
            }
            DType::Int32 => {
                let mut result_data = vec![i32::MIN; result_size];
                let mut result_indices = vec![0; new_shape.len()];

                for item in self.iter_with_meta::<i32>() {
                    let i = item.indices;
                    let val = *item.value;
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                result_indices[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            result_indices[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut result_linear_idx = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        result_linear_idx += result_indices[j] * stride;
                        stride *= new_shape[j];
                    }

                    result_data[result_linear_idx] = result_data[result_linear_idx].max(val);
                }
                Tensor::from_vec(result_data, new_shape)
            }
            DType::Int64 => {
                let mut result_data = vec![i64::MIN; result_size];
                let mut result_indices = vec![0; new_shape.len()];

                for item in self.iter_with_meta::<i64>() {
                    let i = item.indices;
                    let val = *item.value;
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                result_indices[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            result_indices[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut result_linear_idx = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        result_linear_idx += result_indices[j] * stride;
                        stride *= new_shape[j];
                    }

                    result_data[result_linear_idx] = result_data[result_linear_idx].max(val);
                }
                Tensor::from_vec(result_data, new_shape)
            }
            DType::Uint8 => {
                let mut result_data = vec![u8::MIN; result_size];
                let mut result_indices = vec![0; new_shape.len()];

                for item in self.iter_with_meta::<u8>() {
                    let i = item.indices;
                    let val = *item.value;
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                result_indices[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            result_indices[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut result_linear_idx = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        result_linear_idx += result_indices[j] * stride;
                        stride *= new_shape[j];
                    }

                    result_data[result_linear_idx] = result_data[result_linear_idx].max(val);
                }
                Tensor::from_vec(result_data, new_shape)
            }
            DType::Uint16 => {
                let mut result_data = vec![u16::MIN; result_size];
                let mut result_indices = vec![0; new_shape.len()];

                for item in self.iter_with_meta::<u16>() {
                    let i = item.indices;
                    let val = *item.value;
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                result_indices[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            result_indices[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut result_linear_idx = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        result_linear_idx += result_indices[j] * stride;
                        stride *= new_shape[j];
                    }

                    result_data[result_linear_idx] = result_data[result_linear_idx].max(val);
                }
                Tensor::from_vec(result_data, new_shape)
            }
            DType::Uint32 => {
                let mut result_data = vec![u32::MIN; result_size];
                let mut result_indices = vec![0; new_shape.len()];

                for item in self.iter_with_meta::<u32>() {
                    let i = item.indices;
                    let val = *item.value;
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                result_indices[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            result_indices[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut result_linear_idx = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        result_linear_idx += result_indices[j] * stride;
                        stride *= new_shape[j];
                    }

                    result_data[result_linear_idx] = result_data[result_linear_idx].max(val);
                }
                Tensor::from_vec(result_data, new_shape)
            }
            DType::Uint64 => {
                let mut result_data = vec![u64::MIN; result_size];
                let mut result_indices = vec![0; new_shape.len()];

                for item in self.iter_with_meta::<u64>() {
                    let i = item.indices;
                    let val = *item.value;
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                result_indices[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            result_indices[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut result_linear_idx = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        result_linear_idx += result_indices[j] * stride;
                        stride *= new_shape[j];
                    }

                    result_data[result_linear_idx] = result_data[result_linear_idx].max(val);
                }
                Tensor::from_vec(result_data, new_shape)
            }
            _ => anyhow::bail!("Max not supported for dtype {:?}", self.dtype()),
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::Tensor;

    #[test]
    fn test_max_1d() {
        // Basic 1D max test
        let tensor = Tensor::from_vec(vec![3.0f32, 1.0, 4.0, 1.0, 5.0], vec![5]).unwrap();
        let result = tensor.max(0).unwrap();
        assert_eq!(result.dims(), &[] as &[usize]);
        assert_eq!(result.to_scalar::<f32>().unwrap(), 5.0);

        // 1D max with keepdim
        let result_keepdim = tensor.max_keepdim(0).unwrap();
        assert_eq!(result_keepdim.dims(), &[1]);
        assert_eq!(result_keepdim.to_vec::<f32>().unwrap(), vec![5.0]);
    }

    #[test]
    fn test_max_1d_edge_cases() {
        // Single element
        let tensor = Tensor::from_vec(vec![42.0f32], vec![1]).unwrap();
        let result = tensor.max(0).unwrap();
        assert_eq!(result.to_scalar::<f32>().unwrap(), 42.0);

        // All same values
        let tensor = Tensor::from_vec(vec![5.0f32; 10], vec![10]).unwrap();
        let result = tensor.max(0).unwrap();
        assert_eq!(result.to_scalar::<f32>().unwrap(), 5.0);

        // Negative values
        let tensor = Tensor::from_vec(vec![-3.0f32, -1.0, -4.0, -1.0, -5.0], vec![5]).unwrap();
        let result = tensor.max(0).unwrap();
        assert_eq!(result.to_scalar::<f32>().unwrap(), -1.0);

        // Mixed positive and negative
        let tensor = Tensor::from_vec(vec![-2.0f32, 3.0, -1.0, 4.0], vec![4]).unwrap();
        let result = tensor.max(0).unwrap();
        assert_eq!(result.to_scalar::<f32>().unwrap(), 4.0);
    }

    #[test]
    fn test_max_2d() {
        // 2D tensor max along different dimensions
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        // Max along dimension 0 (rows)
        let result = tensor.max(0).unwrap();
        assert_eq!(result.dims(), &[3]);
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![4.0, 5.0, 6.0]);

        // Max along dimension 1 (columns)
        let result = tensor.max(1).unwrap();
        assert_eq!(result.dims(), &[2]);
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![3.0, 6.0]);

        // Max with keepdim
        let result_keepdim = tensor.max_keepdim(0).unwrap();
        assert_eq!(result_keepdim.dims(), &[1, 3]);
        assert_eq!(
            result_keepdim.to_vec2::<f32>().unwrap(),
            vec![vec![4.0, 5.0, 6.0]]
        );

        let result_keepdim = tensor.max_keepdim(1).unwrap();
        assert_eq!(result_keepdim.dims(), &[2, 1]);
        assert_eq!(
            result_keepdim.to_vec2::<f32>().unwrap(),
            vec![vec![3.0], vec![6.0]]
        );
    }

    #[test]
    fn test_max_2d_edge_cases() {
        // Single row
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = tensor.max(0).unwrap();
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);

        // Single column
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = tensor.max(1).unwrap();
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);

        // Square matrix with negative values
        let tensor = Tensor::from_vec(vec![-1.0f32, -2.0, -3.0, -4.0], vec![2, 2]).unwrap();
        let result = tensor.max(0).unwrap();
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![-1.0, -2.0]);
        let result = tensor.max(1).unwrap();
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![-1.0, -3.0]);
    }

    #[test]
    fn test_max_3d() {
        // 3D tensor: 2x3x2
        let tensor = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 3, 2],
        )
        .unwrap();

        // Max along dimension 0
        let result = tensor.max(0).unwrap();
        assert_eq!(result.dims(), &[3, 2]);
        let expected = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        assert_eq!(result.to_vec2::<f32>().unwrap(), expected);

        // Max along dimension 1
        let result = tensor.max(1).unwrap();
        assert_eq!(result.dims(), &[2, 2]);
        let expected = vec![vec![5.0, 6.0], vec![11.0, 12.0]];
        assert_eq!(result.to_vec2::<f32>().unwrap(), expected);

        // Max along dimension 2
        let result = tensor.max(2).unwrap();
        assert_eq!(result.dims(), &[2, 3]);
        let expected = vec![vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]];
        assert_eq!(result.to_vec2::<f32>().unwrap(), expected);
    }

    #[test]
    fn test_max_3d_keepdim() {
        let tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2],
        )
        .unwrap();

        // Max along dimension 0 with keepdim
        let result = tensor.max_keepdim(0).unwrap();
        assert_eq!(result.dims(), &[1, 2, 2]);
        let expected = vec![vec![vec![5.0, 6.0], vec![7.0, 8.0]]];
        assert_eq!(result.to_vec3::<f32>().unwrap(), expected);

        // Max along dimension 1 with keepdim
        let result = tensor.max_keepdim(1).unwrap();
        assert_eq!(result.dims(), &[2, 1, 2]);
        let expected = vec![vec![vec![3.0, 4.0]], vec![vec![7.0, 8.0]]];
        assert_eq!(result.to_vec3::<f32>().unwrap(), expected);

        // Max along dimension 2 with keepdim
        let result = tensor.max_keepdim(2).unwrap();
        assert_eq!(result.dims(), &[2, 2, 1]);
        let expected = vec![vec![vec![2.0], vec![4.0]], vec![vec![6.0], vec![8.0]]];
        assert_eq!(result.to_vec3::<f32>().unwrap(), expected);
    }

    #[test]
    fn test_max_non_contiguous() {
        // Create a 2x3 tensor and transpose it to make it non-contiguous
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let transposed = tensor.clone().permute([1, 0]).unwrap();
        // Transposed tensor is now [[1, 4], [2, 5], [3, 6]] with shape [3, 2]

        // Max along dimension 0 of transposed tensor (reduce rows, keep columns)
        let result = transposed.max(0).unwrap();
        assert_eq!(result.dims(), &[2]);
        // Column 0: max(1, 2, 3) = 3, Column 1: max(4, 5, 6) = 6
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![3.0, 6.0]);

        // Max along dimension 1 of transposed tensor (reduce columns, keep rows)
        let result = transposed.max(1).unwrap();
        assert_eq!(result.dims(), &[3]);
        // Row 0: max(1, 4) = 4, Row 1: max(2, 5) = 5, Row 2: max(3, 6) = 6
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_max_non_contiguous_3d() {
        // Create a 2x2x2 tensor and permute it
        let tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2],
        )
        .unwrap();
        // Original: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        let permuted = tensor.clone().permute([2, 0, 1]).unwrap(); // Shape becomes [2, 2, 2]
                                                                   // Permuted: [[[1, 3], [5, 7]], [[2, 4], [6, 8]]]

        // Max along dimension 0 of permuted tensor (reduce first dim)
        let result = permuted.max(0).unwrap();
        assert_eq!(result.dims(), &[2, 2]);
        // Position [0,0]: max(1, 2) = 2, [0,1]: max(3, 4) = 4
        // Position [1,0]: max(5, 6) = 6, [1,1]: max(7, 8) = 8
        let expected = vec![vec![2.0, 4.0], vec![6.0, 8.0]];
        assert_eq!(result.to_vec2::<f32>().unwrap(), expected);

        // Max along dimension 1 of permuted tensor (reduce second dim)
        let result = permuted.max(1).unwrap();
        assert_eq!(result.dims(), &[2, 2]);
        // Slice 0: [[1, 3], [5, 7]] -> max along rows: [max(1,5), max(3,7)] = [5, 7]
        // Slice 1: [[2, 4], [6, 8]] -> max along rows: [max(2,6), max(4,8)] = [6, 8]
        let expected = vec![vec![5.0, 7.0], vec![6.0, 8.0]];
        assert_eq!(result.to_vec2::<f32>().unwrap(), expected);
    }

    #[test]
    fn test_max_different_dtypes() {
        // Test with i32
        let tensor_i32 = Tensor::from_vec(vec![3i32, 1, 4, 1, 5], vec![5]).unwrap();
        let result = tensor_i32.max(0).unwrap();
        assert_eq!(result.to_scalar::<i32>().unwrap(), 5);

        // Test with f64
        let tensor_f64 = Tensor::from_vec(vec![3.0f64, 1.0, 4.0, 1.0, 5.0], vec![5]).unwrap();
        let result = tensor_f64.max(0).unwrap();
        assert_eq!(result.to_scalar::<f64>().unwrap(), 5.0);

        // Test with u32
        let tensor_u32 = Tensor::from_vec(vec![3u32, 1, 4, 1, 5], vec![5]).unwrap();
        let result = tensor_u32.max(0).unwrap();
        assert_eq!(result.to_scalar::<u32>().unwrap(), 5);
    }

    #[test]
    fn test_max_large_tensor() {
        // Test with larger tensor to ensure performance
        let size = 1000;
        let mut data = vec![0.0f32; size];
        for (i, item) in data.iter_mut().enumerate() {
            *item = (i as f32) * 0.1;
        }
        data[500] = 1000.0; // Maximum value

        let tensor = Tensor::from_vec(data, vec![size]).unwrap();
        let result = tensor.max(0).unwrap();
        assert_eq!(result.to_scalar::<f32>().unwrap(), 1000.0);
    }

    #[test]
    fn test_max_special_values() {
        // Test with infinity and NaN
        let tensor =
            Tensor::from_vec(vec![f32::NEG_INFINITY, 1.0, f32::INFINITY, 2.0], vec![4]).unwrap();
        let result = tensor.max(0).unwrap();
        assert_eq!(result.to_scalar::<f32>().unwrap(), f32::INFINITY);

        // Test with NaN (NaN behavior in max operations)
        let tensor = Tensor::from_vec(vec![1.0f32, f32::NAN, 2.0, 3.0], vec![4]).unwrap();
        let result = tensor.max(0).unwrap();
        // NaN comparisons are tricky, but typically NaN should propagate or be handled specially
        let result_val = result.to_scalar::<f32>().unwrap();
        assert!(result_val.is_nan() || result_val == 3.0);
    }

    #[test]
    fn test_max_empty_tensor_error() {
        // Test that empty tensor throws an error
        let tensor = Tensor::from_vec(Vec::<f32>::new(), vec![0]).unwrap();
        let result = tensor.max(0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot find max of empty tensor"));
    }

    #[test]
    fn test_max_vs_min_consistency() {
        // Test that max and min give consistent results on the same data
        let tensor = Tensor::from_vec(vec![1.0f32, 5.0, 2.0, 8.0, 3.0, 7.0], vec![2, 3]).unwrap();

        let max_result = tensor.max(1).unwrap();
        let min_result = tensor.min(1).unwrap();

        assert_eq!(max_result.to_vec::<f32>().unwrap(), vec![5.0, 8.0]);
        assert_eq!(min_result.to_vec::<f32>().unwrap(), vec![1.0, 3.0]);

        // Ensure max >= min for each position
        let max_vals = max_result.to_vec::<f32>().unwrap();
        let min_vals = min_result.to_vec::<f32>().unwrap();
        for (max_val, min_val) in max_vals.iter().zip(min_vals.iter()) {
            assert!(max_val >= min_val);
        }
    }

    #[test]
    fn test_max_all() {
        // Test max_all function
        let tensor = Tensor::from_vec(vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0], vec![2, 3]).unwrap();
        let result: f32 = tensor.max_all().unwrap();
        assert_eq!(result, 9.0);

        // Test with negative values
        let tensor = Tensor::from_vec(vec![-3.0f32, -1.0, -4.0, -1.0, -5.0], vec![5]).unwrap();
        let result: f32 = tensor.max_all().unwrap();
        assert_eq!(result, -1.0);

        // Test with mixed positive and negative
        let tensor = Tensor::from_vec(vec![-2.0f32, 3.0, -1.0, 4.0], vec![4]).unwrap();
        let result: f32 = tensor.max_all().unwrap();
        assert_eq!(result, 4.0);

        // Test with single element
        let tensor = Tensor::from_vec(vec![42.0f32], vec![1]).unwrap();
        let result: f32 = tensor.max_all().unwrap();
        assert_eq!(result, 42.0);

        // Test with all same values
        let tensor = Tensor::from_vec(vec![5.0f32; 10], vec![10]).unwrap();
        let result: f32 = tensor.max_all().unwrap();
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_max_all_different_dtypes() {
        // Test with i32
        let tensor_i32 = Tensor::from_vec(vec![3i32, 1, 4, 1, 5], vec![5]).unwrap();
        let result: i32 = tensor_i32.max_all().unwrap();
        assert_eq!(result, 5);

        // Test with f64
        let tensor_f64 = Tensor::from_vec(vec![3.0f64, 1.0, 4.0, 1.0, 5.0], vec![5]).unwrap();
        let result: f64 = tensor_f64.max_all().unwrap();
        assert_eq!(result, 5.0);

        // Test with u32
        let tensor_u32 = Tensor::from_vec(vec![3u32, 1, 4, 1, 5], vec![5]).unwrap();
        let result: u32 = tensor_u32.max_all().unwrap();
        assert_eq!(result, 5);
    }

    #[test]
    fn test_max_all_special_values() {
        // Test with infinity
        let tensor =
            Tensor::from_vec(vec![f32::NEG_INFINITY, 1.0, f32::INFINITY, 2.0], vec![4]).unwrap();
        let result: f32 = tensor.max_all().unwrap();
        assert_eq!(result, f32::INFINITY);

        // Test with NaN (NaN behavior in max operations)
        let tensor = Tensor::from_vec(vec![1.0f32, f32::NAN, 2.0, 3.0], vec![4]).unwrap();
        let result: f32 = tensor.max_all().unwrap();
        // NaN comparisons are tricky, but typically NaN should propagate or be handled specially
        let result_val = result;
        assert!(result_val.is_nan() || result_val == 3.0);
    }

    #[test]
    fn test_max_all_empty_tensor_error() {
        // Test that empty tensor throws an error
        let tensor = Tensor::from_vec(Vec::<f32>::new(), vec![0]).unwrap();
        let result: Result<f32, anyhow::Error> = tensor.max_all();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot find max of empty tensor"));
    }

    #[test]
    fn test_max_all_large_tensor() {
        // Test with larger tensor to ensure performance
        let size = 1000;
        let mut data = vec![0.0f32; size];
        for (i, item) in data.iter_mut().enumerate() {
            *item = (i as f32) * 0.1;
        }
        data[500] = 1000.0; // Maximum value

        let tensor = Tensor::from_vec(data, vec![size]).unwrap();
        let result: f32 = tensor.max_all().unwrap();
        assert_eq!(result, 1000.0);
    }
}
