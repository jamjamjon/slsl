use anyhow::Result;
use half::{bf16, f16};

use crate::{global_backend, DType, Dim, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    pub fn max_argmax<D: Dim + Clone>(&self, dim: D) -> Result<(Tensor, Tensor)> {
        self.max_argmax_impl(dim, false)
    }

    pub fn max_argmax_keepdim<D: Dim + Clone>(&self, dim: D) -> Result<(Tensor, Tensor)> {
        self.max_argmax_impl(dim, true)
    }

    pub fn max_argmax_impl<D: Dim + Clone>(
        &self,
        dim: D,
        keepdim: bool,
    ) -> Result<(Tensor, Tensor)> {
        let dim_index = dim.to_dim(self.rank())?;
        if self.shape()[dim_index] == 0 {
            anyhow::bail!("Cannot find max of dimension with size 0");
        }

        if self.is_contiguous() && self.can_reduce_over_last_dims(&[dim_index]) {
            let backend = global_backend();
            let shape = self.shape();
            let reduce_size = shape[dim_index];
            let output_size = self.numel() / reduce_size;
            let (new_shape, _) =
                crate::reduce::reduce_shape_stride(self.shape, &[dim_index], keepdim);

            match self.dtype() {
                DType::Fp32 => {
                    let data = self.as_slice::<f32>()?;

                    let mut out_vals = UninitVec::<f32>::new(output_size);
                    let mut out_idxs = UninitVec::<u64>::new(output_size);

                    let dst_vals = out_vals.as_mut_slice();
                    let dst_idxs = out_idxs.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (maxv, max_idx) = backend.max_vi_f32(&data[start..end]);
                        dst_vals[i] = maxv;
                        dst_idxs[i] = max_idx;
                    }

                    let out_vals = unsafe { out_vals.finalize() };
                    let out_idxs = unsafe { out_idxs.finalize() };

                    Ok((
                        Tensor::from_vec(out_vals, new_shape)?,
                        Tensor::from_vec(out_idxs, new_shape)?,
                    ))
                }
                DType::Fp64 => {
                    let data = self.as_slice::<f64>()?;
                    let mut out_vals = UninitVec::<f64>::new(output_size);
                    let mut out_idxs = UninitVec::<u64>::new(output_size);

                    let dst_vals = out_vals.as_mut_slice();
                    let dst_idxs = out_idxs.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (maxv, max_idx) = backend.max_vi_f64(&data[start..end]);
                        dst_vals[i] = maxv;
                        dst_idxs[i] = max_idx;
                    }

                    let out_vals = unsafe { out_vals.finalize() };
                    let out_idxs = unsafe { out_idxs.finalize() };

                    Ok((
                        Tensor::from_vec(out_vals, new_shape)?,
                        Tensor::from_vec(out_idxs, new_shape)?,
                    ))
                }
                DType::Fp16 => {
                    let data = self.as_slice::<f16>()?;
                    let mut out_vals = UninitVec::<f16>::new(output_size);
                    let mut out_idxs = UninitVec::<u64>::new(output_size);

                    let dst_vals = out_vals.as_mut_slice();
                    let dst_idxs = out_idxs.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (maxv, max_idx) = backend.max_vi_f16(&data[start..end]);
                        dst_vals[i] = maxv;
                        dst_idxs[i] = max_idx;
                    }

                    let out_vals = unsafe { out_vals.finalize() };
                    let out_idxs = unsafe { out_idxs.finalize() };

                    Ok((
                        Tensor::from_vec(out_vals, new_shape)?,
                        Tensor::from_vec(out_idxs, new_shape)?,
                    ))
                }
                DType::Bf16 => {
                    let data = self.as_slice::<bf16>()?;
                    let mut out_vals = UninitVec::<bf16>::new(output_size);
                    let mut out_idxs = UninitVec::<u64>::new(output_size);

                    let dst_vals = out_vals.as_mut_slice();
                    let dst_idxs = out_idxs.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (maxv, max_idx) = backend.max_vi_bf16(&data[start..end]);
                        dst_vals[i] = maxv;
                        dst_idxs[i] = max_idx;
                    }

                    let out_vals = unsafe { out_vals.finalize() };
                    let out_idxs = unsafe { out_idxs.finalize() };

                    Ok((
                        Tensor::from_vec(out_vals, new_shape)?,
                        Tensor::from_vec(out_idxs, new_shape)?,
                    ))
                }
                DType::Int8 => {
                    let data = self.as_slice::<i8>()?;
                    let mut out_vals = UninitVec::<i8>::new(output_size);
                    let mut out_idxs = UninitVec::<u64>::new(output_size);

                    let dst_vals = out_vals.as_mut_slice();
                    let dst_idxs = out_idxs.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (maxv, max_idx) = backend.max_vi_i8(&data[start..end]);
                        dst_vals[i] = maxv;
                        dst_idxs[i] = max_idx;
                    }

                    let out_vals = unsafe { out_vals.finalize() };
                    let out_idxs = unsafe { out_idxs.finalize() };

                    Ok((
                        Tensor::from_vec(out_vals, new_shape)?,
                        Tensor::from_vec(out_idxs, new_shape)?,
                    ))
                }
                DType::Int16 => {
                    let data = self.as_slice::<i16>()?;
                    let mut out_vals = UninitVec::<i16>::new(output_size);
                    let mut out_idxs = UninitVec::<u64>::new(output_size);

                    let dst_vals = out_vals.as_mut_slice();
                    let dst_idxs = out_idxs.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (maxv, max_idx) = backend.max_vi_i16(&data[start..end]);
                        dst_vals[i] = maxv;
                        dst_idxs[i] = max_idx;
                    }

                    let out_vals = unsafe { out_vals.finalize() };
                    let out_idxs = unsafe { out_idxs.finalize() };

                    Ok((
                        Tensor::from_vec(out_vals, new_shape)?,
                        Tensor::from_vec(out_idxs, new_shape)?,
                    ))
                }
                DType::Int32 => {
                    let data = self.as_slice::<i32>()?;
                    let mut out_vals = UninitVec::<i32>::new(output_size);
                    let mut out_idxs = UninitVec::<u64>::new(output_size);

                    let dst_vals = out_vals.as_mut_slice();
                    let dst_idxs = out_idxs.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (maxv, max_idx) = backend.max_vi_i32(&data[start..end]);
                        dst_vals[i] = maxv;
                        dst_idxs[i] = max_idx;
                    }

                    let out_vals = unsafe { out_vals.finalize() };
                    let out_idxs = unsafe { out_idxs.finalize() };

                    Ok((
                        Tensor::from_vec(out_vals, new_shape)?,
                        Tensor::from_vec(out_idxs, new_shape)?,
                    ))
                }
                DType::Int64 => {
                    let data = self.as_slice::<i64>()?;
                    let mut out_vals = UninitVec::<i64>::new(output_size);
                    let mut out_idxs = UninitVec::<u64>::new(output_size);

                    let dst_vals = out_vals.as_mut_slice();
                    let dst_idxs = out_idxs.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (maxv, max_idx) = backend.max_vi_i64(&data[start..end]);
                        dst_vals[i] = maxv;
                        dst_idxs[i] = max_idx;
                    }

                    let out_vals = unsafe { out_vals.finalize() };
                    let out_idxs = unsafe { out_idxs.finalize() };

                    Ok((
                        Tensor::from_vec(out_vals, new_shape)?,
                        Tensor::from_vec(out_idxs, new_shape)?,
                    ))
                }
                DType::Uint8 => {
                    let data = self.as_slice::<u8>()?;
                    let mut out_vals = UninitVec::<u8>::new(output_size);
                    let mut out_idxs = UninitVec::<u64>::new(output_size);

                    let dst_vals = out_vals.as_mut_slice();
                    let dst_idxs = out_idxs.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (maxv, max_idx) = backend.max_vi_u8(&data[start..end]);
                        dst_vals[i] = maxv;
                        dst_idxs[i] = max_idx;
                    }

                    let out_vals = unsafe { out_vals.finalize() };
                    let out_idxs = unsafe { out_idxs.finalize() };

                    Ok((
                        Tensor::from_vec(out_vals, new_shape)?,
                        Tensor::from_vec(out_idxs, new_shape)?,
                    ))
                }
                DType::Uint16 => {
                    let data = self.as_slice::<u16>()?;
                    let mut out_vals = UninitVec::<u16>::new(output_size);
                    let mut out_idxs = UninitVec::<u64>::new(output_size);

                    let dst_vals = out_vals.as_mut_slice();
                    let dst_idxs = out_idxs.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (maxv, max_idx) = backend.max_vi_u16(&data[start..end]);
                        dst_vals[i] = maxv;
                        dst_idxs[i] = max_idx;
                    }

                    let out_vals = unsafe { out_vals.finalize() };
                    let out_idxs = unsafe { out_idxs.finalize() };

                    Ok((
                        Tensor::from_vec(out_vals, new_shape)?,
                        Tensor::from_vec(out_idxs, new_shape)?,
                    ))
                }
                DType::Uint32 => {
                    let data = self.as_slice::<u32>()?;
                    let mut out_vals = UninitVec::<u32>::new(output_size);
                    let mut out_idxs = UninitVec::<u64>::new(output_size);

                    let dst_vals = out_vals.as_mut_slice();
                    let dst_idxs = out_idxs.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (maxv, max_idx) = backend.max_vi_u32(&data[start..end]);
                        dst_vals[i] = maxv;
                        dst_idxs[i] = max_idx;
                    }

                    let out_vals = unsafe { out_vals.finalize() };
                    let out_idxs = unsafe { out_idxs.finalize() };

                    Ok((
                        Tensor::from_vec(out_vals, new_shape)?,
                        Tensor::from_vec(out_idxs, new_shape)?,
                    ))
                }
                DType::Uint64 => {
                    let data = self.as_slice::<u64>()?;
                    let mut out_vals = UninitVec::<u64>::new(output_size);
                    let mut out_idxs = UninitVec::<u64>::new(output_size);

                    let dst_vals = out_vals.as_mut_slice();
                    let dst_idxs = out_idxs.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (maxv, max_idx) = backend.max_vi_u64(&data[start..end]);
                        dst_vals[i] = maxv;
                        dst_idxs[i] = max_idx;
                    }

                    let out_vals = unsafe { out_vals.finalize() };
                    let out_idxs = unsafe { out_idxs.finalize() };

                    Ok((
                        Tensor::from_vec(out_vals, new_shape)?,
                        Tensor::from_vec(out_idxs, new_shape)?,
                    ))
                }
                _ => anyhow::bail!("Max+Argmax not supported for dtype {:?}", self.dtype()),
            }
        } else {
            // Non-contiguous/general implementation: single pass update
            let (new_shape, _) =
                crate::reduce::reduce_shape_stride(self.shape, &[dim_index], keepdim);

            let result_size = new_shape.iter().product();

            match self.dtype() {
                DType::Fp32 => {
                    let mut maxs = vec![f32::NEG_INFINITY; result_size];
                    let mut argmaxs = vec![0u64; result_size];

                    for item in self.iter_with_meta::<f32>() {
                        let idx = item.indices;
                        let val = *item.value;

                        // build output coordinates (skip or set reduced dim to 0)
                        let mut out_coords = Vec::with_capacity(new_shape.len());
                        for j in 0..self.rank() {
                            if j == dim_index {
                                if keepdim {
                                    out_coords.push(0usize);
                                } else {
                                    continue;
                                }
                            } else {
                                out_coords.push(idx[j]);
                            }
                        }

                        // convert multi-dim coords to linear index
                        let mut out_linear = 0usize;
                        for (k, &c) in out_coords.iter().enumerate() {
                            out_linear = out_linear * new_shape[k] + c;
                        }

                        if val > maxs[out_linear] {
                            maxs[out_linear] = val;
                            argmaxs[out_linear] = idx[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(maxs, new_shape)?,
                        Tensor::from_vec(argmaxs, new_shape)?,
                    ))
                }
                DType::Fp64 => {
                    let mut maxs = vec![f64::NEG_INFINITY; result_size];
                    let mut argmaxs = vec![0u64; result_size];

                    for item in self.iter_with_meta::<f64>() {
                        let idx = item.indices;
                        let val = *item.value;

                        let mut out_coords = Vec::with_capacity(new_shape.len());
                        for j in 0..self.rank() {
                            if j == dim_index {
                                if keepdim {
                                    out_coords.push(0usize);
                                } else {
                                    continue;
                                }
                            } else {
                                out_coords.push(idx[j]);
                            }
                        }

                        let mut out_linear = 0usize;
                        for (k, &c) in out_coords.iter().enumerate() {
                            out_linear = out_linear * new_shape[k] + c;
                        }

                        if val > maxs[out_linear] {
                            maxs[out_linear] = val;
                            argmaxs[out_linear] = idx[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(maxs, new_shape)?,
                        Tensor::from_vec(argmaxs, new_shape)?,
                    ))
                }
                DType::Fp16 => {
                    let mut maxs = vec![f16::from_f32(f32::NEG_INFINITY); result_size];
                    let mut argmaxs = vec![0u64; result_size];

                    for item in self.iter_with_meta::<f16>() {
                        let idx = item.indices;
                        let val = *item.value;

                        let mut out_coords = Vec::with_capacity(new_shape.len());
                        for j in 0..self.rank() {
                            if j == dim_index {
                                if keepdim {
                                    out_coords.push(0usize);
                                } else {
                                    continue;
                                }
                            } else {
                                out_coords.push(idx[j]);
                            }
                        }

                        let mut out_linear = 0usize;
                        for (k, &c) in out_coords.iter().enumerate() {
                            out_linear = out_linear * new_shape[k] + c;
                        }

                        if val > maxs[out_linear] {
                            maxs[out_linear] = val;
                            argmaxs[out_linear] = idx[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(maxs, new_shape)?,
                        Tensor::from_vec(argmaxs, new_shape)?,
                    ))
                }
                DType::Bf16 => {
                    let mut maxs = vec![bf16::from_f32(f32::NEG_INFINITY); result_size];
                    let mut argmaxs = vec![0u64; result_size];

                    for item in self.iter_with_meta::<bf16>() {
                        let idx = item.indices;
                        let val = *item.value;

                        let mut out_coords = Vec::with_capacity(new_shape.len());
                        for j in 0..self.rank() {
                            if j == dim_index {
                                if keepdim {
                                    out_coords.push(0usize);
                                } else {
                                    continue;
                                }
                            } else {
                                out_coords.push(idx[j]);
                            }
                        }

                        let mut out_linear = 0usize;
                        for (k, &c) in out_coords.iter().enumerate() {
                            out_linear = out_linear * new_shape[k] + c;
                        }

                        if val > maxs[out_linear] {
                            maxs[out_linear] = val;
                            argmaxs[out_linear] = idx[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(maxs, new_shape)?,
                        Tensor::from_vec(argmaxs, new_shape)?,
                    ))
                }
                DType::Int8 => {
                    let mut maxs = vec![i8::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];

                    for item in self.iter_with_meta::<i8>() {
                        let idx = item.indices;
                        let val = *item.value;

                        let mut out_coords = Vec::with_capacity(new_shape.len());
                        for j in 0..self.rank() {
                            if j == dim_index {
                                if keepdim {
                                    out_coords.push(0usize);
                                } else {
                                    continue;
                                }
                            } else {
                                out_coords.push(idx[j]);
                            }
                        }

                        let mut out_linear = 0usize;
                        for (k, &c) in out_coords.iter().enumerate() {
                            out_linear = out_linear * new_shape[k] + c;
                        }

                        if val > maxs[out_linear] {
                            maxs[out_linear] = val;
                            argmaxs[out_linear] = idx[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(maxs, new_shape)?,
                        Tensor::from_vec(argmaxs, new_shape)?,
                    ))
                }
                DType::Int16 => {
                    let mut maxs = vec![i16::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];

                    for item in self.iter_with_meta::<i16>() {
                        let idx = item.indices;
                        let val = *item.value;

                        let mut out_coords = Vec::with_capacity(new_shape.len());
                        for j in 0..self.rank() {
                            if j == dim_index {
                                if keepdim {
                                    out_coords.push(0usize);
                                } else {
                                    continue;
                                }
                            } else {
                                out_coords.push(idx[j]);
                            }
                        }

                        let mut out_linear = 0usize;
                        for (k, &c) in out_coords.iter().enumerate() {
                            out_linear = out_linear * new_shape[k] + c;
                        }

                        if val > maxs[out_linear] {
                            maxs[out_linear] = val;
                            argmaxs[out_linear] = idx[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(maxs, new_shape)?,
                        Tensor::from_vec(argmaxs, new_shape)?,
                    ))
                }
                DType::Int32 => {
                    let mut maxs = vec![i32::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];

                    for item in self.iter_with_meta::<i32>() {
                        let idx = item.indices;
                        let val = *item.value;

                        let mut out_coords = Vec::with_capacity(new_shape.len());
                        for j in 0..self.rank() {
                            if j == dim_index {
                                if keepdim {
                                    out_coords.push(0usize);
                                } else {
                                    continue;
                                }
                            } else {
                                out_coords.push(idx[j]);
                            }
                        }

                        let mut out_linear = 0usize;
                        for (k, &c) in out_coords.iter().enumerate() {
                            out_linear = out_linear * new_shape[k] + c;
                        }

                        if val > maxs[out_linear] {
                            maxs[out_linear] = val;
                            argmaxs[out_linear] = idx[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(maxs, new_shape)?,
                        Tensor::from_vec(argmaxs, new_shape)?,
                    ))
                }
                DType::Int64 => {
                    let mut maxs = vec![i64::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];

                    for item in self.iter_with_meta::<i64>() {
                        let idx = item.indices;
                        let val = *item.value;

                        let mut out_coords = Vec::with_capacity(new_shape.len());
                        for j in 0..self.rank() {
                            if j == dim_index {
                                if keepdim {
                                    out_coords.push(0usize);
                                } else {
                                    continue;
                                }
                            } else {
                                out_coords.push(idx[j]);
                            }
                        }

                        let mut out_linear = 0usize;
                        for (k, &c) in out_coords.iter().enumerate() {
                            out_linear = out_linear * new_shape[k] + c;
                        }

                        if val > maxs[out_linear] {
                            maxs[out_linear] = val;
                            argmaxs[out_linear] = idx[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(maxs, new_shape)?,
                        Tensor::from_vec(argmaxs, new_shape)?,
                    ))
                }
                DType::Uint8 => {
                    let mut maxs = vec![u8::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];

                    for item in self.iter_with_meta::<u8>() {
                        let idx = item.indices;
                        let val = *item.value;

                        let mut out_coords = Vec::with_capacity(new_shape.len());
                        for j in 0..self.rank() {
                            if j == dim_index {
                                if keepdim {
                                    out_coords.push(0usize);
                                } else {
                                    continue;
                                }
                            } else {
                                out_coords.push(idx[j]);
                            }
                        }

                        let mut out_linear = 0usize;
                        for (k, &c) in out_coords.iter().enumerate() {
                            out_linear = out_linear * new_shape[k] + c;
                        }

                        if val > maxs[out_linear] {
                            maxs[out_linear] = val;
                            argmaxs[out_linear] = idx[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(maxs, new_shape)?,
                        Tensor::from_vec(argmaxs, new_shape)?,
                    ))
                }
                DType::Uint16 => {
                    let mut maxs = vec![u16::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];

                    for item in self.iter_with_meta::<u16>() {
                        let idx = item.indices;
                        let val = *item.value;

                        let mut out_coords = Vec::with_capacity(new_shape.len());
                        for j in 0..self.rank() {
                            if j == dim_index {
                                if keepdim {
                                    out_coords.push(0usize);
                                } else {
                                    continue;
                                }
                            } else {
                                out_coords.push(idx[j]);
                            }
                        }

                        let mut out_linear = 0usize;
                        for (k, &c) in out_coords.iter().enumerate() {
                            out_linear = out_linear * new_shape[k] + c;
                        }

                        if val > maxs[out_linear] {
                            maxs[out_linear] = val;
                            argmaxs[out_linear] = idx[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(maxs, new_shape)?,
                        Tensor::from_vec(argmaxs, new_shape)?,
                    ))
                }
                DType::Uint32 => {
                    let mut maxs = vec![u32::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];

                    for item in self.iter_with_meta::<u32>() {
                        let idx = item.indices;
                        let val = *item.value;

                        let mut out_coords = Vec::with_capacity(new_shape.len());
                        for j in 0..self.rank() {
                            if j == dim_index {
                                if keepdim {
                                    out_coords.push(0usize);
                                } else {
                                    continue;
                                }
                            } else {
                                out_coords.push(idx[j]);
                            }
                        }

                        let mut out_linear = 0usize;
                        for (k, &c) in out_coords.iter().enumerate() {
                            out_linear = out_linear * new_shape[k] + c;
                        }

                        if val > maxs[out_linear] {
                            maxs[out_linear] = val;
                            argmaxs[out_linear] = idx[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(maxs, new_shape)?,
                        Tensor::from_vec(argmaxs, new_shape)?,
                    ))
                }
                DType::Uint64 => {
                    let mut maxs = vec![u64::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];

                    for item in self.iter_with_meta::<u64>() {
                        let idx = item.indices;
                        let val = *item.value;

                        let mut out_coords = Vec::with_capacity(new_shape.len());
                        for j in 0..self.rank() {
                            if j == dim_index {
                                if keepdim {
                                    out_coords.push(0usize);
                                } else {
                                    continue;
                                }
                            } else {
                                out_coords.push(idx[j]);
                            }
                        }

                        let mut out_linear = 0usize;
                        for (k, &c) in out_coords.iter().enumerate() {
                            out_linear = out_linear * new_shape[k] + c;
                        }

                        if val > maxs[out_linear] {
                            maxs[out_linear] = val;
                            argmaxs[out_linear] = idx[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(maxs, new_shape)?,
                        Tensor::from_vec(argmaxs, new_shape)?,
                    ))
                }
                _ => anyhow::bail!("Max+Argmax not supported for dtype {:?}", self.dtype()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use anyhow::Result;

    #[test]
    fn test_max_argmax_1d_basic() -> Result<()> {
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let tensor = Tensor::from_vec(data, [7])?;

        let (max_result, argmax_result) = tensor.max_argmax(0)?;

        // Should return scalars (empty shape)
        assert_eq!(max_result.dims(), &[] as &[usize]);
        assert_eq!(argmax_result.dims(), &[] as &[usize]);

        let max_val = max_result.to_scalar::<f32>()?;
        let argmax_val = argmax_result.to_scalar::<u64>()?;

        // Maximum value is 9.0 at index 5
        assert_eq!(max_val, 9.0);
        assert_eq!(argmax_val, 5);

        Ok(())
    }

    #[test]
    fn test_max_argmax_2d_dim0() -> Result<()> {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 8.0, 1.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        let (max_result, argmax_result) = tensor.max_argmax(0)?;

        assert_eq!(max_result.dims(), &[3]);
        assert_eq!(argmax_result.dims(), &[3]);

        let max_vals = max_result.to_vec::<f32>()?;
        let argmax_vals = argmax_result.to_vec::<u64>()?;

        // Max along dim 0: [max(1,2), max(5,8), max(3,1)] = [2, 8, 3]
        assert_eq!(max_vals, &[2.0, 8.0, 3.0]);
        // Argmax along dim 0: [argmax(1,2), argmax(5,8), argmax(3,1)] = [1, 1, 0]
        assert_eq!(argmax_vals, &[1, 1, 0]);

        Ok(())
    }

    #[test]
    fn test_max_argmax_2d_dim1() -> Result<()> {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 8.0, 1.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        let (max_result, argmax_result) = tensor.max_argmax(1)?;

        assert_eq!(max_result.dims(), &[2]);
        assert_eq!(argmax_result.dims(), &[2]);

        let max_vals = max_result.to_vec::<f32>()?;
        let argmax_vals = argmax_result.to_vec::<u64>()?;

        // Max along dim 1: [max(1,5,3), max(2,8,1)] = [5, 8]
        assert_eq!(max_vals, &[5.0, 8.0]);
        // Argmax along dim 1: [argmax(1,5,3), argmax(2,8,1)] = [1, 1]
        assert_eq!(argmax_vals, &[1, 1]);

        Ok(())
    }

    #[test]
    fn test_max_argmax_3d_basic() -> Result<()> {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [2, 3, 4])?;

        // Test along dimension 0
        let (max_result, argmax_result) = tensor.max_argmax(0)?;
        assert_eq!(max_result.dims(), &[3, 4]);
        assert_eq!(argmax_result.dims(), &[3, 4]);

        // Test along dimension 1
        let (max_result, argmax_result) = tensor.max_argmax(1)?;
        assert_eq!(max_result.dims(), &[2, 4]);
        assert_eq!(argmax_result.dims(), &[2, 4]);

        // Test along dimension 2
        let (max_result, argmax_result) = tensor.max_argmax(2)?;
        assert_eq!(max_result.dims(), &[2, 3]);
        assert_eq!(argmax_result.dims(), &[2, 3]);

        Ok(())
    }

    #[test]
    fn test_max_argmax_keepdim_1d() -> Result<()> {
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0];
        let tensor = Tensor::from_vec(data, [5])?;

        let (max_result, argmax_result) = tensor.max_argmax_keepdim(0)?;

        // Should keep dimension as [1]
        assert_eq!(max_result.dims(), &[1]);
        assert_eq!(argmax_result.dims(), &[1]);

        let max_val = max_result.to_vec::<f32>()?[0];
        let argmax_val = argmax_result.to_vec::<u64>()?[0];

        // Maximum value is 5.0 at index 4
        assert_eq!(max_val, 5.0);
        assert_eq!(argmax_val, 4);

        Ok(())
    }

    #[test]
    fn test_max_argmax_keepdim_2d() -> Result<()> {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 8.0, 1.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        // Test keepdim along dimension 0
        let (max_result, argmax_result) = tensor.max_argmax_keepdim(0)?;
        assert_eq!(max_result.dims(), &[1, 3]);
        assert_eq!(argmax_result.dims(), &[1, 3]);

        // Test keepdim along dimension 1
        let (max_result, argmax_result) = tensor.max_argmax_keepdim(1)?;
        assert_eq!(max_result.dims(), &[2, 1]);
        assert_eq!(argmax_result.dims(), &[2, 1]);

        Ok(())
    }

    #[test]
    fn test_max_argmax_keepdim_3d() -> Result<()> {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [2, 3, 4])?;

        // Test keepdim along different dimensions
        let (max_result, argmax_result) = tensor.max_argmax_keepdim(0)?;
        assert_eq!(max_result.dims(), &[1, 3, 4]);
        assert_eq!(argmax_result.dims(), &[1, 3, 4]);

        let (max_result, argmax_result) = tensor.max_argmax_keepdim(1)?;
        assert_eq!(max_result.dims(), &[2, 1, 4]);
        assert_eq!(argmax_result.dims(), &[2, 1, 4]);

        let (max_result, argmax_result) = tensor.max_argmax_keepdim(2)?;
        assert_eq!(max_result.dims(), &[2, 3, 1]);
        assert_eq!(argmax_result.dims(), &[2, 3, 1]);

        Ok(())
    }

    #[test]
    fn test_max_argmax_non_contiguous_2d() -> Result<()> {
        // Test max_argmax with non-contiguous tensor using permute
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        // Create non-contiguous tensor by permuting dimensions
        let permuted = tensor.clone().permute([1, 0])?; // [3, 2]

        // Test max_argmax along different dimensions
        let (max_result, argmax_result) = permuted.max_argmax(0)?;
        assert_eq!(max_result.dims(), &[2]);
        assert_eq!(argmax_result.dims(), &[2]);

        let max_vals = max_result.to_vec::<f32>()?;
        let argmax_vals = argmax_result.to_vec::<u64>()?;

        // After permute: [[1,4], [2,5], [3,6]]
        // Max along dim 0: [max(1,2,3), max(4,5,6)] = [3, 6]
        assert_eq!(max_vals, &[3.0, 6.0]);
        // Argmax along dim 0: indices of max values (3 is at index 2, 6 is at index 2)
        assert_eq!(argmax_vals, &[2, 2]);

        let (max_result, argmax_result) = permuted.max_argmax(1)?;
        assert_eq!(max_result.dims(), &[3]);
        assert_eq!(argmax_result.dims(), &[3]);

        let max_vals = max_result.to_vec::<f32>()?;
        let argmax_vals = argmax_result.to_vec::<u64>()?;

        // Max along dim 1: [max(1,4), max(2,5), max(3,6)] = [4, 5, 6]
        assert_eq!(max_vals, &[4.0, 5.0, 6.0]);
        // Argmax along dim 1: indices of max values (4 is at index 1, 5 is at index 1, 6 is at index 1)
        assert_eq!(argmax_vals, &[1, 1, 1]);

        Ok(())
    }

    #[test]
    fn test_max_argmax_non_contiguous_3d() -> Result<()> {
        // Test max_argmax with 3D non-contiguous tensor
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [2, 3, 4])?;

        // Create non-contiguous tensor by permuting dimensions
        let permuted = tensor.clone().permute([2, 0, 1])?; // [4, 2, 3]

        // Test max_argmax along different dimensions
        let (max_result, argmax_result) = permuted.max_argmax(0)?;
        assert_eq!(max_result.dims(), &[2, 3]);
        assert_eq!(argmax_result.dims(), &[2, 3]);

        let (max_result, argmax_result) = permuted.max_argmax(1)?;
        assert_eq!(max_result.dims(), &[4, 3]);
        assert_eq!(argmax_result.dims(), &[4, 3]);

        let (max_result, argmax_result) = permuted.max_argmax(2)?;
        assert_eq!(max_result.dims(), &[4, 2]);
        assert_eq!(argmax_result.dims(), &[4, 2]);

        Ok(())
    }

    #[test]
    fn test_max_argmax_keepdim_non_contiguous() -> Result<()> {
        // Test max_argmax_keepdim with non-contiguous tensor
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data, [2, 2, 2])?;

        // Create non-contiguous tensor
        let permuted = tensor.clone().permute([2, 1, 0])?; // [2, 2, 2]

        // Test max_argmax_keepdim along different dimensions
        let (max_result, argmax_result) = permuted.max_argmax_keepdim(0)?;
        assert_eq!(max_result.dims(), &[1, 2, 2]);
        assert_eq!(argmax_result.dims(), &[1, 2, 2]);

        let (max_result, argmax_result) = permuted.max_argmax_keepdim(1)?;
        assert_eq!(max_result.dims(), &[2, 1, 2]);
        assert_eq!(argmax_result.dims(), &[2, 1, 2]);

        let (max_result, argmax_result) = permuted.max_argmax_keepdim(2)?;
        assert_eq!(max_result.dims(), &[2, 2, 1]);
        assert_eq!(argmax_result.dims(), &[2, 2, 1]);

        Ok(())
    }

    #[test]
    fn test_max_argmax_different_data_types() -> Result<()> {
        // Test max_argmax with different data types

        // Test with i32
        let data_i32 = vec![5i32, 1, 9, 3, 7, 2];
        let tensor_i32 = Tensor::from_vec(data_i32, [2, 3])?;
        let (max_result, argmax_result) = tensor_i32.max_argmax(1)?;

        let max_vals = max_result.to_vec::<i32>()?;
        let argmax_vals = argmax_result.to_vec::<u64>()?;

        // Max along dim 1: [max(5,1,9), max(3,7,2)] = [9, 7]
        assert_eq!(max_vals, &[9, 7]);
        // Argmax along dim 1: [argmax(5,1,9), argmax(3,7,2)] = [2, 1]
        assert_eq!(argmax_vals, &[2, 1]);

        // Test with u32
        let data_u32 = vec![10u32, 20, 5, 15];
        let tensor_u32 = Tensor::from_vec(data_u32, [2, 2])?;
        let (max_result, argmax_result) = tensor_u32.max_argmax(0)?;

        let max_vals = max_result.to_vec::<u32>()?;
        let argmax_vals = argmax_result.to_vec::<u64>()?;

        // Max along dim 0: [max(10,5), max(20,15)] = [10, 20]
        assert_eq!(max_vals, &[10, 20]);
        // Argmax along dim 0: [argmax(10,5), argmax(20,15)] = [0, 0]
        assert_eq!(argmax_vals, &[0, 0]);

        Ok(())
    }

    #[test]
    fn test_max_argmax_special_values() -> Result<()> {
        // Test max_argmax with special floating point values

        // Test with infinity
        let data_inf = vec![1.0f32, f32::INFINITY, 3.0, f32::NEG_INFINITY];
        let tensor_inf = Tensor::from_vec(data_inf, [4])?;
        let (max_result, argmax_result) = tensor_inf.max_argmax(0)?;

        let _max_val = max_result.to_scalar::<f32>()?;
        let argmax_val = argmax_result.to_scalar::<u64>()?;

        // INFINITY is the maximum value at index 1
        assert_eq!(_max_val, f32::INFINITY);
        assert_eq!(argmax_val, 1);

        // Test with NaN
        let data_nan = vec![1.0f32, f32::NAN, 3.0];
        let tensor_nan = Tensor::from_vec(data_nan, [3])?;
        let (max_result, argmax_result) = tensor_nan.max_argmax(0)?;

        let _max_val = max_result.to_scalar::<f32>()?;
        let argmax_val = argmax_result.to_scalar::<u64>()?;

        // NaN behavior: typically returns NaN or first non-NaN value
        // The exact behavior depends on the backend implementation
        assert!(argmax_val < 3);

        Ok(())
    }

    #[test]
    fn test_max_argmax_edge_cases() -> Result<()> {
        // Test various edge cases

        // Single element tensor
        let single = Tensor::from_vec(vec![42.0f32], [1])?;
        let (max_result, argmax_result) = single.max_argmax(0)?;

        let max_val = max_result.to_scalar::<f32>()?;
        let argmax_val = argmax_result.to_scalar::<u64>()?;

        assert_eq!(max_val, 42.0);
        assert_eq!(argmax_val, 0);

        // Tensor with all same values
        let same = Tensor::from_vec(vec![5.0f32, 5.0, 5.0, 5.0], [2, 2])?;
        let (max_result, argmax_result) = same.max_argmax(0)?;

        let max_vals = max_result.to_vec::<f32>()?;
        let argmax_vals = argmax_result.to_vec::<u64>()?;

        // Should return first occurrence (index 0) for both positions
        assert_eq!(max_vals, &[5.0, 5.0]);
        assert_eq!(argmax_vals, &[0, 0]);

        Ok(())
    }

    #[test]
    fn test_max_argmax_rectangular_tensors() -> Result<()> {
        // Test max_argmax with rectangular (non-square) tensors

        // 1x5 tensor
        let data_1x5 = vec![5.0f32, 1.0, 9.0, 3.0, 7.0];
        let tensor_1x5 = Tensor::from_vec(data_1x5, [1, 5])?;

        let (max_result, argmax_result) = tensor_1x5.max_argmax(0)?;
        assert_eq!(max_result.dims(), &[5]);
        assert_eq!(argmax_result.dims(), &[5]);

        let (max_result, argmax_result) = tensor_1x5.max_argmax(1)?;
        assert_eq!(max_result.dims(), &[1]);
        assert_eq!(argmax_result.dims(), &[1]);

        let max_val = max_result.to_vec::<f32>()?[0];
        let argmax_val = argmax_result.to_vec::<u64>()?[0];
        assert_eq!(max_val, 9.0); // max value 9.0 at index 2
        assert_eq!(argmax_val, 2);

        // 5x1 tensor
        let data_5x1 = vec![10.0f32, 20.0, 5.0, 30.0, 15.0];
        let tensor_5x1 = Tensor::from_vec(data_5x1, [5, 1])?;

        let (max_result, argmax_result) = tensor_5x1.max_argmax(0)?;
        assert_eq!(max_result.dims(), &[1]);
        assert_eq!(argmax_result.dims(), &[1]);

        let max_val = max_result.to_vec::<f32>()?[0];
        let argmax_val = argmax_result.to_vec::<u64>()?[0];
        assert_eq!(max_val, 30.0); // max value 30.0 at index 3
        assert_eq!(argmax_val, 3);

        let (max_result, argmax_result) = tensor_5x1.max_argmax(1)?;
        assert_eq!(max_result.dims(), &[5]);
        assert_eq!(argmax_result.dims(), &[5]);

        Ok(())
    }

    #[test]
    fn test_max_argmax_consistency_with_individual_ops() -> Result<()> {
        // Test that max_argmax results are consistent with individual max and argmax operations
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 4])?;

        // Test along dimension 0
        let (max_result, argmax_result) = tensor.max_argmax(0)?;
        let individual_max = tensor.max(0)?;
        let individual_argmax = tensor.argmax(0)?;

        let max_vals = max_result.to_vec::<f32>()?;
        let argmax_vals = argmax_result.to_vec::<u64>()?;
        let individual_max_vals = individual_max.to_vec::<f32>()?;
        let individual_argmax_vals = individual_argmax.to_vec::<u64>()?;

        for (i, (&max_val, &argmax_val)) in max_vals.iter().zip(argmax_vals.iter()).enumerate() {
            assert_eq!(max_val, individual_max_vals[i]);
            assert_eq!(argmax_val, individual_argmax_vals[i]);
        }

        // Test along dimension 1
        let (max_result, argmax_result) = tensor.max_argmax(1)?;
        let individual_max = tensor.max(1)?;
        let individual_argmax = tensor.argmax(1)?;

        let max_vals = max_result.to_vec::<f32>()?;
        let argmax_vals = argmax_result.to_vec::<u64>()?;
        let individual_max_vals = individual_max.to_vec::<f32>()?;
        let individual_argmax_vals = individual_argmax.to_vec::<u64>()?;

        for (i, (&max_val, &argmax_val)) in max_vals.iter().zip(argmax_vals.iter()).enumerate() {
            assert_eq!(max_val, individual_max_vals[i]);
            assert_eq!(argmax_val, individual_argmax_vals[i]);
        }

        Ok(())
    }

    #[test]
    fn test_max_argmax_large_tensor() -> Result<()> {
        // Test max_argmax with a larger tensor
        let size = 1000;
        let data: Vec<f32> = (0..size).map(|i| (i % 100) as f32).collect();
        let tensor = Tensor::from_vec(data, [10, 100])?;

        let (max_result, argmax_result) = tensor.max_argmax(1)?;

        assert_eq!(max_result.dims(), &[10]);
        assert_eq!(argmax_result.dims(), &[10]);

        let max_vals = max_result.to_vec::<f32>()?;
        let argmax_vals = argmax_result.to_vec::<u64>()?;

        // Each row should have max=99.0 and argmax=99
        for (&max_val, &argmax_val) in max_vals.iter().zip(argmax_vals.iter()) {
            assert_eq!(max_val, 99.0);
            assert_eq!(argmax_val, 99);
        }

        Ok(())
    }

    #[test]
    fn test_max_argmax_empty_tensor_error() -> Result<()> {
        // Test that max_argmax fails gracefully with empty tensor
        let empty_tensor = Tensor::from_vec(Vec::<f32>::new(), [0])?;

        let result = empty_tensor.max_argmax(0);
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Cannot find max of dimension with size 0"));

        Ok(())
    }

    #[test]
    fn test_max_argmax_invalid_dimension() -> Result<()> {
        // Test max_argmax with invalid dimension
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, [3])?;

        // Should fail for dimension >= rank
        assert!(tensor.max_argmax(1).is_err());
        assert!(tensor.max_argmax(2).is_err());

        Ok(())
    }

    #[test]
    fn test_max_argmax_first_occurrence() -> Result<()> {
        // Test that max_argmax returns the first occurrence of maximum values
        let data = vec![3.0f32, 9.0, 4.0, 9.0, 5.0, 9.0, 2.0];
        let tensor = Tensor::from_vec(data, [7])?;

        let (max_result, argmax_result) = tensor.max_argmax(0)?;

        let max_val = max_result.to_scalar::<f32>()?;
        let argmax_val = argmax_result.to_scalar::<u64>()?;

        // Maximum value is 9.0, first occurrence at index 1
        assert_eq!(max_val, 9.0);
        assert_eq!(argmax_val, 1);

        Ok(())
    }
}
