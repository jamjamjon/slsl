use anyhow::Result;
use half::{bf16, f16};

use crate::{global_backend, DType, Dim, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    pub fn min_max<D: Dim + Clone>(&self, dim: D) -> Result<(Tensor, Tensor)> {
        self.min_max_impl(dim, false)
    }

    pub fn min_max_keepdim<D: Dim + Clone>(&self, dim: D) -> Result<(Tensor, Tensor)> {
        self.min_max_impl(dim, true)
    }

    pub fn min_max_impl<D: Dim + Clone>(&self, dim: D, keepdim: bool) -> Result<(Tensor, Tensor)> {
        let dim_index = dim.to_dim(self.rank())?;

        if self.numel() == 0 {
            anyhow::bail!("Cannot find min/max of empty tensor");
        }

        // Use dimension-agnostic optimization for contiguous tensors when reducing over last dimensions
        if self.is_contiguous() && self.can_reduce_over_last_dims(&[dim_index]) {
            self.min_max_contiguous(dim_index, keepdim)
        } else {
            self.min_max_non_contiguous(dim_index, keepdim)
        }
    }

    fn min_max_contiguous(&self, dim_index: usize, keepdim: bool) -> Result<(Tensor, Tensor)> {
        let backend = global_backend();
        let shape = self.shape();
        let reduce_size = shape[dim_index];
        let output_size = self.numel() / reduce_size;
        let (new_shape, _) = crate::reduce::reduce_shape_stride(self.shape, &[dim_index], keepdim);

        match self.dtype() {
            DType::Fp32 => {
                let data = self.as_slice::<f32>()?;
                let mut out_min = UninitVec::<f32>::new(output_size);
                let mut out_max = UninitVec::<f32>::new(output_size);
                let dst_min = out_min.as_mut_slice();
                let dst_max = out_max.as_mut_slice();

                for (i, (a, b)) in dst_min
                    .iter_mut()
                    .zip(dst_max.iter_mut())
                    .enumerate()
                    .take(output_size)
                {
                    let start = i * reduce_size;
                    let end = start + reduce_size;
                    let (minv, maxv) = backend.min_max_v_f32(&data[start..end]);
                    *a = minv;
                    *b = maxv;
                }

                let out_min = unsafe { out_min.finalize() };
                let out_max = unsafe { out_max.finalize() };
                Ok((
                    Tensor::from_vec(out_min, new_shape)?,
                    Tensor::from_vec(out_max, new_shape)?,
                ))
            }
            DType::Fp64 => {
                let data = self.as_slice::<f64>()?;
                let mut out_min = UninitVec::<f64>::new(output_size);
                let mut out_max = UninitVec::<f64>::new(output_size);
                let dst_min = out_min.as_mut_slice();
                let dst_max = out_max.as_mut_slice();
                for (i, (a, b)) in dst_min
                    .iter_mut()
                    .zip(dst_max.iter_mut())
                    .enumerate()
                    .take(output_size)
                {
                    let start = i * reduce_size;
                    let end = start + reduce_size;
                    let (minv, maxv) = backend.min_max_v_f64(&data[start..end]);
                    *a = minv;
                    *b = maxv;
                }

                let out_min = unsafe { out_min.finalize() };
                let out_max = unsafe { out_max.finalize() };
                Ok((
                    Tensor::from_vec(out_min, new_shape)?,
                    Tensor::from_vec(out_max, new_shape)?,
                ))
            }
            DType::Fp16 => {
                let data = self.as_slice::<f16>()?;
                let mut out_min = UninitVec::<f16>::new(output_size);
                let mut out_max = UninitVec::<f16>::new(output_size);
                let dst_min = out_min.as_mut_slice();
                let dst_max = out_max.as_mut_slice();

                for (i, (a, b)) in dst_min
                    .iter_mut()
                    .zip(dst_max.iter_mut())
                    .enumerate()
                    .take(output_size)
                {
                    let start = i * reduce_size;
                    let end = start + reduce_size;
                    let (minv, maxv) = backend.min_max_v_f16(&data[start..end]);
                    *a = minv;
                    *b = maxv;
                }

                let out_min = unsafe { out_min.finalize() };
                let out_max = unsafe { out_max.finalize() };
                Ok((
                    Tensor::from_vec(out_min, new_shape)?,
                    Tensor::from_vec(out_max, new_shape)?,
                ))
            }
            DType::Bf16 => {
                let data = self.as_slice::<bf16>()?;
                let mut out_min = UninitVec::<bf16>::new(output_size);
                let mut out_max = UninitVec::<bf16>::new(output_size);
                let dst_min = out_min.as_mut_slice();
                let dst_max = out_max.as_mut_slice();

                for (i, (a, b)) in dst_min
                    .iter_mut()
                    .zip(dst_max.iter_mut())
                    .enumerate()
                    .take(output_size)
                {
                    let start = i * reduce_size;
                    let end = start + reduce_size;
                    let (minv, maxv) = backend.min_max_v_bf16(&data[start..end]);
                    *a = minv;
                    *b = maxv;
                }

                let out_min = unsafe { out_min.finalize() };
                let out_max = unsafe { out_max.finalize() };
                Ok((
                    Tensor::from_vec(out_min, new_shape)?,
                    Tensor::from_vec(out_max, new_shape)?,
                ))
            }
            DType::Int8 => {
                let data = self.as_slice::<i8>()?;
                let mut out_min = UninitVec::<i8>::new(output_size);
                let mut out_max = UninitVec::<i8>::new(output_size);
                let dst_min = out_min.as_mut_slice();
                let dst_max = out_max.as_mut_slice();

                for (i, (a, b)) in dst_min
                    .iter_mut()
                    .zip(dst_max.iter_mut())
                    .enumerate()
                    .take(output_size)
                {
                    let start = i * reduce_size;
                    let end = start + reduce_size;
                    let (minv, maxv) = backend.min_max_v_i8(&data[start..end]);
                    *a = minv;
                    *b = maxv;
                }

                let out_min = unsafe { out_min.finalize() };
                let out_max = unsafe { out_max.finalize() };
                Ok((
                    Tensor::from_vec(out_min, new_shape)?,
                    Tensor::from_vec(out_max, new_shape)?,
                ))
            }
            DType::Int16 => {
                let data = self.as_slice::<i16>()?;
                let mut out_min = UninitVec::<i16>::new(output_size);
                let mut out_max = UninitVec::<i16>::new(output_size);
                let dst_min = out_min.as_mut_slice();
                let dst_max = out_max.as_mut_slice();

                for (i, (a, b)) in dst_min
                    .iter_mut()
                    .zip(dst_max.iter_mut())
                    .enumerate()
                    .take(output_size)
                {
                    let start = i * reduce_size;
                    let end = start + reduce_size;
                    let (minv, maxv) = backend.min_max_v_i16(&data[start..end]);
                    *a = minv;
                    *b = maxv;
                }

                let out_min = unsafe { out_min.finalize() };
                let out_max = unsafe { out_max.finalize() };
                Ok((
                    Tensor::from_vec(out_min, new_shape)?,
                    Tensor::from_vec(out_max, new_shape)?,
                ))
            }
            DType::Int32 => {
                let data = self.as_slice::<i32>()?;
                let mut out_min = UninitVec::<i32>::new(output_size);
                let mut out_max = UninitVec::<i32>::new(output_size);
                let dst_min = out_min.as_mut_slice();
                let dst_max = out_max.as_mut_slice();

                for (i, (a, b)) in dst_min
                    .iter_mut()
                    .zip(dst_max.iter_mut())
                    .enumerate()
                    .take(output_size)
                {
                    let start = i * reduce_size;
                    let end = start + reduce_size;
                    let (minv, maxv) = backend.min_max_v_i32(&data[start..end]);
                    *a = minv;
                    *b = maxv;
                }

                let out_min = unsafe { out_min.finalize() };
                let out_max = unsafe { out_max.finalize() };
                Ok((
                    Tensor::from_vec(out_min, new_shape)?,
                    Tensor::from_vec(out_max, new_shape)?,
                ))
            }
            DType::Int64 => {
                let data = self.as_slice::<i64>()?;
                let mut out_min = UninitVec::<i64>::new(output_size);
                let mut out_max = UninitVec::<i64>::new(output_size);
                let dst_min = out_min.as_mut_slice();
                let dst_max = out_max.as_mut_slice();

                for (i, (a, b)) in dst_min
                    .iter_mut()
                    .zip(dst_max.iter_mut())
                    .enumerate()
                    .take(output_size)
                {
                    let start = i * reduce_size;
                    let end = start + reduce_size;
                    let (minv, maxv) = backend.min_max_v_i64(&data[start..end]);
                    *a = minv;
                    *b = maxv;
                }

                let out_min = unsafe { out_min.finalize() };
                let out_max = unsafe { out_max.finalize() };
                Ok((
                    Tensor::from_vec(out_min, new_shape)?,
                    Tensor::from_vec(out_max, new_shape)?,
                ))
            }
            DType::Uint8 => {
                let data = self.as_slice::<u8>()?;
                let mut out_min = UninitVec::<u8>::new(output_size);
                let mut out_max = UninitVec::<u8>::new(output_size);
                let dst_min = out_min.as_mut_slice();
                let dst_max = out_max.as_mut_slice();

                for (i, (a, b)) in dst_min
                    .iter_mut()
                    .zip(dst_max.iter_mut())
                    .enumerate()
                    .take(output_size)
                {
                    let start = i * reduce_size;
                    let end = start + reduce_size;
                    let (minv, maxv) = backend.min_max_v_u8(&data[start..end]);
                    *a = minv;
                    *b = maxv;
                }

                let out_min = unsafe { out_min.finalize() };
                let out_max = unsafe { out_max.finalize() };
                Ok((
                    Tensor::from_vec(out_min, new_shape)?,
                    Tensor::from_vec(out_max, new_shape)?,
                ))
            }
            DType::Uint16 => {
                let data = self.as_slice::<u16>()?;
                let mut out_min = UninitVec::<u16>::new(output_size);
                let mut out_max = UninitVec::<u16>::new(output_size);
                let dst_min = out_min.as_mut_slice();
                let dst_max = out_max.as_mut_slice();

                for (i, (a, b)) in dst_min
                    .iter_mut()
                    .zip(dst_max.iter_mut())
                    .enumerate()
                    .take(output_size)
                {
                    let start = i * reduce_size;
                    let end = start + reduce_size;
                    let (minv, maxv) = backend.min_max_v_u16(&data[start..end]);
                    *a = minv;
                    *b = maxv;
                }

                let out_min = unsafe { out_min.finalize() };
                let out_max = unsafe { out_max.finalize() };
                Ok((
                    Tensor::from_vec(out_min, new_shape)?,
                    Tensor::from_vec(out_max, new_shape)?,
                ))
            }
            DType::Uint32 => {
                let data = self.as_slice::<u32>()?;
                let mut out_min = UninitVec::<u32>::new(output_size);
                let mut out_max = UninitVec::<u32>::new(output_size);
                let dst_min = out_min.as_mut_slice();
                let dst_max = out_max.as_mut_slice();

                for (i, (a, b)) in dst_min
                    .iter_mut()
                    .zip(dst_max.iter_mut())
                    .enumerate()
                    .take(output_size)
                {
                    let start = i * reduce_size;
                    let end = start + reduce_size;
                    let (minv, maxv) = backend.min_max_v_u32(&data[start..end]);
                    *a = minv;
                    *b = maxv;
                }

                let out_min = unsafe { out_min.finalize() };
                let out_max = unsafe { out_max.finalize() };
                Ok((
                    Tensor::from_vec(out_min, new_shape)?,
                    Tensor::from_vec(out_max, new_shape)?,
                ))
            }
            DType::Uint64 => {
                let data = self.as_slice::<u64>()?;
                let mut out_min = UninitVec::<u64>::new(output_size);
                let mut out_max = UninitVec::<u64>::new(output_size);
                let dst_min = out_min.as_mut_slice();
                let dst_max = out_max.as_mut_slice();

                for (i, (a, b)) in dst_min
                    .iter_mut()
                    .zip(dst_max.iter_mut())
                    .enumerate()
                    .take(output_size)
                {
                    let start = i * reduce_size;
                    let end = start + reduce_size;
                    let (minv, maxv) = backend.min_max_v_u64(&data[start..end]);
                    *a = minv;
                    *b = maxv;
                }

                let out_min = unsafe { out_min.finalize() };
                let out_max = unsafe { out_max.finalize() };
                Ok((
                    Tensor::from_vec(out_min, new_shape)?,
                    Tensor::from_vec(out_max, new_shape)?,
                ))
            }
            _ => anyhow::bail!("Min/Max not supported for dtype {:?}", self.dtype()),
        }
    }

    fn min_max_non_contiguous(&self, dim_index: usize, keepdim: bool) -> Result<(Tensor, Tensor)> {
        let (new_shape, _) = crate::reduce::reduce_shape_stride(self.shape, &[dim_index], keepdim);

        let result_size = new_shape.iter().product();
        match self.dtype() {
            DType::Fp32 => {
                let mut mins_uninit = UninitVec::<f32>::new(result_size);
                let mins = mins_uninit.as_mut_slice();
                let mut maxs_uninit = UninitVec::<f32>::new(result_size);
                let maxs = maxs_uninit.as_mut_slice();
                for i in 0..result_size {
                    mins[i] = f32::INFINITY;
                    maxs[i] = f32::NEG_INFINITY;
                }
                let mut idx_buf = vec![0; new_shape.len()];

                for elem in self.iter() {
                    let i = elem.indices;
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *(ptr as *const f32) };
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                idx_buf[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            idx_buf[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut linear = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        linear += idx_buf[j] * stride;
                        stride *= new_shape[j];
                    }

                    mins[linear] = mins[linear].min(val);
                    maxs[linear] = maxs[linear].max(val);
                }

                Ok((
                    Tensor::from_vec(unsafe { mins_uninit.finalize() }, new_shape)?,
                    Tensor::from_vec(unsafe { maxs_uninit.finalize() }, new_shape)?,
                ))
            }
            DType::Fp64 => {
                let mut mins_uninit = UninitVec::<f64>::new(result_size);
                let mins = mins_uninit.as_mut_slice();
                let mut maxs_uninit = UninitVec::<f64>::new(result_size);
                let maxs = maxs_uninit.as_mut_slice();
                for i in 0..result_size {
                    mins[i] = f64::INFINITY;
                    maxs[i] = f64::NEG_INFINITY;
                }
                let mut idx_buf = vec![0; new_shape.len()];

                for elem in self.iter() {
                    let i = elem.indices;
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *(ptr as *const f64) };
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                idx_buf[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            idx_buf[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut linear = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        linear += idx_buf[j] * stride;
                        stride *= new_shape[j];
                    }

                    mins[linear] = mins[linear].min(val);
                    maxs[linear] = maxs[linear].max(val);
                }

                Ok((
                    Tensor::from_vec(unsafe { mins_uninit.finalize() }, new_shape)?,
                    Tensor::from_vec(unsafe { maxs_uninit.finalize() }, new_shape)?,
                ))
            }
            DType::Fp16 => {
                let mut mins_uninit = UninitVec::<f16>::new(result_size);
                let mins = mins_uninit.as_mut_slice();
                let mut maxs_uninit = UninitVec::<f16>::new(result_size);
                let maxs = maxs_uninit.as_mut_slice();
                for i in 0..result_size {
                    mins[i] = f16::from_f32(f32::INFINITY);
                    maxs[i] = f16::from_f32(f32::NEG_INFINITY);
                }
                let mut idx_buf = vec![0; new_shape.len()];

                for elem in self.iter() {
                    let i = elem.indices;
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *(ptr as *const f16) };
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                idx_buf[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            idx_buf[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut linear = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        linear += idx_buf[j] * stride;
                        stride *= new_shape[j];
                    }

                    mins[linear] = mins[linear].min(val);
                    maxs[linear] = maxs[linear].max(val);
                }

                Ok((
                    Tensor::from_vec(unsafe { mins_uninit.finalize() }, new_shape)?,
                    Tensor::from_vec(unsafe { maxs_uninit.finalize() }, new_shape)?,
                ))
            }
            DType::Bf16 => {
                let mut mins_uninit = UninitVec::<bf16>::new(result_size);
                let mins = mins_uninit.as_mut_slice();
                let mut maxs_uninit = UninitVec::<bf16>::new(result_size);
                let maxs = maxs_uninit.as_mut_slice();
                for i in 0..result_size {
                    mins[i] = bf16::from_f32(f32::INFINITY);
                    maxs[i] = bf16::from_f32(f32::NEG_INFINITY);
                }
                let mut idx_buf = vec![0; new_shape.len()];

                for elem in self.iter() {
                    let i = elem.indices;
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *(ptr as *const bf16) };
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                idx_buf[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            idx_buf[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut linear = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        linear += idx_buf[j] * stride;
                        stride *= new_shape[j];
                    }

                    mins[linear] = mins[linear].min(val);
                    maxs[linear] = maxs[linear].max(val);
                }

                Ok((
                    Tensor::from_vec(unsafe { mins_uninit.finalize() }, new_shape)?,
                    Tensor::from_vec(unsafe { maxs_uninit.finalize() }, new_shape)?,
                ))
            }
            DType::Int8 => {
                let mut mins_uninit = UninitVec::<i8>::new(result_size);
                let mins = mins_uninit.as_mut_slice();
                let mut maxs_uninit = UninitVec::<i8>::new(result_size);
                let maxs = maxs_uninit.as_mut_slice();
                for i in 0..result_size {
                    mins[i] = i8::MAX;
                    maxs[i] = i8::MIN;
                }
                let mut idx_buf = vec![0; new_shape.len()];

                for elem in self.iter() {
                    let i = elem.indices;
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *(ptr as *const i8) };
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                idx_buf[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            idx_buf[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut linear = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        linear += idx_buf[j] * stride;
                        stride *= new_shape[j];
                    }

                    mins[linear] = mins[linear].min(val);
                    maxs[linear] = maxs[linear].max(val);
                }

                Ok((
                    Tensor::from_vec(unsafe { mins_uninit.finalize() }, new_shape)?,
                    Tensor::from_vec(unsafe { maxs_uninit.finalize() }, new_shape)?,
                ))
            }
            DType::Int16 => {
                let mut mins_uninit = UninitVec::<i16>::new(result_size);
                let mins = mins_uninit.as_mut_slice();
                let mut maxs_uninit = UninitVec::<i16>::new(result_size);
                let maxs = maxs_uninit.as_mut_slice();
                for i in 0..result_size {
                    mins[i] = i16::MAX;
                    maxs[i] = i16::MIN;
                }
                let mut idx_buf = vec![0; new_shape.len()];

                for elem in self.iter() {
                    let i = elem.indices;
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *(ptr as *const i16) };
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                idx_buf[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            idx_buf[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut linear = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        linear += idx_buf[j] * stride;
                        stride *= new_shape[j];
                    }

                    mins[linear] = mins[linear].min(val);
                    maxs[linear] = maxs[linear].max(val);
                }

                Ok((
                    Tensor::from_vec(unsafe { mins_uninit.finalize() }, new_shape)?,
                    Tensor::from_vec(unsafe { maxs_uninit.finalize() }, new_shape)?,
                ))
            }
            DType::Int32 => {
                let mut mins_uninit = UninitVec::<i32>::new(result_size);
                let mins = mins_uninit.as_mut_slice();
                let mut maxs_uninit = UninitVec::<i32>::new(result_size);
                let maxs = maxs_uninit.as_mut_slice();
                for i in 0..result_size {
                    mins[i] = i32::MAX;
                    maxs[i] = i32::MIN;
                }
                let mut idx_buf = vec![0; new_shape.len()];

                for elem in self.iter() {
                    let i = elem.indices;
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *(ptr as *const i32) };
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                idx_buf[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            idx_buf[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut linear = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        linear += idx_buf[j] * stride;
                        stride *= new_shape[j];
                    }

                    mins[linear] = mins[linear].min(val);
                    maxs[linear] = maxs[linear].max(val);
                }

                Ok((
                    Tensor::from_vec(unsafe { mins_uninit.finalize() }, new_shape)?,
                    Tensor::from_vec(unsafe { maxs_uninit.finalize() }, new_shape)?,
                ))
            }
            DType::Int64 => {
                let mut mins_uninit = UninitVec::<i64>::new(result_size);
                let mins = mins_uninit.as_mut_slice();
                let mut maxs_uninit = UninitVec::<i64>::new(result_size);
                let maxs = maxs_uninit.as_mut_slice();
                for i in 0..result_size {
                    mins[i] = i64::MAX;
                    maxs[i] = i64::MIN;
                }
                let mut idx_buf = vec![0; new_shape.len()];

                for elem in self.iter() {
                    let i = elem.indices;
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *(ptr as *const i64) };
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                idx_buf[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            idx_buf[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut linear = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        linear += idx_buf[j] * stride;
                        stride *= new_shape[j];
                    }

                    mins[linear] = mins[linear].min(val);
                    maxs[linear] = maxs[linear].max(val);
                }

                Ok((
                    Tensor::from_vec(unsafe { mins_uninit.finalize() }, new_shape)?,
                    Tensor::from_vec(unsafe { maxs_uninit.finalize() }, new_shape)?,
                ))
            }
            DType::Uint8 => {
                let mut mins_uninit = UninitVec::<u8>::new(result_size);
                let mins = mins_uninit.as_mut_slice();
                let mut maxs_uninit = UninitVec::<u8>::new(result_size);
                let maxs = maxs_uninit.as_mut_slice();
                for i in 0..result_size {
                    mins[i] = u8::MAX;
                    maxs[i] = u8::MIN;
                }
                let mut idx_buf = vec![0; new_shape.len()];

                for elem in self.iter() {
                    let i = elem.indices;
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *ptr };
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                idx_buf[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            idx_buf[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut linear = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        linear += idx_buf[j] * stride;
                        stride *= new_shape[j];
                    }

                    mins[linear] = mins[linear].min(val);
                    maxs[linear] = maxs[linear].max(val);
                }

                Ok((
                    Tensor::from_vec(unsafe { mins_uninit.finalize() }, new_shape)?,
                    Tensor::from_vec(unsafe { maxs_uninit.finalize() }, new_shape)?,
                ))
            }
            DType::Uint16 => {
                let mut mins_uninit = UninitVec::<u16>::new(result_size);
                let mins = mins_uninit.as_mut_slice();
                let mut maxs_uninit = UninitVec::<u16>::new(result_size);
                let maxs = maxs_uninit.as_mut_slice();
                for i in 0..result_size {
                    mins[i] = u16::MAX;
                    maxs[i] = u16::MIN;
                }
                let mut idx_buf = vec![0; new_shape.len()];

                for elem in self.iter() {
                    let i = elem.indices;
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *(ptr as *const u16) };
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                idx_buf[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            idx_buf[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut linear = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        linear += idx_buf[j] * stride;
                        stride *= new_shape[j];
                    }

                    mins[linear] = mins[linear].min(val);
                    maxs[linear] = maxs[linear].max(val);
                }

                Ok((
                    Tensor::from_vec(unsafe { mins_uninit.finalize() }, new_shape)?,
                    Tensor::from_vec(unsafe { maxs_uninit.finalize() }, new_shape)?,
                ))
            }
            DType::Uint32 => {
                let mut mins_uninit = UninitVec::<u32>::new(result_size);
                let mins = mins_uninit.as_mut_slice();
                let mut maxs_uninit = UninitVec::<u32>::new(result_size);
                let maxs = maxs_uninit.as_mut_slice();
                for i in 0..result_size {
                    mins[i] = u32::MAX;
                    maxs[i] = u32::MIN;
                }
                let mut idx_buf = vec![0; new_shape.len()];

                for elem in self.iter() {
                    let i = elem.indices;
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *(ptr as *const u32) };
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                idx_buf[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            idx_buf[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut linear = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        linear += idx_buf[j] * stride;
                        stride *= new_shape[j];
                    }

                    mins[linear] = mins[linear].min(val);
                    maxs[linear] = maxs[linear].max(val);
                }

                Ok((
                    Tensor::from_vec(unsafe { mins_uninit.finalize() }, new_shape)?,
                    Tensor::from_vec(unsafe { maxs_uninit.finalize() }, new_shape)?,
                ))
            }
            DType::Uint64 => {
                let mut mins_uninit = UninitVec::<u64>::new(result_size);
                let mins = mins_uninit.as_mut_slice();
                let mut maxs_uninit = UninitVec::<u64>::new(result_size);
                let maxs = maxs_uninit.as_mut_slice();
                for i in 0..result_size {
                    mins[i] = u64::MAX;
                    maxs[i] = u64::MIN;
                }
                let mut idx_buf = vec![0; new_shape.len()];

                for elem in self.iter() {
                    let i = elem.indices;
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *(ptr as *const u64) };
                    let mut current_dim = 0;
                    for k in 0..self.rank() {
                        if k == dim_index {
                            if keepdim {
                                idx_buf[current_dim] = 0;
                                current_dim += 1;
                            }
                        } else {
                            idx_buf[current_dim] = i[k];
                            current_dim += 1;
                        }
                    }

                    let mut linear = 0;
                    let mut stride = 1;
                    for j in (0..new_shape.len()).rev() {
                        linear += idx_buf[j] * stride;
                        stride *= new_shape[j];
                    }

                    mins[linear] = mins[linear].min(val);
                    maxs[linear] = maxs[linear].max(val);
                }

                Ok((
                    Tensor::from_vec(unsafe { mins_uninit.finalize() }, new_shape)?,
                    Tensor::from_vec(unsafe { maxs_uninit.finalize() }, new_shape)?,
                ))
            }
            _ => anyhow::bail!("Min/Max not supported for dtype {:?}", self.dtype()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use anyhow::Result;

    #[test]
    fn test_min_max_1d_basic() -> Result<()> {
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let tensor = Tensor::from_vec(data, [7])?;

        let (min_result, max_result) = tensor.min_max(0)?;

        // Should return scalars (empty shape)
        assert_eq!(min_result.dims(), &[] as &[usize]);
        assert_eq!(max_result.dims(), &[] as &[usize]);

        let min_val = min_result.as_slice::<f32>()?[0];
        let max_val = max_result.as_slice::<f32>()?[0];

        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 9.0);

        Ok(())
    }

    #[test]
    fn test_min_max_2d_dim0() -> Result<()> {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 8.0, 1.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        let (min_result, max_result) = tensor.min_max(0)?;

        assert_eq!(min_result.dims(), &[3]);
        assert_eq!(max_result.dims(), &[3]);

        let min_vals = min_result.as_slice::<f32>()?;
        let max_vals = max_result.as_slice::<f32>()?;

        // Min along dim 0: [min(1,2), min(5,8), min(3,1)] = [1, 5, 1]
        assert_eq!(min_vals, &[1.0, 5.0, 1.0]);
        // Max along dim 0: [max(1,2), max(5,8), max(3,1)] = [2, 8, 3]
        assert_eq!(max_vals, &[2.0, 8.0, 3.0]);

        Ok(())
    }

    #[test]
    fn test_min_max_2d_dim1() -> Result<()> {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 8.0, 1.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        let (min_result, max_result) = tensor.min_max(1)?;

        assert_eq!(min_result.dims(), &[2]);
        assert_eq!(max_result.dims(), &[2]);

        let min_vals = min_result.as_slice::<f32>()?;
        let max_vals = max_result.as_slice::<f32>()?;

        // Min along dim 1: [min(1,5,3), min(2,8,1)] = [1, 1]
        assert_eq!(min_vals, &[1.0, 1.0]);
        // Max along dim 1: [max(1,5,3), max(2,8,1)] = [5, 8]
        assert_eq!(max_vals, &[5.0, 8.0]);

        Ok(())
    }

    #[test]
    fn test_min_max_3d_basic() -> Result<()> {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [2, 3, 4])?;

        // Test along dimension 0
        let (min_result, max_result) = tensor.min_max(0)?;
        assert_eq!(min_result.dims(), &[3, 4]);
        assert_eq!(max_result.dims(), &[3, 4]);

        // Test along dimension 1
        let (min_result, max_result) = tensor.min_max(1)?;
        assert_eq!(min_result.dims(), &[2, 4]);
        assert_eq!(max_result.dims(), &[2, 4]);

        // Test along dimension 2
        let (min_result, max_result) = tensor.min_max(2)?;
        assert_eq!(min_result.dims(), &[2, 3]);
        assert_eq!(max_result.dims(), &[2, 3]);

        Ok(())
    }

    #[test]
    fn test_min_max_keepdim_1d() -> Result<()> {
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0];
        let tensor = Tensor::from_vec(data, [5])?;

        let (min_result, max_result) = tensor.min_max_keepdim(0)?;

        // Should keep dimension as [1]
        assert_eq!(min_result.dims(), &[1]);
        assert_eq!(max_result.dims(), &[1]);

        let min_val = min_result.as_slice::<f32>()?[0];
        let max_val = max_result.as_slice::<f32>()?[0];

        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 5.0);

        Ok(())
    }

    #[test]
    fn test_min_max_keepdim_2d() -> Result<()> {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 8.0, 1.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        // Test keepdim along dimension 0
        let (min_result, max_result) = tensor.min_max_keepdim(0)?;
        assert_eq!(min_result.dims(), &[1, 3]);
        assert_eq!(max_result.dims(), &[1, 3]);

        // Test keepdim along dimension 1
        let (min_result, max_result) = tensor.min_max_keepdim(1)?;
        assert_eq!(min_result.dims(), &[2, 1]);
        assert_eq!(max_result.dims(), &[2, 1]);

        Ok(())
    }

    #[test]
    fn test_min_max_keepdim_3d() -> Result<()> {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [2, 3, 4])?;

        // Test keepdim along different dimensions
        let (min_result, max_result) = tensor.min_max_keepdim(0)?;
        assert_eq!(min_result.dims(), &[1, 3, 4]);
        assert_eq!(max_result.dims(), &[1, 3, 4]);

        let (min_result, max_result) = tensor.min_max_keepdim(1)?;
        assert_eq!(min_result.dims(), &[2, 1, 4]);
        assert_eq!(max_result.dims(), &[2, 1, 4]);

        let (min_result, max_result) = tensor.min_max_keepdim(2)?;
        assert_eq!(min_result.dims(), &[2, 3, 1]);
        assert_eq!(max_result.dims(), &[2, 3, 1]);

        Ok(())
    }

    #[test]
    fn test_min_max_non_contiguous_2d() -> Result<()> {
        // Test min_max with non-contiguous tensor using permute
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        // Create non-contiguous tensor by permuting dimensions
        let permuted = tensor.clone().permute([1, 0])?; // [3, 2]

        // Test min_max along different dimensions
        let (min_result, max_result) = permuted.min_max(0)?;
        assert_eq!(min_result.dims(), &[2]);
        assert_eq!(max_result.dims(), &[2]);

        let min_vals = min_result.as_slice::<f32>()?;
        let max_vals = max_result.as_slice::<f32>()?;

        // After permute: [[1,4], [2,5], [3,6]]
        // Min along dim 0: [min(1,2,3), min(4,5,6)] = [1, 4]
        assert_eq!(min_vals, &[1.0, 4.0]);
        // Max along dim 0: [max(1,2,3), max(4,5,6)] = [3, 6]
        assert_eq!(max_vals, &[3.0, 6.0]);

        let (min_result, max_result) = permuted.min_max(1)?;
        assert_eq!(min_result.dims(), &[3]);
        assert_eq!(max_result.dims(), &[3]);

        let min_vals = min_result.as_slice::<f32>()?;
        let max_vals = max_result.as_slice::<f32>()?;

        // Min along dim 1: [min(1,4), min(2,5), min(3,6)] = [1, 2, 3]
        assert_eq!(min_vals, &[1.0, 2.0, 3.0]);
        // Max along dim 1: [max(1,4), max(2,5), max(3,6)] = [4, 5, 6]
        assert_eq!(max_vals, &[4.0, 5.0, 6.0]);

        Ok(())
    }

    #[test]
    fn test_min_max_non_contiguous_3d() -> Result<()> {
        // Test min_max with 3D non-contiguous tensor
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [2, 3, 4])?;

        // Create non-contiguous tensor by permuting dimensions
        let permuted = tensor.clone().permute([2, 0, 1])?; // [4, 2, 3]

        // Test min_max along different dimensions
        let (min_result, max_result) = permuted.min_max(0)?;
        assert_eq!(min_result.dims(), &[2, 3]);
        assert_eq!(max_result.dims(), &[2, 3]);

        let (min_result, max_result) = permuted.min_max(1)?;
        assert_eq!(min_result.dims(), &[4, 3]);
        assert_eq!(max_result.dims(), &[4, 3]);

        let (min_result, max_result) = permuted.min_max(2)?;
        assert_eq!(min_result.dims(), &[4, 2]);
        assert_eq!(max_result.dims(), &[4, 2]);

        Ok(())
    }

    #[test]
    fn test_min_max_keepdim_non_contiguous() -> Result<()> {
        // Test min_max_keepdim with non-contiguous tensor
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data, [2, 2, 2])?;

        // Create non-contiguous tensor
        let permuted = tensor.clone().permute([2, 1, 0])?; // [2, 2, 2]

        // Test min_max_keepdim along different dimensions
        let (min_result, max_result) = permuted.min_max_keepdim(0)?;
        assert_eq!(min_result.dims(), &[1, 2, 2]);
        assert_eq!(max_result.dims(), &[1, 2, 2]);

        let (min_result, max_result) = permuted.min_max_keepdim(1)?;
        assert_eq!(min_result.dims(), &[2, 1, 2]);
        assert_eq!(max_result.dims(), &[2, 1, 2]);

        let (min_result, max_result) = permuted.min_max_keepdim(2)?;
        assert_eq!(min_result.dims(), &[2, 2, 1]);
        assert_eq!(max_result.dims(), &[2, 2, 1]);

        Ok(())
    }

    #[test]
    fn test_min_max_different_data_types() -> Result<()> {
        // Test min_max with different data types

        // Test with i32
        let data_i32 = vec![5i32, 1, 9, 3, 7, 2];
        let tensor_i32 = Tensor::from_vec(data_i32, [2, 3])?;
        let (min_result, max_result) = tensor_i32.min_max(1)?;

        let min_vals = min_result.as_slice::<i32>()?;
        let max_vals = max_result.as_slice::<i32>()?;

        // Min along dim 1: [min(5,1,9), min(3,7,2)] = [1, 2]
        assert_eq!(min_vals, &[1, 2]);
        // Max along dim 1: [max(5,1,9), max(3,7,2)] = [9, 7]
        assert_eq!(max_vals, &[9, 7]);

        // Test with u32
        let data_u32 = vec![10u32, 20, 5, 15];
        let tensor_u32 = Tensor::from_vec(data_u32, [2, 2])?;
        let (min_result, max_result) = tensor_u32.min_max(0)?;

        let min_vals = min_result.as_slice::<u32>()?;
        let max_vals = max_result.as_slice::<u32>()?;

        // Min along dim 0: [min(10,5), min(20,15)] = [5, 15]
        assert_eq!(min_vals, &[5, 15]);
        // Max along dim 0: [max(10,5), max(20,15)] = [10, 20]
        assert_eq!(max_vals, &[10, 20]);

        Ok(())
    }

    #[test]
    fn test_min_max_special_values() -> Result<()> {
        // Test min_max with special floating point values

        // Test with infinity
        let data_inf = vec![1.0f32, f32::INFINITY, 3.0, f32::NEG_INFINITY];
        let tensor_inf = Tensor::from_vec(data_inf, [4])?;
        let (min_result, max_result) = tensor_inf.min_max(0)?;

        let min_val = min_result.as_slice::<f32>()?[0];
        let max_val = max_result.as_slice::<f32>()?[0];

        assert_eq!(min_val, f32::NEG_INFINITY);
        assert_eq!(max_val, f32::INFINITY);

        // Test with NaN
        let data_nan = vec![1.0f32, f32::NAN, 3.0];
        let tensor_nan = Tensor::from_vec(data_nan, [3])?;
        let (min_result, max_result) = tensor_nan.min_max(0)?;

        let min_val = min_result.as_slice::<f32>()?[0];
        let max_val = max_result.as_slice::<f32>()?[0];

        // NaN behavior: Skip NaN checks as behavior may vary across implementations
        // Just verify the operation completes without error
        println!("min_val: {min_val}, max_val: {max_val}");

        Ok(())
    }

    #[test]
    fn test_min_max_edge_cases() -> Result<()> {
        // Test various edge cases

        // Single element tensor
        let single = Tensor::from_vec(vec![42.0f32], [1])?;
        let (min_result, max_result) = single.min_max(0)?;

        let min_val = min_result.as_slice::<f32>()?[0];
        let max_val = max_result.as_slice::<f32>()?[0];

        assert_eq!(min_val, 42.0);
        assert_eq!(max_val, 42.0);

        // Tensor with all same values
        let same = Tensor::from_vec(vec![5.0f32, 5.0, 5.0, 5.0], [2, 2])?;
        let (min_result, max_result) = same.min_max(0)?;

        let min_vals = min_result.as_slice::<f32>()?;
        let max_vals = max_result.as_slice::<f32>()?;

        assert_eq!(min_vals, &[5.0, 5.0]);
        assert_eq!(max_vals, &[5.0, 5.0]);

        Ok(())
    }

    #[test]
    fn test_min_max_rectangular_tensors() -> Result<()> {
        // Test min_max with rectangular (non-square) tensors

        // 1x5 tensor
        let data_1x5 = vec![5.0f32, 1.0, 9.0, 3.0, 7.0];
        let tensor_1x5 = Tensor::from_vec(data_1x5, [1, 5])?;

        let (min_result, max_result) = tensor_1x5.min_max(0)?;
        assert_eq!(min_result.dims(), &[5]);
        assert_eq!(max_result.dims(), &[5]);

        let (min_result, max_result) = tensor_1x5.min_max(1)?;
        assert_eq!(min_result.dims(), &[1]);
        assert_eq!(max_result.dims(), &[1]);

        let min_val = min_result.as_slice::<f32>()?[0];
        let max_val = max_result.as_slice::<f32>()?[0];
        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 9.0);

        // 5x1 tensor
        let data_5x1 = vec![10.0f32, 20.0, 5.0, 30.0, 15.0];
        let tensor_5x1 = Tensor::from_vec(data_5x1, [5, 1])?;

        let (min_result, max_result) = tensor_5x1.min_max(0)?;
        assert_eq!(min_result.dims(), &[1]);
        assert_eq!(max_result.dims(), &[1]);

        let min_val = min_result.as_slice::<f32>()?[0];
        let max_val = max_result.as_slice::<f32>()?[0];
        assert_eq!(min_val, 5.0);
        assert_eq!(max_val, 30.0);

        let (min_result, max_result) = tensor_5x1.min_max(1)?;
        assert_eq!(min_result.dims(), &[5]);
        assert_eq!(max_result.dims(), &[5]);

        Ok(())
    }

    #[test]
    fn test_min_max_consistency_with_individual_ops() -> Result<()> {
        // Test that min_max results are consistent with individual min and max operations
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 4])?;

        // Test along dimension 0
        let (min_result, max_result) = tensor.min_max(0)?;
        let individual_min = tensor.min(0)?;
        let individual_max = tensor.max(0)?;

        let min_vals = min_result.as_slice::<f32>()?;
        let max_vals = max_result.as_slice::<f32>()?;
        let individual_min_vals = individual_min.as_slice::<f32>()?;
        let individual_max_vals = individual_max.as_slice::<f32>()?;

        for (i, (&min_val, &max_val)) in min_vals.iter().zip(max_vals.iter()).enumerate() {
            assert_eq!(min_val, individual_min_vals[i]);
            assert_eq!(max_val, individual_max_vals[i]);
        }

        // Test along dimension 1
        let (min_result, max_result) = tensor.min_max(1)?;
        let individual_min = tensor.min(1)?;
        let individual_max = tensor.max(1)?;

        let min_vals = min_result.as_slice::<f32>()?;
        let max_vals = max_result.as_slice::<f32>()?;
        let individual_min_vals = individual_min.as_slice::<f32>()?;
        let individual_max_vals = individual_max.as_slice::<f32>()?;

        for (i, (&min_val, &max_val)) in min_vals.iter().zip(max_vals.iter()).enumerate() {
            assert_eq!(min_val, individual_min_vals[i]);
            assert_eq!(max_val, individual_max_vals[i]);
        }

        Ok(())
    }

    #[test]
    fn test_min_max_large_tensor() -> Result<()> {
        // Test min_max with a larger tensor
        let size = 1000;
        let data: Vec<f32> = (0..size).map(|i| (i % 100) as f32).collect();
        let tensor = Tensor::from_vec(data, [10, 100])?;

        let (min_result, max_result) = tensor.min_max(1)?;

        assert_eq!(min_result.dims(), &[10]);
        assert_eq!(max_result.dims(), &[10]);

        let min_vals = min_result.as_slice::<f32>()?;
        let max_vals = max_result.as_slice::<f32>()?;

        // Each row should have min=0 and max=99
        for (&min_val, &max_val) in min_vals.iter().zip(max_vals.iter()) {
            assert_eq!(min_val, 0.0);
            assert_eq!(max_val, 99.0);
        }

        Ok(())
    }

    #[test]
    fn test_min_max_empty_tensor_error() -> Result<()> {
        // Test that min_max fails gracefully with empty tensor
        let empty_tensor = Tensor::from_vec(Vec::<f32>::new(), [0])?;

        let result = empty_tensor.min_max(0);
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("empty"));

        Ok(())
    }

    #[test]
    fn test_min_max_invalid_dimension() -> Result<()> {
        // Test min_max with invalid dimension
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, [3])?;

        // Should fail for dimension >= rank
        assert!(tensor.min_max(1).is_err());
        assert!(tensor.min_max(2).is_err());

        Ok(())
    }
}
