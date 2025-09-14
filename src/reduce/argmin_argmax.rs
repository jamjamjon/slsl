use anyhow::Result;
use half::{bf16, f16};

use crate::{global_backend, DType, Dim, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    pub fn argmin_argmax<D: Dim + Clone>(&self, dim: D) -> Result<(Tensor, Tensor)> {
        self.argmin_argmax_impl(dim, false)
    }

    pub fn argmin_argmax_keepdim<D: Dim + Clone>(&self, dim: D) -> Result<(Tensor, Tensor)> {
        self.argmin_argmax_impl(dim, true)
    }

    pub fn argmin_argmax_impl<D: Dim + Clone>(
        &self,
        dim: D,
        keepdim: bool,
    ) -> Result<(Tensor, Tensor)> {
        let dim_index = dim.to_dim(self.rank())?;

        if self.shape()[dim_index] == 0 {
            anyhow::bail!("Cannot find argmin/argmax of dimension with size 0");
        }

        // Use dimension-agnostic optimization when possible
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

                    let mut out_argmin = UninitVec::<u64>::new(output_size);
                    let mut out_argmax = UninitVec::<u64>::new(output_size);

                    let dst_argmin = out_argmin.as_mut_slice();
                    let dst_argmax = out_argmax.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (min_idx, max_idx) = backend.min_max_i_f32(&data[start..end]);

                        dst_argmin[i] = min_idx;
                        dst_argmax[i] = max_idx;
                    }

                    let out_argmin = unsafe { out_argmin.finalize() };
                    let out_argmax = unsafe { out_argmax.finalize() };

                    Ok((
                        Tensor::from_vec(out_argmin, new_shape)?,
                        Tensor::from_vec(out_argmax, new_shape)?,
                    ))
                }
                DType::Fp64 => {
                    let data = self.as_slice::<f64>()?;

                    let mut out_argmin = UninitVec::<u64>::new(output_size);
                    let mut out_argmax = UninitVec::<u64>::new(output_size);

                    let dst_argmin = out_argmin.as_mut_slice();
                    let dst_argmax = out_argmax.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (min_idx, max_idx) = backend.min_max_i_f64(&data[start..end]);

                        dst_argmin[i] = min_idx;
                        dst_argmax[i] = max_idx;
                    }

                    let out_argmin = unsafe { out_argmin.finalize() };
                    let out_argmax = unsafe { out_argmax.finalize() };

                    Ok((
                        Tensor::from_vec(out_argmin, new_shape)?,
                        Tensor::from_vec(out_argmax, new_shape)?,
                    ))
                }
                DType::Fp16 => {
                    let data = self.as_slice::<f16>()?;

                    let mut out_argmin = UninitVec::<u64>::new(output_size);
                    let mut out_argmax = UninitVec::<u64>::new(output_size);

                    let dst_argmin = out_argmin.as_mut_slice();
                    let dst_argmax = out_argmax.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (min_idx, max_idx) = backend.min_max_i_f16(&data[start..end]);

                        dst_argmin[i] = min_idx;
                        dst_argmax[i] = max_idx;
                    }

                    let out_argmin = unsafe { out_argmin.finalize() };
                    let out_argmax = unsafe { out_argmax.finalize() };

                    Ok((
                        Tensor::from_vec(out_argmin, new_shape)?,
                        Tensor::from_vec(out_argmax, new_shape)?,
                    ))
                }
                DType::Bf16 => {
                    let data = self.as_slice::<bf16>()?;

                    let mut out_argmin = UninitVec::<u64>::new(output_size);
                    let mut out_argmax = UninitVec::<u64>::new(output_size);

                    let dst_argmin = out_argmin.as_mut_slice();
                    let dst_argmax = out_argmax.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (min_idx, max_idx) = backend.min_max_i_bf16(&data[start..end]);

                        dst_argmin[i] = min_idx;
                        dst_argmax[i] = max_idx;
                    }

                    let out_argmin = unsafe { out_argmin.finalize() };
                    let out_argmax = unsafe { out_argmax.finalize() };

                    Ok((
                        Tensor::from_vec(out_argmin, new_shape)?,
                        Tensor::from_vec(out_argmax, new_shape)?,
                    ))
                }
                DType::Int8 => {
                    let data = self.as_slice::<i8>()?;

                    let mut out_argmin = UninitVec::<u64>::new(output_size);
                    let mut out_argmax = UninitVec::<u64>::new(output_size);

                    let dst_argmin = out_argmin.as_mut_slice();
                    let dst_argmax = out_argmax.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (min_idx, max_idx) = backend.min_max_i_i8(&data[start..end]);

                        dst_argmin[i] = min_idx;
                        dst_argmax[i] = max_idx;
                    }

                    let out_argmin = unsafe { out_argmin.finalize() };
                    let out_argmax = unsafe { out_argmax.finalize() };

                    Ok((
                        Tensor::from_vec(out_argmin, new_shape)?,
                        Tensor::from_vec(out_argmax, new_shape)?,
                    ))
                }
                DType::Int16 => {
                    let data = self.as_slice::<i16>()?;

                    let mut out_argmin = UninitVec::<u64>::new(output_size);
                    let mut out_argmax = UninitVec::<u64>::new(output_size);

                    let dst_argmin = out_argmin.as_mut_slice();
                    let dst_argmax = out_argmax.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (min_idx, max_idx) = backend.min_max_i_i16(&data[start..end]);

                        dst_argmin[i] = min_idx;
                        dst_argmax[i] = max_idx;
                    }

                    let out_argmin = unsafe { out_argmin.finalize() };
                    let out_argmax = unsafe { out_argmax.finalize() };

                    Ok((
                        Tensor::from_vec(out_argmin, new_shape)?,
                        Tensor::from_vec(out_argmax, new_shape)?,
                    ))
                }
                DType::Int32 => {
                    let data = self.as_slice::<i32>()?;

                    let mut out_argmin = UninitVec::<u64>::new(output_size);
                    let mut out_argmax = UninitVec::<u64>::new(output_size);

                    let dst_argmin = out_argmin.as_mut_slice();
                    let dst_argmax = out_argmax.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (min_idx, max_idx) = backend.min_max_i_i32(&data[start..end]);

                        dst_argmin[i] = min_idx;
                        dst_argmax[i] = max_idx;
                    }

                    let out_argmin = unsafe { out_argmin.finalize() };
                    let out_argmax = unsafe { out_argmax.finalize() };

                    Ok((
                        Tensor::from_vec(out_argmin, new_shape)?,
                        Tensor::from_vec(out_argmax, new_shape)?,
                    ))
                }
                DType::Int64 => {
                    let data = self.as_slice::<i64>()?;

                    let mut out_argmin = UninitVec::<u64>::new(output_size);
                    let mut out_argmax = UninitVec::<u64>::new(output_size);

                    let dst_argmin = out_argmin.as_mut_slice();
                    let dst_argmax = out_argmax.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (min_idx, max_idx) = backend.min_max_i_i64(&data[start..end]);

                        dst_argmin[i] = min_idx;
                        dst_argmax[i] = max_idx;
                    }

                    let out_argmin = unsafe { out_argmin.finalize() };
                    let out_argmax = unsafe { out_argmax.finalize() };

                    Ok((
                        Tensor::from_vec(out_argmin, new_shape)?,
                        Tensor::from_vec(out_argmax, new_shape)?,
                    ))
                }
                DType::Uint8 => {
                    let data = self.as_slice::<u8>()?;

                    let mut out_argmin = UninitVec::<u64>::new(output_size);
                    let mut out_argmax = UninitVec::<u64>::new(output_size);

                    let dst_argmin = out_argmin.as_mut_slice();
                    let dst_argmax = out_argmax.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (min_idx, max_idx) = backend.min_max_i_u8(&data[start..end]);

                        dst_argmin[i] = min_idx;
                        dst_argmax[i] = max_idx;
                    }

                    let out_argmin = unsafe { out_argmin.finalize() };
                    let out_argmax = unsafe { out_argmax.finalize() };

                    Ok((
                        Tensor::from_vec(out_argmin, new_shape)?,
                        Tensor::from_vec(out_argmax, new_shape)?,
                    ))
                }
                DType::Uint16 => {
                    let data = self.as_slice::<u16>()?;

                    let mut out_argmin = UninitVec::<u64>::new(output_size);
                    let mut out_argmax = UninitVec::<u64>::new(output_size);

                    let dst_argmin = out_argmin.as_mut_slice();
                    let dst_argmax = out_argmax.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (min_idx, max_idx) = backend.min_max_i_u16(&data[start..end]);

                        dst_argmin[i] = min_idx;
                        dst_argmax[i] = max_idx;
                    }

                    let out_argmin = unsafe { out_argmin.finalize() };
                    let out_argmax = unsafe { out_argmax.finalize() };

                    Ok((
                        Tensor::from_vec(out_argmin, new_shape)?,
                        Tensor::from_vec(out_argmax, new_shape)?,
                    ))
                }
                DType::Uint32 => {
                    let data = self.as_slice::<u32>()?;

                    let mut out_argmin = UninitVec::<u64>::new(output_size);
                    let mut out_argmax = UninitVec::<u64>::new(output_size);

                    let dst_argmin = out_argmin.as_mut_slice();
                    let dst_argmax = out_argmax.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (min_idx, max_idx) = backend.min_max_i_u32(&data[start..end]);

                        dst_argmin[i] = min_idx;
                        dst_argmax[i] = max_idx;
                    }

                    let out_argmin = unsafe { out_argmin.finalize() };
                    let out_argmax = unsafe { out_argmax.finalize() };

                    Ok((
                        Tensor::from_vec(out_argmin, new_shape)?,
                        Tensor::from_vec(out_argmax, new_shape)?,
                    ))
                }
                DType::Uint64 => {
                    let data = self.as_slice::<u64>()?;

                    let mut out_argmin = UninitVec::<u64>::new(output_size);
                    let mut out_argmax = UninitVec::<u64>::new(output_size);

                    let dst_argmin = out_argmin.as_mut_slice();
                    let dst_argmax = out_argmax.as_mut_slice();

                    for i in 0..output_size {
                        let start = i * reduce_size;
                        let end = start + reduce_size;
                        let (min_idx, max_idx) = backend.min_max_i_u64(&data[start..end]);

                        dst_argmin[i] = min_idx;
                        dst_argmax[i] = max_idx;
                    }

                    let out_argmin = unsafe { out_argmin.finalize() };
                    let out_argmax = unsafe { out_argmax.finalize() };

                    Ok((
                        Tensor::from_vec(out_argmin, new_shape)?,
                        Tensor::from_vec(out_argmax, new_shape)?,
                    ))
                }
                _ => anyhow::bail!("Argmin/Argmax not supported for dtype {:?}", self.dtype()),
            }
        } else {
            let (new_shape, _) = crate::reduce_shape_stride(self.shape, &[dim_index], keepdim);

            let result_size = new_shape.iter().product();
            macro_rules! noncontig_argmin_argmax {
                ($t:ty, $min_init:expr, $max_init:expr) => {{
                    let mut mins = vec![$min_init; result_size];
                    let mut maxs = vec![$max_init; result_size];
                    let mut argmins = vec![0u64; result_size];
                    let mut argmaxs = vec![0u64; result_size];
                    let mut idx_buf = vec![0; new_shape.len()];

                    for elem in self.iter() {
                        let i = elem.indices;
                        let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                        let val = unsafe { *(ptr as *const $t) };
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

                        if val < mins[linear] {
                            mins[linear] = val;
                            argmins[linear] = i[dim_index] as u64;
                        }
                        if val > maxs[linear] {
                            maxs[linear] = val;
                            argmaxs[linear] = i[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(argmins, new_shape)?,
                        Tensor::from_vec(argmaxs, new_shape)?,
                    ))
                }};
            }
            match self.dtype() {
                DType::Fp32 => noncontig_argmin_argmax!(f32, f32::INFINITY, f32::NEG_INFINITY),
                DType::Fp64 => noncontig_argmin_argmax!(f64, f64::INFINITY, f64::NEG_INFINITY),
                DType::Fp16 => noncontig_argmin_argmax!(
                    f16,
                    f16::from_f32(f32::INFINITY),
                    f16::from_f32(f32::NEG_INFINITY)
                ),
                DType::Bf16 => noncontig_argmin_argmax!(
                    bf16,
                    bf16::from_f32(f32::INFINITY),
                    bf16::from_f32(f32::NEG_INFINITY)
                ),
                DType::Int8 => noncontig_argmin_argmax!(i8, i8::MAX, i8::MIN),
                DType::Int16 => noncontig_argmin_argmax!(i16, i16::MAX, i16::MIN),
                DType::Int32 => noncontig_argmin_argmax!(i32, i32::MAX, i32::MIN),
                DType::Int64 => noncontig_argmin_argmax!(i64, i64::MAX, i64::MIN),
                DType::Uint8 => noncontig_argmin_argmax!(u8, u8::MAX, u8::MIN),
                DType::Uint16 => noncontig_argmin_argmax!(u16, u16::MAX, u16::MIN),
                DType::Uint32 => noncontig_argmin_argmax!(u32, u32::MAX, u32::MIN),
                DType::Uint64 => noncontig_argmin_argmax!(u64, u64::MAX, u64::MIN),
                _ => anyhow::bail!("Argmin/Argmax not supported for dtype {:?}", self.dtype()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use anyhow::Result;

    #[test]
    fn test_argmin_argmax_1d_basic() -> Result<()> {
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let tensor = Tensor::from_vec(data, [7])?;

        let (argmin_result, argmax_result) = tensor.argmin_argmax(0)?;

        // Should return scalars (empty shape)
        assert_eq!(argmin_result.dims(), &[] as &[usize]);
        assert_eq!(argmax_result.dims(), &[] as &[usize]);

        let argmin_val = argmin_result.as_slice::<u64>()?[0];
        let argmax_val = argmax_result.as_slice::<u64>()?[0];

        // First occurrence of minimum value 1.0 is at index 1
        assert_eq!(argmin_val, 1);
        // Maximum value 9.0 is at index 5
        assert_eq!(argmax_val, 5);

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_2d_dim0() -> Result<()> {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 8.0, 1.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        let (argmin_result, argmax_result) = tensor.argmin_argmax(0)?;

        assert_eq!(argmin_result.dims(), &[3]);
        assert_eq!(argmax_result.dims(), &[3]);

        let argmin_vals = argmin_result.as_slice::<u64>()?;
        let argmax_vals = argmax_result.as_slice::<u64>()?;

        // Argmin along dim 0: [argmin(1,2), argmin(5,8), argmin(3,1)] = [0, 0, 1]
        assert_eq!(argmin_vals, &[0, 0, 1]);
        // Argmax along dim 0: [argmax(1,2), argmax(5,8), argmax(3,1)] = [1, 1, 0]
        assert_eq!(argmax_vals, &[1, 1, 0]);

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_2d_dim1() -> Result<()> {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 8.0, 1.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        let (argmin_result, argmax_result) = tensor.argmin_argmax(1)?;

        assert_eq!(argmin_result.dims(), &[2]);
        assert_eq!(argmax_result.dims(), &[2]);

        let argmin_vals = argmin_result.as_slice::<u64>()?;
        let argmax_vals = argmax_result.as_slice::<u64>()?;

        // Argmin along dim 1: [argmin(1,5,3), argmin(2,8,1)] = [0, 2]
        assert_eq!(argmin_vals, &[0, 2]);
        // Argmax along dim 1: [argmax(1,5,3), argmax(2,8,1)] = [1, 1]
        assert_eq!(argmax_vals, &[1, 1]);

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_3d_basic() -> Result<()> {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [2, 3, 4])?;

        // Test along dimension 0
        let (argmin_result, argmax_result) = tensor.argmin_argmax(0)?;
        assert_eq!(argmin_result.dims(), &[3, 4]);
        assert_eq!(argmax_result.dims(), &[3, 4]);

        // Test along dimension 1
        let (argmin_result, argmax_result) = tensor.argmin_argmax(1)?;
        assert_eq!(argmin_result.dims(), &[2, 4]);
        assert_eq!(argmax_result.dims(), &[2, 4]);

        // Test along dimension 2
        let (argmin_result, argmax_result) = tensor.argmin_argmax(2)?;
        assert_eq!(argmin_result.dims(), &[2, 3]);
        assert_eq!(argmax_result.dims(), &[2, 3]);

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_keepdim_1d() -> Result<()> {
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0];
        let tensor = Tensor::from_vec(data, [5])?;

        let (argmin_result, argmax_result) = tensor.argmin_argmax_keepdim(0)?;

        // Should keep dimension as [1]
        assert_eq!(argmin_result.dims(), &[1]);
        assert_eq!(argmax_result.dims(), &[1]);

        let argmin_val = argmin_result.as_slice::<u64>()?[0];
        let argmax_val = argmax_result.as_slice::<u64>()?[0];

        // First occurrence of minimum value 1.0 is at index 1
        assert_eq!(argmin_val, 1);
        // Maximum value 5.0 is at index 4
        assert_eq!(argmax_val, 4);

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_keepdim_2d() -> Result<()> {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 8.0, 1.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        // Test keepdim along dimension 0
        let (argmin_result, argmax_result) = tensor.argmin_argmax_keepdim(0)?;
        assert_eq!(argmin_result.dims(), &[1, 3]);
        assert_eq!(argmax_result.dims(), &[1, 3]);

        // Test keepdim along dimension 1
        let (argmin_result, argmax_result) = tensor.argmin_argmax_keepdim(1)?;
        assert_eq!(argmin_result.dims(), &[2, 1]);
        assert_eq!(argmax_result.dims(), &[2, 1]);

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_keepdim_3d() -> Result<()> {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [2, 3, 4])?;

        // Test keepdim along different dimensions
        let (argmin_result, argmax_result) = tensor.argmin_argmax_keepdim(0)?;
        assert_eq!(argmin_result.dims(), &[1, 3, 4]);
        assert_eq!(argmax_result.dims(), &[1, 3, 4]);

        let (argmin_result, argmax_result) = tensor.argmin_argmax_keepdim(1)?;
        assert_eq!(argmin_result.dims(), &[2, 1, 4]);
        assert_eq!(argmax_result.dims(), &[2, 1, 4]);

        let (argmin_result, argmax_result) = tensor.argmin_argmax_keepdim(2)?;
        assert_eq!(argmin_result.dims(), &[2, 3, 1]);
        assert_eq!(argmax_result.dims(), &[2, 3, 1]);

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_non_contiguous_2d() -> Result<()> {
        // Test argmin_argmax with non-contiguous tensor using permute
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3])?;

        // Create non-contiguous tensor by permuting dimensions
        let permuted = tensor.clone().permute([1, 0])?; // [3, 2]

        // Test argmin_argmax along different dimensions
        let (argmin_result, argmax_result) = permuted.argmin_argmax(0)?;
        assert_eq!(argmin_result.dims(), &[2]);
        assert_eq!(argmax_result.dims(), &[2]);

        let argmin_vals = argmin_result.as_slice::<u64>()?;
        let argmax_vals = argmax_result.as_slice::<u64>()?;

        // After permute: [[1,4], [2,5], [3,6]]
        // Argmin along dim 0: [argmin(1,2,3), argmin(4,5,6)] = [0, 0]
        assert_eq!(argmin_vals, &[0, 0]);
        // Argmax along dim 0: [argmax(1,2,3), argmax(4,5,6)] = [2, 2]
        assert_eq!(argmax_vals, &[2, 2]);

        let (argmin_result, argmax_result) = permuted.argmin_argmax(1)?;
        assert_eq!(argmin_result.dims(), &[3]);
        assert_eq!(argmax_result.dims(), &[3]);

        let argmin_vals = argmin_result.as_slice::<u64>()?;
        let argmax_vals = argmax_result.as_slice::<u64>()?;

        // Argmin along dim 1: [argmin(1,4), argmin(2,5), argmin(3,6)] = [0, 0, 0]
        assert_eq!(argmin_vals, &[0, 0, 0]);
        // Argmax along dim 1: [argmax(1,4), argmax(2,5), argmax(3,6)] = [1, 1, 1]
        assert_eq!(argmax_vals, &[1, 1, 1]);

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_non_contiguous_3d() -> Result<()> {
        // Test argmin_argmax with 3D non-contiguous tensor
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [2, 3, 4])?;

        // Create non-contiguous tensor by permuting dimensions
        let permuted = tensor.clone().permute([2, 0, 1])?; // [4, 2, 3]

        // Test argmin_argmax along different dimensions
        let (argmin_result, argmax_result) = permuted.argmin_argmax(0)?;
        assert_eq!(argmin_result.dims(), &[2, 3]);
        assert_eq!(argmax_result.dims(), &[2, 3]);

        let (argmin_result, argmax_result) = permuted.argmin_argmax(1)?;
        assert_eq!(argmin_result.dims(), &[4, 3]);
        assert_eq!(argmax_result.dims(), &[4, 3]);

        let (argmin_result, argmax_result) = permuted.argmin_argmax(2)?;
        assert_eq!(argmin_result.dims(), &[4, 2]);
        assert_eq!(argmax_result.dims(), &[4, 2]);

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_keepdim_non_contiguous() -> Result<()> {
        // Test argmin_argmax_keepdim with non-contiguous tensor
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data, [2, 2, 2])?;

        // Create non-contiguous tensor
        let permuted = tensor.clone().permute([2, 1, 0])?; // [2, 2, 2]

        // Test argmin_argmax_keepdim along different dimensions
        let (argmin_result, argmax_result) = permuted.argmin_argmax_keepdim(0)?;
        assert_eq!(argmin_result.dims(), &[1, 2, 2]);
        assert_eq!(argmax_result.dims(), &[1, 2, 2]);

        let (argmin_result, argmax_result) = permuted.argmin_argmax_keepdim(1)?;
        assert_eq!(argmin_result.dims(), &[2, 1, 2]);
        assert_eq!(argmax_result.dims(), &[2, 1, 2]);

        let (argmin_result, argmax_result) = permuted.argmin_argmax_keepdim(2)?;
        assert_eq!(argmin_result.dims(), &[2, 2, 1]);
        assert_eq!(argmax_result.dims(), &[2, 2, 1]);

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_different_data_types() -> Result<()> {
        // Test argmin_argmax with different data types

        // Test with i32
        let data_i32 = vec![5i32, 1, 9, 3, 7, 2];
        let tensor_i32 = Tensor::from_vec(data_i32, [2, 3])?;
        let (argmin_result, argmax_result) = tensor_i32.argmin_argmax(1)?;

        let argmin_vals = argmin_result.as_slice::<u64>()?;
        let argmax_vals = argmax_result.as_slice::<u64>()?;

        // Argmin along dim 1: [argmin(5,1,9), argmin(3,7,2)] = [1, 2]
        assert_eq!(argmin_vals, &[1, 2]);
        // Argmax along dim 1: [argmax(5,1,9), argmax(3,7,2)] = [2, 1]
        assert_eq!(argmax_vals, &[2, 1]);

        // Test with u32
        let data_u32 = vec![10u32, 20, 5, 15];
        let tensor_u32 = Tensor::from_vec(data_u32, [2, 2])?;
        let (argmin_result, argmax_result) = tensor_u32.argmin_argmax(0)?;

        let argmin_vals = argmin_result.as_slice::<u64>()?;
        let argmax_vals = argmax_result.as_slice::<u64>()?;

        // Argmin along dim 0: [argmin(10,5), argmin(20,15)] = [1, 1]
        assert_eq!(argmin_vals, &[1, 1]);
        // Argmax along dim 0: [argmax(10,5), argmax(20,15)] = [0, 0]
        assert_eq!(argmax_vals, &[0, 0]);

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_special_values() -> Result<()> {
        // Test argmin_argmax with special floating point values

        // Test with infinity
        let data_inf = vec![1.0f32, f32::INFINITY, 3.0, f32::NEG_INFINITY];
        let tensor_inf = Tensor::from_vec(data_inf, [4])?;
        let (argmin_result, argmax_result) = tensor_inf.argmin_argmax(0)?;

        let argmin_val = argmin_result.as_slice::<u64>()?[0];
        let argmax_val = argmax_result.as_slice::<u64>()?[0];

        // NEG_INFINITY is at index 3, INFINITY is at index 1
        assert_eq!(argmin_val, 3);
        assert_eq!(argmax_val, 1);

        // Test with NaN
        let data_nan = vec![1.0f32, f32::NAN, 3.0];
        let tensor_nan = Tensor::from_vec(data_nan, [3])?;
        let (argmin_result, argmax_result) = tensor_nan.argmin_argmax(0)?;

        let argmin_val = argmin_result.as_slice::<u64>()?[0];
        let argmax_val = argmax_result.as_slice::<u64>()?[0];

        // NaN behavior: typically returns the index of NaN or first non-NaN value
        // The exact behavior depends on the backend implementation
        assert!(argmin_val < 3);
        assert!(argmax_val < 3);

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_edge_cases() -> Result<()> {
        // Test various edge cases

        // Single element tensor
        let single = Tensor::from_vec(vec![42.0f32], [1])?;
        let (argmin_result, argmax_result) = single.argmin_argmax(0)?;

        let argmin_val = argmin_result.as_slice::<u64>()?[0];
        let argmax_val = argmax_result.as_slice::<u64>()?[0];

        assert_eq!(argmin_val, 0);
        assert_eq!(argmax_val, 0);

        // Tensor with all same values
        let same = Tensor::from_vec(vec![5.0f32, 5.0, 5.0, 5.0], [2, 2])?;
        let (argmin_result, argmax_result) = same.argmin_argmax(0)?;

        let argmin_vals = argmin_result.as_slice::<u64>()?;
        let argmax_vals = argmax_result.as_slice::<u64>()?;

        // Should return first occurrence (index 0) for both min and max
        assert_eq!(argmin_vals, &[0, 0]);
        assert_eq!(argmax_vals, &[0, 0]);

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_rectangular_tensors() -> Result<()> {
        // Test argmin_argmax with rectangular (non-square) tensors

        // 1x5 tensor
        let data_1x5 = vec![5.0f32, 1.0, 9.0, 3.0, 7.0];
        let tensor_1x5 = Tensor::from_vec(data_1x5, [1, 5])?;

        let (argmin_result, argmax_result) = tensor_1x5.argmin_argmax(0)?;
        assert_eq!(argmin_result.dims(), &[5]);
        assert_eq!(argmax_result.dims(), &[5]);

        let (argmin_result, argmax_result) = tensor_1x5.argmin_argmax(1)?;
        assert_eq!(argmin_result.dims(), &[1]);
        assert_eq!(argmax_result.dims(), &[1]);

        let argmin_val = argmin_result.as_slice::<u64>()?[0];
        let argmax_val = argmax_result.as_slice::<u64>()?[0];
        assert_eq!(argmin_val, 1); // min value 1.0 at index 1
        assert_eq!(argmax_val, 2); // max value 9.0 at index 2

        // 5x1 tensor
        let data_5x1 = vec![10.0f32, 20.0, 5.0, 30.0, 15.0];
        let tensor_5x1 = Tensor::from_vec(data_5x1, [5, 1])?;

        let (argmin_result, argmax_result) = tensor_5x1.argmin_argmax(0)?;
        assert_eq!(argmin_result.dims(), &[1]);
        assert_eq!(argmax_result.dims(), &[1]);

        let argmin_val = argmin_result.as_slice::<u64>()?[0];
        let argmax_val = argmax_result.as_slice::<u64>()?[0];
        assert_eq!(argmin_val, 2); // min value 5.0 at index 2
        assert_eq!(argmax_val, 3); // max value 30.0 at index 3

        let (argmin_result, argmax_result) = tensor_5x1.argmin_argmax(1)?;
        assert_eq!(argmin_result.dims(), &[5]);
        assert_eq!(argmax_result.dims(), &[5]);

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_consistency_with_individual_ops() -> Result<()> {
        // Test that argmin_argmax results are consistent with individual argmin and argmax operations
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 4])?;

        // Test along dimension 0
        let (argmin_result, argmax_result) = tensor.argmin_argmax(0)?;
        let individual_argmin = tensor.argmin(0)?;
        let individual_argmax = tensor.argmax(0)?;

        let argmin_vals = argmin_result.as_slice::<u64>()?;
        let argmax_vals = argmax_result.as_slice::<u64>()?;
        let individual_argmin_vals = individual_argmin.as_slice::<u64>()?;
        let individual_argmax_vals = individual_argmax.as_slice::<u64>()?;

        for (i, (&argmin_val, &argmax_val)) in
            argmin_vals.iter().zip(argmax_vals.iter()).enumerate()
        {
            assert_eq!(argmin_val, individual_argmin_vals[i]);
            assert_eq!(argmax_val, individual_argmax_vals[i]);
        }

        // Test along dimension 1
        let (argmin_result, argmax_result) = tensor.argmin_argmax(1)?;
        let individual_argmin = tensor.argmin(1)?;
        let individual_argmax = tensor.argmax(1)?;

        let argmin_vals = argmin_result.as_slice::<u64>()?;
        let argmax_vals = argmax_result.as_slice::<u64>()?;
        let individual_argmin_vals = individual_argmin.as_slice::<u64>()?;
        let individual_argmax_vals = individual_argmax.as_slice::<u64>()?;

        for (i, (&argmin_val, &argmax_val)) in
            argmin_vals.iter().zip(argmax_vals.iter()).enumerate()
        {
            assert_eq!(argmin_val, individual_argmin_vals[i]);
            assert_eq!(argmax_val, individual_argmax_vals[i]);
        }

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_large_tensor() -> Result<()> {
        // Test argmin_argmax with a larger tensor
        let size = 1000;
        let data: Vec<f32> = (0..size).map(|i| (i % 100) as f32).collect();
        let tensor = Tensor::from_vec(data, [10, 100])?;

        let (argmin_result, argmax_result) = tensor.argmin_argmax(1)?;

        assert_eq!(argmin_result.dims(), &[10]);
        assert_eq!(argmax_result.dims(), &[10]);

        let argmin_vals = argmin_result.as_slice::<u64>()?;
        let argmax_vals = argmax_result.as_slice::<u64>()?;

        // Each row should have argmin=0 and argmax=99
        for (&argmin_val, &argmax_val) in argmin_vals.iter().zip(argmax_vals.iter()) {
            assert_eq!(argmin_val, 0);
            assert_eq!(argmax_val, 99);
        }

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_empty_tensor_error() -> Result<()> {
        // Test that argmin_argmax fails gracefully with empty tensor
        let empty_tensor = Tensor::from_vec(Vec::<f32>::new(), [0])?;

        let result = empty_tensor.argmin_argmax(0);
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("size 0"));

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_invalid_dimension() -> Result<()> {
        // Test argmin_argmax with invalid dimension
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, [3])?;

        // Should fail for dimension >= rank
        assert!(tensor.argmin_argmax(1).is_err());
        assert!(tensor.argmin_argmax(2).is_err());

        Ok(())
    }

    #[test]
    fn test_argmin_argmax_first_occurrence() -> Result<()> {
        // Test that argmin_argmax returns the first occurrence of min/max values
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 5.0, 2.0];
        let tensor = Tensor::from_vec(data, [7])?;

        let (argmin_result, argmax_result) = tensor.argmin_argmax(0)?;

        let argmin_val = argmin_result.as_slice::<u64>()?[0];
        let argmax_val = argmax_result.as_slice::<u64>()?[0];

        // First occurrence of minimum value 1.0 is at index 1
        assert_eq!(argmin_val, 1);
        // First occurrence of maximum value 5.0 is at index 4
        assert_eq!(argmax_val, 4);

        Ok(())
    }
}
