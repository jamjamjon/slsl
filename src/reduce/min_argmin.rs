use anyhow::Result;
use half::{bf16, f16};

use crate::{global_backend, DType, Dim, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    pub fn min_argmin<D: Dim + Clone>(&self, dim: D) -> Result<(Tensor, Tensor)> {
        self.min_argmin_impl(dim, false)
    }

    pub fn min_argmin_keepdim<D: Dim + Clone>(&self, dim: D) -> Result<(Tensor, Tensor)> {
        self.min_argmin_impl(dim, true)
    }

    pub fn min_argmin_impl<D: Dim + Clone>(
        &self,
        dim: D,
        keepdim: bool,
    ) -> Result<(Tensor, Tensor)> {
        let dim_index = dim.to_dim(self.rank())?;
        if self.shape()[dim_index] == 0 {
            anyhow::bail!("Cannot find min of dimension with size 0");
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
                        let (minv, min_idx) = backend.min_vi_f32(&data[start..end]);
                        dst_vals[i] = minv;
                        dst_idxs[i] = min_idx;
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
                        let (minv, min_idx) = backend.min_vi_f64(&data[start..end]);
                        dst_vals[i] = minv;
                        dst_idxs[i] = min_idx;
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
                        let (minv, min_idx) = backend.min_vi_f16(&data[start..end]);
                        dst_vals[i] = minv;
                        dst_idxs[i] = min_idx;
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
                        let (minv, min_idx) = backend.min_vi_bf16(&data[start..end]);
                        dst_vals[i] = minv;
                        dst_idxs[i] = min_idx;
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
                        let (minv, min_idx) = backend.min_vi_i8(&data[start..end]);
                        dst_vals[i] = minv;
                        dst_idxs[i] = min_idx;
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
                        let (minv, min_idx) = backend.min_vi_i16(&data[start..end]);
                        dst_vals[i] = minv;
                        dst_idxs[i] = min_idx;
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
                        let (minv, min_idx) = backend.min_vi_i32(&data[start..end]);
                        dst_vals[i] = minv;
                        dst_idxs[i] = min_idx;
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
                        let (minv, min_idx) = backend.min_vi_i64(&data[start..end]);
                        dst_vals[i] = minv;
                        dst_idxs[i] = min_idx;
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
                        let (minv, min_idx) = backend.min_vi_u8(&data[start..end]);
                        dst_vals[i] = minv;
                        dst_idxs[i] = min_idx;
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
                        let (minv, min_idx) = backend.min_vi_u16(&data[start..end]);
                        dst_vals[i] = minv;
                        dst_idxs[i] = min_idx;
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
                        let (minv, min_idx) = backend.min_vi_u32(&data[start..end]);
                        dst_vals[i] = minv;
                        dst_idxs[i] = min_idx;
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
                        let (minv, min_idx) = backend.min_vi_u64(&data[start..end]);
                        dst_vals[i] = minv;
                        dst_idxs[i] = min_idx;
                    }

                    let out_vals = unsafe { out_vals.finalize() };
                    let out_idxs = unsafe { out_idxs.finalize() };

                    Ok((
                        Tensor::from_vec(out_vals, new_shape)?,
                        Tensor::from_vec(out_idxs, new_shape)?,
                    ))
                }
                _ => anyhow::bail!("Min+Argmin not supported for dtype {:?}", self.dtype()),
            }
        } else {
            // General case: non-contiguous or not reducing last dim
            let (new_shape, _) =
                crate::reduce::reduce_shape_stride(self.shape, &[dim_index], keepdim);
            let result_size = new_shape.iter().product::<usize>();

            match self.dtype() {
                DType::Fp32 => {
                    let mut mins = vec![f32::INFINITY; result_size];
                    let mut argmins = vec![0u64; result_size];

                    for item in self.iter_with_meta::<f32>() {
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

                        // convert multi-dim coords to linear index
                        let mut out_linear = 0usize;
                        for (k, &c) in out_coords.iter().enumerate() {
                            out_linear = out_linear * new_shape[k] + c;
                        }

                        if val < mins[out_linear] {
                            mins[out_linear] = val;
                            argmins[out_linear] = idx[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(mins, new_shape)?,
                        Tensor::from_vec(argmins, new_shape)?,
                    ))
                }
                DType::Fp64 => {
                    let mut mins = vec![f64::INFINITY; result_size];
                    let mut argmins = vec![0u64; result_size];

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

                        if val < mins[out_linear] {
                            mins[out_linear] = val;
                            argmins[out_linear] = idx[dim_index] as u64;
                        }
                    }

                    Ok((
                        Tensor::from_vec(mins, new_shape)?,
                        Tensor::from_vec(argmins, new_shape)?,
                    ))
                }
                _ => {
                    // Typed dispatch using iter_with_meta::<T>() to avoid untyped iteration
                    macro_rules! dispatch {
                        ($ty:ty) => {{
                            let mut mins = vec![f64::INFINITY; result_size];
                            let mut argmins = vec![0u64; result_size];
                            for item in self.iter_with_meta::<$ty>() {
                                let idx = item.indices;
                                let val_f64 = (*item.value) as f64;
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
                                if val_f64 < mins[out_linear] {
                                    mins[out_linear] = val_f64;
                                    argmins[out_linear] = idx[dim_index] as u64;
                                }
                            }
                            Ok((
                                Tensor::from_vec(
                                    mins.into_iter().map(|v| v as $ty).collect::<Vec<$ty>>(),
                                    new_shape,
                                )?,
                                Tensor::from_vec(argmins, new_shape)?,
                            ))
                        }};
                    }
                    match self.dtype() {
                        DType::Int8 => dispatch!(i8),
                        DType::Int16 => dispatch!(i16),
                        DType::Int32 => dispatch!(i32),
                        DType::Int64 => dispatch!(i64),
                        DType::Uint8 => dispatch!(u8),
                        DType::Uint16 => dispatch!(u16),
                        DType::Uint32 => dispatch!(u32),
                        DType::Uint64 => dispatch!(u64),
                        DType::Fp16 => {
                            let mut mins = vec![f64::INFINITY; result_size];
                            let mut argmins = vec![0u64; result_size];
                            for item in self.iter_with_meta::<f16>() {
                                let idx = item.indices;
                                let val_f64 = item.value.to_f64();
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
                                if val_f64 < mins[out_linear] {
                                    mins[out_linear] = val_f64;
                                    argmins[out_linear] = idx[dim_index] as u64;
                                }
                            }
                            let vals: Vec<f16> = mins.into_iter().map(f16::from_f64).collect();
                            Ok((
                                Tensor::from_vec(vals, new_shape)?,
                                Tensor::from_vec(argmins, new_shape)?,
                            ))
                        }
                        DType::Bf16 => {
                            let mut mins = vec![f64::INFINITY; result_size];
                            let mut argmins = vec![0u64; result_size];
                            for item in self.iter_with_meta::<bf16>() {
                                let idx = item.indices;
                                let val_f64 = item.value.to_f64();
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
                                if val_f64 < mins[out_linear] {
                                    mins[out_linear] = val_f64;
                                    argmins[out_linear] = idx[dim_index] as u64;
                                }
                            }
                            let vals: Vec<bf16> = mins.into_iter().map(bf16::from_f64).collect();
                            Ok((
                                Tensor::from_vec(vals, new_shape)?,
                                Tensor::from_vec(argmins, new_shape)?,
                            ))
                        }
                        _ => anyhow::bail!("Min+Argmin not supported for dtype {:?}", self.dtype()),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use anyhow::Result;

    #[test]
    fn test_min_argmin_1d_basic() -> Result<()> {
        let tensor = Tensor::from_vec(vec![3.0f32, 1.0, 4.0, 2.0], [4])?;
        let (min_vals, min_indices) = tensor.min_argmin(0)?;

        assert_eq!(min_vals.to_scalar::<f32>()?, 1.0);
        assert_eq!(min_indices.to_scalar::<u64>()?, 1);
        Ok(())
    }

    #[test]
    fn test_min_argmin_1d_duplicate_values() -> Result<()> {
        let tensor = Tensor::from_vec(vec![2.0f32, 1.0, 1.0, 3.0], [4])?;
        let (min_vals, min_indices) = tensor.min_argmin(0)?;

        assert_eq!(min_vals.to_scalar::<f32>()?, 1.0);
        assert_eq!(min_indices.to_scalar::<u64>()?, 1); // First occurrence
        Ok(())
    }

    #[test]
    fn test_min_argmin_1d_negative_values() -> Result<()> {
        let tensor = Tensor::from_vec(vec![-1.0f32, -3.0, 2.0, -2.0], [4])?;
        let (min_vals, min_indices) = tensor.min_argmin(0)?;

        assert_eq!(min_vals.to_scalar::<f32>()?, -3.0);
        assert_eq!(min_indices.to_scalar::<u64>()?, 1);
        Ok(())
    }

    #[test]
    fn test_min_argmin_2d_dim0() -> Result<()> {
        // Test case: 2x3 tensor
        // [[1.0, 4.0, 2.0],
        //  [3.0, 2.0, 5.0]]
        // Reduce along dimension 0 (rows)
        let tensor = Tensor::from_vec(vec![1.0f32, 4.0, 2.0, 3.0, 2.0, 5.0], [2, 3])?;
        let (min_vals, min_indices) = tensor.min_argmin(0)?;

        assert_eq!(min_vals.to_vec::<f32>()?, vec![1.0, 2.0, 2.0]);
        assert_eq!(min_indices.to_vec::<u64>()?, vec![0, 1, 0]);
        Ok(())
    }

    #[test]
    fn test_min_argmin_2d_dim1() -> Result<()> {
        // Test case: 2x3 tensor
        // [[1.0, 4.0, 2.0],
        //  [3.0, 2.0, 5.0]]
        // Reduce along dimension 1 (columns)
        let tensor = Tensor::from_vec(vec![1.0f32, 4.0, 2.0, 3.0, 2.0, 5.0], [2, 3])?;
        let (min_vals, min_indices) = tensor.min_argmin(1)?;

        assert_eq!(min_vals.to_vec::<f32>()?, vec![1.0, 2.0]);
        assert_eq!(min_indices.to_vec::<u64>()?, vec![0, 1]);
        Ok(())
    }

    #[test]
    fn test_min_argmin_3d() -> Result<()> {
        // Test case: 2x2x2 tensor
        // [[[1.0, 2.0],
        //   [3.0, 4.0]],
        //  [[5.0, 0.5],
        //   [7.0, 8.0]]]
        // Reduce along dimension 0 (depth)
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 0.5, 7.0, 8.0], [2, 2, 2])?;
        let (min_vals, min_indices) = tensor.min_argmin(0)?;

        // Expected: [[1.0, 0.5], [3.0, 4.0]]
        // Indices:  [[0, 1], [0, 0]]
        assert_eq!(min_vals.shape().as_slice(), &[2, 2]);
        assert_eq!(
            min_vals.flatten_all()?.to_vec::<f32>()?,
            vec![1.0, 0.5, 3.0, 4.0]
        );
        assert_eq!(
            min_indices.flatten_all()?.to_vec::<u64>()?,
            vec![0, 1, 0, 0]
        );
        Ok(())
    }

    #[test]
    fn test_min_argmin_keepdim_1d() -> Result<()> {
        let tensor = Tensor::from_vec(vec![3.0f32, 1.0, 4.0, 2.0], [4])?;
        let (min_vals, min_indices) = tensor.min_argmin_keepdim(0)?;

        assert_eq!(min_vals.shape().as_slice(), &[1]);
        assert_eq!(min_indices.shape().as_slice(), &[1]);
        assert_eq!(min_vals.to_vec::<f32>()?, vec![1.0]);
        assert_eq!(min_indices.to_vec::<u64>()?, vec![1]);
        Ok(())
    }

    #[test]
    fn test_min_argmin_keepdim_2d() -> Result<()> {
        // Test case: 2x3 tensor with keepdim=true
        // [[1.0, 4.0, 2.0],
        //  [3.0, 2.0, 5.0]]
        // Reduce along dimension 0 with keepdim
        let tensor = Tensor::from_vec(vec![1.0f32, 4.0, 2.0, 3.0, 2.0, 5.0], [2, 3])?;
        let (min_vals, min_indices) = tensor.min_argmin_keepdim(0)?;

        // Shape should be [1, 3] instead of [3]
        assert_eq!(min_vals.shape().as_slice(), &[1, 3]);
        assert_eq!(min_indices.shape().as_slice(), &[1, 3]);
        assert_eq!(
            min_vals.flatten_all()?.to_vec::<f32>()?,
            vec![1.0, 2.0, 2.0]
        );
        assert_eq!(min_indices.flatten_all()?.to_vec::<u64>()?, vec![0, 1, 0]);

        // Test along dimension 1 with keepdim
        let (min_vals, min_indices) = tensor.min_argmin_keepdim(1)?;
        assert_eq!(min_vals.shape().as_slice(), &[2, 1]);
        assert_eq!(min_indices.shape().as_slice(), &[2, 1]);
        assert_eq!(min_vals.flatten_all()?.to_vec::<f32>()?, vec![1.0, 2.0]);
        assert_eq!(min_indices.flatten_all()?.to_vec::<u64>()?, vec![0, 1]);
        Ok(())
    }

    #[test]
    fn test_min_argmin_single_element() -> Result<()> {
        let tensor = Tensor::from_vec(vec![42.0f32], [1])?;
        let (min_vals, min_indices) = tensor.min_argmin(0)?;

        assert_eq!(min_vals.to_scalar::<f32>()?, 42.0);
        assert_eq!(min_indices.to_scalar::<u64>()?, 0);
        Ok(())
    }

    #[test]
    fn test_min_argmin_empty_tensor() -> Result<()> {
        let tensor = Tensor::from_vec(Vec::<f32>::new(), [0])?;
        let result = tensor.min_argmin(0);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_min_argmin_different_dtypes() -> Result<()> {
        // Test with i32
        let tensor_i32 = Tensor::from_vec(vec![3i32, 1, 4, 2], [4])?;
        let (min_vals, min_indices) = tensor_i32.min_argmin(0)?;
        assert_eq!(min_vals.to_scalar::<i32>()?, 1);
        assert_eq!(min_indices.to_scalar::<u64>()?, 1);

        // Test with f64
        let tensor_f64 = Tensor::from_vec(vec![3.0f64, 1.0, 4.0, 2.0], [4])?;
        let (min_vals, min_indices) = tensor_f64.min_argmin(0)?;
        assert_eq!(min_vals.to_scalar::<f64>()?, 1.0);
        assert_eq!(min_indices.to_scalar::<u64>()?, 1);

        // Test with u8
        let tensor_u8 = Tensor::from_vec(vec![3u8, 1, 4, 2], [4])?;
        let (min_vals, min_indices) = tensor_u8.min_argmin(0)?;
        assert_eq!(min_vals.to_scalar::<u8>()?, 1);
        assert_eq!(min_indices.to_scalar::<u64>()?, 1);
        Ok(())
    }

    #[test]
    fn test_min_argmin_invalid_dim() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3])?;
        let result = tensor.min_argmin(1); // Invalid dimension for 1D tensor
        assert!(result.is_err());
        Ok(())
    }
}
