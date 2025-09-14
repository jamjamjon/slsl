use anyhow::Result;
use half::{bf16, f16};

use crate::{global_backend, DType, Dim, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    #[inline(always)]
    pub fn argmin<D: Dim + Clone>(&self, dim: D) -> Result<Tensor> {
        self.argmin_impl(dim, false)
    }

    #[inline(always)]
    pub fn argmin_keepdim<D: Dim + Clone>(&self, dim: D) -> Result<Tensor> {
        self.argmin_impl(dim, true)
    }

    #[inline(always)]
    pub fn argmin_impl<D: Dim + Clone>(&self, dim: D, keepdim: bool) -> Result<Tensor> {
        let dim_index = dim.to_dim(self.rank())?;
        if self.shape()[dim_index] == 0 {
            anyhow::bail!("Cannot find argmin of dimension with size 0");
        }

        if self.is_contiguous() && self.can_reduce_over_last_dims(&[dim_index]) {
            let backend = global_backend();
            let shape = self.shape();
            let reduce_size = shape[dim_index];
            let output_size = self.numel() / reduce_size;
            let (new_shape, _) = crate::reduce_shape_stride(self.shape, &[dim_index], keepdim);

            match self.dtype() {
                DType::Fp32 => {
                    let data = self.as_slice::<f32>()?;
                    let out = UninitVec::<u64>::new(output_size).init_with(|dst| {
                        for (i, item) in dst.iter_mut().enumerate().take(output_size) {
                            let start = i * reduce_size;
                            let end = start + reduce_size;
                            let (_minv, idx) = backend.min_vi_f32(&data[start..end]);
                            *item = idx;
                        }
                    });
                    Tensor::from_vec(out, new_shape)
                }
                DType::Fp64 => {
                    let data = self.as_slice::<f64>()?;
                    let out = UninitVec::<u64>::new(output_size).init_with(|dst| {
                        for (i, item) in dst.iter_mut().enumerate().take(output_size) {
                            let start = i * reduce_size;
                            let end = start + reduce_size;
                            let (_minv, idx) = backend.min_vi_f64(&data[start..end]);
                            *item = idx;
                        }
                    });
                    Tensor::from_vec(out, new_shape)
                }
                DType::Fp16 => {
                    let data = self.as_slice::<f16>()?;
                    let out = UninitVec::<u64>::new(output_size).init_with(|dst| {
                        for (i, item) in dst.iter_mut().enumerate().take(output_size) {
                            let start = i * reduce_size;
                            let end = start + reduce_size;
                            let (_minv, idx) = backend.min_vi_f16(&data[start..end]);
                            *item = idx;
                        }
                    });
                    Tensor::from_vec(out, new_shape)
                }
                DType::Bf16 => {
                    let data = self.as_slice::<bf16>()?;
                    let out = UninitVec::<u64>::new(output_size).init_with(|dst| {
                        for (i, item) in dst.iter_mut().enumerate().take(output_size) {
                            let start = i * reduce_size;
                            let end = start + reduce_size;
                            let (_minv, idx) = backend.min_vi_bf16(&data[start..end]);
                            *item = idx;
                        }
                    });
                    Tensor::from_vec(out, new_shape)
                }
                DType::Int8 => {
                    let data = self.as_slice::<i8>()?;
                    let out = UninitVec::<u64>::new(output_size).init_with(|dst| {
                        for (i, item) in dst.iter_mut().enumerate().take(output_size) {
                            let start = i * reduce_size;
                            let end = start + reduce_size;
                            let (_minv, idx) = backend.min_vi_i8(&data[start..end]);
                            *item = idx;
                        }
                    });
                    Tensor::from_vec(out, new_shape)
                }
                DType::Int16 => {
                    let data = self.as_slice::<i16>()?;
                    let out = UninitVec::<u64>::new(output_size).init_with(|dst| {
                        for (i, item) in dst.iter_mut().enumerate().take(output_size) {
                            let start = i * reduce_size;
                            let end = start + reduce_size;
                            let (_minv, idx) = backend.min_vi_i16(&data[start..end]);
                            *item = idx;
                        }
                    });
                    Tensor::from_vec(out, new_shape)
                }
                DType::Int32 => {
                    let data = self.as_slice::<i32>()?;
                    let out = UninitVec::<u64>::new(output_size).init_with(|dst| {
                        for (i, item) in dst.iter_mut().enumerate().take(output_size) {
                            let start = i * reduce_size;
                            let end = start + reduce_size;
                            let (_minv, idx) = backend.min_vi_i32(&data[start..end]);
                            *item = idx;
                        }
                    });
                    Tensor::from_vec(out, new_shape)
                }
                DType::Int64 => {
                    let data = self.as_slice::<i64>()?;
                    let out = UninitVec::<u64>::new(output_size).init_with(|dst| {
                        for (i, item) in dst.iter_mut().enumerate().take(output_size) {
                            let start = i * reduce_size;
                            let end = start + reduce_size;
                            let (_minv, idx) = backend.min_vi_i64(&data[start..end]);
                            *item = idx;
                        }
                    });
                    Tensor::from_vec(out, new_shape)
                }
                DType::Uint8 => {
                    let data = self.as_slice::<u8>()?;
                    let out = UninitVec::<u64>::new(output_size).init_with(|dst| {
                        for (i, item) in dst.iter_mut().enumerate().take(output_size) {
                            let start = i * reduce_size;
                            let end = start + reduce_size;
                            let (_minv, idx) = backend.min_vi_u8(&data[start..end]);
                            *item = idx;
                        }
                    });
                    Tensor::from_vec(out, new_shape)
                }
                DType::Uint16 => {
                    let data = self.as_slice::<u16>()?;
                    let out = UninitVec::<u64>::new(output_size).init_with(|dst| {
                        for (i, item) in dst.iter_mut().enumerate().take(output_size) {
                            let start = i * reduce_size;
                            let end = start + reduce_size;
                            let (_minv, idx) = backend.min_vi_u16(&data[start..end]);
                            *item = idx;
                        }
                    });
                    Tensor::from_vec(out, new_shape)
                }
                DType::Uint32 => {
                    let data = self.as_slice::<u32>()?;
                    let out = UninitVec::<u64>::new(output_size).init_with(|dst| {
                        for (i, item) in dst.iter_mut().enumerate().take(output_size) {
                            let start = i * reduce_size;
                            let end = start + reduce_size;
                            let (_minv, idx) = backend.min_vi_u32(&data[start..end]);
                            *item = idx;
                        }
                    });
                    Tensor::from_vec(out, new_shape)
                }
                DType::Uint64 => {
                    let data = self.as_slice::<u64>()?;
                    let out = UninitVec::<u64>::new(output_size).init_with(|dst| {
                        for (i, item) in dst.iter_mut().enumerate().take(output_size) {
                            let start = i * reduce_size;
                            let end = start + reduce_size;
                            let (_minv, idx) = backend.min_vi_u64(&data[start..end]);
                            *item = idx;
                        }
                    });
                    Tensor::from_vec(out, new_shape)
                }
                _ => anyhow::bail!("Argmin not supported for dtype {:?}", self.dtype()),
            }
        } else {
            let (new_shape, _) =
                crate::reduce::reduce_shape_stride(self.shape, &[dim_index], keepdim);
            let result_size = new_shape.iter().product();
            match self.dtype() {
                DType::Fp32 => {
                    let mut mins = vec![f32::INFINITY; result_size];
                    let mut argmins = vec![0u64; result_size];
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

                        if val < mins[linear] {
                            mins[linear] = val;
                            argmins[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmins, new_shape)
                }
                DType::Fp64 => {
                    let mut mins = vec![f64::INFINITY; result_size];
                    let mut argmins = vec![0u64; result_size];
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

                        if val < mins[linear] {
                            mins[linear] = val;
                            argmins[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmins, new_shape)
                }
                DType::Fp16 => {
                    let mut mins = vec![f16::from_f32(f32::INFINITY); result_size];
                    let mut argmins = vec![0u64; result_size];
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

                        if val < mins[linear] {
                            mins[linear] = val;
                            argmins[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmins, new_shape)
                }
                DType::Bf16 => {
                    let mut mins = vec![bf16::from_f32(f32::INFINITY); result_size];
                    let mut argmins = vec![0u64; result_size];
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

                        if val < mins[linear] {
                            mins[linear] = val;
                            argmins[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmins, new_shape)
                }
                DType::Int8 => {
                    let mut mins = vec![i8::MAX; result_size];
                    let mut argmins = vec![0u64; result_size];
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

                        if val < mins[linear] {
                            mins[linear] = val;
                            argmins[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmins, new_shape)
                }
                DType::Int16 => {
                    let mut mins = vec![i16::MAX; result_size];
                    let mut argmins = vec![0u64; result_size];
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

                        if val < mins[linear] {
                            mins[linear] = val;
                            argmins[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmins, new_shape)
                }
                DType::Int32 => {
                    let mut mins = vec![i32::MAX; result_size];
                    let mut argmins = vec![0u64; result_size];
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

                        if val < mins[linear] {
                            mins[linear] = val;
                            argmins[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmins, new_shape)
                }
                DType::Int64 => {
                    let mut mins = vec![i64::MAX; result_size];
                    let mut argmins = vec![0u64; result_size];
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

                        if val < mins[linear] {
                            mins[linear] = val;
                            argmins[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmins, new_shape)
                }
                DType::Uint8 => {
                    let mut mins = vec![u8::MAX; result_size];
                    let mut argmins = vec![0u64; result_size];
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

                        if val < mins[linear] {
                            mins[linear] = val;
                            argmins[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmins, new_shape)
                }
                DType::Uint16 => {
                    let mut mins = vec![u16::MAX; result_size];
                    let mut argmins = vec![0u64; result_size];
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

                        if val < mins[linear] {
                            mins[linear] = val;
                            argmins[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmins, new_shape)
                }
                DType::Uint32 => {
                    let mut mins = vec![u32::MAX; result_size];
                    let mut argmins = vec![0u64; result_size];
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

                        if val < mins[linear] {
                            mins[linear] = val;
                            argmins[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmins, new_shape)
                }
                DType::Uint64 => {
                    let mut mins = vec![u64::MAX; result_size];
                    let mut argmins = vec![0u64; result_size];
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

                        if val < mins[linear] {
                            mins[linear] = val;
                            argmins[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmins, new_shape)
                }
                _ => anyhow::bail!("Argmin not supported for dtype {:?}", self.dtype()),
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::Tensor;

    #[test]
    fn test_argmin_1d_basic() {
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0];
        let tensor = Tensor::from_vec(data, [5]).unwrap();
        let result = tensor.argmin(0).unwrap();
        assert_eq!(result.dims(), &[] as &[usize]); // 0-dimensional tensor
        assert_eq!(result.to_scalar::<u64>().unwrap(), 1); // First occurrence of minimum
    }

    #[test]
    fn test_argmin_1d_edge_cases() {
        // Single element
        let tensor = Tensor::from_vec(vec![42.0f32], [1]).unwrap();
        let result = tensor.argmin(0).unwrap();
        assert_eq!(result.to_scalar::<u64>().unwrap(), 0);

        // All same values
        let tensor = Tensor::from_vec(vec![2.0f32; 5], [5]).unwrap();
        let result = tensor.argmin(0).unwrap();
        assert_eq!(result.to_scalar::<u64>().unwrap(), 0); // First occurrence

        // Negative values
        let tensor = Tensor::from_vec(vec![-1.0f32, -3.0, -2.0], [3]).unwrap();
        let result = tensor.argmin(0).unwrap();
        assert_eq!(result.to_scalar::<u64>().unwrap(), 1);
    }

    #[test]
    fn test_argmin_1d_special_values() {
        // With infinity
        let tensor = Tensor::from_vec(vec![1.0f32, f32::NEG_INFINITY, 2.0], [3]).unwrap();
        let result = tensor.argmin(0).unwrap();
        assert_eq!(result.to_scalar::<u64>().unwrap(), 1);

        // With NaN (NaN behavior may be implementation-specific)
        let tensor = Tensor::from_vec(vec![1.0f32, f32::NAN, 2.0], [3]).unwrap();
        let result = tensor.argmin(0).unwrap();
        // NaN comparison behavior - typically returns first non-NaN or specific behavior
        let result_val = result.to_scalar::<u64>().unwrap();
        assert!(result_val <= 2); // Should be a valid index
    }

    #[test]
    fn test_argmin_2d_axis0() {
        let data = vec![3.0f32, 1.0, 4.0, 2.0, 5.0, 1.0, 1.0, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, [3, 3]).unwrap();
        let result = tensor.argmin(0).unwrap();
        assert_eq!(result.dims(), &[3]);
        assert_eq!(result.to_vec::<u64>().unwrap(), vec![2, 0, 1]); // [1.0, 1.0, 1.0] at indices [2, 0, 1]
    }

    #[test]
    fn test_argmin_2d_axis1() {
        let data = vec![3.0f32, 1.0, 4.0, 2.0, 5.0, 1.0, 1.0, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, [3, 3]).unwrap();
        let result = tensor.argmin(1).unwrap();
        assert_eq!(result.dims(), &[3]);
        assert_eq!(result.to_vec::<u64>().unwrap(), vec![1, 2, 0]); // Min in each row
    }

    #[test]
    fn test_argmin_2d_keepdim() {
        let data = vec![3.0f32, 1.0, 4.0, 2.0];
        let tensor = Tensor::from_vec(data, [2, 2]).unwrap();

        let result = tensor.argmin_keepdim(0).unwrap();
        assert_eq!(result.dims(), &[1, 2]);
        assert_eq!(result.to_vec2::<u64>().unwrap(), vec![vec![0, 0]]); // [1.0, 1.0] at indices [0, 0]

        let result = tensor.argmin_keepdim(1).unwrap();
        assert_eq!(result.dims(), &[2, 1]);
        assert_eq!(result.to_vec2::<u64>().unwrap(), vec![vec![1], vec![1]]); // Min in each row
    }

    #[test]
    fn test_argmin_3d() {
        let data = (0..24).map(|x| x as f32).collect::<Vec<_>>();
        let mut data_modified = data.clone();
        data_modified[5] = -1.0; // Make element at [0,1,1] the global minimum
        data_modified[18] = -0.5; // Make element at [2,0,0] second smallest

        let tensor = Tensor::from_vec(data_modified, [2, 3, 4]).unwrap();

        // Test argmin along different axes
        let result = tensor.argmin(0).unwrap(); // Along first dimension
        assert_eq!(result.dims(), &[3, 4]);

        let result = tensor.argmin(1).unwrap(); // Along second dimension
        assert_eq!(result.dims(), &[2, 4]);

        let result = tensor.argmin(2).unwrap(); // Along third dimension
        assert_eq!(result.dims(), &[2, 3]);
    }

    #[test]
    fn test_argmin_non_contiguous() {
        // Create a tensor and make it non-contiguous using permute
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, 6.0, 7.0, 8.0, 9.0];
        let tensor = Tensor::from_vec(data, [3, 3]).unwrap();
        let permuted = tensor.clone().permute([1, 0]).unwrap(); // Transpose to make non-contiguous

        let result = permuted.argmin(0).unwrap();
        assert_eq!(result.dims(), &[3]);
        // After permutation: columns become rows
        // Original: [[1,2,3], [4,0.5,6], [7,8,9]]
        // Permuted: [[1,4,7], [2,0.5,8], [3,6,9]]
        // Argmin along axis 0: [0, 1, 0] (indices of minimum in each column)
        assert_eq!(result.to_vec::<u64>().unwrap(), vec![0, 1, 0]);
    }

    #[test]
    fn test_argmin_different_dtypes() {
        // Test with i32
        let data_i32 = vec![3i32, 1, 4, 1, 5];
        let tensor_i32 = Tensor::from_vec(data_i32, [5]).unwrap();
        let result_i32 = tensor_i32.argmin(0).unwrap();
        assert_eq!(result_i32.to_scalar::<u64>().unwrap(), 1);

        // Test with f64
        let data_f64 = vec![3.0f64, 1.0, 4.0, 1.0, 5.0];
        let tensor_f64 = Tensor::from_vec(data_f64, [5]).unwrap();
        let result_f64 = tensor_f64.argmin(0).unwrap();
        assert_eq!(result_f64.to_scalar::<u64>().unwrap(), 1);

        // Test with u32
        let data_u32 = vec![3u32, 1, 4, 1, 5];
        let tensor_u32 = Tensor::from_vec(data_u32, [5]).unwrap();
        let result_u32 = tensor_u32.argmin(0).unwrap();
        assert_eq!(result_u32.to_scalar::<u64>().unwrap(), 1);
    }

    #[test]
    fn test_argmin_large_tensor() {
        let size = 1000;
        let mut data = (0..size).map(|x| x as f32).collect::<Vec<_>>();
        data[500] = -1.0; // Set minimum at index 500

        let tensor = Tensor::from_vec(data, [size]).unwrap();
        let result = tensor.argmin(0).unwrap();
        assert_eq!(result.to_scalar::<u64>().unwrap(), 500);
    }

    #[test]
    fn test_argmin_rectangular_2d() {
        // Test non-square 2D tensor
        let data = vec![5.0f32, 2.0, 8.0, 1.0, 3.0, 7.0];
        let tensor = Tensor::from_vec(data, [2, 3]).unwrap();

        let result = tensor.argmin(0).unwrap();
        assert_eq!(result.dims(), &[3]);
        assert_eq!(result.to_vec::<u64>().unwrap(), vec![1, 0, 1]); // [1.0, 2.0, 7.0] at indices [1, 0, 1]

        let result = tensor.argmin(1).unwrap();
        assert_eq!(result.dims(), &[2]);
        assert_eq!(result.to_vec::<u64>().unwrap(), vec![1, 0]); // Min in each row: [2.0, 1.0] at indices [1, 0]
    }

    #[test]
    #[should_panic(expected = "Cannot find argmin of dimension with size 0")]
    fn test_argmin_empty_dimension() {
        let tensor = Tensor::zeros::<f32>([0, 3]).unwrap();
        let _ = tensor.argmin(0).unwrap();
    }

    #[test]
    fn test_argmin_consistency_with_min() {
        // Verify that argmin returns the correct index by comparing with actual min values
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 0.5];
        let tensor = Tensor::from_vec(data.clone(), [2, 3]).unwrap();

        let argmin_result = tensor.argmin(1).unwrap();
        let argmin_indices = argmin_result.to_vec::<u64>().unwrap();

        // Manually verify the indices are correct
        for (row, &argmin_idx) in argmin_indices.iter().enumerate() {
            let row_start = row * 3;
            let row_data = &data[row_start..row_start + 3];
            let min_val = row_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            assert_eq!(row_data[argmin_idx as usize], min_val);
        }
    }

    #[test]
    fn test_argmin_3d_complex() {
        // More complex 3D test with known values
        let data = vec![
            // First 2x3 slice
            9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, // Second 2x3 slice
            3.0, 2.0, 1.0, 0.0, -1.0, -2.0,
        ];
        let tensor = Tensor::from_vec(data, [2, 2, 3]).unwrap();

        // Test argmin along axis 2 (innermost)
        let result = tensor.argmin(2).unwrap();
        assert_eq!(result.dims(), &[2, 2]);
        let result_vec = result.to_vec2::<u64>().unwrap();
        // Expected: [[2, 2], [2, 2]] (last element in each row is smallest)
        assert_eq!(result_vec, vec![vec![2, 2], vec![2, 2]]);
    }
}
