use anyhow::Result;
use half::{bf16, f16};

use crate::{global_backend, DType, Dim, OpsTrait, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    pub fn argmax<D: Dim + Clone>(&self, dim: D) -> Result<Tensor> {
        self.argmax_impl(dim, false)
    }

    pub fn argmax_keepdim<D: Dim + Clone>(&self, dim: D) -> Result<Tensor> {
        self.argmax_impl(dim, true)
    }

    pub fn argmax_impl<D: Dim + Clone>(&self, dim: D, keepdim: bool) -> Result<Tensor> {
        let dim_index = dim.to_dim(self.rank())?;
        if self.shape()[dim_index] == 0 {
            anyhow::bail!("Cannot find argmax of dimension with size 0");
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
                            let (_maxv, idx) = backend.max_vi_f32(&data[start..end]);
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
                            let (_maxv, idx) = backend.max_vi_f64(&data[start..end]);
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
                            let (_maxv, idx) = backend.max_vi_f16(&data[start..end]);
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
                            let (_maxv, idx) = backend.max_vi_bf16(&data[start..end]);
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
                            let (_maxv, idx) = backend.max_vi_i8(&data[start..end]);
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
                            let (_maxv, idx) = backend.max_vi_i16(&data[start..end]);
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
                            let (_maxv, idx) = backend.max_vi_i32(&data[start..end]);
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
                            let (_maxv, idx) = backend.max_vi_i64(&data[start..end]);
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
                            let (_maxv, idx) = backend.max_vi_u8(&data[start..end]);
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
                            let (_maxv, idx) = backend.max_vi_u16(&data[start..end]);
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
                            let (_maxv, idx) = backend.max_vi_u32(&data[start..end]);
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
                            let (_maxv, idx) = backend.max_vi_u64(&data[start..end]);
                            *item = idx;
                        }
                    });
                    Tensor::from_vec(out, new_shape)
                }
                _ => anyhow::bail!("Argmax not supported for dtype {:?}", self.dtype()),
            }
        } else {
            let new_shape = if keepdim {
                let mut shape = self.shape().as_slice().to_vec();
                shape[dim_index] = 1;
                shape
            } else {
                let mut shape = self.shape().as_slice().to_vec();
                shape.remove(dim_index);
                shape
            };

            let result_size = new_shape.iter().product();
            match self.dtype() {
                DType::Fp32 => {
                    let mut maxs = vec![f32::NEG_INFINITY; result_size];
                    let mut argmaxs = vec![0u64; result_size];
                    let mut idx_buf = vec![0; new_shape.len()];

                    for item in self.iter_with_meta::<f32>() {
                        let i = item.indices;
                        let val = *item.value;
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

                        if val > maxs[linear] {
                            maxs[linear] = val;
                            argmaxs[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmaxs, new_shape)
                }
                DType::Fp64 => {
                    let mut maxs = vec![f64::NEG_INFINITY; result_size];
                    let mut argmaxs = vec![0u64; result_size];
                    let mut idx_buf = vec![0; new_shape.len()];

                    for item in self.iter_with_meta::<f64>() {
                        let i = item.indices;
                        let val = *item.value;
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

                        if val > maxs[linear] {
                            maxs[linear] = val;
                            argmaxs[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmaxs, new_shape)
                }
                DType::Fp16 => {
                    let mut maxs = vec![f16::from_f32(f32::NEG_INFINITY); result_size];
                    let mut argmaxs = vec![0u64; result_size];
                    let mut idx_buf = vec![0; new_shape.len()];

                    for item in self.iter_with_meta::<f16>() {
                        let i = item.indices;
                        let val = *item.value;
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

                        if val > maxs[linear] {
                            maxs[linear] = val;
                            argmaxs[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmaxs, new_shape)
                }
                DType::Bf16 => {
                    let mut maxs = vec![bf16::from_f32(f32::NEG_INFINITY); result_size];
                    let mut argmaxs = vec![0u64; result_size];
                    let mut idx_buf = vec![0; new_shape.len()];

                    for item in self.iter_with_meta::<bf16>() {
                        let i = item.indices;
                        let val = *item.value;
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

                        if val > maxs[linear] {
                            maxs[linear] = val;
                            argmaxs[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmaxs, new_shape)
                }
                DType::Int8 => {
                    let mut maxs = vec![i8::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];
                    let mut idx_buf = vec![0; new_shape.len()];

                    for item in self.iter_with_meta::<i8>() {
                        let i = item.indices;
                        let val = *item.value;
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

                        if val > maxs[linear] {
                            maxs[linear] = val;
                            argmaxs[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmaxs, new_shape)
                }
                DType::Int16 => {
                    let mut maxs = vec![i16::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];
                    let mut idx_buf = vec![0; new_shape.len()];

                    for item in self.iter_with_meta::<i16>() {
                        let i = item.indices;
                        let val = *item.value;
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

                        if val > maxs[linear] {
                            maxs[linear] = val;
                            argmaxs[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmaxs, new_shape)
                }
                DType::Int32 => {
                    let mut maxs = vec![i32::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];
                    let mut idx_buf = vec![0; new_shape.len()];

                    for item in self.iter_with_meta::<i32>() {
                        let i = item.indices;
                        let val = *item.value;
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

                        if val > maxs[linear] {
                            maxs[linear] = val;
                            argmaxs[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmaxs, new_shape)
                }
                DType::Int64 => {
                    let mut maxs = vec![i64::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];
                    let mut idx_buf = vec![0; new_shape.len()];

                    for item in self.iter_with_meta::<i64>() {
                        let i = item.indices;
                        let val = *item.value;
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

                        if val > maxs[linear] {
                            maxs[linear] = val;
                            argmaxs[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmaxs, new_shape)
                }
                DType::Uint8 => {
                    let mut maxs = vec![u8::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];
                    let mut idx_buf = vec![0; new_shape.len()];

                    for item in self.iter_with_meta::<u8>() {
                        let i = item.indices;
                        let val = *item.value;
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

                        if val > maxs[linear] {
                            maxs[linear] = val;
                            argmaxs[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmaxs, new_shape)
                }
                DType::Uint16 => {
                    let mut maxs = vec![u16::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];
                    let mut idx_buf = vec![0; new_shape.len()];

                    for item in self.iter_with_meta::<u16>() {
                        let i = item.indices;
                        let val = *item.value;
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

                        if val > maxs[linear] {
                            maxs[linear] = val;
                            argmaxs[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmaxs, new_shape)
                }
                DType::Uint32 => {
                    let mut maxs = vec![u32::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];
                    let mut idx_buf = vec![0; new_shape.len()];

                    for item in self.iter_with_meta::<u32>() {
                        let i = item.indices;
                        let val = *item.value;
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

                        if val > maxs[linear] {
                            maxs[linear] = val;
                            argmaxs[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmaxs, new_shape)
                }
                DType::Uint64 => {
                    let mut maxs = vec![u64::MIN; result_size];
                    let mut argmaxs = vec![0u64; result_size];
                    let mut idx_buf = vec![0; new_shape.len()];

                    for item in self.iter_with_meta::<u64>() {
                        let i = item.indices;
                        let val = *item.value;
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

                        if val > maxs[linear] {
                            maxs[linear] = val;
                            argmaxs[linear] = i[dim_index] as u64;
                        }
                    }

                    Tensor::from_vec(argmaxs, new_shape)
                }
                _ => anyhow::bail!("Argmax not supported for dtype {:?}", self.dtype()),
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::Tensor;

    #[test]
    fn test_argmax_1d_basic() {
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0];
        let tensor = Tensor::from_vec(data, [5]).unwrap();
        let result = tensor.argmax(0).unwrap();
        assert_eq!(result.dims(), &[] as &[usize]);
        assert_eq!(result.to_scalar::<u64>().unwrap(), 4); // Maximum at index 4
    }

    #[test]
    fn test_argmax_1d_edge_cases() {
        // Single element
        let tensor = Tensor::from_vec(vec![42.0f32], [1]).unwrap();
        let result = tensor.argmax(0).unwrap();
        assert_eq!(result.to_scalar::<u64>().unwrap(), 0);

        // All same values
        let tensor = Tensor::from_vec(vec![2.0f32; 5], [5]).unwrap();
        let result = tensor.argmax(0).unwrap();
        assert_eq!(result.to_scalar::<u64>().unwrap(), 0); // First occurrence

        // Negative values
        let tensor = Tensor::from_vec(vec![-1.0f32, -3.0, -2.0], [3]).unwrap();
        let result = tensor.argmax(0).unwrap();
        assert_eq!(result.to_scalar::<u64>().unwrap(), 0); // -1.0 is maximum
    }

    #[test]
    fn test_argmax_1d_special_values() {
        // With infinity
        let tensor = Tensor::from_vec(vec![1.0f32, f32::INFINITY, 2.0], [3]).unwrap();
        let result = tensor.argmax(0).unwrap();
        assert_eq!(result.to_scalar::<u64>().unwrap(), 1);

        // With NaN (NaN behavior may be implementation-specific)
        let tensor = Tensor::from_vec(vec![1.0f32, f32::NAN, 2.0], [3]).unwrap();
        let result = tensor.argmax(0).unwrap();
        // NaN comparison behavior - typically returns first non-NaN or specific behavior
        let result_val = result.to_scalar::<u64>().unwrap();
        assert!(result_val <= 2); // Should be a valid index
    }

    #[test]
    fn test_argmax_2d_axis0() {
        let data = vec![3.0f32, 1.0, 4.0, 2.0, 5.0, 1.0, 1.0, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, [3, 3]).unwrap();
        let result = tensor.argmax(0).unwrap();
        assert_eq!(result.dims(), &[3]);
        assert_eq!(result.to_vec::<u64>().unwrap(), vec![0, 1, 0]); // [3.0, 5.0, 4.0] at indices [0, 1, 0]
    }

    #[test]
    fn test_argmax_2d_axis1() {
        let data = vec![3.0f32, 1.0, 4.0, 2.0, 5.0, 1.0, 1.0, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, [3, 3]).unwrap();
        let result = tensor.argmax(1).unwrap();
        assert_eq!(result.dims(), &[3]);
        assert_eq!(result.to_vec::<u64>().unwrap(), vec![2, 1, 2]); // Max in each row
    }

    #[test]
    fn test_argmax_2d_keepdim() {
        let data = vec![3.0f32, 1.0, 4.0, 2.0];
        let tensor = Tensor::from_vec(data, [2, 2]).unwrap();

        let result = tensor.argmax_keepdim(0).unwrap();
        assert_eq!(result.dims(), &[1, 2]);
        assert_eq!(result.to_vec2::<u64>().unwrap(), vec![vec![1, 1]]); // [4.0, 2.0] at indices [1, 1]

        let result = tensor.argmax_keepdim(1).unwrap();
        assert_eq!(result.dims(), &[2, 1]);
        assert_eq!(result.to_vec2::<u64>().unwrap(), vec![vec![0], vec![0]]); // Max in each row: [3.0, 4.0] at indices [0, 0]
    }

    #[test]
    fn test_argmax_3d() {
        let data = (0..24).map(|x| x as f32).collect::<Vec<_>>();
        let mut data_modified = data.clone();
        data_modified[5] = 100.0; // Make element at [0,1,1] the global maximum
        data_modified[18] = 50.0; // Make element at [1,2,0] second largest

        let tensor = Tensor::from_vec(data_modified, [2, 3, 4]).unwrap();

        // Test argmax along different axes
        let result = tensor.argmax(0).unwrap(); // Along first dimension
        assert_eq!(result.dims(), &[3, 4]);

        let result = tensor.argmax(1).unwrap(); // Along second dimension
        assert_eq!(result.dims(), &[2, 4]);

        let result = tensor.argmax(2).unwrap(); // Along third dimension
        assert_eq!(result.dims(), &[2, 3]);
    }

    #[test]
    fn test_argmax_non_contiguous() {
        // Create a tensor and make it non-contiguous using permute
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 9.5, 6.0, 7.0, 8.0, 9.0];
        let tensor = Tensor::from_vec(data, [3, 3]).unwrap();
        let permuted = tensor.clone().permute([1, 0]).unwrap(); // Transpose to make non-contiguous

        let result = permuted.argmax(0).unwrap();
        assert_eq!(result.dims(), &[3]);
        // After permutation: columns become rows
        // Original: [[1,2,3], [4,9.5,6], [7,8,9]]
        // Permuted: [[1,4,7], [2,9.5,8], [3,6,9]]
        // Argmax along axis 0: [2, 1, 2] (indices of maximum in each column)
        assert_eq!(result.to_vec::<u64>().unwrap(), vec![2, 1, 2]);
    }

    #[test]
    fn test_argmax_different_dtypes() {
        // Test with i32
        let data_i32 = vec![3i32, 1, 4, 1, 5];
        let tensor_i32 = Tensor::from_vec(data_i32, [5]).unwrap();
        let result_i32 = tensor_i32.argmax(0).unwrap();
        assert_eq!(result_i32.to_scalar::<u64>().unwrap(), 4);

        // Test with f64
        let data_f64 = vec![3.0f64, 1.0, 4.0, 1.0, 5.0];
        let tensor_f64 = Tensor::from_vec(data_f64, [5]).unwrap();
        let result_f64 = tensor_f64.argmax(0).unwrap();
        assert_eq!(result_f64.to_scalar::<u64>().unwrap(), 4);

        // Test with u32
        let data_u32 = vec![3u32, 1, 4, 1, 5];
        let tensor_u32 = Tensor::from_vec(data_u32, [5]).unwrap();
        let result_u32 = tensor_u32.argmax(0).unwrap();
        assert_eq!(result_u32.to_scalar::<u64>().unwrap(), 4);
    }

    #[test]
    fn test_argmax_large_tensor() {
        let size = 1000;
        let mut data = (0..size).map(|x| x as f32).collect::<Vec<_>>();
        data[500] = 2000.0; // Set maximum at index 500

        let tensor = Tensor::from_vec(data, [size]).unwrap();
        let result = tensor.argmax(0).unwrap();
        assert_eq!(result.to_scalar::<u64>().unwrap(), 500);
    }

    #[test]
    fn test_argmax_rectangular_2d() {
        // Test non-square 2D tensor
        let data = vec![5.0f32, 2.0, 8.0, 1.0, 3.0, 7.0];
        let tensor = Tensor::from_vec(data, [2, 3]).unwrap();

        let result = tensor.argmax(0).unwrap();
        assert_eq!(result.dims(), &[3]);
        assert_eq!(result.to_vec::<u64>().unwrap(), vec![0, 1, 0]); // [5.0, 3.0, 8.0] at indices [0, 1, 0]

        let result = tensor.argmax(1).unwrap();
        assert_eq!(result.dims(), &[2]);
        assert_eq!(result.to_vec::<u64>().unwrap(), vec![2, 2]); // Max in each row: [8.0, 7.0] at indices [2, 2]
    }

    #[test]
    #[should_panic(expected = "Cannot find argmax of dimension with size 0")]
    fn test_argmax_empty_dimension() {
        let tensor = Tensor::zeros::<f32>([0, 3]).unwrap();
        let _ = tensor.argmax(0).unwrap();
    }

    #[test]
    fn test_argmax_consistency_with_max() {
        // Verify that argmax returns the correct index by comparing with actual max values
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 0.5];
        let tensor = Tensor::from_vec(data.clone(), [2, 3]).unwrap();

        let argmax_result = tensor.argmax(1).unwrap();
        let argmax_indices = argmax_result.to_vec::<u64>().unwrap();

        // Manually verify the indices are correct
        for (row, &argmax_idx) in argmax_indices.iter().enumerate() {
            let row_start = row * 3;
            let row_data = &data[row_start..row_start + 3];
            let max_val = row_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            assert_eq!(row_data[argmax_idx as usize], max_val);
        }
    }

    #[test]
    fn test_argmax_3d_complex() {
        // More complex 3D test with known values
        let data = vec![
            // First 2x3 slice
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, // Second 2x3 slice
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let tensor = Tensor::from_vec(data, [2, 2, 3]).unwrap();

        // Test argmax along axis 2 (innermost)
        let result = tensor.argmax(2).unwrap();
        assert_eq!(result.dims(), &[2, 2]);
        let result_vec = result.to_vec2::<u64>().unwrap();
        // Expected: [[2, 2], [2, 2]] (last element in each row is largest)
        assert_eq!(result_vec, vec![vec![2, 2], vec![2, 2]]);
    }

    #[test]
    fn test_argmax_argmin_consistency() {
        // Test that argmax and argmin work consistently on the same data
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0];
        let tensor = Tensor::from_vec(data, [5]).unwrap();

        let argmax_result = tensor.argmax(0).unwrap();
        let argmin_result = tensor.argmin(0).unwrap();

        assert_eq!(argmax_result.to_scalar::<u64>().unwrap(), 4); // Maximum at index 4 (value 5.0)
        assert_eq!(argmin_result.to_scalar::<u64>().unwrap(), 1); // Minimum at index 1 (value 1.0)

        // Ensure they're different (unless all values are the same)
        assert_ne!(
            argmax_result.to_scalar::<u64>().unwrap(),
            argmin_result.to_scalar::<u64>().unwrap()
        );
    }
}
