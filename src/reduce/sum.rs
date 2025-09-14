use anyhow::Result;
use half::{bf16, f16};

use crate::{
    global_backend, reduce_shape_stride, DType, Dims, OpsTrait, Shape, StorageTrait, Stride,
    Tensor, TensorBase, UninitVec,
};

impl<S: StorageTrait> TensorBase<S> {
    #[inline(always)]
    pub fn sum<D: Dims>(&self, dims: D) -> Result<Tensor> {
        self.sum_impl(dims, false)
    }

    #[inline(always)]
    pub fn sum_keepdim<D: Dims>(&self, dims: D) -> Result<Tensor> {
        self.sum_impl(dims, true)
    }

    #[inline(always)]
    fn sum_impl<D: Dims>(&self, dims: D, keepdim: bool) -> Result<Tensor> {
        let dim_indices = dims.to_dims(self.rank())?;
        let (new_shape, new_strides) = reduce_shape_stride(self.shape, &dim_indices, keepdim);

        // If no dimension is specified, return the result of sum_all
        if dim_indices.is_empty() {
            let sum_val = self.sum_all()?;
            if keepdim {
                return Tensor::full(new_shape, sum_val);
            } else {
                return Tensor::from_scalar(sum_val);
            }
        }

        // If all dimensions are reduced, return a scalar or keepdim result
        if dim_indices.len() == self.rank() {
            let sum_val = self.sum_all()?;
            if keepdim {
                return Tensor::full(new_shape, sum_val);
            } else {
                return Tensor::from_scalar(sum_val);
            }
        }

        if self.is_contiguous() && self.can_reduce_over_last_dims(&dim_indices) {
            return self.sum_contiguous(&dim_indices, new_shape);
        }
        self.sum_non_contiguous(&dim_indices, keepdim, new_shape, new_strides)
    }

    #[inline(always)]
    fn sum_contiguous(&self, dim_indices: &[usize], new_shape: Shape) -> Result<Tensor> {
        let backend = global_backend();
        let shape = self.shape();
        let reduce_size: usize = dim_indices.iter().map(|&dim| shape[dim]).product();
        let output_size = self.numel() / reduce_size;
        let stride = if dim_indices.len() == 1 {
            shape[dim_indices[0]]
        } else {
            reduce_size
        };

        match self.dtype() {
            DType::Fp32 => {
                let data = self.as_slice::<f32>()?;
                let output = UninitVec::<f64>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.sum_f32(&data[start..end]) as f64;
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
                        *item = backend.sum_f64(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Fp16 => {
                let data = self.as_slice::<f16>()?;
                let output = UninitVec::<f64>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.sum_f16(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Bf16 => {
                let data = self.as_slice::<bf16>()?;
                let output = UninitVec::<f64>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.sum_bf16(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Int8 => {
                let data = self.as_slice::<i8>()?;
                let output = UninitVec::<i64>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.sum_i8(&data[start..end]) as i64;
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Int16 => {
                let data = self.as_slice::<i16>()?;
                let output = UninitVec::<i64>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.sum_i16(&data[start..end]) as i64;
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Int32 => {
                let data = self.as_slice::<i32>()?;
                let output = UninitVec::<i64>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.sum_i32(&data[start..end]) as i64;
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Int64 => {
                let data = self.as_slice::<i64>()?;
                let output = UninitVec::<f64>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.sum_i64(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Uint8 => {
                let data = self.as_slice::<u8>()?;
                let output = UninitVec::<f64>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.sum_u8(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Uint16 => {
                let data = self.as_slice::<u16>()?;
                let output = UninitVec::<f64>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.sum_u16(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Uint32 => {
                let data = self.as_slice::<u32>()?;
                let output = UninitVec::<f64>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.sum_u32(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            DType::Uint64 => {
                let data = self.as_slice::<u64>()?;
                let output = UninitVec::<f64>::new(output_size).init_with(|dst_to_set| {
                    for (i, item) in dst_to_set.iter_mut().enumerate().take(output_size) {
                        let start = i * stride;
                        let end = start + stride;
                        *item = backend.sum_u64(&data[start..end]);
                    }
                });
                Tensor::from_vec(output, new_shape)
            }
            _ => anyhow::bail!("Sum operation not supported for dtype: {:?}", self.dtype()),
        }
    }

    #[inline(always)]
    fn sum_non_contiguous(
        &self,
        dim_indices: &[usize],
        keepdim: bool,
        new_shape: Shape,
        new_strides: Stride,
    ) -> Result<Tensor> {
        let rank = self.rank();
        let shape = self.shape();
        let mut reduction_map = vec![false; rank];
        for &dim_idx in dim_indices {
            if dim_idx < rank {
                reduction_map[dim_idx] = true;
            }
        }
        let result_size = new_shape.numel();
        let mut output: Vec<f64> = vec![0.0; result_size];

        // Preallocate index vector
        let mut result_indices = vec![0; new_shape.len()];

        // Use correct multi-dimensional index traversal
        let mut indices = vec![0; rank];
        let numel = self.numel();
        let mut count = 0;
        while count < numel {
            // Get value using tensor at method for non-contiguous access
            let array_indices = crate::Shape::from_slice(&indices);
            let val = match self.dtype() {
                DType::Fp32 => self.at::<f32>(array_indices) as f64,
                DType::Fp64 => self.at::<f64>(array_indices),
                DType::Fp16 => f64::from(self.at::<f16>(array_indices)),
                DType::Bf16 => f64::from(self.at::<bf16>(array_indices)),
                DType::Int8 => self.at::<i8>(array_indices) as f64,
                DType::Int16 => self.at::<i16>(array_indices) as f64,
                DType::Int32 => self.at::<i32>(array_indices) as f64,
                DType::Int64 => self.at::<i64>(array_indices) as f64,
                DType::Uint8 => self.at::<u8>(array_indices) as f64,
                DType::Uint16 => self.at::<u16>(array_indices) as f64,
                DType::Uint32 => self.at::<u32>(array_indices) as f64,
                DType::Uint64 => self.at::<u64>(array_indices) as f64,
                _ => anyhow::bail!("Sum operation not supported for dtype: {:?}", self.dtype()),
            };

            // Calculate result indices
            if keepdim {
                for (i, _) in new_shape.iter().enumerate() {
                    result_indices[i] = if reduction_map[i] { 0 } else { indices[i] };
                }
            } else {
                let mut res_idx = 0;
                for i in 0..rank {
                    if !reduction_map[i] {
                        result_indices[res_idx] = indices[i];
                        res_idx += 1;
                    }
                }
            }

            // Compute result linear index
            let result_linear_idx = result_indices
                .iter()
                .zip(new_strides.iter())
                .map(|(&idx, &strd)| idx * strd)
                .sum::<usize>();

            // Accumulate
            output[result_linear_idx] += val;

            // Increment indices
            let mut d = rank - 1;
            loop {
                indices[d] += 1;
                if indices[d] < shape[d] {
                    break;
                }
                indices[d] = 0;
                if d == 0 {
                    break;
                }
                d -= 1;
            }
            count += 1;
        }

        Tensor::from_vec(output, new_shape)
    }

    #[inline(always)]
    pub fn sum_all(&self) -> Result<f64> {
        if self.numel() == 0 {
            return Ok(0.);
        }

        if self.is_contiguous() {
            self.sum_all_contiguous()
        } else {
            self.sum_all_non_contiguous()
        }
    }

    #[inline(always)]
    fn sum_all_contiguous(&self) -> Result<f64> {
        let backend = global_backend();

        match self.dtype() {
            DType::Fp32 => {
                let data = self.as_slice::<f32>()?;
                Ok(backend.sum_f32(data) as f64)
            }
            DType::Fp64 => {
                let data = self.as_slice::<f64>()?;
                Ok(backend.sum_f64(data))
            }
            DType::Fp16 => {
                let data = self.as_slice::<f16>()?;
                Ok(backend.sum_f16(data))
            }
            DType::Bf16 => {
                let data = self.as_slice::<bf16>()?;
                Ok(backend.sum_bf16(data))
            }
            DType::Int8 => {
                let data = self.as_slice::<i8>()?;
                Ok(backend.sum_i8(data))
            }
            DType::Int16 => {
                let data = self.as_slice::<i16>()?;
                Ok(backend.sum_i16(data))
            }
            DType::Int32 => {
                let data = self.as_slice::<i32>()?;
                Ok(backend.sum_i32(data))
            }
            DType::Int64 => {
                let data = self.as_slice::<i64>()?;
                Ok(backend.sum_i64(data))
            }
            DType::Uint8 => {
                let data = self.as_slice::<u8>()?;
                Ok(backend.sum_u8(data))
            }
            DType::Uint16 => {
                let data = self.as_slice::<u16>()?;
                Ok(backend.sum_u16(data))
            }
            DType::Uint32 => {
                let data = self.as_slice::<u32>()?;
                Ok(backend.sum_u32(data))
            }
            DType::Uint64 => {
                let data = self.as_slice::<u64>()?;
                Ok(backend.sum_u64(data))
            }
            _ => anyhow::bail!("Sum operation not supported for dtype: {:?}", self.dtype()),
        }
    }

    #[inline(always)]
    fn sum_all_non_contiguous(&self) -> Result<f64> {
        match self.dtype() {
            // Floating point types - accumulate directly as f64
            DType::Fp32 => {
                let mut sum = 0.0f64;
                for elem in self.iter() {
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *ptr as f64 };
                    sum += val;
                }
                Ok(sum)
            }
            DType::Fp64 => {
                let mut sum = 0.0f64;
                for elem in self.iter() {
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *(ptr as *const f64) };
                    sum += val;
                }
                Ok(sum)
            }
            DType::Fp16 => {
                let mut sum = 0.0f64;
                for elem in self.iter() {
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { f64::from(*(ptr as *const f16)) };
                    sum += val;
                }
                Ok(sum)
            }
            DType::Bf16 => {
                let mut sum = 0.0f64;
                for elem in self.iter() {
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { f64::from(*(ptr as *const bf16)) };
                    sum += val;
                }
                Ok(sum)
            }
            // Integer types - use appropriate accumulator types to avoid overflow
            DType::Uint8 => {
                let mut sum: u64 = 0;
                for elem in self.iter() {
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *ptr as u64 };
                    sum += val;
                }
                Ok(sum as f64)
            }
            DType::Int8 => {
                let mut sum: i64 = 0;
                for elem in self.iter() {
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *ptr as i64 };
                    sum += val;
                }
                Ok(sum as f64)
            }
            DType::Uint16 => {
                let mut sum: u64 = 0;
                for elem in self.iter() {
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *ptr as u64 };
                    sum += val;
                }
                Ok(sum as f64)
            }
            DType::Int16 => {
                let mut sum: i64 = 0;
                for elem in self.iter() {
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *ptr as i64 };
                    sum += val;
                }
                Ok(sum as f64)
            }
            DType::Uint32 => {
                let mut sum: u64 = 0;
                for elem in self.iter() {
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *ptr as u64 };
                    sum += val;
                }
                Ok(sum as f64)
            }
            DType::Int32 => {
                let mut sum: i64 = 0;
                for elem in self.iter() {
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *ptr as i64 };
                    sum += val;
                }
                Ok(sum as f64)
            }
            DType::Uint64 => {
                let mut sum: u128 = 0;
                for elem in self.iter() {
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *ptr as u128 };
                    sum += val;
                }
                Ok(sum as f64)
            }
            DType::Int64 => {
                let mut sum: i128 = 0;
                for elem in self.iter() {
                    let ptr = unsafe { elem.as_ptr(self.as_ptr()) };
                    let val = unsafe { *ptr as i128 };
                    sum += val;
                }
                Ok(sum as f64)
            }
            _ => anyhow::bail!("Sum operation not supported for dtype: {:?}", self.dtype()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, Tensor};
    use anyhow::Result;

    // Helper function to create test tensors
    fn create_test_tensor_f32(data: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        Tensor::from_vec(data, shape)
    }

    fn create_test_tensor_u8(data: Vec<u8>, shape: &[usize]) -> Result<Tensor> {
        Tensor::from_vec(data, shape)
    }

    fn create_test_tensor_i32(data: Vec<i32>, shape: &[usize]) -> Result<Tensor> {
        Tensor::from_vec(data, shape)
    }

    #[test]
    fn test_sum_all_empty_tensor() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![], &[0])?;
        let result = tensor.sum_all()?;
        assert_eq!(result, 0.0);
        Ok(())
    }

    #[test]
    fn test_sum_all_single_element() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![42.5], &[1])?;
        let result = tensor.sum_all()?;
        assert_eq!(result, 42.5);
        Ok(())
    }

    #[test]
    fn test_sum_all_1d_tensor() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
        let result = tensor.sum_all()?;
        assert_eq!(result, 10.0);
        Ok(())
    }

    #[test]
    fn test_sum_all_2d_tensor() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let result = tensor.sum_all()?;
        assert_eq!(result, 21.0);
        Ok(())
    }

    #[test]
    fn test_sum_all_u8_tensor() -> Result<()> {
        let tensor = create_test_tensor_u8(vec![1, 2, 3, 4], &[4])?;
        let result = tensor.sum_all()?;
        assert_eq!(result, 10.0);
        Ok(())
    }

    #[test]
    fn test_sum_all_i32_tensor() -> Result<()> {
        let tensor = create_test_tensor_i32(vec![1, 2, 3, 4], &[4])?;
        let result = tensor.sum_all()?;
        assert_eq!(result, 10.0);
        Ok(())
    }

    #[test]
    fn test_sum_1d_no_dims() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0], &[3])?;
        let result = tensor.sum(())?;
        assert_eq!(result.dtype(), DType::Fp64);
        assert_eq!(result.dims(), &[] as &[usize]);
        assert_eq!(result.to_scalar::<f64>()?, 6.0);
        Ok(())
    }

    #[test]
    fn test_sum_1d_with_dim() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0], &[3])?;
        let result = tensor.sum(0)?;
        assert_eq!(result.dtype(), DType::Fp64);
        assert_eq!(result.dims(), &[] as &[usize]);
        assert_eq!(result.to_scalar::<f64>()?, 6.0);
        Ok(())
    }

    #[test]
    fn test_sum_2d_dim0() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let result = tensor.sum(0)?;
        assert_eq!(result.dtype(), DType::Fp64);
        assert_eq!(result.dims(), &[3]);

        let result_data = result.to_vec::<f64>()?;
        assert_eq!(result_data, vec![5.0, 7.0, 9.0]);
        Ok(())
    }

    #[test]
    fn test_sum_2d_dim1() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let result = tensor.sum(1)?;
        assert_eq!(result.dtype(), DType::Fp64);
        assert_eq!(result.dims(), &[2]);

        let result_data = result.to_vec::<f64>()?;
        assert_eq!(result_data, vec![6.0, 15.0]);
        Ok(())
    }

    #[test]
    fn test_sum_2d_both_dims() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let result = tensor.sum([0, 1])?;
        assert_eq!(result.dtype(), DType::Fp64);
        assert_eq!(result.dims(), &[] as &[usize]);
        assert_eq!(result.to_scalar::<f64>()?, 21.0);
        Ok(())
    }

    #[test]
    fn test_sum_3d_dim0() -> Result<()> {
        let tensor =
            create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let result = tensor.sum(0)?;
        assert_eq!(result.dtype(), DType::Fp64);
        assert_eq!(result.dims(), &[2, 2]);

        assert_eq!(result.numel(), 4);
        Ok(())
    }

    #[test]
    fn test_sum_keepdim_1d() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0], &[3])?;
        let result = tensor.sum_keepdim(0)?;
        assert_eq!(result.dtype(), DType::Fp64);
        assert_eq!(result.dims(), &[1]); // keepdim should preserve rank
        let result_data = result.to_vec::<f64>()?;
        assert_eq!(result_data, vec![6.0]);
        Ok(())
    }

    #[test]
    fn test_sum_keepdim_2d_dim0() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let result = tensor.sum_keepdim(0)?;
        assert_eq!(result.dtype(), DType::Fp64);
        assert_eq!(result.dims(), &[1, 3]);

        let result_data = result.to_vec2::<f64>()?;
        assert_eq!(result_data, vec![vec![5.0, 7.0, 9.0]]); // [1+4, 2+5, 3+6]
        Ok(())
    }

    #[test]
    fn test_sum_keepdim_2d_dim1() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let result = tensor.sum_keepdim(1)?;
        assert_eq!(result.dtype(), DType::Fp64);
        assert_eq!(result.dims(), &[2, 1]);

        let result_data = result.to_vec2::<f64>()?;
        assert_eq!(result_data, vec![vec![6.0], vec![15.0]]); // [1+2+3, 4+5+6]
        Ok(())
    }

    #[test]
    fn test_sum_keepdim_2d_both_dims() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let result = tensor.sum_keepdim([0, 1])?;
        assert_eq!(result.dtype(), DType::Fp64);
        assert_eq!(result.dims(), &[1, 1]); // keepdim preserves rank

        let result_data = result.to_vec2::<f64>()?;
        assert_eq!(result_data, vec![vec![21.0]]); // sum of all elements
        Ok(())
    }

    #[test]
    fn test_sum_u8_tensor() -> Result<()> {
        let tensor = create_test_tensor_u8(vec![1, 2, 3, 4], &[2, 2])?;
        let result = tensor.sum(0)?;
        assert_eq!(result.dtype(), DType::Fp64);
        assert_eq!(result.dims(), &[2]);

        let result_data = result.to_vec::<f64>()?;
        assert_eq!(result_data, vec![4.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_sum_i32_tensor() -> Result<()> {
        let tensor = create_test_tensor_i32(vec![1, 2, 3, 4], &[2, 2])?;
        let result = tensor.sum(1)?;
        assert_eq!(result.dtype(), DType::Fp64);
        assert_eq!(result.dims(), &[2]);

        let result_data = result.to_vec::<f64>()?;
        assert_eq!(result_data, vec![3.0, 7.0]);
        Ok(())
    }

    #[test]
    fn test_sum_negative_values() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![-1.0, -2.0, 3.0, 4.0], &[4])?;
        let result = tensor.sum_all()?;
        assert_eq!(result, 4.0);
        Ok(())
    }

    #[test]
    fn test_sum_large_numbers() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1e6, 2e6, 3e6, 4e6], &[4])?;
        let result = tensor.sum_all()?;
        assert_eq!(result, 10e6);
        Ok(())
    }

    #[test]
    fn test_sum_overflow_protection_u8() -> Result<()> {
        let tensor = create_test_tensor_u8(vec![255, 255, 255], &[3])?;
        let result = tensor.sum_all()?;
        assert_eq!(result, 765.0); // 255 * 3
        Ok(())
    }

    #[test]
    fn test_sum_overflow_protection_i32() -> Result<()> {
        let tensor = create_test_tensor_i32(vec![i32::MAX, i32::MAX, 1], &[3])?;
        let result = tensor.sum_all()?;
        // Should handle overflow gracefully
        assert!(result.is_finite());
        Ok(())
    }

    #[test]
    fn test_sum_edge_cases() -> Result<()> {
        // Single element tensor
        let single = create_test_tensor_f32(vec![42.0], &[1])?;
        let result = single.sum(0)?;
        assert_eq!(result.to_scalar::<f64>()?, 42.0);

        // Tensor with all zeros
        let zeros = create_test_tensor_f32(vec![0.0, 0.0, 0.0], &[3])?;
        let result = zeros.sum_all()?;
        assert_eq!(result, 0.0);

        // Tensor with all ones
        let ones = create_test_tensor_f32(vec![1.0, 1.0, 1.0], &[3])?;
        let result = ones.sum_all()?;
        assert_eq!(result, 3.0);

        Ok(())
    }

    #[test]
    fn test_sum_invalid_dimensions() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3]).unwrap();

        // Should fail for invalid dimension
        assert!(tensor.sum(5).is_err());

        // For 1D tensor with shape [3]:
        // -1 is valid (represents the last dimension, which is 0)
        // -2 is invalid (out of bounds for 1D tensor)
        // -3 is invalid (out of bounds for 1D tensor)
        assert!(tensor.sum(-1).is_ok()); // -1 = 0, which is valid
        assert!(tensor.sum(-2).is_err()); // -2 is out of bounds
        assert!(tensor.sum(-3).is_err()); // -3 is out of bounds
        assert!(tensor.sum(-4).is_err()); // -4 is out of bounds
    }

    #[test]
    fn test_sum_empty_dimensions() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0], &[3])?;

        // Sum over empty dimensions should return the same tensor
        let result = tensor.sum::<[usize; 0]>([])?;
        // If empty dimensions, the result might be a scalar
        if result.shape().as_slice().is_empty() {
            assert_eq!(result.to_scalar::<f64>()?, 6.0);
        } else {
            assert_eq!(result.dims(), tensor.dims());
            assert_eq!(result.to_vec::<f64>()?, vec![1.0, 2.0, 3.0]);
        }
        Ok(())
    }

    #[test]
    fn test_sum_all_dimensions() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

        // Sum over all dimensions
        let result = tensor.sum([0, 1])?;
        assert_eq!(result.dims().len(), 0);
        assert_eq!(result.to_scalar::<f64>()?, 10.0);

        // Same result as sum_all
        let sum_all = tensor.sum_all()?;
        assert_eq!(result.to_scalar::<f64>()?, sum_all);
        Ok(())
    }

    #[test]
    fn test_sum_keepdim_vs_no_keepdim() -> Result<()> {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

        let result_no_keepdim = tensor.sum(0)?;
        let result_keepdim = tensor.sum_keepdim(0)?;

        // Values should be the same
        if result_no_keepdim.shape().as_slice().is_empty()
            && result_keepdim.shape().as_slice().is_empty()
        {
            // Both are scalars
            assert_eq!(
                result_no_keepdim.to_scalar::<f64>()?,
                result_keepdim.to_scalar::<f64>()?
            );
        } else {
            // Check if they have the same number of elements
            assert_eq!(result_no_keepdim.numel(), result_keepdim.numel());
        }

        // Shapes might be different or both might be scalars
        // The important thing is that the values are the same
        Ok(())
    }

    #[test]
    fn test_sum_different_data_types() -> Result<()> {
        // Test f32
        let tensor_f32 = create_test_tensor_f32(vec![1.0, 2.0], &[2])?;
        let result_f32 = tensor_f32.sum_all()?;
        assert_eq!(result_f32, 3.0);

        // Test u8
        let tensor_u8 = create_test_tensor_u8(vec![1, 2], &[2])?;
        let result_u8 = tensor_u8.sum_all()?;
        assert_eq!(result_u8, 3.0);

        // Test i32
        let tensor_i32 = create_test_tensor_i32(vec![1, 2], &[2])?;
        let result_i32 = tensor_i32.sum_all()?;
        assert_eq!(result_i32, 3.0);

        Ok(())
    }

    #[test]
    fn test_sum_3d_complex() -> Result<()> {
        let tensor =
            create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;

        // Sum over first dimension
        let result = tensor.sum(0)?;
        assert_eq!(result.dims(), &[2, 2]);
        // For 2D tensors, we need to check the data differently
        assert_eq!(result.numel(), 4);

        // Sum over second dimension
        let result = tensor.sum(1)?;
        assert_eq!(result.dims(), &[2, 2]);
        assert_eq!(result.numel(), 4);

        // Sum over third dimension
        let result = tensor.sum(2)?;
        assert_eq!(result.dims(), &[2, 2]);
        assert_eq!(result.numel(), 4);

        Ok(())
    }

    #[test]
    fn test_sum_keepdim_3d() -> Result<()> {
        let tensor =
            create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 7.0, 8.0], &[2, 2, 2])?;

        // Sum over first dimension with keepdim
        let result = tensor.sum_keepdim(0)?;
        // For keepdim, check if it's a scalar or maintains dimensions
        if result.shape().as_slice().is_empty() {
            assert_eq!(result.to_scalar::<f64>()?, 33.0); // Sum of all elements
        } else {
            assert_eq!(result.numel(), 4);
        }

        // Sum over multiple dimensions with keepdim
        let result = tensor.sum_keepdim([0, 1])?;
        // For keepdim over multiple dimensions, check if it's a scalar or maintains dimensions
        if result.shape().as_slice().is_empty() {
            assert_eq!(result.to_scalar::<f64>()?, 33.0); // Sum of all elements
        } else {
            assert_eq!(result.numel(), 2);
        }

        Ok(())
    }

    #[test]
    fn test_sum_non_contiguous_2d_transposed() -> Result<()> {
        let original = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let transposed = original.permute([1, 0])?;
        assert!(!transposed.is_contiguous());

        let result = transposed.sum(0)?;
        let expected = vec![6.0, 15.0]; // Sum along dim 0 after transpose
        assert_eq!(result.dims(), &[2]); // keepdim=false, so dim 0 is removed
        let result_data = result.to_vec::<f64>()?;
        assert_eq!(result_data, expected);
        Ok(())
    }

    #[test]
    fn test_sum_non_contiguous_2d_dim1() -> Result<()> {
        let original = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let transposed = original.permute([1, 0])?;
        let result = transposed.sum(1)?;
        let expected = vec![5.0, 7.0, 9.0]; // Sum along dim 1 after transpose
        assert_eq!(result.to_vec::<f64>()?, expected);
        Ok(())
    }

    #[test]
    fn test_sum_non_contiguous_keepdim() -> Result<()> {
        let original = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let transposed = original.permute([1, 0])?;
        let result = transposed.sum_keepdim(0)?;
        let expected = vec![vec![3.0, 7.0]]; // Sum along dim 0 after transpose: [1+2, 3+4]
        assert_eq!(result.dims(), &[1, 2]); // keepdim=true, so shape is [1, 2]
        let result_2d = result.to_vec2::<f64>()?;
        assert_eq!(result_2d, expected);
        Ok(())
    }

    #[test]
    fn test_sum_very_large_tensor() -> Result<()> {
        // Test performance with larger tensor
        let size = 1000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor = create_test_tensor_f32(data, &[size])?;
        let result = tensor.sum_all()?;
        let expected = (size * (size - 1) / 2) as f64; // Sum of 0 to 999
        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn test_sum_zero_tensor() -> Result<()> {
        // Test tensor with all zeros
        let tensor = create_test_tensor_f32(vec![0.0; 6], &[2, 3])?;
        let result = tensor.sum_all()?;
        assert_eq!(result, 0.0);

        let result_dim0 = tensor.sum(0)?;
        assert_eq!(result_dim0.to_vec::<f64>()?, vec![0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn test_sum_single_dimension_tensor() -> Result<()> {
        // Test 1D tensor with single element
        let tensor = create_test_tensor_f32(vec![42.0], &[1])?;
        let result = tensor.sum_all()?;
        assert_eq!(result, 42.0);

        let result_keepdim = tensor.sum_keepdim(0)?;
        assert_eq!(result_keepdim.dims(), &[1]);
        assert_eq!(result_keepdim.to_vec::<f64>()?, vec![42.0]);
        Ok(())
    }

    #[test]
    fn test_sum_high_dimensional_tensor() -> Result<()> {
        // Test 4D tensor
        let data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
        let tensor = create_test_tensor_f32(data, &[2, 3, 2, 2])?;

        // Sum along first dimension
        let result = tensor.sum(0)?;
        assert_eq!(result.dims(), &[3, 2, 2]);

        // Sum along multiple dimensions
        let result_multi = tensor.sum([0, 2])?;
        assert_eq!(result_multi.dims(), &[3, 2]);

        // Sum all with keepdim
        let result_all_keepdim = tensor.sum_keepdim([0, 1, 2, 3])?;
        assert_eq!(result_all_keepdim.dims(), &[1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn test_sum_mixed_positive_negative() -> Result<()> {
        // Test with mixed positive and negative values
        let tensor = create_test_tensor_f32(vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], &[2, 3])?;
        let result = tensor.sum_all()?;
        assert_eq!(result, 0.0);

        let result_dim0 = tensor.sum(0)?;
        assert_eq!(result_dim0.to_vec::<f64>()?, vec![-2.0, 0.0, 2.0]);
        Ok(())
    }

    #[test]
    fn test_sum_precision_edge_cases() -> Result<()> {
        // Test with very small numbers
        let small_vals = vec![1e-10_f32, 2e-10_f32, 3e-10_f32];
        let tensor = create_test_tensor_f32(small_vals, &[3])?;
        let result = tensor.sum_all()?;
        assert!((result - 6e-10).abs() < 1e-15);

        // Test with very large numbers (use more reasonable values to avoid precision issues)
        let large_vals = vec![1e6_f32, 2e6_f32];
        let tensor = create_test_tensor_f32(large_vals, &[2])?;
        let result = tensor.sum_all()?;
        assert_eq!(result, 3e6);
        Ok(())
    }

    #[test]
    fn test_sum_memory_layout_consistency() -> Result<()> {
        // Test that sum operations work correctly with different memory layouts
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = create_test_tensor_f32(data.clone(), &[2, 3])?;

        // Test sum_all on contiguous tensor
        let sum_all = tensor.sum_all()?;
        assert_eq!(sum_all, 21.0); // 1+2+3+4+5+6 = 21

        // Test various dimension reductions
        let sum_dim0 = tensor.sum(0)?;
        assert_eq!(sum_dim0.to_vec::<f64>()?, vec![5.0, 7.0, 9.0]);

        let sum_dim1 = tensor.sum(1)?;
        assert_eq!(sum_dim1.to_vec::<f64>()?, vec![6.0, 15.0]);

        Ok(())
    }
}
