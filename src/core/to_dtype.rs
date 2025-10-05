use anyhow::Result;
use half::{bf16, f16};

use crate::{DType, Storage, StorageTrait, Tensor, TensorBase, TensorElement};

impl<S: StorageTrait> TensorBase<S> {
    /// Convert tensor to a different data type
    ///
    /// This method creates a new tensor with the specified data type,
    /// performing element-wise type conversion. The operation is optimized
    /// for both contiguous and non-contiguous tensors.
    ///
    /// # Type Parameters
    /// * `T` - Target tensor element type implementing `TensorElement`
    ///
    /// # Returns
    /// * `Result<Tensor>` - New tensor with converted data type
    ///
    /// # Performance
    /// - Fast path for same-type conversion (zero-cost clone)
    /// - Optimized memory layout for contiguous tensors
    /// - Efficient strided access for non-contiguous tensors
    ///
    /// # Example
    /// ```rust
    /// use slsl::Tensor;
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let tensor_f32 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3])?;
    /// let tensor_i32 = tensor_f32.to_dtype::<i32>()?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn to_dtype<T: TensorElement>(&self) -> Result<Tensor> {
        // Fast path: if target type matches current type, clone the tensor
        if T::DTYPE == self.dtype {
            return self.to_contiguous();
        }

        // Handle empty tensors
        if self.numel() == 0 {
            let empty_storage = Storage::new(0, std::mem::align_of::<T>())?;
            return Ok(Tensor {
                storage: empty_storage,
                ptr: std::ptr::NonNull::dangling(),
                dtype: T::DTYPE,
                shape: self.shape,
                strides: Self::compute_contiguous_strides(&self.shape),
                offset_bytes: 0,
            });
        }

        // Dispatch based on source data type
        match self.dtype {
            DType::Fp32 => self.convert_from_typed::<f32, T>(),
            DType::Fp64 => self.convert_from_typed::<f64, T>(),
            DType::Fp16 => self.convert_from_typed::<f16, T>(),
            DType::Bf16 => self.convert_from_typed::<bf16, T>(),
            DType::Int8 => self.convert_from_typed::<i8, T>(),
            DType::Int16 => self.convert_from_typed::<i16, T>(),
            DType::Int32 => self.convert_from_typed::<i32, T>(),
            DType::Int64 => self.convert_from_typed::<i64, T>(),
            DType::Uint8 => self.convert_from_typed::<u8, T>(),
            DType::Uint16 => self.convert_from_typed::<u16, T>(),
            DType::Uint32 => self.convert_from_typed::<u32, T>(),
            DType::Uint64 => self.convert_from_typed::<u64, T>(),
            DType::Bool => self.convert_from_typed::<bool, T>(),
            _ => anyhow::bail!("Unsupported source dtype: {:?}", self.dtype),
        }
    }

    /// Type-specific conversion implementation
    ///
    /// This method handles the actual data conversion from source type `S` to target type `T`.
    /// It optimizes for both contiguous and non-contiguous tensor layouts.
    #[inline(always)]
    fn convert_from_typed<Src: TensorElement + Copy, T: TensorElement + Copy>(
        &self,
    ) -> Result<Tensor> {
        let total_bytes = self.numel() * std::mem::size_of::<T>();
        let alignment = std::mem::align_of::<T>();
        let new_storage = Storage::new(total_bytes, alignment)?;

        // Get typed pointers
        let dst_ptr = new_storage.ptr().as_ptr() as *mut T;

        // Create new tensor with contiguous layout
        let new_tensor = Tensor {
            storage: new_storage,
            ptr: unsafe { std::ptr::NonNull::new_unchecked(dst_ptr as *mut u8) },
            dtype: T::DTYPE,
            shape: self.shape,
            strides: Self::compute_contiguous_strides(&self.shape),
            offset_bytes: 0,
        };

        unsafe {
            if self.is_contiguous() {
                self.convert_contiguous_data::<Src, T>(dst_ptr)?
            } else {
                self.convert_strided_data::<Src, T>(dst_ptr)?
            }
        }

        Ok(new_tensor)
    }

    /// Optimized conversion for small contiguous data (< 4K elements)
    #[inline(always)]
    unsafe fn convert_small_contiguous<Src: TensorElement + Copy, T: TensorElement + Copy>(
        src_ptr: *const Src,
        dst_ptr: *mut T,
        numel: usize,
    ) -> Result<()> {
        // For small data, use 8-element unrolling to reduce overhead
        const UNROLL: usize = 8;
        let chunks_end = (numel / UNROLL) * UNROLL;

        for chunk in 0..(numel / UNROLL) {
            let i = chunk * UNROLL;
            *dst_ptr.add(i) = T::from_f32((*src_ptr.add(i)).to_f32());
            *dst_ptr.add(i + 1) = T::from_f32((*src_ptr.add(i + 1)).to_f32());
            *dst_ptr.add(i + 2) = T::from_f32((*src_ptr.add(i + 2)).to_f32());
            *dst_ptr.add(i + 3) = T::from_f32((*src_ptr.add(i + 3)).to_f32());
            *dst_ptr.add(i + 4) = T::from_f32((*src_ptr.add(i + 4)).to_f32());
            *dst_ptr.add(i + 5) = T::from_f32((*src_ptr.add(i + 5)).to_f32());
            *dst_ptr.add(i + 6) = T::from_f32((*src_ptr.add(i + 6)).to_f32());
            *dst_ptr.add(i + 7) = T::from_f32((*src_ptr.add(i + 7)).to_f32());
        }

        // Process remainder
        for i in chunks_end..numel {
            *dst_ptr.add(i) = T::from_f32((*src_ptr.add(i)).to_f32());
        }

        Ok(())
    }

    /// Optimized conversion for large contiguous data (>= 64K elements)
    #[inline(always)]
    unsafe fn convert_large_contiguous<Src: TensorElement + Copy, T: TensorElement + Copy>(
        src_ptr: *const Src,
        dst_ptr: *mut T,
        numel: usize,
    ) -> Result<()> {
        use std::any::TypeId;

        // Special fast path for f32 <-> f64 (direct cast, no intermediate conversion)
        let src_type = TypeId::of::<Src>();
        let dst_type = TypeId::of::<T>();

        // Special handling for common direct conversions (no intermediate f32)
        if src_type == TypeId::of::<f32>() && dst_type == TypeId::of::<f64>() {
            let src = src_ptr as *const f32;
            let dst = dst_ptr as *mut f64;
            for i in 0..numel {
                *dst.add(i) = *src.add(i) as f64;
            }
            return Ok(());
        }

        if src_type == TypeId::of::<f64>() && dst_type == TypeId::of::<f32>() {
            let src = src_ptr as *const f64;
            let dst = dst_ptr as *mut f32;
            for i in 0..numel {
                *dst.add(i) = *src.add(i) as f32;
            }
            return Ok(());
        }

        if src_type == TypeId::of::<u8>() && dst_type == TypeId::of::<f32>() {
            let src = src_ptr as *const u8;
            let dst = dst_ptr as *mut f32;
            for i in 0..numel {
                *dst.add(i) = *src.add(i) as f32;
            }
            return Ok(());
        }

        if src_type == TypeId::of::<f32>() && dst_type == TypeId::of::<u8>() {
            let src = src_ptr as *const f32;
            let dst = dst_ptr as *mut u8;
            for i in 0..numel {
                *dst.add(i) = *src.add(i) as u8;
            }
            return Ok(());
        }

        if src_type == TypeId::of::<i64>() && dst_type == TypeId::of::<f32>() {
            let src = src_ptr as *const i64;
            let dst = dst_ptr as *mut f32;
            for i in 0..numel {
                *dst.add(i) = *src.add(i) as f32;
            }
            return Ok(());
        }

        if src_type == TypeId::of::<f32>() && dst_type == TypeId::of::<i64>() {
            let src = src_ptr as *const f32;
            let dst = dst_ptr as *mut i64;
            // Unroll for better performance
            const UNROLL: usize = 8;
            let chunks = numel / UNROLL;
            let chunks_end = chunks * UNROLL;

            for chunk in 0..chunks {
                let i = chunk * UNROLL;
                *dst.add(i) = *src.add(i) as i64;
                *dst.add(i + 1) = *src.add(i + 1) as i64;
                *dst.add(i + 2) = *src.add(i + 2) as i64;
                *dst.add(i + 3) = *src.add(i + 3) as i64;
                *dst.add(i + 4) = *src.add(i + 4) as i64;
                *dst.add(i + 5) = *src.add(i + 5) as i64;
                *dst.add(i + 6) = *src.add(i + 6) as i64;
                *dst.add(i + 7) = *src.add(i + 7) as i64;
            }

            for i in chunks_end..numel {
                *dst.add(i) = *src.add(i) as i64;
            }
            return Ok(());
        }

        if src_type == TypeId::of::<i64>() && dst_type == TypeId::of::<f32>() {
            let src = src_ptr as *const i64;
            let dst = dst_ptr as *mut f32;
            // Unroll for better performance
            const UNROLL: usize = 8;
            let chunks = numel / UNROLL;
            let chunks_end = chunks * UNROLL;

            for chunk in 0..chunks {
                let i = chunk * UNROLL;
                *dst.add(i) = *src.add(i) as f32;
                *dst.add(i + 1) = *src.add(i + 1) as f32;
                *dst.add(i + 2) = *src.add(i + 2) as f32;
                *dst.add(i + 3) = *src.add(i + 3) as f32;
                *dst.add(i + 4) = *src.add(i + 4) as f32;
                *dst.add(i + 5) = *src.add(i + 5) as f32;
                *dst.add(i + 6) = *src.add(i + 6) as f32;
                *dst.add(i + 7) = *src.add(i + 7) as f32;
            }

            for i in chunks_end..numel {
                *dst.add(i) = *src.add(i) as f32;
            }
            return Ok(());
        }

        // For large data, use 32-element unrolling with prefetching
        const UNROLL: usize = 32;
        let chunks_end = (numel / UNROLL) * UNROLL;

        let mut i = 0;
        while i < chunks_end {
            // Prefetch next cache lines
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                std::arch::x86_64::_mm_prefetch(
                    src_ptr.add(i + 128) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }

            // Unroll 32 elements for maximum throughput
            for offset in 0..32 {
                *dst_ptr.add(i + offset) = T::from_f32((*src_ptr.add(i + offset)).to_f32());
            }

            i += UNROLL;
        }

        // Process remainder
        for i in chunks_end..numel {
            *dst_ptr.add(i) = T::from_f32((*src_ptr.add(i)).to_f32());
        }

        Ok(())
    }

    /// Convert contiguous tensor data with optimized memory access (medium-sized data)
    #[inline(always)]
    unsafe fn convert_medium_contiguous<Src: TensorElement + Copy, T: TensorElement + Copy>(
        src_ptr: *const Src,
        dst_ptr: *mut T,
        numel: usize,
    ) -> Result<()> {
        // Medium data: 16-element unrolling
        const UNROLL: usize = 16;
        let chunks_end = (numel / UNROLL) * UNROLL;
        let mut i = 0;

        while i < chunks_end {
            // Prefetch hint for next cache line (64 bytes ahead)
            #[cfg(target_arch = "x86_64")]
            {
                std::arch::x86_64::_mm_prefetch(
                    src_ptr.add(i + 64) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }

            // Manually unroll the loop for 16 elements
            // Load phase - better cache utilization
            let src_val0 = *src_ptr.add(i);
            let src_val1 = *src_ptr.add(i + 1);
            let src_val2 = *src_ptr.add(i + 2);
            let src_val3 = *src_ptr.add(i + 3);
            let src_val4 = *src_ptr.add(i + 4);
            let src_val5 = *src_ptr.add(i + 5);
            let src_val6 = *src_ptr.add(i + 6);
            let src_val7 = *src_ptr.add(i + 7);
            let src_val8 = *src_ptr.add(i + 8);
            let src_val9 = *src_ptr.add(i + 9);
            let src_val10 = *src_ptr.add(i + 10);
            let src_val11 = *src_ptr.add(i + 11);
            let src_val12 = *src_ptr.add(i + 12);
            let src_val13 = *src_ptr.add(i + 13);
            let src_val14 = *src_ptr.add(i + 14);
            let src_val15 = *src_ptr.add(i + 15);

            // Convert and store phase
            *dst_ptr.add(i) = T::from_f32(src_val0.to_f32());
            *dst_ptr.add(i + 1) = T::from_f32(src_val1.to_f32());
            *dst_ptr.add(i + 2) = T::from_f32(src_val2.to_f32());
            *dst_ptr.add(i + 3) = T::from_f32(src_val3.to_f32());
            *dst_ptr.add(i + 4) = T::from_f32(src_val4.to_f32());
            *dst_ptr.add(i + 5) = T::from_f32(src_val5.to_f32());
            *dst_ptr.add(i + 6) = T::from_f32(src_val6.to_f32());
            *dst_ptr.add(i + 7) = T::from_f32(src_val7.to_f32());
            *dst_ptr.add(i + 8) = T::from_f32(src_val8.to_f32());
            *dst_ptr.add(i + 9) = T::from_f32(src_val9.to_f32());
            *dst_ptr.add(i + 10) = T::from_f32(src_val10.to_f32());
            *dst_ptr.add(i + 11) = T::from_f32(src_val11.to_f32());
            *dst_ptr.add(i + 12) = T::from_f32(src_val12.to_f32());
            *dst_ptr.add(i + 13) = T::from_f32(src_val13.to_f32());
            *dst_ptr.add(i + 14) = T::from_f32(src_val14.to_f32());
            *dst_ptr.add(i + 15) = T::from_f32(src_val15.to_f32());

            i += UNROLL;
        }

        // Process remainder with smaller unroll
        let remainder_start = chunks_end;
        let remainder = numel - remainder_start;
        if remainder >= 8 {
            let i = remainder_start;
            let src_val0 = *src_ptr.add(i);
            let src_val1 = *src_ptr.add(i + 1);
            let src_val2 = *src_ptr.add(i + 2);
            let src_val3 = *src_ptr.add(i + 3);
            let src_val4 = *src_ptr.add(i + 4);
            let src_val5 = *src_ptr.add(i + 5);
            let src_val6 = *src_ptr.add(i + 6);
            let src_val7 = *src_ptr.add(i + 7);

            *dst_ptr.add(i) = T::from_f32(src_val0.to_f32());
            *dst_ptr.add(i + 1) = T::from_f32(src_val1.to_f32());
            *dst_ptr.add(i + 2) = T::from_f32(src_val2.to_f32());
            *dst_ptr.add(i + 3) = T::from_f32(src_val3.to_f32());
            *dst_ptr.add(i + 4) = T::from_f32(src_val4.to_f32());
            *dst_ptr.add(i + 5) = T::from_f32(src_val5.to_f32());
            *dst_ptr.add(i + 6) = T::from_f32(src_val6.to_f32());
            *dst_ptr.add(i + 7) = T::from_f32(src_val7.to_f32());
        }

        // Process final remainder
        let final_start = if remainder >= 8 {
            remainder_start + 8
        } else {
            remainder_start
        };
        for i in final_start..numel {
            let src_val = *src_ptr.add(i);
            *dst_ptr.add(i) = T::from_f32(src_val.to_f32());
        }

        Ok(())
    }

    /// Main entry point for contiguous conversion - dispatches to size-optimized functions
    #[inline(always)]
    unsafe fn convert_contiguous_data<Src: TensorElement + Copy, T: TensorElement + Copy>(
        &self,
        dst_ptr: *mut T,
    ) -> Result<()> {
        let src_ptr = self.as_ptr() as *const Src;
        let numel = self.numel();

        // Dispatch to size-optimized functions
        if numel >= 65536 {
            Self::convert_large_contiguous::<Src, T>(src_ptr, dst_ptr, numel)
        } else if numel < 4096 {
            Self::convert_small_contiguous::<Src, T>(src_ptr, dst_ptr, numel)
        } else {
            Self::convert_medium_contiguous::<Src, T>(src_ptr, dst_ptr, numel)
        }
    }

    /// Convert non-contiguous tensor data using strided access
    #[inline(always)]
    unsafe fn convert_strided_data<Src: TensorElement + Copy, T: TensorElement + Copy>(
        &self,
        dst_ptr: *mut T,
    ) -> Result<()> {
        let src_base_ptr = self
            .storage
            .as_storage()
            .ptr()
            .as_ptr()
            .add(self.offset_bytes) as *const Src;
        let mut dst_idx = 0;

        // 2D case (most common for non-contiguous tensors)
        if self.rank() == 2 {
            let dims = self.dims();
            let strides = self.strides().as_slice();
            let dim0 = dims[0];
            let dim1 = dims[1];
            let stride0 = strides[0];
            let stride1 = strides[1];

            // Unroll inner loop for better performance
            const INNER_UNROLL: usize = 4;

            for i in 0..dim0 {
                let row_offset = i * stride0;
                let inner_chunks = dim1 / INNER_UNROLL;

                // Process unrolled chunks for better ILP
                for chunk in 0..inner_chunks {
                    let j = chunk * INNER_UNROLL;
                    let offset0 = row_offset + j * stride1;
                    let offset1 = row_offset + (j + 1) * stride1;
                    let offset2 = row_offset + (j + 2) * stride1;
                    let offset3 = row_offset + (j + 3) * stride1;

                    let src_val0 = *src_base_ptr.add(offset0);
                    let src_val1 = *src_base_ptr.add(offset1);
                    let src_val2 = *src_base_ptr.add(offset2);
                    let src_val3 = *src_base_ptr.add(offset3);

                    *dst_ptr.add(dst_idx) = T::from_f32(src_val0.to_f32());
                    *dst_ptr.add(dst_idx + 1) = T::from_f32(src_val1.to_f32());
                    *dst_ptr.add(dst_idx + 2) = T::from_f32(src_val2.to_f32());
                    *dst_ptr.add(dst_idx + 3) = T::from_f32(src_val3.to_f32());
                    dst_idx += INNER_UNROLL;
                }

                // Process remainder
                for j in (inner_chunks * INNER_UNROLL)..dim1 {
                    let src_offset_bytes = row_offset + j * stride1;
                    let src_val = *src_base_ptr.add(src_offset_bytes);
                    *dst_ptr.add(dst_idx) = T::from_f32(src_val.to_f32());
                    dst_idx += 1;
                }
            }
            return Ok(());
        }

        // 3D case (common in batch processing)
        if self.rank() == 3 {
            let dims = self.dims();
            let strides = self.strides().as_slice();
            let dim0 = dims[0];
            let dim1 = dims[1];
            let dim2 = dims[2];
            let stride0 = strides[0];
            let stride1 = strides[1];
            let stride2 = strides[2];

            for i in 0..dim0 {
                let batch_offset = i * stride0;
                for j in 0..dim1 {
                    let row_offset = batch_offset + j * stride1;
                    for k in 0..dim2 {
                        let src_offset_bytes = row_offset + k * stride2;
                        let src_val = *src_base_ptr.add(src_offset_bytes);
                        *dst_ptr.add(dst_idx) = T::from_f32(src_val.to_f32());
                        dst_idx += 1;
                    }
                }
            }
            return Ok(());
        }

        // General case for higher dimensions
        let mut indices = vec![0; self.rank()];
        let dims = self.dims();
        let strides = self.strides().as_slice();

        loop {
            // Calculate source offset using strides
            let mut src_offset_bytes = 0;
            for (dim_idx, &index) in indices.iter().enumerate() {
                src_offset_bytes += index * strides[dim_idx];
            }

            // Perform type conversion
            let src_val = *src_base_ptr.add(src_offset_bytes / std::mem::size_of::<Src>());
            let dst_val = T::from_f32(src_val.to_f32());
            *dst_ptr.add(dst_idx) = dst_val;
            dst_idx += 1;

            // Increment indices (row-major order)
            let mut carry = true;
            for dim in (0..self.rank()).rev() {
                if carry {
                    indices[dim] += 1;
                    if indices[dim] < dims[dim] {
                        carry = false;
                    } else {
                        indices[dim] = 0;
                    }
                }
            }

            if carry {
                break; // All dimensions exhausted
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_dtype_f32_to_f64() {
        let data = vec![1.5f32, -2.5, 3.7, -4.2];
        let tensor = Tensor::from_vec(data.clone(), [2, 2]).unwrap();

        let result = tensor.to_dtype::<f64>().unwrap();
        assert_eq!(result.dtype(), DType::Fp64);
        assert_eq!(result.shape(), tensor.shape());

        let result_data = result.as_slice::<f64>().unwrap();
        for (i, &val) in data.iter().enumerate() {
            assert!((result_data[i] - val as f64).abs() < 1e-10);
        }
    }

    #[test]
    fn test_to_dtype_f64_to_f32() {
        let data = vec![1.5f64, -2.5, 3.7, -4.2];
        let tensor = Tensor::from_vec(data.clone(), [2, 2]).unwrap();

        let result = tensor.to_dtype::<f32>().unwrap();
        assert_eq!(result.dtype(), DType::Fp32);
        assert_eq!(result.shape(), tensor.shape());

        let result_data = result.as_slice::<f32>().unwrap();
        for (i, &val) in data.iter().enumerate() {
            assert!((result_data[i] - val as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn test_to_dtype_f32_to_u8() {
        let data = vec![0.0f32, 50.5, 127.3, 255.0];
        let tensor = Tensor::from_vec(data.clone(), [2, 2]).unwrap();

        let result = tensor.to_dtype::<u8>().unwrap();
        assert_eq!(result.dtype(), DType::Uint8);

        let result_data = result.as_slice::<u8>().unwrap();
        assert_eq!(result_data, &[0u8, 50, 127, 255]);
    }

    #[test]
    fn test_to_dtype_u8_to_f32() {
        let data = vec![0u8, 50, 127, 255];
        let tensor = Tensor::from_vec(data.clone(), [2, 2]).unwrap();

        let result = tensor.to_dtype::<f32>().unwrap();
        assert_eq!(result.dtype(), DType::Fp32);

        let result_data = result.as_slice::<f32>().unwrap();
        for (i, &val) in data.iter().enumerate() {
            assert_eq!(result_data[i], val as f32);
        }
    }

    #[test]
    fn test_to_dtype_i32_to_f32() {
        let data = vec![-100i32, 0, 100, 200];
        let tensor = Tensor::from_vec(data.clone(), [2, 2]).unwrap();

        let result = tensor.to_dtype::<f32>().unwrap();
        assert_eq!(result.dtype(), DType::Fp32);

        let result_data = result.as_slice::<f32>().unwrap();
        for (i, &val) in data.iter().enumerate() {
            assert_eq!(result_data[i], val as f32);
        }
    }

    #[test]
    fn test_to_dtype_non_contiguous() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3]).unwrap();

        // Create non-contiguous tensor via permute
        let permuted = tensor.permute([1, 0]).unwrap();
        assert!(!permuted.is_contiguous());

        let result = permuted.to_dtype::<f64>().unwrap();
        assert_eq!(result.dtype(), DType::Fp64);
        assert_eq!(result.shape(), permuted.shape());

        // Verify data is correctly converted and laid out
        let result_data = result.as_slice::<f64>().unwrap();
        let expected = [1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((result_data[i] - exp).abs() < 1e-10);
        }
    }

    #[test]
    fn test_to_dtype_same_type() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data.clone(), [2, 2]).unwrap();

        // Converting to same type should be efficient (fast path)
        let result = tensor.to_dtype::<f32>().unwrap();
        assert_eq!(result.dtype(), DType::Fp32);

        let result_data = result.as_slice::<f32>().unwrap();
        assert_eq!(result_data, data.as_slice());
    }

    #[test]
    fn test_to_dtype_3d_tensor() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data.clone(), [2, 3, 4]).unwrap();

        let result = tensor.to_dtype::<i32>().unwrap();
        assert_eq!(result.dtype(), DType::Int32);
        assert_eq!(result.shape(), tensor.shape());

        let result_data = result.as_slice::<i32>().unwrap();
        for (i, &val) in data.iter().enumerate() {
            assert_eq!(result_data[i], val as i32);
        }
    }

    #[test]
    fn test_to_dtype_edge_cases() {
        // Test with single element
        let tensor = Tensor::from_vec(vec![42.0f32], [1]).unwrap();
        let result = tensor.to_dtype::<i64>().unwrap();
        assert_eq!(result.as_slice::<i64>().unwrap()[0], 42i64);

        // Test with large values
        let tensor = Tensor::from_vec(vec![1e6f32, -1e6], [2]).unwrap();
        let result = tensor.to_dtype::<f64>().unwrap();
        let data = result.as_slice::<f64>().unwrap();
        assert!((data[0] - 1e6).abs() < 1.0);
        assert!((data[1] + 1e6).abs() < 1.0);
    }
}
