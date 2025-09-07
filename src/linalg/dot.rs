use anyhow::Result;

use crate::{
    backend::{global_backend, r#impl::OpsTrait},
    DType, StorageTrait, TensorBase,
};

impl<S: StorageTrait> TensorBase<S> {
    /// Compute dot product between two 1D tensors
    ///
    /// # Arguments
    /// * `other` - The other tensor to compute dot product with
    ///
    /// # Returns
    /// * `Result<f64>` - The scalar dot product result as f64 (safe for all numeric types)
    ///
    /// # Errors
    /// * Returns error if tensors are not 1D
    /// * Returns error if vector lengths don't match
    /// * Returns error if dtypes don't match
    /// * Returns error if dtype is not supported
    pub fn dot(&self, other: &Self) -> Result<f64> {
        // For 1D tensors, compute vector dot product
        if self.rank() != 1 || other.rank() != 1 {
            anyhow::bail!("dot() only supports 1D tensors, use matmul() for matrices");
        }

        if self.shape[0] != other.shape[0] {
            anyhow::bail!(
                "Vector lengths must match for dot product: {} vs {}",
                self.shape[0],
                other.shape[0]
            );
        }

        if self.dtype != other.dtype {
            anyhow::bail!("Both tensors must have the same dtype for dot product");
        }

        // Check if both tensors are contiguous for optimized path
        if self.is_contiguous() && other.is_contiguous() {
            self.dot_contiguous(other)
        } else {
            self.dot_non_contiguous(other)
        }
    }

    /// Optimized dot product for contiguous tensors
    #[inline(always)]
    fn dot_contiguous(&self, other: &Self) -> Result<f64> {
        let backend = global_backend();

        match self.dtype {
            DType::Fp32 => {
                let a_data = self.as_slice::<f32>()?;
                let b_data = other.as_slice::<f32>()?;
                let result = backend.dot_f32(a_data, b_data);
                Ok(result)
            }
            DType::Fp64 => {
                let a_data = self.as_slice::<f64>()?;
                let b_data = other.as_slice::<f64>()?;
                let result = backend.dot_f64(a_data, b_data);
                Ok(result)
            }
            DType::Int8 => {
                let a_data = self.as_slice::<i8>()?;
                let b_data = other.as_slice::<i8>()?;
                let result = backend.dot_i8(a_data, b_data);
                Ok(result)
            }
            DType::Int16 => {
                let a_data = self.as_slice::<i16>()?;
                let b_data = other.as_slice::<i16>()?;
                let result = backend.dot_i16(a_data, b_data);
                Ok(result as f64)
            }
            DType::Int32 => {
                let a_data = self.as_slice::<i32>()?;
                let b_data = other.as_slice::<i32>()?;
                let result = backend.dot_i32(a_data, b_data);
                Ok(result as f64)
            }
            DType::Int64 => {
                let a_data = self.as_slice::<i64>()?;
                let b_data = other.as_slice::<i64>()?;
                let result = backend.dot_i64(a_data, b_data);
                Ok(result as f64)
            }
            DType::Uint8 => {
                let a_data = self.as_slice::<u8>()?;
                let b_data = other.as_slice::<u8>()?;
                let result = backend.dot_u8(a_data, b_data);
                Ok(result as f64)
            }
            DType::Uint16 => {
                let a_data = self.as_slice::<u16>()?;
                let b_data = other.as_slice::<u16>()?;
                let result = backend.dot_u16(a_data, b_data);
                Ok(result as f64)
            }
            DType::Uint32 => {
                let a_data = self.as_slice::<u32>()?;
                let b_data = other.as_slice::<u32>()?;
                let result = backend.dot_u32(a_data, b_data);
                Ok(result as f64)
            }
            DType::Uint64 => {
                let a_data = self.as_slice::<u64>()?;
                let b_data = other.as_slice::<u64>()?;
                let result = backend.dot_u64(a_data, b_data);
                Ok(result as f64)
            }
            DType::Fp16 => {
                let a_data = self.as_slice::<half::f16>()?;
                let b_data = other.as_slice::<half::f16>()?;
                let result = backend.dot_f16(a_data, b_data);
                Ok(result)
            }
            DType::Bf16 => {
                let a_data = self.as_slice::<half::bf16>()?;
                let b_data = other.as_slice::<half::bf16>()?;
                let result = backend.dot_bf16(a_data, b_data);
                Ok(result)
            }
            _ => anyhow::bail!("Unsupported dtype for dot operation: {:?}", self.dtype),
        }
    }

    /// Dot product for non-contiguous tensors using iter() for better performance
    #[inline(always)]
    fn dot_non_contiguous(&self, other: &Self) -> Result<f64> {
        let mut sum = 0.0f64;

        match self.dtype {
            DType::Fp32 => {
                for (self_elem, other_elem) in self.iter().zip(other.iter()) {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let a_val = unsafe { *(self_ptr as *const f32) };
                    let b_val = unsafe { *(other_ptr as *const f32) };
                    sum += (a_val * b_val) as f64;
                }
            }
            DType::Fp64 => {
                for (self_elem, other_elem) in self.iter().zip(other.iter()) {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let a_val = unsafe { *(self_ptr as *const f64) };
                    let b_val = unsafe { *(other_ptr as *const f64) };
                    sum += a_val * b_val;
                }
            }
            DType::Int8 => {
                for (self_elem, other_elem) in self.iter().zip(other.iter()) {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let a_val = unsafe { *(self_ptr as *const i8) };
                    let b_val = unsafe { *(other_ptr as *const i8) };
                    sum += (a_val as f64) * (b_val as f64);
                }
            }
            DType::Int16 => {
                for (self_elem, other_elem) in self.iter().zip(other.iter()) {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let a_val = unsafe { *(self_ptr as *const i16) };
                    let b_val = unsafe { *(other_ptr as *const i16) };
                    sum += (a_val as f64) * (b_val as f64);
                }
            }
            DType::Int32 => {
                for (self_elem, other_elem) in self.iter().zip(other.iter()) {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let a_val = unsafe { *(self_ptr as *const i32) };
                    let b_val = unsafe { *(other_ptr as *const i32) };
                    sum += (a_val as f64) * (b_val as f64);
                }
            }
            DType::Int64 => {
                for (self_elem, other_elem) in self.iter().zip(other.iter()) {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let a_val = unsafe { *(self_ptr as *const i64) };
                    let b_val = unsafe { *(other_ptr as *const i64) };
                    sum += (a_val as f64) * (b_val as f64);
                }
            }
            DType::Uint8 => {
                for (self_elem, other_elem) in self.iter().zip(other.iter()) {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let a_val = unsafe { *self_ptr };
                    let b_val = unsafe { *other_ptr };
                    sum += (a_val as f64) * (b_val as f64);
                }
            }
            DType::Uint16 => {
                for (self_elem, other_elem) in self.iter().zip(other.iter()) {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let a_val = unsafe { *(self_ptr as *const u16) };
                    let b_val = unsafe { *(other_ptr as *const u16) };
                    sum += (a_val as f64) * (b_val as f64);
                }
            }
            DType::Uint32 => {
                for (self_elem, other_elem) in self.iter().zip(other.iter()) {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let a_val = unsafe { *(self_ptr as *const u32) };
                    let b_val = unsafe { *(other_ptr as *const u32) };
                    sum += (a_val as f64) * (b_val as f64);
                }
            }
            DType::Uint64 => {
                for (self_elem, other_elem) in self.iter().zip(other.iter()) {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let a_val = unsafe { *(self_ptr as *const u64) };
                    let b_val = unsafe { *(other_ptr as *const u64) };
                    sum += (a_val as f64) * (b_val as f64);
                }
            }
            DType::Fp16 => {
                for (self_elem, other_elem) in self.iter().zip(other.iter()) {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let a_val = unsafe { *(self_ptr as *const half::f16) };
                    let b_val = unsafe { *(other_ptr as *const half::f16) };
                    sum += (a_val.to_f32() * b_val.to_f32()) as f64;
                }
            }
            DType::Bf16 => {
                for (self_elem, other_elem) in self.iter().zip(other.iter()) {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let a_val = unsafe { *(self_ptr as *const half::bf16) };
                    let b_val = unsafe { *(other_ptr as *const half::bf16) };
                    sum += (a_val.to_f32() * b_val.to_f32()) as f64;
                }
            }
            DType::Bool => {
                anyhow::bail!("dot() does not support bool tensors");
            }
            _ => anyhow::bail!("Unsupported dtype for dot operation: {:?}", self.dtype),
        }

        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_dot_product_basic() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3])?;
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3])?;

        let result = a.dot(&b)?;
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

        Ok(())
    }

    #[test]
    fn test_dot_product_u8_overflow_safe() -> Result<()> {
        // Test that u8 values can safely multiply without overflow
        let a = Tensor::from_vec(vec![255u8, 255u8], vec![2])?;
        let b = Tensor::from_vec(vec![255u8, 255u8], vec![2])?;

        let result = a.dot(&b)?;
        assert_eq!(result, 130050.0); // 255*255 + 255*255 = 65025 + 65025 = 130050

        Ok(())
    }

    #[test]
    fn test_dot_product_different_dtypes() -> Result<()> {
        // Test f64
        let a = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], vec![3])?;
        let b = Tensor::from_vec(vec![4.0f64, 5.0, 6.0], vec![3])?;
        let result = a.dot(&b)?;
        assert_eq!(result, 32.0);

        // Test i32
        let a = Tensor::from_vec(vec![1i32, 2, 3], vec![3])?;
        let b = Tensor::from_vec(vec![4i32, 5, 6], vec![3])?;
        let result = a.dot(&b)?;
        assert_eq!(result, 32.0);

        Ok(())
    }

    #[test]
    fn test_dot_product_errors() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0], vec![2])?;
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3])?;

        // Shape mismatch should fail
        assert!(a.dot(&b).is_err());

        // Dtype mismatch should fail
        let c = Tensor::from_vec(vec![1i32, 2], vec![2])?;
        assert!(a.dot(&c).is_err());

        Ok(())
    }

    #[test]
    fn test_dot_product_non_contiguous() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2])?;

        // Create non-contiguous views
        let view1 = a.reshape(vec![4])?;
        let view2 = b.reshape(vec![4])?;

        let result = view1.dot(&view2)?;
        assert_eq!(result, 70.0); // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70

        Ok(())
    }

    #[test]
    fn test_dot_product_non_contiguous_permute_f16() -> Result<()> {
        // Test with f16 data type using reshape to create non-contiguous tensor
        let data_a = vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(4.0),
        ];
        let data_b = vec![
            half::f16::from_f32(0.5),
            half::f16::from_f32(1.5),
            half::f16::from_f32(2.5),
            half::f16::from_f32(3.5),
        ];

        let tensor_a = Tensor::from_vec(data_a, vec![2, 2])?;
        let tensor_b = Tensor::from_vec(data_b, vec![2, 2])?;

        // Create non-contiguous views using reshape
        let view_a = tensor_a.reshape(vec![4])?;
        let view_b = tensor_b.reshape(vec![4])?;

        let result = view_a.dot(&view_b)?;
        // dot = 1*0.5 + 2*1.5 + 3*2.5 + 4*3.5 = 0.5 + 3.0 + 7.5 + 14.0 = 25.0
        assert!((result - 25.0).abs() < 1e-3); // Use tolerance for f16 precision

        Ok(())
    }

    #[test]
    fn test_dot_product_non_contiguous_permute_bf16() -> Result<()> {
        // Test with bf16 data type using reshape to create non-contiguous tensor
        let data_a = vec![
            half::bf16::from_f32(2.0),
            half::bf16::from_f32(4.0),
            half::bf16::from_f32(6.0),
            half::bf16::from_f32(8.0),
        ];
        let data_b = vec![
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(3.0),
            half::bf16::from_f32(5.0),
            half::bf16::from_f32(7.0),
        ];

        let tensor_a = Tensor::from_vec(data_a, vec![2, 2])?;
        let tensor_b = Tensor::from_vec(data_b, vec![2, 2])?;

        // Create non-contiguous views using reshape
        let view_a = tensor_a.reshape(vec![4])?;
        let view_b = tensor_b.reshape(vec![4])?;

        let result = view_a.dot(&view_b)?;
        // dot = 2*1 + 4*3 + 6*5 + 8*7 = 2 + 12 + 30 + 56 = 100
        assert!((result - 100.0).abs() < 1e-3); // Use tolerance for bf16 precision

        Ok(())
    }

    #[test]
    fn test_dot_product_non_contiguous_complex_permute() -> Result<()> {
        // Test with 2D tensor using permute to create non-contiguous tensor, then use view to get 1D
        let data_a = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let data_b = vec![
            12.0f32, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ];

        // Create 2x6 tensors first
        let tensor_a = Tensor::from_vec(data_a, vec![2, 6])?;
        let tensor_b = Tensor::from_vec(data_b, vec![2, 6])?;

        // Use permute to create non-contiguous tensors, then reshape to 1D
        let permuted_a = tensor_a.permute(vec![1, 0])?;
        let permuted_b = tensor_b.permute(vec![1, 0])?;

        // Verify tensors are non-contiguous after permute
        assert!(!permuted_a.is_contiguous());
        assert!(!permuted_b.is_contiguous());

        // Flatten to 1D for dot product
        let flat_a = permuted_a.flatten_all()?;
        let flat_b = permuted_b.flatten_all()?;

        let result = flat_a.dot(&flat_b)?;
        // After permute [1,0] on 2x6, data order changes from [1,2,3,4,5,6,7,8,9,10,11,12]
        // to [1,7,2,8,3,9,4,10,5,11,6,12] and [12,11,10,9,8,7,6,5,4,3,2,1] to [12,6,11,5,10,4,9,3,8,2,7,1]
        // dot = 1*12 + 7*6 + 2*11 + 8*5 + 3*10 + 9*4 + 4*9 + 10*3 + 5*8 + 11*2 + 6*7 + 12*1
        //     = 12 + 42 + 22 + 40 + 30 + 36 + 36 + 30 + 40 + 22 + 42 + 12 = 364
        assert!((result - 364.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_dot_product_non_contiguous_large_integers() -> Result<()> {
        // Test with larger integer values to ensure no overflow in intermediate calculations
        let data_a = vec![1000i64, 2000, 3000, 4000];
        let data_b = vec![5000i64, 6000, 7000, 8000];

        let tensor_a = Tensor::from_vec(data_a, vec![2, 2])?;
        let tensor_b = Tensor::from_vec(data_b, vec![2, 2])?;

        // Create views using reshape to 1D
        let view_a = tensor_a.reshape(vec![4])?;
        let view_b = tensor_b.reshape(vec![4])?;

        let result = view_a.dot(&view_b)?;
        // dot = 1000*5000 + 2000*6000 + 3000*7000 + 4000*8000 = 5M + 12M + 21M + 32M = 70M
        assert_eq!(result, 70000000.0);

        Ok(())
    }

    #[test]
    fn test_dot_product_non_contiguous_single_element() -> Result<()> {
        // Test single element non-contiguous tensor (edge case)
        let data_a = vec![7.0f32];
        let data_b = vec![3.0f32];

        let tensor_a = Tensor::from_vec(data_a, vec![1, 1])?;
        let tensor_b = Tensor::from_vec(data_b, vec![1, 1])?;

        // Even with permute, single element should work
        let permuted_a = tensor_a.permute(vec![1, 0])?.reshape(vec![1])?;
        let permuted_b = tensor_b.permute(vec![1, 0])?.reshape(vec![1])?;

        let result = permuted_a.dot(&permuted_b)?;
        assert_eq!(result, 21.0); // 7 * 3 = 21

        Ok(())
    }

    #[test]
    fn test_dot_product_non_contiguous_permute_f32() -> Result<()> {
        // Create 2D tensor and use reshape to create view
        let data_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

        let tensor_a = Tensor::from_vec(data_a, vec![2, 3])?;
        let tensor_b = Tensor::from_vec(data_b, vec![2, 3])?;

        // Create views using reshape to 1D
        let view_a = tensor_a.reshape(vec![6])?;
        let view_b = tensor_b.reshape(vec![6])?;

        let result = view_a.dot(&view_b)?;
        // dot = 1*7 + 2*8 + 3*9 + 4*10 + 5*11 + 6*12 = 7 + 16 + 27 + 40 + 55 + 72 = 217
        assert_eq!(result, 217.0);

        Ok(())
    }

    #[test]
    fn test_dot_product_non_contiguous_permute_i64() -> Result<()> {
        // Test with i64 data type and larger values
        let data_a = vec![1000i64, 2000, 3000, 4000];
        let data_b = vec![5000i64, 6000, 7000, 8000];

        let tensor_a = Tensor::from_vec(data_a, vec![2, 2])?;
        let tensor_b = Tensor::from_vec(data_b, vec![2, 2])?;

        // Create views using reshape to 1D
        let view_a = tensor_a.reshape(vec![4])?;
        let view_b = tensor_b.reshape(vec![4])?;

        let result = view_a.dot(&view_b)?;
        // dot = 1000*5000 + 2000*6000 + 3000*7000 + 4000*8000 = 5M + 12M + 21M + 32M = 70M
        assert_eq!(result, 70000000.0);

        Ok(())
    }

    #[test]
    fn test_dot_product_non_contiguous_mixed_signs() -> Result<()> {
        // Test with mixed positive and negative values
        let data_a = vec![-1.0f32, 2.0, -3.0, 4.0, -5.0, 6.0];
        let data_b = vec![6.0f32, -5.0, 4.0, -3.0, 2.0, -1.0];

        let tensor_a = Tensor::from_vec(data_a, vec![2, 3])?;
        let tensor_b = Tensor::from_vec(data_b, vec![2, 3])?;

        // Create views using reshape to 1D
        let view_a = tensor_a.reshape(vec![6])?;
        let view_b = tensor_b.reshape(vec![6])?;

        let result = view_a.dot(&view_b)?;
        // dot = (-1)*6 + 2*(-5) + (-3)*4 + 4*(-3) + (-5)*2 + 6*(-1) = -6 - 10 - 12 - 12 - 10 - 6 = -56
        assert_eq!(result, -56.0);

        Ok(())
    }

    #[test]
    fn test_dot_product_non_contiguous_error_cases() -> Result<()> {
        // Test error cases with non-contiguous tensors
        let data_a = vec![1.0f32, 2.0, 3.0, 4.0];
        let data_b = vec![5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0]; // Different size

        let tensor_a = Tensor::from_vec(data_a, vec![2, 2])?;
        let tensor_b = Tensor::from_vec(data_b, vec![2, 3])?;

        let view_a = tensor_a.reshape(vec![4])?;
        let view_b = tensor_b.reshape(vec![6])?;

        // Should fail due to size mismatch
        assert!(view_a.dot(&view_b).is_err());

        // Test dtype mismatch with non-contiguous tensors
        let data_c = vec![1i32, 2, 3, 4];
        let tensor_c = Tensor::from_vec(data_c, vec![2, 2])?;
        let view_c = tensor_c.reshape(vec![4])?;

        // Should fail due to dtype mismatch
        assert!(view_a.dot(&view_c).is_err());

        Ok(())
    }

    #[test]
    fn test_dot_product_empty() -> Result<()> {
        let a = Tensor::from_vec(Vec::<f32>::new(), vec![0])?;
        let b = Tensor::from_vec(Vec::<f32>::new(), vec![0])?;

        let result = a.dot(&b)?;
        assert_eq!(result, 0.0); // Empty dot product should return 0

        Ok(())
    }

    #[test]
    fn test_dot_product_single_element() -> Result<()> {
        let a = Tensor::from_vec(vec![5.0f32], vec![1])?;
        let b = Tensor::from_vec(vec![3.0f32], vec![1])?;

        let result = a.dot(&b)?;
        assert_eq!(result, 15.0); // 5 * 3 = 15

        Ok(())
    }

    #[test]
    fn test_dot_product_truly_non_contiguous_permute() -> Result<()> {
        // Test truly non-contiguous tensors using permute on 2D tensor
        let data_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let data_b = vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let tensor_a = Tensor::from_vec(data_a, vec![2, 4])?;
        let tensor_b = Tensor::from_vec(data_b, vec![2, 4])?;

        // Create non-contiguous tensors using permute to change dimension order
        let permuted_a = tensor_a.permute(vec![1, 0])?;
        let permuted_b = tensor_b.permute(vec![1, 0])?;

        // Verify tensors are truly non-contiguous after permute
        assert!(!permuted_a.is_contiguous());
        assert!(!permuted_b.is_contiguous());

        // Flatten to 1D for dot product
        let flat_a = permuted_a.flatten_all()?;
        let flat_b = permuted_b.flatten_all()?;

        let result = flat_a.dot(&flat_b)?;
        // After permute [1,0] on 2x4, data order changes from [1,2,3,4,5,6,7,8] to [1,5,2,6,3,7,4,8]
        // and [8,7,6,5,4,3,2,1] to [8,4,7,3,6,2,5,1]
        // dot = 1*8 + 5*4 + 2*7 + 6*3 + 3*6 + 7*2 + 4*5 + 8*1 = 8 + 20 + 14 + 18 + 18 + 14 + 20 + 8 = 120
        assert_eq!(result, 120.0);

        Ok(())
    }

    #[test]
    fn test_dot_product_non_contiguous_3d_permute() -> Result<()> {
        // Test with 3D tensor using permute to create non-contiguous layout
        let data_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let data_b = vec![9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let tensor_a = Tensor::from_vec(data_a, vec![3, 3])?;
        let tensor_b = Tensor::from_vec(data_b, vec![3, 3])?;

        // Create non-contiguous tensors using permute
        let permuted_a = tensor_a.permute(vec![1, 0])?;
        let permuted_b = tensor_b.permute(vec![1, 0])?;

        // Verify tensors are non-contiguous after permute
        assert!(!permuted_a.is_contiguous());
        assert!(!permuted_b.is_contiguous());

        // Flatten to 1D for dot product
        let flat_a = permuted_a.flatten_all()?;
        let flat_b = permuted_b.flatten_all()?;

        let result = flat_a.dot(&flat_b)?;
        // After permute [1,0] on 3x3, data order changes from [1,2,3,4,5,6,7,8,9] to [1,4,7,2,5,8,3,6,9]
        // and [9,8,7,6,5,4,3,2,1] to [9,6,3,8,5,2,7,4,1]
        // dot = 1*9 + 4*6 + 7*3 + 2*8 + 5*5 + 8*2 + 3*7 + 6*4 + 9*1 = 9 + 24 + 21 + 16 + 25 + 16 + 21 + 24 + 9 = 165
        assert_eq!(result, 165.0);

        Ok(())
    }
}
