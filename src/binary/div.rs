use anyhow::Result;
use half::{bf16, f16};

use crate::{
    backend::{global_backend, r#impl::OpsTrait},
    DType, StorageTrait, Tensor, TensorBase, TensorElement, TensorView, UninitVec,
};

impl<S: StorageTrait> TensorBase<S> {
    /// Divide two tensors element-wise (shapes must match exactly)
    #[inline(always)]
    pub fn div<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        // Check dtype compatibility
        if self.dtype() != other.dtype() {
            anyhow::bail!(
                "Dtype mismatch for div operation: {:?} vs {:?}",
                self.dtype(),
                other.dtype()
            );
        }

        // Check shape compatibility - no automatic broadcasting
        if self.shape() != other.shape() {
            anyhow::bail!(
                "Shape mismatch for div operation: {:?} vs {:?}. Use broadcast_to() explicitly if needed.",
                self.dims(),
                other.dims()
            );
        }

        // Handle empty tensors
        let numel = self.numel();
        if numel == 0 {
            let empty_storage = crate::Storage::new(0, self.dtype().size_in_bytes())?;
            return Ok(Tensor {
                storage: empty_storage,
                ptr: std::ptr::NonNull::dangling(),
                dtype: self.dtype(),
                shape: self.shape,
                strides: Self::compute_contiguous_strides(self.shape()),
                offset_bytes: 0,
            });
        }

        // Check if both tensors are contiguous
        if self.is_contiguous() && other.is_contiguous() {
            self.div_contiguous(other)
        } else {
            self.div_non_contiguous(other)
        }
    }

    /// Optimized division for contiguous tensors using backend acceleration
    #[inline(always)]
    fn div_contiguous<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        let numel = self.numel();
        let backend = global_backend();

        match self.dtype() {
            DType::Fp32 => {
                let a = self.as_slice::<f32>()?;
                let b = other.as_slice::<f32>()?;
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    backend.v_div_f32(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let a = self.as_slice::<f64>()?;
                let b = other.as_slice::<f64>()?;
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    backend.v_div_f64(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let a = self.as_slice::<f16>()?;
                let b = other.as_slice::<f16>()?;
                let out = UninitVec::<f16>::new(numel).init_with(|dst| {
                    backend.v_div_f16(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let a = self.as_slice::<bf16>()?;
                let b = other.as_slice::<bf16>()?;
                let out = UninitVec::<bf16>::new(numel).init_with(|dst| {
                    backend.v_div_bf16(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Division not supported for dtype: {:?}", self.dtype()),
        }
    }

    /// Fallback division for non-contiguous tensors using element-wise iteration
    #[inline(always)]
    fn div_non_contiguous<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        let numel = self.numel();

        match self.dtype() {
            DType::Fp32 => {
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<f32>] as *mut [f32])
                };

                for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate() {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let self_val = unsafe { *(self_ptr as *const f32) };
                    let other_val = unsafe { *(other_ptr as *const f32) };
                    dst_to_set[idx] = self_val / other_val;
                }

                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Fp64 => {
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<f64>] as *mut [f64])
                };

                for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate() {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let self_val = unsafe { *(self_ptr as *const f64) };
                    let other_val = unsafe { *(other_ptr as *const f64) };
                    dst_to_set[idx] = self_val / other_val;
                }

                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Fp16 => {
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<f16>] as *mut [f16])
                };

                for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate() {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let self_val = unsafe { *(self_ptr as *const f16) };
                    let other_val = unsafe { *(other_ptr as *const f16) };
                    dst_to_set[idx] = f16::from_f32(self_val.to_f32() / other_val.to_f32());
                }

                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Bf16 => {
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<bf16>] as *mut [bf16])
                };

                for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate() {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let self_val = unsafe { *(self_ptr as *const bf16) };
                    let other_val = unsafe { *(other_ptr as *const bf16) };
                    dst_to_set[idx] = bf16::from_f32(self_val.to_f32() / other_val.to_f32());
                }

                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            _ => anyhow::bail!("Division not supported for dtype: {:?}", self.dtype()),
        }
    }

    /// Divide all elements of the tensor by a scalar
    #[inline(always)]
    pub fn div_scalar<T: TensorElement + std::ops::Div<Output = T>>(
        &self,
        scalar: T,
    ) -> Result<Tensor> {
        // Check dtype compatibility - scalar type must match tensor dtype exactly
        if T::DTYPE != self.dtype() {
            anyhow::bail!(
                "Scalar type mismatch for div_scalar operation: scalar dtype {:?} vs tensor dtype {:?}",
                T::DTYPE,
                self.dtype()
            );
        }

        let numel = self.shape().numel();

        // Handle empty tensors
        if numel == 0 {
            let empty_storage = crate::Storage::new(0, self.dtype().size_in_bytes())?;
            return Ok(Tensor {
                storage: empty_storage,
                ptr: std::ptr::NonNull::dangling(),
                dtype: self.dtype(),
                shape: self.shape,
                strides: Self::compute_contiguous_strides(self.shape()),
                offset_bytes: 0,
            });
        }

        // Since types match exactly, we can use the scalar directly without conversion
        if self.is_contiguous() {
            self.div_scalar_contiguous(scalar)
        } else {
            self.div_scalar_non_contiguous(scalar)
        }
    }

    /// Direct scalar division for contiguous tensors - no type conversion needed
    #[inline(always)]
    fn div_scalar_contiguous<T: TensorElement + std::ops::Div<Output = T>>(
        &self,
        scalar: T,
    ) -> Result<Tensor> {
        let numel = self.shape().numel();
        let backend = global_backend();

        // Since types match exactly, we can use the scalar directly
        match self.dtype() {
            DType::Fp32 => {
                let input_data = self.as_slice::<f32>()?;
                let s = unsafe { std::mem::transmute_copy::<T, f32>(&scalar) };
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<f32>] as *mut [f32])
                };
                backend.v_div_scalar_f32(input_data, s, dst_to_set);
                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Fp64 => {
                let input_data = self.as_slice::<f64>()?;
                let s = unsafe { std::mem::transmute_copy::<T, f64>(&scalar) };
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<f64>] as *mut [f64])
                };
                backend.v_div_scalar_f64(input_data, s, dst_to_set);
                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Fp16 => {
                let input_data = self.as_slice::<f16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, f16>(&scalar) };
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<f16>] as *mut [f16])
                };
                backend.v_div_scalar_f16(input_data, s, dst_to_set);
                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Bf16 => {
                let input_data = self.as_slice::<bf16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, bf16>(&scalar) };
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<bf16>] as *mut [bf16])
                };
                backend.v_div_scalar_bf16(input_data, s, dst_to_set);
                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            _ => anyhow::bail!(
                "Scalar division not supported for dtype: {:?}",
                self.dtype()
            ),
        }
    }

    /// Direct scalar division for non-contiguous tensors - no type conversion needed
    #[inline(always)]
    fn div_scalar_non_contiguous<T: TensorElement + std::ops::Div<Output = T>>(
        &self,
        scalar: T,
    ) -> Result<Tensor> {
        // Since types match exactly, we can use the scalar directly
        match self.dtype() {
            DType::Fp32 => {
                let s = unsafe { std::mem::transmute_copy::<T, f32>(&scalar) };
                self.map_non_contiguous::<f32>(|&x| x / s)
            }
            DType::Fp64 => {
                let s = unsafe { std::mem::transmute_copy::<T, f64>(&scalar) };
                self.map_non_contiguous::<f64>(|&x| x / s)
            }
            DType::Fp16 => {
                let s = unsafe { std::mem::transmute_copy::<T, f16>(&scalar) };
                self.map_non_contiguous::<f16>(|&x| x / s)
            }
            DType::Bf16 => {
                let s = unsafe { std::mem::transmute_copy::<T, bf16>(&scalar) };
                self.map_non_contiguous::<bf16>(|&x| x / s)
            }
            _ => anyhow::bail!(
                "Scalar division not supported for dtype: {:?}",
                self.dtype()
            ),
        }
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Div<&TensorBase<S2>> for &TensorBase<S1> {
    type Output = Tensor;
    fn div(self, other: &TensorBase<S2>) -> Self::Output {
        TensorBase::div(self, other).expect("Tensor division failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Div<TensorBase<S2>> for &TensorBase<S1> {
    type Output = Tensor;
    fn div(self, other: TensorBase<S2>) -> Self::Output {
        TensorBase::div(self, &other).expect("Tensor division failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Div<&TensorBase<S2>> for TensorBase<S1> {
    type Output = Tensor;
    fn div(self, other: &TensorBase<S2>) -> Self::Output {
        TensorBase::div(&self, other).expect("Tensor division failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Div<TensorBase<S2>> for TensorBase<S1> {
    type Output = Tensor;
    fn div(self, other: TensorBase<S2>) -> Self::Output {
        TensorBase::div(&self, &other).expect("Tensor division failed")
    }
}

// Scalar division operator implementations using macro to avoid recursion
macro_rules! impl_scalar_div {
    ($($scalar_type:ty),+) => {
        $(
            impl std::ops::Div<$scalar_type> for &Tensor {
                type Output = Tensor;
                fn div(self, other: $scalar_type) -> Self::Output {
                    self.div_scalar(other).unwrap()
                }
            }

            impl std::ops::Div<$scalar_type> for Tensor {
                type Output = Tensor;
                fn div(self, other: $scalar_type) -> Self::Output {
                    self.div_scalar(other).unwrap()
                }
            }
        )+
    };
}

// Generate scalar division implementations for common numeric types
impl_scalar_div!(f32, f64, i32, i64, u32, u64, i16, u16, i8, u8, f16, bf16);

// Additional implementations for TensorView (TensorBase<&Storage>)
macro_rules! impl_scalar_div_view {
    ($($scalar_type:ty),+) => {
        $(
            impl std::ops::Div<$scalar_type> for &TensorView<'_> {
                type Output = Tensor;
                fn div(self, other: $scalar_type) -> Self::Output {
                    self.div_scalar(other).unwrap()
                }
            }

            impl std::ops::Div<$scalar_type> for TensorView<'_> {
                type Output = Tensor;
                fn div(self, other: $scalar_type) -> Self::Output {
                    self.div_scalar(other).unwrap()
                }
            }
        )+
    };
}

impl_scalar_div_view!(f32, f64, i32, i64, u32, u64, i16, u16, i8, u8, f16, bf16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_div_basic() -> Result<()> {
        let a = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![2.0f32, 4.0, 5.0, 8.0], vec![2, 2])?;

        let result = a.div(&b)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[5.0, 5.0, 6.0, 5.0]);

        Ok(())
    }

    #[test]
    fn test_tensor_div_scalar() -> Result<()> {
        let a = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2])?;

        let result = a.div_scalar(2.0f32)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[5.0, 10.0, 15.0, 20.0]);

        Ok(())
    }

    #[test]
    fn test_tensor_div_different_dtypes() -> Result<()> {
        // Test f64
        let a = Tensor::from_vec(vec![10.0f64, 20.0, 30.0, 40.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![2.0f64, 4.0, 5.0, 8.0], vec![2, 2])?;
        let result = a.div(&b)?;
        let data = result.as_slice::<f64>()?;
        assert_eq!(data, &[5.0, 5.0, 6.0, 5.0]);

        Ok(())
    }

    #[test]
    fn test_tensor_div_half_precision() -> Result<()> {
        // Test f16
        let a = Tensor::from_vec(vec![f16::from_f32(10.0), f16::from_f32(20.0)], vec![2])?;
        let b = Tensor::from_vec(vec![f16::from_f32(2.0), f16::from_f32(4.0)], vec![2])?;
        let result = a.div(&b)?;
        let data = result.as_slice::<f16>()?;
        assert!((data[0].to_f32() - 5.0).abs() < 1e-3);
        assert!((data[1].to_f32() - 5.0).abs() < 1e-3);

        // Test bf16
        let a = Tensor::from_vec(vec![bf16::from_f32(10.0), bf16::from_f32(20.0)], vec![2])?;
        let b = Tensor::from_vec(vec![bf16::from_f32(2.0), bf16::from_f32(4.0)], vec![2])?;
        let result = a.div(&b)?;
        let data = result.as_slice::<bf16>()?;
        assert!((data[0].to_f32() - 5.0).abs() < 1e-2);
        assert!((data[1].to_f32() - 5.0).abs() < 1e-2);

        Ok(())
    }

    #[test]
    fn test_tensor_div_errors() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0], vec![2])?;
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3])?;

        // Shape mismatch should fail
        assert!(a.div(&b).is_err());

        // Dtype mismatch should fail
        let c = Tensor::from_vec(vec![1i32, 2], vec![2])?;
        assert!(a.div(&c).is_err());

        Ok(())
    }

    #[test]
    fn test_tensor_div_empty() -> Result<()> {
        let a = Tensor::from_vec(Vec::<f32>::new(), vec![0])?;
        let b = Tensor::from_vec(Vec::<f32>::new(), vec![0])?;

        let result = a.div(&b)?;
        assert_eq!(result.numel(), 0);
        assert_eq!(result.dims(), &[0]);

        // Test scalar division with empty tensor
        let result = a.div_scalar(2.0f32)?;
        assert_eq!(result.numel(), 0);

        Ok(())
    }

    #[test]
    fn test_operator_overloading() -> Result<()> {
        let a = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![2.0f32, 4.0, 5.0, 8.0], vec![2, 2])?;

        // Test tensor / tensor
        let result1 = &a / &b;
        let data1 = result1.as_slice::<f32>()?;
        assert_eq!(data1, &[5.0, 5.0, 6.0, 5.0]);

        // Test tensor / scalar
        let result2 = &a / 2.0f32;
        let data2 = result2.as_slice::<f32>()?;
        assert_eq!(data2, &[5.0, 10.0, 15.0, 20.0]);

        Ok(())
    }

    #[test]
    fn test_edge_cases() -> Result<()> {
        // Single element tensors
        let a = Tensor::from_vec(vec![15.0f32], vec![1])?;
        let b = Tensor::from_vec(vec![3.0f32], vec![1])?;
        let result = a.div(&b)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[5.0]);

        // Scalar division with single element
        let result = a.div_scalar(3.0f32)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[5.0]);

        Ok(())
    }

    #[test]
    fn test_tensor_view_operations() -> Result<()> {
        let a = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3])?;
        let b = Tensor::from_vec(vec![2.0f32, 4.0, 5.0, 8.0, 10.0, 12.0], vec![2, 3])?;

        // Create views
        let view1 = a.reshape(vec![3, 2])?;
        let view2 = b.reshape(vec![3, 2])?;

        // Test division with views
        let result = view1.div(&view2)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[5.0, 5.0, 6.0, 5.0, 5.0, 5.0]);

        Ok(())
    }

    #[test]
    fn test_performance_comparison() -> Result<()> {
        use std::time::Instant;

        let size = 1000;
        let a = Tensor::from_vec((0..size).map(|i| (i + 1) as f32).collect(), vec![size])?;
        let b = Tensor::from_vec((0..size).map(|i| (i + 2) as f32).collect(), vec![size])?;

        let start = Instant::now();
        let _result = a.div(&b)?;
        let elapsed = start.elapsed();

        println!("   Dividing {size} elements took: {elapsed:?}");

        // Should complete reasonably quickly
        assert!(elapsed.as_secs() < 1);

        Ok(())
    }
}
