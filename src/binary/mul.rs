use anyhow::Result;
use half::{bf16, f16};

use crate::{
    backend::{global_backend, r#impl::OpsTrait},
    compute_broadcast_shape, DType, StorageTrait, Tensor, TensorBase, TensorElement, TensorView,
    UninitVec,
};

impl<S: StorageTrait> TensorBase<S> {
    /// Multiply two tensors element-wise (shapes must match exactly)
    #[inline(always)]
    pub fn mul<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        // Check shape compatibility
        debug_assert_eq!(
            self.shape(),
            other.shape(),
            "Shape mismatch for mul operation: {:?} vs {:?}",
            self.dims(),
            other.dims()
        );
        debug_assert_eq!(
            self.dtype(),
            other.dtype(),
            "Dtype mismatch for mul operation: {:?} vs {:?}",
            self.dtype(),
            other.dtype()
        );

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
            self.mul_contiguous(other)
        } else {
            self.mul_non_contiguous(other)
        }
    }

    /// Multiply two tensors element-wise with broadcasting support
    ///
    /// If shapes don't match, attempts to broadcast them to a compatible shape.
    #[inline(always)]
    pub fn broadcast_mul<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        // Check dtype compatibility
        if self.dtype() != other.dtype() {
            anyhow::bail!(
                "Dtype mismatch for mul operation: {:?} vs {:?}",
                self.dtype(),
                other.dtype()
            );
        }

        // Handle empty tensors
        if self.numel() == 0 || other.numel() == 0 {
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

        // Check if shapes match exactly
        if self.shape() == other.shape() {
            return self.mul(other);
        }

        // Try to broadcast shapes
        let broadcast_shape = compute_broadcast_shape(self.shape(), other.shape())?;

        // Broadcast both tensors to the target shape
        let self_broadcasted = self.broadcast_to(broadcast_shape)?;
        let other_broadcasted = other.broadcast_to(broadcast_shape)?;

        // Perform multiplication on broadcasted tensors
        self_broadcasted.mul(&other_broadcasted)
    }

    /// Optimized multiplication for contiguous tensors using backend acceleration
    #[inline(always)]
    fn mul_contiguous<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        let numel = self.numel();
        let backend = global_backend();

        match self.dtype() {
            DType::Fp32 => {
                let a = self.as_slice::<f32>()?;
                let b = other.as_slice::<f32>()?;
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    backend.v_mul_f32(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let a = self.as_slice::<f64>()?;
                let b = other.as_slice::<f64>()?;
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    backend.v_mul_f64(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let a = self.as_slice::<f16>()?;
                let b = other.as_slice::<f16>()?;
                let out = UninitVec::<f16>::new(numel).init_with(|dst| {
                    backend.v_mul_f16(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let a = self.as_slice::<bf16>()?;
                let b = other.as_slice::<bf16>()?;
                let out = UninitVec::<bf16>::new(numel).init_with(|dst| {
                    backend.v_mul_bf16(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int8 => {
                let a = self.as_slice::<i8>()?;
                let b = other.as_slice::<i8>()?;
                let out = UninitVec::<i8>::new(numel).init_with(|dst| {
                    backend.v_mul_i8(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int16 => {
                let a = self.as_slice::<i16>()?;
                let b = other.as_slice::<i16>()?;
                let out = UninitVec::<i16>::new(numel).init_with(|dst| {
                    backend.v_mul_i16(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int32 => {
                let a = self.as_slice::<i32>()?;
                let b = other.as_slice::<i32>()?;
                let out = UninitVec::<i32>::new(numel).init_with(|dst| {
                    backend.v_mul_i32(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int64 => {
                let a = self.as_slice::<i64>()?;
                let b = other.as_slice::<i64>()?;
                let out = UninitVec::<i64>::new(numel).init_with(|dst| {
                    backend.v_mul_i64(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint8 => {
                let a = self.as_slice::<u8>()?;
                let b = other.as_slice::<u8>()?;
                let out = UninitVec::<u8>::new(numel).init_with(|dst| {
                    backend.v_mul_u8(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint16 => {
                let a = self.as_slice::<u16>()?;
                let b = other.as_slice::<u16>()?;
                let out = UninitVec::<u16>::new(numel).init_with(|dst| {
                    backend.v_mul_u16(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint32 => {
                let a = self.as_slice::<u32>()?;
                let b = other.as_slice::<u32>()?;
                let out = UninitVec::<u32>::new(numel).init_with(|dst| {
                    backend.v_mul_u32(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint64 => {
                let a = self.as_slice::<u64>()?;
                let b = other.as_slice::<u64>()?;
                let out = UninitVec::<u64>::new(numel).init_with(|dst| {
                    backend.v_mul_u64(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Multiplication not supported for dtype: {:?}", self.dtype()),
        }
    }

    /// Fallback multiplication for non-contiguous tensors using element-wise iteration
    #[inline(always)]
    fn mul_non_contiguous<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        match self.dtype() {
            DType::Fp32 => {
                let numel = self.numel();
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    for (idx, (self_item, other_item)) in self
                        .iter_with_meta::<f32>()
                        .zip(other.iter_with_meta::<f32>())
                        .enumerate()
                    {
                        dst[idx] = *self_item.value * *other_item.value;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let numel = self.numel();
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    for (idx, (self_item, other_item)) in self
                        .iter_with_meta::<f64>()
                        .zip(other.iter_with_meta::<f64>())
                        .enumerate()
                    {
                        dst[idx] = *self_item.value * *other_item.value;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let numel = self.numel();
                let out = UninitVec::<f16>::new(numel).init_with(|dst| {
                    for (idx, (self_item, other_item)) in self
                        .iter_with_meta::<f16>()
                        .zip(other.iter_with_meta::<f16>())
                        .enumerate()
                    {
                        dst[idx] =
                            f16::from_f32(self_item.value.to_f32() * other_item.value.to_f32());
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let numel = self.numel();
                let out = UninitVec::<bf16>::new(numel).init_with(|dst| {
                    for (idx, (self_item, other_item)) in self
                        .iter_with_meta::<bf16>()
                        .zip(other.iter_with_meta::<bf16>())
                        .enumerate()
                    {
                        dst[idx] =
                            bf16::from_f32(self_item.value.to_f32() * other_item.value.to_f32());
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int8 => {
                let numel = self.numel();
                let out = UninitVec::<i8>::new(numel).init_with(|dst| {
                    for (idx, (self_item, other_item)) in self
                        .iter_with_meta::<i8>()
                        .zip(other.iter_with_meta::<i8>())
                        .enumerate()
                    {
                        dst[idx] = *self_item.value * *other_item.value;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int16 => {
                let numel = self.numel();
                let out = UninitVec::<i16>::new(numel).init_with(|dst| {
                    for (idx, (self_item, other_item)) in self
                        .iter_with_meta::<i16>()
                        .zip(other.iter_with_meta::<i16>())
                        .enumerate()
                    {
                        dst[idx] = *self_item.value * *other_item.value;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int32 => {
                let numel = self.numel();
                let out = UninitVec::<i32>::new(numel).init_with(|dst| {
                    for (idx, (self_item, other_item)) in self
                        .iter_with_meta::<i32>()
                        .zip(other.iter_with_meta::<i32>())
                        .enumerate()
                    {
                        dst[idx] = *self_item.value * *other_item.value;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int64 => {
                let numel = self.numel();
                let out = UninitVec::<i64>::new(numel).init_with(|dst| {
                    for (idx, (self_item, other_item)) in self
                        .iter_with_meta::<i64>()
                        .zip(other.iter_with_meta::<i64>())
                        .enumerate()
                    {
                        dst[idx] = *self_item.value * *other_item.value;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint8 => {
                let numel = self.numel();
                let out = UninitVec::<u8>::new(numel).init_with(|dst| {
                    for (idx, (self_item, other_item)) in self
                        .iter_with_meta::<u8>()
                        .zip(other.iter_with_meta::<u8>())
                        .enumerate()
                    {
                        dst[idx] = *self_item.value * *other_item.value;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint16 => {
                let numel = self.numel();
                let out = UninitVec::<u16>::new(numel).init_with(|dst| {
                    for (idx, (self_item, other_item)) in self
                        .iter_with_meta::<u16>()
                        .zip(other.iter_with_meta::<u16>())
                        .enumerate()
                    {
                        dst[idx] = *self_item.value * *other_item.value;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint32 => {
                let numel = self.numel();
                let out = UninitVec::<u32>::new(numel).init_with(|dst| {
                    for (idx, (self_item, other_item)) in self
                        .iter_with_meta::<u32>()
                        .zip(other.iter_with_meta::<u32>())
                        .enumerate()
                    {
                        dst[idx] = *self_item.value * *other_item.value;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint64 => {
                let numel = self.numel();
                let out = UninitVec::<u64>::new(numel).init_with(|dst| {
                    for (idx, (self_item, other_item)) in self
                        .iter_with_meta::<u64>()
                        .zip(other.iter_with_meta::<u64>())
                        .enumerate()
                    {
                        dst[idx] = *self_item.value * *other_item.value;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Multiplication not supported for dtype: {:?}", self.dtype()),
        }
    }
    /// Multiply all elements of the tensor by a scalar
    #[inline(always)]
    pub fn mul_scalar<T: TensorElement + std::ops::Mul<Output = T>>(
        &self,
        scalar: T,
    ) -> Result<Tensor> {
        // Check dtype compatibility - scalar type must match tensor dtype exactly
        debug_assert_eq!(
            T::DTYPE,
            self.dtype(),
            "Scalar type mismatch for mul_scalar operation: scalar dtype {:?} vs tensor dtype {:?}",
            T::DTYPE,
            self.dtype()
        );

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
            self.mul_scalar_contiguous(scalar)
        } else {
            self.mul_scalar_non_contiguous(scalar)
        }
    }

    /// Direct scalar multiplication for contiguous tensors - no type conversion needed
    #[inline(always)]
    fn mul_scalar_contiguous<T: TensorElement + std::ops::Mul<Output = T>>(
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
                let mut output = UninitVec::new(numel);
                backend.v_mul_scalar_f32(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Fp64 => {
                let input_data = self.as_slice::<f64>()?;
                let s = unsafe { std::mem::transmute_copy::<T, f64>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_mul_scalar_f64(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Fp16 => {
                let input_data = self.as_slice::<f16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, f16>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_mul_scalar_f16(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Bf16 => {
                let input_data = self.as_slice::<bf16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, bf16>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_mul_scalar_bf16(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Int8 => {
                let input_data = self.as_slice::<i8>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i8>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_mul_scalar_i8(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Int16 => {
                let input_data = self.as_slice::<i16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i16>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_mul_scalar_i16(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Int32 => {
                let input_data = self.as_slice::<i32>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i32>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_mul_scalar_i32(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Int64 => {
                let input_data = self.as_slice::<i64>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i64>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_mul_scalar_i64(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Uint8 => {
                let input_data = self.as_slice::<u8>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u8>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_mul_scalar_u8(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Uint16 => {
                let input_data = self.as_slice::<u16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u16>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_mul_scalar_u16(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Uint32 => {
                let input_data = self.as_slice::<u32>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u32>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_mul_scalar_u32(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Uint64 => {
                let input_data = self.as_slice::<u64>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u64>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_mul_scalar_u64(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            _ => anyhow::bail!(
                "Scalar multiplication not supported for dtype: {:?}",
                self.dtype()
            ),
        }
    }

    /// Direct scalar multiplication for non-contiguous tensors - no type conversion needed
    #[inline(always)]
    fn mul_scalar_non_contiguous<T: TensorElement + std::ops::Mul<Output = T>>(
        &self,
        scalar: T,
    ) -> Result<Tensor> {
        // Since types match exactly, we can use the scalar directly
        match self.dtype() {
            DType::Fp32 => {
                let s = unsafe { std::mem::transmute_copy::<T, f32>(&scalar) };
                self.map_non_contiguous::<f32>(|&x| x * s)
            }
            DType::Fp64 => {
                let s = unsafe { std::mem::transmute_copy::<T, f64>(&scalar) };
                self.map_non_contiguous::<f64>(|&x| x * s)
            }
            DType::Fp16 => {
                let s = unsafe { std::mem::transmute_copy::<T, f16>(&scalar) };
                self.map_non_contiguous::<f16>(|&x| x * s)
            }
            DType::Bf16 => {
                let s = unsafe { std::mem::transmute_copy::<T, bf16>(&scalar) };
                self.map_non_contiguous::<bf16>(|&x| x * s)
            }
            DType::Int8 => {
                let s = unsafe { std::mem::transmute_copy::<T, i8>(&scalar) };
                self.map_non_contiguous::<i8>(|&x| x * s)
            }
            DType::Int16 => {
                let s = unsafe { std::mem::transmute_copy::<T, i16>(&scalar) };
                self.map_non_contiguous::<i16>(|&x| x * s)
            }
            DType::Int32 => {
                let s = unsafe { std::mem::transmute_copy::<T, i32>(&scalar) };
                self.map_non_contiguous::<i32>(|&x| x * s)
            }
            DType::Int64 => {
                let s = unsafe { std::mem::transmute_copy::<T, i64>(&scalar) };
                self.map_non_contiguous::<i64>(|&x| x * s)
            }
            DType::Uint8 => {
                let s = unsafe { std::mem::transmute_copy::<T, u8>(&scalar) };
                self.map_non_contiguous::<u8>(|&x| x * s)
            }
            DType::Uint16 => {
                let s = unsafe { std::mem::transmute_copy::<T, u16>(&scalar) };
                self.map_non_contiguous::<u16>(|&x| x * s)
            }
            DType::Uint32 => {
                let s = unsafe { std::mem::transmute_copy::<T, u32>(&scalar) };
                self.map_non_contiguous::<u32>(|&x| x * s)
            }
            DType::Uint64 => {
                let s = unsafe { std::mem::transmute_copy::<T, u64>(&scalar) };
                self.map_non_contiguous::<u64>(|&x| x * s)
            }
            DType::Bool => anyhow::bail!("Scalar multiplication not supported for Bool dtype"),
            _ => anyhow::bail!(
                "Scalar multiplication not supported for dtype: {:?}",
                self.dtype()
            ),
        }
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Mul<&TensorBase<S2>> for &TensorBase<S1> {
    type Output = Tensor;
    fn mul(self, other: &TensorBase<S2>) -> Self::Output {
        TensorBase::broadcast_mul(self, other).expect("Tensor multiplication failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Mul<TensorBase<S2>> for &TensorBase<S1> {
    type Output = Tensor;
    fn mul(self, other: TensorBase<S2>) -> Self::Output {
        TensorBase::broadcast_mul(self, &other).expect("Tensor multiplication failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Mul<&TensorBase<S2>> for TensorBase<S1> {
    type Output = Tensor;
    fn mul(self, other: &TensorBase<S2>) -> Self::Output {
        TensorBase::broadcast_mul(&self, other).expect("Tensor multiplication failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Mul<TensorBase<S2>> for TensorBase<S1> {
    type Output = Tensor;
    fn mul(self, other: TensorBase<S2>) -> Self::Output {
        TensorBase::broadcast_mul(&self, &other).expect("Tensor multiplication failed")
    }
}

// Scalar multiplication operator implementations using macro to avoid recursion
macro_rules! impl_scalar_mul {
    ($($scalar_type:ty),+) => {
        $(
            impl std::ops::Mul<$scalar_type> for &Tensor {
                type Output = Tensor;
                fn mul(self, other: $scalar_type) -> Self::Output {
                    self.mul_scalar(other).unwrap()
                }
            }

            impl std::ops::Mul<$scalar_type> for Tensor {
                type Output = Tensor;
                fn mul(self, other: $scalar_type) -> Self::Output {
                    self.mul_scalar(other).unwrap()
                }
            }
        )+
    };
}

// Generate scalar multiplication implementations for common numeric types
impl_scalar_mul!(f32, f64, i32, i64, u32, u64, i16, u16, i8, u8);

// Additional implementations for TensorView (TensorBase<&Storage>)
macro_rules! impl_scalar_mul_view {
    ($($scalar_type:ty),+) => {
        $(
            impl std::ops::Mul<$scalar_type> for &TensorView<'_> {
                type Output = Tensor;
                fn mul(self, other: $scalar_type) -> Self::Output {
                    self.mul_scalar(other).unwrap()
                }
            }

            impl std::ops::Mul<$scalar_type> for TensorView<'_> {
                type Output = Tensor;
                fn mul(self, other: $scalar_type) -> Self::Output {
                    self.mul_scalar(other).unwrap()
                }
            }
        )+
    };
}

impl_scalar_mul_view!(f32, f64, i32, i64, u32, u64, i16, u16, i8, u8);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_mul_basic() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0], vec![2, 2])?;

        let result = a.mul(&b)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[2.0, 6.0, 12.0, 20.0]);

        Ok(())
    }

    #[test]
    fn test_tensor_mul_scalar() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2])?;

        let result = a.mul_scalar(2.0f32)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[2.0, 4.0, 6.0, 8.0]);

        Ok(())
    }

    #[test]
    fn test_tensor_mul_different_dtypes() -> Result<()> {
        // Test f64
        let a = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![2.0f64, 3.0, 4.0, 5.0], vec![2, 2])?;
        let result = a.mul(&b)?;
        let data = result.as_slice::<f64>()?;
        assert_eq!(data, &[2.0, 6.0, 12.0, 20.0]);

        // Test i32
        let a = Tensor::from_vec(vec![2i32, 3, 4, 5], vec![2, 2])?;
        let b = Tensor::from_vec(vec![1i32, 2, 3, 4], vec![2, 2])?;
        let result = a.mul(&b)?;
        let data = result.as_slice::<i32>()?;
        assert_eq!(data, &[2, 6, 12, 20]);

        // Test u8
        let a = Tensor::from_vec(vec![2u8, 3, 4, 5], vec![2, 2])?;
        let b = Tensor::from_vec(vec![1u8, 2, 3, 4], vec![2, 2])?;
        let result = a.mul(&b)?;
        let data = result.as_slice::<u8>()?;
        assert_eq!(data, &[2, 6, 12, 20]);

        Ok(())
    }

    #[test]
    fn test_tensor_mul_half_precision() -> Result<()> {
        // Test f16
        let a = Tensor::from_vec(vec![f16::from_f32(2.0), f16::from_f32(3.0)], vec![2])?;
        let b = Tensor::from_vec(vec![f16::from_f32(1.5), f16::from_f32(2.0)], vec![2])?;
        let result = a.mul(&b)?;
        let data = result.as_slice::<f16>()?;
        assert!((data[0].to_f32() - 3.0).abs() < 1e-3);
        assert!((data[1].to_f32() - 6.0).abs() < 1e-3);

        // Test bf16
        let a = Tensor::from_vec(vec![bf16::from_f32(2.0), bf16::from_f32(3.0)], vec![2])?;
        let b = Tensor::from_vec(vec![bf16::from_f32(1.5), bf16::from_f32(2.0)], vec![2])?;
        let result = a.mul(&b)?;
        let data = result.as_slice::<bf16>()?;
        assert!((data[0].to_f32() - 3.0).abs() < 1e-2);
        assert!((data[1].to_f32() - 6.0).abs() < 1e-2);

        Ok(())
    }

    #[test]
    fn test_tensor_mul_same_types() -> Result<()> {
        // Test same dtypes and shapes - should work fine
        let a = Tensor::from_vec(vec![2.0f32, 3.0], vec![2])?;
        let b = Tensor::from_vec(vec![4.0f32, 5.0], vec![2])?;
        let result = a.mul(&b)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[8.0, 15.0]);

        // Test multiplication with ones
        let _a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3])?;
        let _b = Tensor::from_vec(vec![1.0f32, 1.0], vec![2])?;
        // Note: This would trigger debug_assert in debug mode, but passes in release
        // We're testing the happy path here

        Ok(())
    }

    #[test]
    fn test_tensor_mul_empty() -> Result<()> {
        let a = Tensor::from_vec(Vec::<f32>::new(), vec![0])?;
        let b = Tensor::from_vec(Vec::<f32>::new(), vec![0])?;

        let result = a.mul(&b)?;
        assert_eq!(result.numel(), 0);
        assert_eq!(result.dims(), &[0]);

        // Test scalar multiplication with empty tensor
        let result = a.mul_scalar(2.0f32)?;
        assert_eq!(result.numel(), 0);

        Ok(())
    }

    #[test]
    fn test_operator_overloading() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0], vec![2, 2])?;

        // Test tensor * tensor
        let result1 = &a * &b;
        let data1 = result1.as_slice::<f32>()?;
        assert_eq!(data1, &[2.0, 6.0, 12.0, 20.0]);

        // Test tensor * scalar
        let result2 = &a * 2.0f32;
        let data2 = result2.as_slice::<f32>()?;
        assert_eq!(data2, &[2.0, 4.0, 6.0, 8.0]);

        Ok(())
    }

    #[test]
    fn test_edge_cases() -> Result<()> {
        // Single element tensors
        let a = Tensor::from_vec(vec![5.0f32], vec![1])?;
        let b = Tensor::from_vec(vec![3.0f32], vec![1])?;
        let result = a.mul(&b)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[15.0]);

        // Scalar multiplication with single element
        let result = a.mul_scalar(2.0f32)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[10.0]);

        Ok(())
    }

    #[test]
    fn test_tensor_view_operations() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let b = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0], vec![2, 3])?;

        // Create views
        let view1 = a.reshape(vec![3, 2])?;
        let view2 = b.reshape(vec![3, 2])?;

        // Test multiplication with views
        let result = view1.mul(&view2)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[2.0, 6.0, 12.0, 20.0, 30.0, 42.0]);

        Ok(())
    }

    #[test]
    fn test_performance_comparison() -> Result<()> {
        use std::time::Instant;

        let size = 1000;
        let a = Tensor::from_vec((0..size).map(|i| i as f32).collect(), vec![size])?;
        let b = Tensor::from_vec((0..size).map(|i| (i + 1) as f32).collect(), vec![size])?;

        let start = Instant::now();
        let _result = a.mul(&b)?;
        let elapsed = start.elapsed();

        println!("   Multiplying {size} elements took: {elapsed:?}");

        // Should complete reasonably quickly
        assert!(elapsed.as_secs() < 1);

        Ok(())
    }

    #[test]
    fn test_broadcast_mul_same_shape() -> Result<()> {
        let a = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0], [2, 2])?;
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2])?;

        let result = a.broadcast_mul(&b)?;
        assert_eq!(result.to_flat_vec::<f32>()?, vec![2.0, 6.0, 12.0, 20.0]);
        Ok(())
    }

    #[test]
    fn test_broadcast_mul_scalar() -> Result<()> {
        let a = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0], [2, 2])?;
        let b = Tensor::from_vec(vec![2.0f32], [1, 1])?;

        let result = a.broadcast_mul(&b)?;
        assert_eq!(result.to_flat_vec::<f32>()?, vec![4.0, 6.0, 8.0, 10.0]);
        Ok(())
    }

    #[test]
    fn test_broadcast_mul_row_vector() -> Result<()> {
        let a = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0], [2, 2])?;
        let b = Tensor::from_vec(vec![2.0f32, 3.0], [1, 2])?;

        let result = a.broadcast_mul(&b)?;
        assert_eq!(result.to_flat_vec::<f32>()?, vec![4.0, 9.0, 8.0, 15.0]);
        Ok(())
    }

    #[test]
    fn test_broadcast_mul_column_vector() -> Result<()> {
        let a = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0], [2, 2])?;
        let b = Tensor::from_vec(vec![2.0f32, 3.0], [2, 1])?;

        let result = a.broadcast_mul(&b)?;
        assert_eq!(result.to_flat_vec::<f32>()?, vec![4.0, 6.0, 12.0, 15.0]);
        Ok(())
    }

    #[test]
    fn test_broadcast_mul_incompatible_shapes() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2])?;
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3])?;

        let result = a.broadcast_mul(&b);
        assert!(result.is_err());
        Ok(())
    }
}
