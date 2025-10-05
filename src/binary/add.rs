use anyhow::Result;
use half::{bf16, f16};

use crate::{
    backend::{global_backend, r#impl::OpsTrait},
    compute_broadcast_shape, DType, StorageTrait, Tensor, TensorBase, TensorElement, TensorView,
    UninitVec,
};

impl<S: StorageTrait> TensorBase<S> {
    /// Add two tensors element-wise (shapes must match exactly)
    #[inline(always)]
    pub fn add<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        // Check shape compatibility
        debug_assert_eq!(
            self.shape(),
            other.shape(),
            "Shape mismatch for add operation: {:?} vs {:?}",
            self.dims(),
            other.dims()
        );
        debug_assert_eq!(
            self.dtype(),
            other.dtype(),
            "Dtype mismatch for add operation: {:?} vs {:?}",
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
            self.add_contiguous(other)
        } else {
            self.add_non_contiguous(other)
        }
    }

    /// Add two tensors element-wise with broadcasting support
    ///
    /// If shapes don't match, attempts to broadcast them to a compatible shape.
    #[inline(always)]
    pub fn broadcast_add<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        // Check dtype compatibility
        if self.dtype() != other.dtype() {
            anyhow::bail!(
                "Dtype mismatch for add operation: {:?} vs {:?}",
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
            return self.add(other);
        }

        // Try to broadcast shapes
        let broadcast_shape = compute_broadcast_shape(self.shape(), other.shape())?;

        // Broadcast both tensors to the target shape
        let self_broadcasted = self.broadcast_to(broadcast_shape)?;
        let other_broadcasted = other.broadcast_to(broadcast_shape)?;

        // Perform addition on broadcasted tensors
        self_broadcasted.add(&other_broadcasted)
    }

    /// Optimized addition for contiguous tensors using backend acceleration
    #[inline(always)]
    fn add_contiguous<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        let numel = self.numel();
        let backend = global_backend();

        match self.dtype() {
            DType::Fp32 => {
                let a = self.as_slice::<f32>()?;
                let b = other.as_slice::<f32>()?;
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    backend.v_add_f32(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let a = self.as_slice::<f64>()?;
                let b = other.as_slice::<f64>()?;
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    backend.v_add_f64(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let a = self.as_slice::<f16>()?;
                let b = other.as_slice::<f16>()?;
                let out = UninitVec::<f16>::new(numel).init_with(|dst| {
                    backend.v_add_f16(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let a = self.as_slice::<bf16>()?;
                let b = other.as_slice::<bf16>()?;
                let out = UninitVec::<bf16>::new(numel).init_with(|dst| {
                    backend.v_add_bf16(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int8 => {
                let a = self.as_slice::<i8>()?;
                let b = other.as_slice::<i8>()?;
                let out = UninitVec::<i8>::new(numel).init_with(|dst| {
                    backend.v_add_i8(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int16 => {
                let a = self.as_slice::<i16>()?;
                let b = other.as_slice::<i16>()?;
                let out = UninitVec::<i16>::new(numel).init_with(|dst| {
                    backend.v_add_i16(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int32 => {
                let a = self.as_slice::<i32>()?;
                let b = other.as_slice::<i32>()?;
                let out = UninitVec::<i32>::new(numel).init_with(|dst| {
                    backend.v_add_i32(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int64 => {
                let a = self.as_slice::<i64>()?;
                let b = other.as_slice::<i64>()?;
                let out = UninitVec::<i64>::new(numel).init_with(|dst| {
                    backend.v_add_i64(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint8 => {
                let a = self.as_slice::<u8>()?;
                let b = other.as_slice::<u8>()?;
                let out = UninitVec::<u8>::new(numel).init_with(|dst| {
                    backend.v_add_u8(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint16 => {
                let a = self.as_slice::<u16>()?;
                let b = other.as_slice::<u16>()?;
                let out = UninitVec::<u16>::new(numel).init_with(|dst| {
                    backend.v_add_u16(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint32 => {
                let a = self.as_slice::<u32>()?;
                let b = other.as_slice::<u32>()?;
                let out = UninitVec::<u32>::new(numel).init_with(|dst| {
                    backend.v_add_u32(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint64 => {
                let a = self.as_slice::<u64>()?;
                let b = other.as_slice::<u64>()?;
                let out = UninitVec::<u64>::new(numel).init_with(|dst| {
                    backend.v_add_u64(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Addition not supported for dtype: {:?}", self.dtype()),
        }
    }

    #[inline(always)]
    fn add_non_contiguous<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        match self.dtype() {
            DType::Fp32 => {
                let numel = self.numel();
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    for (idx, (self_item, other_item)) in self
                        .iter_with_meta::<f32>()
                        .zip(other.iter_with_meta::<f32>())
                        .enumerate()
                    {
                        dst[idx] = *self_item.value + *other_item.value;
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
                        dst[idx] = *self_item.value + *other_item.value;
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
                            f16::from_f32(self_item.value.to_f32() + other_item.value.to_f32());
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
                            bf16::from_f32(self_item.value.to_f32() + other_item.value.to_f32());
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
                        dst[idx] = *self_item.value + *other_item.value;
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
                        dst[idx] = *self_item.value + *other_item.value;
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
                        dst[idx] = *self_item.value + *other_item.value;
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
                        dst[idx] = *self_item.value + *other_item.value;
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
                        dst[idx] = *self_item.value + *other_item.value;
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
                        dst[idx] = *self_item.value + *other_item.value;
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
                        dst[idx] = *self_item.value + *other_item.value;
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
                        dst[idx] = *self_item.value + *other_item.value;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Addition not supported for dtype: {:?}", self.dtype()),
        }
    }

    /// Add scalar to tensor
    #[inline(always)]
    pub fn add_scalar<T: TensorElement + std::ops::Add<Output = T>>(
        &self,
        scalar: T,
    ) -> Result<Tensor> {
        // Check dtype compatibility - scalar type must match tensor dtype exactly
        debug_assert_eq!(
            T::DTYPE,
            self.dtype(),
            "Scalar type mismatch for add_scalar operation: scalar dtype {:?} vs tensor dtype {:?}",
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
            self.add_scalar_contiguous(scalar)
        } else {
            self.add_scalar_non_contiguous(scalar)
        }
    }

    /// Direct scalar addition for contiguous tensors - no type conversion needed
    #[inline(always)]
    fn add_scalar_contiguous<T: TensorElement + std::ops::Add<Output = T>>(
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
                backend.v_add_scalar_f32(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Fp64 => {
                let input_data = self.as_slice::<f64>()?;
                let s = unsafe { std::mem::transmute_copy::<T, f64>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_add_scalar_f64(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Fp16 => {
                let input_data = self.as_slice::<f16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, f16>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_add_scalar_f16(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Bf16 => {
                let input_data = self.as_slice::<bf16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, bf16>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_add_scalar_bf16(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Int8 => {
                let input_data = self.as_slice::<i8>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i8>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_add_scalar_i8(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Int16 => {
                let input_data = self.as_slice::<i16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i16>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_add_scalar_i16(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Int32 => {
                let input_data = self.as_slice::<i32>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i32>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_add_scalar_i32(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Int64 => {
                let input_data = self.as_slice::<i64>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i64>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_add_scalar_i64(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Uint8 => {
                let input_data = self.as_slice::<u8>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u8>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_add_scalar_u8(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Uint16 => {
                let input_data = self.as_slice::<u16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u16>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_add_scalar_u16(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Uint32 => {
                let input_data = self.as_slice::<u32>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u32>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_add_scalar_u32(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            DType::Uint64 => {
                let input_data = self.as_slice::<u64>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u64>(&scalar) };
                let mut output = UninitVec::new(numel);
                backend.v_add_scalar_u64(input_data, s, output.as_mut_slice());
                Tensor::from_vec(unsafe { output.finalize() }, self.shape)
            }
            _ => anyhow::bail!(
                "Scalar addition not supported for dtype: {:?}",
                self.dtype()
            ),
        }
    }

    /// Direct scalar addition for non-contiguous tensors - no type conversion needed
    #[inline(always)]
    fn add_scalar_non_contiguous<T: TensorElement + std::ops::Add<Output = T>>(
        &self,
        scalar: T,
    ) -> Result<Tensor> {
        // Since types match exactly, we can use the scalar directly
        match self.dtype() {
            DType::Fp32 => {
                let s = unsafe { std::mem::transmute_copy::<T, f32>(&scalar) };
                self.map_non_contiguous::<f32>(|&x| x + s)
            }
            DType::Fp64 => {
                let s = unsafe { std::mem::transmute_copy::<T, f64>(&scalar) };
                self.map_non_contiguous::<f64>(|&x| x + s)
            }
            DType::Fp16 => {
                let s = unsafe { std::mem::transmute_copy::<T, f16>(&scalar) };
                self.map_non_contiguous::<f16>(|&x| x + s)
            }
            DType::Bf16 => {
                let s = unsafe { std::mem::transmute_copy::<T, bf16>(&scalar) };
                self.map_non_contiguous::<bf16>(|&x| x + s)
            }
            DType::Int8 => {
                let s = unsafe { std::mem::transmute_copy::<T, i8>(&scalar) };
                self.map_non_contiguous::<i8>(|&x| x + s)
            }
            DType::Int16 => {
                let s = unsafe { std::mem::transmute_copy::<T, i16>(&scalar) };
                self.map_non_contiguous::<i16>(|&x| x + s)
            }
            DType::Int32 => {
                let s = unsafe { std::mem::transmute_copy::<T, i32>(&scalar) };
                self.map_non_contiguous::<i32>(|&x| x + s)
            }
            DType::Int64 => {
                let s = unsafe { std::mem::transmute_copy::<T, i64>(&scalar) };
                self.map_non_contiguous::<i64>(|&x| x + s)
            }
            DType::Uint8 => {
                let s = unsafe { std::mem::transmute_copy::<T, u8>(&scalar) };
                self.map_non_contiguous::<u8>(|&x| x + s)
            }
            DType::Uint16 => {
                let s = unsafe { std::mem::transmute_copy::<T, u16>(&scalar) };
                self.map_non_contiguous::<u16>(|&x| x + s)
            }
            DType::Uint32 => {
                let s = unsafe { std::mem::transmute_copy::<T, u32>(&scalar) };
                self.map_non_contiguous::<u32>(|&x| x + s)
            }
            DType::Uint64 => {
                let s = unsafe { std::mem::transmute_copy::<T, u64>(&scalar) };
                self.map_non_contiguous::<u64>(|&x| x + s)
            }
            DType::Bool => anyhow::bail!("Scalar addition not supported for Bool dtype"),
            _ => anyhow::bail!(
                "Scalar addition not supported for dtype: {:?}",
                self.dtype()
            ),
        }
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Add<&TensorBase<S2>> for &TensorBase<S1> {
    type Output = Tensor;
    fn add(self, other: &TensorBase<S2>) -> Self::Output {
        TensorBase::broadcast_add(self, other).expect("Tensor addition failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Add<TensorBase<S2>> for &TensorBase<S1> {
    type Output = Tensor;
    fn add(self, other: TensorBase<S2>) -> Self::Output {
        TensorBase::broadcast_add(self, &other).expect("Tensor addition failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Add<&TensorBase<S2>> for TensorBase<S1> {
    type Output = Tensor;
    fn add(self, other: &TensorBase<S2>) -> Self::Output {
        TensorBase::broadcast_add(&self, other).expect("Tensor addition failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Add<TensorBase<S2>> for TensorBase<S1> {
    type Output = Tensor;
    fn add(self, other: TensorBase<S2>) -> Self::Output {
        TensorBase::broadcast_add(&self, &other).expect("Tensor addition failed")
    }
}

// Scalar addition operator implementations using macro to avoid recursion
macro_rules! impl_scalar_add {
    ($($scalar_type:ty),+) => {
        $(
            impl std::ops::Add<$scalar_type> for &Tensor {
                type Output = Tensor;
                fn add(self, other: $scalar_type) -> Self::Output {
                    self.add_scalar(other).unwrap()
                }
            }

            impl std::ops::Add<$scalar_type> for Tensor {
                type Output = Tensor;
                fn add(self, other: $scalar_type) -> Self::Output {
                    self.add_scalar(other).unwrap()
                }
            }
        )+
    };
}

// Generate scalar addition implementations for common numeric types
impl_scalar_add!(f32, f64, i32, i64, u32, u64, i16, u16, i8, u8);

// Additional implementations for TensorView (TensorBase<&Storage>)
macro_rules! impl_scalar_add_view {
    ($($scalar_type:ty),+) => {
        $(
            impl std::ops::Add<$scalar_type> for &TensorView<'_> {
                type Output = Tensor;
                fn add(self, other: $scalar_type) -> Self::Output {
                    self.add_scalar(other).unwrap()
                }
            }

            impl std::ops::Add<$scalar_type> for TensorView<'_> {
                type Output = Tensor;
                fn add(self, other: $scalar_type) -> Self::Output {
                    self.add_scalar(other).unwrap()
                }
            }
        )+
    };
}

impl_scalar_add_view!(f32, f64, i32, i64, u32, u64, i16, u16, i8, u8);

#[cfg(test)]
mod tests {
    // ✅ Tensor + Tensor
    // ✅ TensorView + TensorView
    // ✅ Tensor + TensorView
    // ✅ TensorView + Tensor
    // ✅ Tensor + Scalar
    // ✅ TensorView + Scalar

    use crate::{DType, Tensor};
    use anyhow::Result;

    #[test]
    fn test_tensor_add() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3])?;
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], [3])?;

        // Test add method
        let result = a.add(&b)?;
        assert_eq!(result.dtype(), DType::Fp32);
        assert_eq!(result.dims(), &[3]);
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[5.0, 7.0, 9.0]);

        // Test + operator
        let result_op = &a + &b;
        let data_op = result_op.as_slice::<f32>()?;
        assert_eq!(data_op, &[5.0, 7.0, 9.0]);

        Ok(())
    }

    #[test]
    fn test_tensor_add_scalar() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3])?;

        // Test add_scalar method
        let result = a.add_scalar(5.0f32)?;
        assert_eq!(result.dtype(), DType::Fp32);
        assert_eq!(result.dims(), &[3]);
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[6.0, 7.0, 8.0]);

        // Test + operator with scalar
        let result_op = &a + 5.0f32;
        let data_op = result_op.as_slice::<f32>()?;
        assert_eq!(data_op, &[6.0, 7.0, 8.0]);

        Ok(())
    }

    #[test]
    fn test_tensor_add_empty() -> Result<()> {
        let a = Tensor::from_vec(Vec::<f32>::new(), [0])?;
        let b = Tensor::from_vec(Vec::<f32>::new(), [0])?;

        let result = a.add(&b)?;
        assert_eq!(result.dtype(), DType::Fp32);
        assert_eq!(result.dims(), &[0]);
        assert_eq!(result.numel(), 0);

        let result_scalar = a.add_scalar(5.0f32)?;
        assert_eq!(result_scalar.dtype(), DType::Fp32);
        assert_eq!(result_scalar.dims(), &[0]);
        assert_eq!(result_scalar.numel(), 0);

        Ok(())
    }

    #[test]
    fn test_tensor_add_same_types() -> Result<()> {
        // Test same dtypes and shapes - should work fine
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3])?;
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], [3])?;
        let result = a.add(&b)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[5.0, 7.0, 9.0]);

        // Test same shapes with different values
        let _a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3])?;
        let _b = Tensor::from_vec(vec![1.0f32, 1.0], [2])?;
        // Note: This would trigger debug_assert in debug mode, but passes in release
        // We're testing the happy path here

        Ok(())
    }

    #[test]
    fn test_tensor_view_operations() -> Result<()> {
        let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let data2 = vec![5.0f32, 6.0, 7.0, 8.0];
        let tensor1 = Tensor::from_vec(data1, vec![2, 2])?;
        let tensor2 = Tensor::from_vec(data2, vec![2, 2])?;

        let view1 = tensor1.clone().reshape(vec![2, 2])?;
        let view2 = tensor2.clone().reshape(vec![2, 2])?;

        // Test view + view (method and operator)
        let result = view1.add(&view2)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[6.0, 8.0, 10.0, 12.0]);

        let result_op = &view1 + &view2;
        let data_op = result_op.as_slice::<f32>()?;
        assert_eq!(data_op, &[6.0, 8.0, 10.0, 12.0]);

        // Test tensor + view (method and operator)
        let result = tensor1.add(&view2)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[6.0, 8.0, 10.0, 12.0]);

        let result_op = &tensor1 + &view2;
        let data_op = result_op.as_slice::<f32>()?;
        assert_eq!(data_op, &[6.0, 8.0, 10.0, 12.0]);

        // Test view + tensor (method and operator)
        let result = view1.add(&tensor2)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[6.0, 8.0, 10.0, 12.0]);

        let result_op = &view1 + &tensor2;
        let data_op = result_op.as_slice::<f32>()?;
        assert_eq!(data_op, &[6.0, 8.0, 10.0, 12.0]);

        // Test view + scalar (method and operator)
        let result = view1.add_scalar(5.0f32)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[6.0, 7.0, 8.0, 9.0]);

        let result_op = &view1 + 5.0f32;
        let data_op = result_op.as_slice::<f32>()?;
        assert_eq!(data_op, &[6.0, 7.0, 8.0, 9.0]);

        Ok(())
    }

    #[test]
    fn test_different_dtypes() -> Result<()> {
        // Test i32
        let int_tensor = Tensor::from_vec(vec![1i32, 2, 3, 4], vec![2, 2])?;
        let int_result = int_tensor.add_scalar(5i32)?;
        let int_data = int_result.as_slice::<i32>()?;
        assert_eq!(int_data, &[6, 7, 8, 9]);

        // Test f64
        let f64_tensor = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2])?;
        let f64_result = f64_tensor.add_scalar(5.0f64)?;
        let f64_data = f64_result.as_slice::<f64>()?;
        assert!((f64_data[0] - 6.0).abs() < 1e-10);
        assert!((f64_data[1] - 7.0).abs() < 1e-10);
        assert!((f64_data[2] - 8.0).abs() < 1e-10);
        assert!((f64_data[3] - 9.0).abs() < 1e-10);

        // Test u8
        let u8_tensor = Tensor::from_vec(vec![1u8, 2, 3, 4], vec![2, 2])?;
        let u8_result = u8_tensor.add_scalar(5u8)?;
        let u8_data = u8_result.as_slice::<u8>()?;
        assert_eq!(u8_data, &[6, 7, 8, 9]);

        Ok(())
    }

    #[test]
    fn test_operator_overloading() -> Result<()> {
        let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let data2 = vec![5.0f32, 6.0, 7.0, 8.0];
        let tensor1 = Tensor::from_vec(data1, vec![2, 2])?;
        let tensor2 = Tensor::from_vec(data2, vec![2, 2])?;

        // Test all + operator combinations
        let result1 = &tensor1 + &tensor2;
        let result2 = tensor1.clone() + tensor2.clone();
        let result3 = &tensor1 + tensor2.clone();
        let result4 = tensor1.clone() + &tensor2;
        let result5 = &tensor1 + 10.0f32;
        let result6 = tensor1 + 5.0f32;

        let data1 = result1.as_slice::<f32>()?;
        let data2 = result2.as_slice::<f32>()?;
        let data3 = result3.as_slice::<f32>()?;
        let data4 = result4.as_slice::<f32>()?;
        let data5 = result5.as_slice::<f32>()?;
        let data6 = result6.as_slice::<f32>()?;

        assert_eq!(data1, &[6.0, 8.0, 10.0, 12.0]);
        assert_eq!(data2, &[6.0, 8.0, 10.0, 12.0]);
        assert_eq!(data3, &[6.0, 8.0, 10.0, 12.0]);
        assert_eq!(data4, &[6.0, 8.0, 10.0, 12.0]);
        assert_eq!(data5, &[11.0, 12.0, 13.0, 14.0]);
        assert_eq!(data6, &[6.0, 7.0, 8.0, 9.0]);

        Ok(())
    }

    #[test]
    fn test_performance_comparison() -> Result<()> {
        let size = 100;
        let data1: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
        let data2: Vec<f32> = (0..size * size).map(|i| (i * 2) as f32).collect();

        let tensor1 = Tensor::from_vec(data1, vec![size, size])?;
        let tensor2 = Tensor::from_vec(data2, vec![size, size])?;

        // Contiguous tensors
        let start = std::time::Instant::now();
        let _result1 = &tensor1 + &tensor2;
        let _contiguous_time = start.elapsed();

        // Non-contiguous tensors
        let view1 = tensor1.clone().reshape(vec![size, size])?;
        let view2 = tensor2.clone().reshape(vec![size, size])?;

        let start = std::time::Instant::now();
        let _result2 = &view1 + &view2;
        let _non_contiguous_time = start.elapsed();

        // Verify correctness
        let result1 = &tensor1 + &tensor2;
        let result2 = &view1 + &view2;

        let data1 = result1.as_slice::<f32>()?;
        let data2 = result2.as_slice::<f32>()?;

        for i in 0..10 {
            assert!((data1[i] - data2[i]).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_edge_cases() -> Result<()> {
        // Single element tensor
        let single = Tensor::from_vec(vec![5.0f32], vec![1])?;
        let result = &single + 3.0f32;
        assert_eq!(result.as_slice::<f32>()?[0], 8.0);

        // High dimensional tensor
        let high_dim = Tensor::from_vec(vec![1.0f32], vec![1, 1, 1, 1, 1, 1, 1, 1])?;
        let result = &high_dim + 10.0f32;
        assert_eq!(result.as_slice::<f32>()?[0], 11.0);

        // Large tensor
        let size = 1000;
        let large_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let large_tensor = Tensor::from_vec(large_data, vec![size])?;
        let result = &large_tensor + 1.0f32;

        let data = result.as_slice::<f32>()?;
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[size - 1], size as f32);

        Ok(())
    }

    #[test]
    fn test_broadcast_add_same_shape() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2])?;
        let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], [2, 2])?;

        let result = a.broadcast_add(&b)?;
        assert_eq!(result.to_flat_vec::<f32>()?, vec![6.0, 8.0, 10.0, 12.0]);
        Ok(())
    }

    #[test]
    fn test_broadcast_add_scalar() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2])?;
        let b = Tensor::from_vec(vec![5.0f32], [1, 1])?;

        let result = a.broadcast_add(&b)?;
        assert_eq!(result.to_flat_vec::<f32>()?, vec![6.0, 7.0, 8.0, 9.0]);
        Ok(())
    }

    #[test]
    fn test_broadcast_add_row_vector() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2])?;
        let b = Tensor::from_vec(vec![10.0f32, 20.0], [1, 2])?;

        let result = a.broadcast_add(&b)?;
        assert_eq!(result.to_flat_vec::<f32>()?, vec![11.0, 22.0, 13.0, 24.0]);
        Ok(())
    }

    #[test]
    fn test_broadcast_add_column_vector() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2])?;
        let b = Tensor::from_vec(vec![10.0f32, 20.0], [2, 1])?;

        let result = a.broadcast_add(&b)?;
        assert_eq!(result.to_flat_vec::<f32>()?, vec![11.0, 12.0, 23.0, 24.0]);
        Ok(())
    }

    #[test]
    fn test_broadcast_add_incompatible_shapes() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2])?;
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3])?;

        let result = a.broadcast_add(&b);
        assert!(result.is_err());
        Ok(())
    }
}
