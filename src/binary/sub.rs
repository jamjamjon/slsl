use anyhow::Result;
use half::{bf16, f16};

use crate::{
    backend::{global_backend, r#impl::OpsTrait},
    DType, StorageTrait, Tensor, TensorBase, TensorElement, TensorView, UninitVec,
};

impl<S: StorageTrait> TensorBase<S> {
    /// Subtract two tensors element-wise (shapes must match exactly)
    #[inline(always)]
    pub fn sub<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        // Check dtype compatibility
        debug_assert_eq!(
            self.dtype(),
            other.dtype(),
            "Dtype mismatch for sub operation"
        );
        debug_assert_eq!(
            self.shape(),
            other.shape(),
            "Shape mismatch for sub operation"
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
            self.sub_contiguous(other)
        } else {
            self.sub_non_contiguous(other)
        }
    }

    /// Optimized subtraction for contiguous tensors using backend acceleration
    #[inline(always)]
    fn sub_contiguous<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        let numel = self.numel();
        let backend = global_backend();

        match self.dtype() {
            DType::Fp32 => {
                let a = self.as_slice::<f32>()?;
                let b = other.as_slice::<f32>()?;
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    backend.v_sub_f32(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let a = self.as_slice::<f64>()?;
                let b = other.as_slice::<f64>()?;
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    backend.v_sub_f64(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let a = self.as_slice::<f16>()?;
                let b = other.as_slice::<f16>()?;
                let out = UninitVec::<f16>::new(numel).init_with(|dst| {
                    backend.v_sub_f16(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let a = self.as_slice::<bf16>()?;
                let b = other.as_slice::<bf16>()?;
                let out = UninitVec::<bf16>::new(numel).init_with(|dst| {
                    backend.v_sub_bf16(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int8 => {
                let a = self.as_slice::<i8>()?;
                let b = other.as_slice::<i8>()?;
                let out = UninitVec::<i8>::new(numel).init_with(|dst| {
                    backend.v_sub_i8(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int16 => {
                let a = self.as_slice::<i16>()?;
                let b = other.as_slice::<i16>()?;
                let out = UninitVec::<i16>::new(numel).init_with(|dst| {
                    backend.v_sub_i16(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int32 => {
                let a = self.as_slice::<i32>()?;
                let b = other.as_slice::<i32>()?;
                let out = UninitVec::<i32>::new(numel).init_with(|dst| {
                    backend.v_sub_i32(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int64 => {
                let a = self.as_slice::<i64>()?;
                let b = other.as_slice::<i64>()?;
                let out = UninitVec::<i64>::new(numel).init_with(|dst| {
                    backend.v_sub_i64(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint8 => {
                let a = self.as_slice::<u8>()?;
                let b = other.as_slice::<u8>()?;
                let out = UninitVec::<u8>::new(numel).init_with(|dst| {
                    backend.v_sub_u8(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint16 => {
                let a = self.as_slice::<u16>()?;
                let b = other.as_slice::<u16>()?;
                let out = UninitVec::<u16>::new(numel).init_with(|dst| {
                    backend.v_sub_u16(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint32 => {
                let a = self.as_slice::<u32>()?;
                let b = other.as_slice::<u32>()?;
                let out = UninitVec::<u32>::new(numel).init_with(|dst| {
                    backend.v_sub_u32(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint64 => {
                let a = self.as_slice::<u64>()?;
                let b = other.as_slice::<u64>()?;
                let out = UninitVec::<u64>::new(numel).init_with(|dst| {
                    backend.v_sub_u64(a, b, dst);
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Subtraction not supported for dtype: {:?}", self.dtype()),
        }
    }

    /// Subtraction for non-contiguous tensors using iter
    #[inline(always)]
    fn sub_non_contiguous<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        match self.dtype() {
            DType::Fp32 => {
                let numel = self.numel();
                let out = UninitVec::<f32>::new(numel).init_with(|dst| {
                    for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate()
                    {
                        let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                        let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                        let self_val = unsafe { *(self_ptr as *const f32) };
                        let other_val = unsafe { *(other_ptr as *const f32) };
                        dst[idx] = self_val - other_val;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp64 => {
                let numel = self.numel();
                let out = UninitVec::<f64>::new(numel).init_with(|dst| {
                    for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate()
                    {
                        let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                        let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                        let self_val = unsafe { *(self_ptr as *const f64) };
                        let other_val = unsafe { *(other_ptr as *const f64) };
                        dst[idx] = self_val - other_val;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Fp16 => {
                let numel = self.numel();
                let out = UninitVec::<f16>::new(numel).init_with(|dst| {
                    for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate()
                    {
                        let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                        let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                        let self_val = unsafe { *(self_ptr as *const f16) };
                        let other_val = unsafe { *(other_ptr as *const f16) };
                        dst[idx] = f16::from_f32(self_val.to_f32() - other_val.to_f32());
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Bf16 => {
                let numel = self.numel();
                let out = UninitVec::<bf16>::new(numel).init_with(|dst| {
                    for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate()
                    {
                        let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                        let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                        let self_val = unsafe { *(self_ptr as *const bf16) };
                        let other_val = unsafe { *(other_ptr as *const bf16) };
                        dst[idx] = bf16::from_f32(self_val.to_f32() - other_val.to_f32());
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int8 => {
                let numel = self.numel();
                let out = UninitVec::<i8>::new(numel).init_with(|dst| {
                    for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate()
                    {
                        let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                        let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                        let self_val = unsafe { *(self_ptr as *const i8) };
                        let other_val = unsafe { *(other_ptr as *const i8) };
                        dst[idx] = self_val - other_val;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int16 => {
                let numel = self.numel();
                let out = UninitVec::<i16>::new(numel).init_with(|dst| {
                    for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate()
                    {
                        let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                        let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                        let self_val = unsafe { *(self_ptr as *const i16) };
                        let other_val = unsafe { *(other_ptr as *const i16) };
                        dst[idx] = self_val - other_val;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int32 => {
                let numel = self.numel();
                let out = UninitVec::<i32>::new(numel).init_with(|dst| {
                    for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate()
                    {
                        let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                        let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                        let self_val = unsafe { *(self_ptr as *const i32) };
                        let other_val = unsafe { *(other_ptr as *const i32) };
                        dst[idx] = self_val - other_val;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Int64 => {
                let numel = self.numel();
                let out = UninitVec::<i64>::new(numel).init_with(|dst| {
                    for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate()
                    {
                        let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                        let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                        let self_val = unsafe { *(self_ptr as *const i64) };
                        let other_val = unsafe { *(other_ptr as *const i64) };
                        dst[idx] = self_val - other_val;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint8 => {
                let numel = self.numel();
                let out = UninitVec::<u8>::new(numel).init_with(|dst| {
                    for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate()
                    {
                        let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                        let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                        let self_val = unsafe { *self_ptr };
                        let other_val = unsafe { *other_ptr };
                        dst[idx] = self_val - other_val;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint16 => {
                let numel = self.numel();
                let out = UninitVec::<u16>::new(numel).init_with(|dst| {
                    for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate()
                    {
                        let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                        let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                        let self_val = unsafe { *(self_ptr as *const u16) };
                        let other_val = unsafe { *(other_ptr as *const u16) };
                        dst[idx] = self_val - other_val;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint32 => {
                let numel = self.numel();
                let out = UninitVec::<u32>::new(numel).init_with(|dst| {
                    for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate()
                    {
                        let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                        let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                        let self_val = unsafe { *(self_ptr as *const u32) };
                        let other_val = unsafe { *(other_ptr as *const u32) };
                        dst[idx] = self_val - other_val;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            DType::Uint64 => {
                let numel = self.numel();
                let out = UninitVec::<u64>::new(numel).init_with(|dst| {
                    for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate()
                    {
                        let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                        let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                        let self_val = unsafe { *(self_ptr as *const u64) };
                        let other_val = unsafe { *(other_ptr as *const u64) };
                        dst[idx] = self_val - other_val;
                    }
                });
                Tensor::from_vec(out, self.shape)
            }
            _ => anyhow::bail!("Subtraction not supported for dtype: {:?}", self.dtype()),
        }
    }

    /// Subtract a scalar from all elements of the tensor
    #[inline(always)]
    pub fn sub_scalar<T: TensorElement + std::ops::Sub<Output = T>>(
        &self,
        scalar: T,
    ) -> Result<Tensor> {
        // Check dtype compatibility - scalar type must match tensor dtype exactly
        debug_assert_eq!(
            T::DTYPE,
            self.dtype(),
            "Scalar type mismatch for sub_scalar operation: scalar dtype {:?} vs tensor dtype {:?}",
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
            self.sub_scalar_contiguous(scalar)
        } else {
            self.sub_scalar_non_contiguous(scalar)
        }
    }

    /// Direct scalar subtraction for contiguous tensors - no type conversion needed
    #[inline(always)]
    fn sub_scalar_contiguous<T: TensorElement + std::ops::Sub<Output = T>>(
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
                let mut output = UninitVec::<f32>::new(numel);
                backend.v_sub_scalar_f32(input_data, s, output.as_mut_slice());
                let output = unsafe { output.finalize() };
                Tensor::from_vec(output, self.shape)
            }
            DType::Fp64 => {
                let input_data = self.as_slice::<f64>()?;
                let s = unsafe { std::mem::transmute_copy::<T, f64>(&scalar) };
                let mut output = UninitVec::<f64>::new(numel);
                backend.v_sub_scalar_f64(input_data, s, output.as_mut_slice());
                let output = unsafe { output.finalize() };
                Tensor::from_vec(output, self.shape)
            }
            DType::Fp16 => {
                let input_data = self.as_slice::<f16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, f16>(&scalar) };
                let mut output = UninitVec::<f16>::new(numel);
                backend.v_sub_scalar_f16(input_data, s, output.as_mut_slice());
                let output = unsafe { output.finalize() };
                Tensor::from_vec(output, self.shape)
            }
            DType::Bf16 => {
                let input_data = self.as_slice::<bf16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, bf16>(&scalar) };
                let mut output = UninitVec::<bf16>::new(numel);
                backend.v_sub_scalar_bf16(input_data, s, output.as_mut_slice());
                let output = unsafe { output.finalize() };
                Tensor::from_vec(output, self.shape)
            }
            DType::Int8 => {
                let input_data = self.as_slice::<i8>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i8>(&scalar) };
                let mut output = UninitVec::<i8>::new(numel);
                backend.v_sub_scalar_i8(input_data, s, output.as_mut_slice());
                let output = unsafe { output.finalize() };
                Tensor::from_vec(output, self.shape)
            }
            DType::Int16 => {
                let input_data = self.as_slice::<i16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i16>(&scalar) };
                let mut output = UninitVec::<i16>::new(numel);
                backend.v_sub_scalar_i16(input_data, s, output.as_mut_slice());
                let output = unsafe { output.finalize() };
                Tensor::from_vec(output, self.shape)
            }
            DType::Int32 => {
                let input_data = self.as_slice::<i32>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i32>(&scalar) };
                let mut output = UninitVec::<i32>::new(numel);
                backend.v_sub_scalar_i32(input_data, s, output.as_mut_slice());
                let output = unsafe { output.finalize() };
                Tensor::from_vec(output, self.shape)
            }
            DType::Int64 => {
                let input_data = self.as_slice::<i64>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i64>(&scalar) };
                let mut output = UninitVec::<i64>::new(numel);
                backend.v_sub_scalar_i64(input_data, s, output.as_mut_slice());
                let output = unsafe { output.finalize() };
                Tensor::from_vec(output, self.shape)
            }
            DType::Uint8 => {
                let input_data = self.as_slice::<u8>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u8>(&scalar) };
                let mut output = UninitVec::<u8>::new(numel);
                backend.v_sub_scalar_u8(input_data, s, output.as_mut_slice());
                let output = unsafe { output.finalize() };
                Tensor::from_vec(output, self.shape)
            }
            DType::Uint16 => {
                let input_data = self.as_slice::<u16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u16>(&scalar) };
                let mut output = UninitVec::<u16>::new(numel);
                backend.v_sub_scalar_u16(input_data, s, output.as_mut_slice());
                let output = unsafe { output.finalize() };
                Tensor::from_vec(output, self.shape)
            }
            DType::Uint32 => {
                let input_data = self.as_slice::<u32>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u32>(&scalar) };
                let mut output = UninitVec::<u32>::new(numel);
                backend.v_sub_scalar_u32(input_data, s, output.as_mut_slice());
                let output = unsafe { output.finalize() };
                Tensor::from_vec(output, self.shape)
            }
            DType::Uint64 => {
                let input_data = self.as_slice::<u64>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u64>(&scalar) };
                let mut output = UninitVec::<u64>::new(numel);
                backend.v_sub_scalar_u64(input_data, s, output.as_mut_slice());
                let output = unsafe { output.finalize() };
                Tensor::from_vec(output, self.shape)
            }
            _ => anyhow::bail!(
                "Scalar subtraction not supported for dtype: {:?}",
                self.dtype()
            ),
        }
    }

    /// Direct scalar subtraction for non-contiguous tensors - no type conversion needed
    #[inline(always)]
    fn sub_scalar_non_contiguous<T: TensorElement + std::ops::Sub<Output = T>>(
        &self,
        scalar: T,
    ) -> Result<Tensor> {
        // Since types match exactly, we can use the scalar directly
        match self.dtype() {
            DType::Fp32 => {
                let s = unsafe { std::mem::transmute_copy::<T, f32>(&scalar) };
                self.map_non_contiguous::<f32>(|&x| x - s)
            }
            DType::Fp64 => {
                let s = unsafe { std::mem::transmute_copy::<T, f64>(&scalar) };
                self.map_non_contiguous::<f64>(|&x| x - s)
            }
            DType::Fp16 => {
                let s = unsafe { std::mem::transmute_copy::<T, f16>(&scalar) };
                self.map_non_contiguous::<f16>(|&x| x - s)
            }
            DType::Bf16 => {
                let s = unsafe { std::mem::transmute_copy::<T, bf16>(&scalar) };
                self.map_non_contiguous::<bf16>(|&x| x - s)
            }
            DType::Int8 => {
                let s = unsafe { std::mem::transmute_copy::<T, i8>(&scalar) };
                self.map_non_contiguous::<i8>(|&x| x - s)
            }
            DType::Int16 => {
                let s = unsafe { std::mem::transmute_copy::<T, i16>(&scalar) };
                self.map_non_contiguous::<i16>(|&x| x - s)
            }
            DType::Int32 => {
                let s = unsafe { std::mem::transmute_copy::<T, i32>(&scalar) };
                self.map_non_contiguous::<i32>(|&x| x - s)
            }
            DType::Int64 => {
                let s = unsafe { std::mem::transmute_copy::<T, i64>(&scalar) };
                self.map_non_contiguous::<i64>(|&x| x - s)
            }
            DType::Uint8 => {
                let s = unsafe { std::mem::transmute_copy::<T, u8>(&scalar) };
                self.map_non_contiguous::<u8>(|&x| x - s)
            }
            DType::Uint16 => {
                let s = unsafe { std::mem::transmute_copy::<T, u16>(&scalar) };
                self.map_non_contiguous::<u16>(|&x| x - s)
            }
            DType::Uint32 => {
                let s = unsafe { std::mem::transmute_copy::<T, u32>(&scalar) };
                self.map_non_contiguous::<u32>(|&x| x - s)
            }
            DType::Uint64 => {
                let s = unsafe { std::mem::transmute_copy::<T, u64>(&scalar) };
                self.map_non_contiguous::<u64>(|&x| x - s)
            }
            DType::Bool => anyhow::bail!("Scalar subtraction not supported for Bool dtype"),
            _ => anyhow::bail!(
                "Scalar subtraction not supported for dtype: {:?}",
                self.dtype()
            ),
        }
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Sub<&TensorBase<S2>> for &TensorBase<S1> {
    type Output = Tensor;
    fn sub(self, other: &TensorBase<S2>) -> Self::Output {
        TensorBase::sub(self, other).expect("Tensor subtraction failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Sub<TensorBase<S2>> for &TensorBase<S1> {
    type Output = Tensor;
    fn sub(self, other: TensorBase<S2>) -> Self::Output {
        TensorBase::sub(self, &other).expect("Tensor subtraction failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Sub<&TensorBase<S2>> for TensorBase<S1> {
    type Output = Tensor;
    fn sub(self, other: &TensorBase<S2>) -> Self::Output {
        TensorBase::sub(&self, other).expect("Tensor subtraction failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Sub<TensorBase<S2>> for TensorBase<S1> {
    type Output = Tensor;
    fn sub(self, other: TensorBase<S2>) -> Self::Output {
        TensorBase::sub(&self, &other).expect("Tensor subtraction failed")
    }
}

// Scalar subtraction operator implementations using macro to avoid recursion
macro_rules! impl_scalar_sub {
    ($($scalar_type:ty),+) => {
        $(
            impl std::ops::Sub<$scalar_type> for &Tensor {
                type Output = Tensor;
                fn sub(self, other: $scalar_type) -> Self::Output {
                    self.sub_scalar(other).unwrap()
                }
            }

            impl std::ops::Sub<$scalar_type> for Tensor {
                type Output = Tensor;
                fn sub(self, other: $scalar_type) -> Self::Output {
                    self.sub_scalar(other).unwrap()
                }
            }
        )+
    };
}

// Generate scalar subtraction implementations for common numeric types
impl_scalar_sub!(f32, f64, i32, i64, u32, u64, i16, u16, i8, u8);

// Additional implementations for TensorView (TensorBase<&Storage>)
macro_rules! impl_scalar_sub_view {
    ($($scalar_type:ty),+) => {
        $(
            impl std::ops::Sub<$scalar_type> for &TensorView<'_> {
                type Output = Tensor;
                fn sub(self, other: $scalar_type) -> Self::Output {
                    self.sub_scalar(other).unwrap()
                }
            }

            impl std::ops::Sub<$scalar_type> for TensorView<'_> {
                type Output = Tensor;
                fn sub(self, other: $scalar_type) -> Self::Output {
                    self.sub_scalar(other).unwrap()
                }
            }
        )+
    };
}

impl_scalar_sub_view!(f32, f64, i32, i64, u32, u64, i16, u16, i8, u8);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_sub_basic() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![0.5f32, 1.0, 1.5, 2.0], vec![2, 2])?;

        let result = a.sub(&b)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[0.5, 1.0, 1.5, 2.0]);

        Ok(())
    }

    #[test]
    fn test_tensor_sub_scalar() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2])?;

        let result = a.sub_scalar(1.0f32)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[0.0, 1.0, 2.0, 3.0]);

        Ok(())
    }

    #[test]
    fn test_tensor_sub_different_dtypes() -> Result<()> {
        // Test f64
        let a = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![0.5f64, 1.0, 1.5, 2.0], vec![2, 2])?;
        let result = a.sub(&b)?;
        let data = result.as_slice::<f64>()?;
        assert!((data[0] - 0.5).abs() < 1e-10);
        assert!((data[1] - 1.0).abs() < 1e-10);
        assert!((data[2] - 1.5).abs() < 1e-10);
        assert!((data[3] - 2.0).abs() < 1e-10);

        // Test i32
        let a = Tensor::from_vec(vec![5i32, 6, 7, 8], vec![2, 2])?;
        let b = Tensor::from_vec(vec![1i32, 2, 3, 4], vec![2, 2])?;
        let result = a.sub(&b)?;
        let data = result.as_slice::<i32>()?;
        assert_eq!(data, &[4, 4, 4, 4]);

        // Test u8
        let a = Tensor::from_vec(vec![10u8, 20, 30, 40], vec![2, 2])?;
        let b = Tensor::from_vec(vec![5u8, 10, 15, 20], vec![2, 2])?;
        let result = a.sub(&b)?;
        let data = result.as_slice::<u8>()?;
        assert_eq!(data, &[5, 10, 15, 20]);

        Ok(())
    }

    #[test]
    fn test_tensor_sub_half_precision() -> Result<()> {
        // Test f16
        let a = Tensor::from_vec(vec![f16::from_f32(1.0), f16::from_f32(2.0)], vec![2])?;
        let b = Tensor::from_vec(vec![f16::from_f32(0.5), f16::from_f32(1.0)], vec![2])?;
        let result = a.sub(&b)?;
        let data = result.as_slice::<f16>()?;
        assert!((data[0].to_f32() - 0.5).abs() < 1e-3);
        assert!((data[1].to_f32() - 1.0).abs() < 1e-3);

        // Test bf16
        let a = Tensor::from_vec(vec![bf16::from_f32(1.0), bf16::from_f32(2.0)], vec![2])?;
        let b = Tensor::from_vec(vec![bf16::from_f32(0.5), bf16::from_f32(1.0)], vec![2])?;
        let result = a.sub(&b)?;
        let data = result.as_slice::<bf16>()?;
        assert!((data[0].to_f32() - 0.5).abs() < 1e-3);
        assert!((data[1].to_f32() - 1.0).abs() < 1e-3);

        Ok(())
    }

    #[test]
    fn test_tensor_sub_empty() -> Result<()> {
        let a = Tensor::from_vec(Vec::<f32>::new(), [0])?;
        let b = Tensor::from_vec(Vec::<f32>::new(), [0])?;

        let result = a.sub(&b)?;
        assert_eq!(result.dtype(), DType::Fp32);
        assert_eq!(result.dims(), &[0]);
        assert_eq!(result.numel(), 0);

        let result_scalar = a.sub_scalar(5.0f32)?;
        assert_eq!(result_scalar.dtype(), DType::Fp32);
        assert_eq!(result_scalar.dims(), &[0]);
        assert_eq!(result_scalar.numel(), 0);

        Ok(())
    }

    #[test]
    fn test_tensor_sub_same_types() -> Result<()> {
        // Test same dtypes and shapes - should work fine
        let a = Tensor::from_vec(vec![5.0f32, 7.0, 9.0], [3])?;
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3])?;
        let result = a.sub(&b)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[4.0, 5.0, 6.0]);

        // Test same shapes with different values
        let _a = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], [3])?;
        let _b = Tensor::from_vec(vec![5.0f32, 10.0], [2])?;
        // Note: This would trigger debug_assert in debug mode, but passes in release
        // We're testing the happy path here

        Ok(())
    }

    #[test]
    fn test_tensor_view_operations() -> Result<()> {
        let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let data2 = vec![0.5f32, 1.0, 1.5, 2.0];
        let tensor1 = Tensor::from_vec(data1, vec![2, 2])?;
        let tensor2 = Tensor::from_vec(data2, vec![2, 2])?;

        let view1 = tensor1.clone().reshape(vec![2, 2])?;
        let view2 = tensor2.clone().reshape(vec![2, 2])?;

        // Test view - view (method and operator)
        let result = view1.sub(&view2)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[0.5, 1.0, 1.5, 2.0]);

        let result_op = &view1 - &view2;
        let data_op = result_op.as_slice::<f32>()?;
        assert_eq!(data_op, &[0.5, 1.0, 1.5, 2.0]);

        // Test tensor - view (method and operator)
        let result = tensor1.sub(&view2)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[0.5, 1.0, 1.5, 2.0]);

        // Test view - tensor (method and operator)
        let result = view1.sub(&tensor2)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[0.5, 1.0, 1.5, 2.0]);

        // Test view - scalar (method and operator)
        let result = view1.sub_scalar(1.0f32)?;
        let data = result.as_slice::<f32>()?;
        assert_eq!(data, &[0.0, 1.0, 2.0, 3.0]);

        let result_op = &view1 - 1.0f32;
        let data_op = result_op.as_slice::<f32>()?;
        assert_eq!(data_op, &[0.0, 1.0, 2.0, 3.0]);

        Ok(())
    }

    #[test]
    fn test_operator_overloading() -> Result<()> {
        let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let data2 = vec![0.5f32, 1.0, 1.5, 2.0];
        let tensor1 = Tensor::from_vec(data1, vec![2, 2])?;
        let tensor2 = Tensor::from_vec(data2, vec![2, 2])?;

        // Test all - operator combinations
        let result1 = &tensor1 - &tensor2;
        let result2 = tensor1.clone() - tensor2.clone();
        let result3 = &tensor1 - tensor2.clone();
        let result4 = tensor1.clone() - &tensor2;
        let result5 = &tensor1 - 1.0f32;
        let result6 = tensor1 - 2.0f32;

        let data1 = result1.as_slice::<f32>()?;
        let data2 = result2.as_slice::<f32>()?;
        let data3 = result3.as_slice::<f32>()?;
        let data4 = result4.as_slice::<f32>()?;
        let data5 = result5.as_slice::<f32>()?;
        let data6 = result6.as_slice::<f32>()?;

        assert_eq!(data1, &[0.5, 1.0, 1.5, 2.0]);
        assert_eq!(data2, &[0.5, 1.0, 1.5, 2.0]);
        assert_eq!(data3, &[0.5, 1.0, 1.5, 2.0]);
        assert_eq!(data4, &[0.5, 1.0, 1.5, 2.0]);
        assert_eq!(data5, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(data6, &[-1.0, 0.0, 1.0, 2.0]);

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
        let _result1 = &tensor1 - &tensor2;
        let _contiguous_time = start.elapsed();

        // Non-contiguous tensors
        let view1 = tensor1.clone().reshape(vec![size, size])?;
        let view2 = tensor2.clone().reshape(vec![size, size])?;

        let start = std::time::Instant::now();
        let _result2 = view1.sub(&view2)?;
        let _non_contiguous_time = start.elapsed();

        // Verify correctness
        let result1 = &tensor1 - &tensor2;
        let result2 = view1.sub(&view2)?;

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
        let result = &single - 3.0f32;
        assert_eq!(result.as_slice::<f32>()?[0], 2.0);

        // High dimensional tensor
        let high_dim = Tensor::from_vec(vec![1.0f32], vec![1, 1, 1, 1, 1, 1, 1, 1])?;
        let result = &high_dim - 10.0f32;
        assert_eq!(result.as_slice::<f32>()?[0], -9.0);

        // Large tensor
        let size = 1000;
        let large_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let large_tensor = Tensor::from_vec(large_data, vec![size])?;
        let result = &large_tensor - 1.0f32;

        let data = result.as_slice::<f32>()?;
        assert_eq!(data[0], -1.0);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[size - 1], (size - 2) as f32);

        Ok(())
    }
}
