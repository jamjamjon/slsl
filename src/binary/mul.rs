use anyhow::Result;
use half::{bf16, f16};

use crate::{
    backend::{global_backend, r#impl::OpsTrait},
    DType, StorageTrait, Tensor, TensorBase, TensorElement, TensorView, UninitVec,
};

impl<S: StorageTrait> TensorBase<S> {
    /// Multiply two tensors element-wise
    #[inline(always)]
    pub fn mul<T: StorageTrait>(&self, other: &TensorBase<T>) -> Result<Tensor> {
        // Check shape compatibility
        if self.shape() != other.shape() {
            anyhow::bail!(
                "Shape mismatch for mul operation: {:?} vs {:?}",
                self.dims(),
                other.dims()
            );
        }

        // Check dtype compatibility
        if self.dtype() != other.dtype() {
            anyhow::bail!(
                "Dtype mismatch for mul operation: {:?} vs {:?}",
                self.dtype(),
                other.dtype()
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
            self.mul_contiguous(other)
        } else {
            self.mul_non_contiguous(other)
        }
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
                    dst_to_set[idx] = self_val * other_val;
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
                    dst_to_set[idx] = self_val * other_val;
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
                    dst_to_set[idx] = f16::from_f32(self_val.to_f32() * other_val.to_f32());
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
                    dst_to_set[idx] = bf16::from_f32(self_val.to_f32() * other_val.to_f32());
                }

                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Int8 => {
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set =
                    unsafe { &mut *(dst_to_set as *mut [std::mem::MaybeUninit<i8>] as *mut [i8]) };

                for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate() {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let self_val = unsafe { *(self_ptr as *const i8) };
                    let other_val = unsafe { *(other_ptr as *const i8) };
                    dst_to_set[idx] = self_val * other_val;
                }

                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Int16 => {
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<i16>] as *mut [i16])
                };

                for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate() {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let self_val = unsafe { *(self_ptr as *const i16) };
                    let other_val = unsafe { *(other_ptr as *const i16) };
                    dst_to_set[idx] = self_val * other_val;
                }

                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Int32 => {
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<i32>] as *mut [i32])
                };

                for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate() {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let self_val = unsafe { *(self_ptr as *const i32) };
                    let other_val = unsafe { *(other_ptr as *const i32) };
                    dst_to_set[idx] = self_val * other_val;
                }

                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Int64 => {
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<i64>] as *mut [i64])
                };

                for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate() {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let self_val = unsafe { *(self_ptr as *const i64) };
                    let other_val = unsafe { *(other_ptr as *const i64) };
                    dst_to_set[idx] = self_val * other_val;
                }

                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Uint8 => {
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set =
                    unsafe { &mut *(dst_to_set as *mut [std::mem::MaybeUninit<u8>] as *mut [u8]) };

                for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate() {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let self_val = unsafe { *self_ptr };
                    let other_val = unsafe { *other_ptr };
                    dst_to_set[idx] = self_val * other_val;
                }

                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Uint16 => {
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<u16>] as *mut [u16])
                };

                for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate() {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let self_val = unsafe { *(self_ptr as *const u16) };
                    let other_val = unsafe { *(other_ptr as *const u16) };
                    dst_to_set[idx] = self_val * other_val;
                }

                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Uint32 => {
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<u32>] as *mut [u32])
                };

                for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate() {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let self_val = unsafe { *(self_ptr as *const u32) };
                    let other_val = unsafe { *(other_ptr as *const u32) };
                    dst_to_set[idx] = self_val * other_val;
                }

                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Uint64 => {
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<u64>] as *mut [u64])
                };

                for (idx, (self_elem, other_elem)) in self.iter().zip(other.iter()).enumerate() {
                    let self_ptr = unsafe { self_elem.as_ptr(self.as_ptr()) };
                    let other_ptr = unsafe { other_elem.as_ptr(other.as_ptr()) };
                    let self_val = unsafe { *(self_ptr as *const u64) };
                    let other_val = unsafe { *(other_ptr as *const u64) };
                    dst_to_set[idx] = self_val * other_val;
                }

                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
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
        if T::DTYPE != self.dtype() {
            anyhow::bail!(
                "Scalar type mismatch for mul_scalar operation: scalar dtype {:?} vs tensor dtype {:?}",
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
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<f32>] as *mut [f32])
                };
                backend.v_mul_scalar_f32(input_data, s, dst_to_set);
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
                backend.v_mul_scalar_f64(input_data, s, dst_to_set);
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
                backend.v_mul_scalar_f16(input_data, s, dst_to_set);
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
                backend.v_mul_scalar_bf16(input_data, s, dst_to_set);
                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Int8 => {
                let input_data = self.as_slice::<i8>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i8>(&scalar) };
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set =
                    unsafe { &mut *(dst_to_set as *mut [std::mem::MaybeUninit<i8>] as *mut [i8]) };
                backend.v_mul_scalar_i8(input_data, s, dst_to_set);
                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Int16 => {
                let input_data = self.as_slice::<i16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i16>(&scalar) };
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<i16>] as *mut [i16])
                };
                backend.v_mul_scalar_i16(input_data, s, dst_to_set);
                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Int32 => {
                let input_data = self.as_slice::<i32>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i32>(&scalar) };
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<i32>] as *mut [i32])
                };
                backend.v_mul_scalar_i32(input_data, s, dst_to_set);
                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Int64 => {
                let input_data = self.as_slice::<i64>()?;
                let s = unsafe { std::mem::transmute_copy::<T, i64>(&scalar) };
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<i64>] as *mut [i64])
                };
                backend.v_mul_scalar_i64(input_data, s, dst_to_set);
                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Uint8 => {
                let input_data = self.as_slice::<u8>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u8>(&scalar) };
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set =
                    unsafe { &mut *(dst_to_set as *mut [std::mem::MaybeUninit<u8>] as *mut [u8]) };
                backend.v_mul_scalar_u8(input_data, s, dst_to_set);
                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Uint16 => {
                let input_data = self.as_slice::<u16>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u16>(&scalar) };
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<u16>] as *mut [u16])
                };
                backend.v_mul_scalar_u16(input_data, s, dst_to_set);
                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Uint32 => {
                let input_data = self.as_slice::<u32>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u32>(&scalar) };
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<u32>] as *mut [u32])
                };
                backend.v_mul_scalar_u32(input_data, s, dst_to_set);
                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
            }
            DType::Uint64 => {
                let input_data = self.as_slice::<u64>()?;
                let s = unsafe { std::mem::transmute_copy::<T, u64>(&scalar) };
                let mut output = Vec::with_capacity(numel);
                let dst_to_set = output.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<u64>] as *mut [u64])
                };
                backend.v_mul_scalar_u64(input_data, s, dst_to_set);
                unsafe { output.set_len(numel) };
                Tensor::from_vec(output, self.shape)
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
        TensorBase::mul(self, other).expect("Tensor multiplication failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Mul<TensorBase<S2>> for &TensorBase<S1> {
    type Output = Tensor;
    fn mul(self, other: TensorBase<S2>) -> Self::Output {
        TensorBase::mul(self, &other).expect("Tensor multiplication failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Mul<&TensorBase<S2>> for TensorBase<S1> {
    type Output = Tensor;
    fn mul(self, other: &TensorBase<S2>) -> Self::Output {
        TensorBase::mul(&self, other).expect("Tensor multiplication failed")
    }
}

impl<S1: StorageTrait, S2: StorageTrait> std::ops::Mul<TensorBase<S2>> for TensorBase<S1> {
    type Output = Tensor;
    fn mul(self, other: TensorBase<S2>) -> Self::Output {
        TensorBase::mul(&self, &other).expect("Tensor multiplication failed")
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
    fn test_tensor_mul_errors() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0], vec![2])?;
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3])?;

        // Shape mismatch should fail
        assert!(a.mul(&b).is_err());

        // Dtype mismatch should fail
        let c = Tensor::from_vec(vec![1i32, 2], vec![2])?;
        assert!(a.mul(&c).is_err());

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
}
