use anyhow::Result;

use crate::{
    backend::{global_backend, r#impl::OpsTrait},
    DType, Dim, StorageTrait, Tensor, TensorBase, TensorElement, UninitVec,
};
use num_traits::Float;

impl<S: StorageTrait> TensorBase<S> {
    /// Compute various norms of a tensor, similar to PyTorch's `torch.linalg.norm`.
    /// Supports L1, L2, and Lp norms for floating-point tensors.
    ///
    /// * `ord` - Order of the norm. Supported values:
    ///   - `None` (default): L2 norm (Euclidean norm)
    ///   - `Some(1.0)`: L1 norm (Manhattan norm) - uses backend asum
    ///   - `Some(2.0)`: L2 norm (Euclidean norm) - uses backend nrm2
    ///   - `Some(p)`: Lp norm where p > 0
    pub fn norm<D: Dim>(&self, dim: D, ord: f32) -> Result<Tensor> {
        match ord {
            1.0 => self.norm1(dim),
            2.0 => self.norm2(dim),
            p if p > 0.0 => self.normp(dim, p),
            _ => anyhow::bail!("Invalid norm order: {}", ord),
        }
    }

    pub fn norm_keepdim<D: Dim>(&self, dim: D, ord: f32) -> Result<Tensor> {
        match ord {
            1.0 => self.norm1_keepdim(dim),
            2.0 => self.norm2_keepdim(dim),
            p if p > 0.0 => self.normp_keepdim(dim, p),
            _ => anyhow::bail!("Invalid norm order: {}", ord),
        }
    }

    #[inline(always)]
    pub fn norm1<D: Dim>(&self, dim: D) -> Result<Tensor> {
        let dim_idx = dim.to_dim(self.rank())?;

        if self.dtype == DType::Bool {
            anyhow::bail!("norm1 does not support Bool dtype");
        }

        if self.is_contiguous() {
            self.norm1_contiguous(dim_idx, false)
        } else {
            self.norm1_non_contiguous(dim_idx, false)
        }
    }

    #[inline(always)]
    pub fn norm1_keepdim<D: Dim>(&self, dim: D) -> Result<Tensor> {
        let dim_idx = dim.to_dim(self.rank())?;

        if self.dtype == DType::Bool {
            anyhow::bail!("norm1 does not support Bool dtype");
        }

        if self.is_contiguous() {
            self.norm1_contiguous(dim_idx, true)
        } else {
            self.norm1_non_contiguous(dim_idx, true)
        }
    }

    #[inline(always)]
    pub fn norm2<D: Dim>(&self, dim: D) -> Result<Tensor> {
        let dim_idx = dim.to_dim(self.rank())?;

        if !self.dtype.is_float() {
            anyhow::bail!(
                "norm2 only supports floating-point types, got {:?}",
                self.dtype
            );
        }

        if self.is_contiguous() {
            self.norm2_contiguous(dim_idx, false)
        } else {
            self.norm2_non_contiguous(dim_idx, false)
        }
    }

    #[inline(always)]
    pub fn norm2_keepdim<D: Dim>(&self, dim: D) -> Result<Tensor> {
        let dim_idx = dim.to_dim(self.rank())?;

        if !self.dtype.is_float() {
            anyhow::bail!(
                "norm2 only supports floating-point types, got {:?}",
                self.dtype
            );
        }

        if self.is_contiguous() {
            self.norm2_contiguous(dim_idx, true)
        } else {
            self.norm2_non_contiguous(dim_idx, true)
        }
    }

    #[inline(always)]
    pub fn normp<D: Dim>(&self, dim: D, p: f32) -> Result<Tensor> {
        let dim_idx = dim.to_dim(self.rank())?;

        if !self.dtype.is_float() {
            anyhow::bail!(
                "normp only supports floating-point types, got {:?}",
                self.dtype
            );
        }

        if self.is_contiguous() {
            self.normp_contiguous(dim_idx, p, false)
        } else {
            self.normp_non_contiguous(dim_idx, p, false)
        }
    }

    #[inline(always)]
    pub fn normp_keepdim<D: Dim>(&self, dim: D, p: f32) -> Result<Tensor> {
        let dim_idx = dim.to_dim(self.rank())?;

        if !self.dtype.is_float() {
            anyhow::bail!(
                "normp only supports floating-point types, got {:?}",
                self.dtype
            );
        }

        if self.is_contiguous() {
            self.normp_contiguous(dim_idx, p, true)
        } else {
            self.normp_non_contiguous(dim_idx, p, true)
        }
    }

    // ========== L1 Norm Implementation ==========

    #[inline(always)]
    fn norm1_contiguous(&self, dim_idx: usize, keepdim: bool) -> Result<Tensor> {
        let output_shape = self._compute_output_shape(dim_idx, keepdim)?;
        let slice_size = self._compute_slice_size(dim_idx);
        let dim_size = self.shape[dim_idx];
        let backend = global_backend();

        match self.dtype {
            DType::Fp32 => {
                let mut result_data = UninitVec::<f32>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_typed::<f32>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = backend.asum_f32(&slice_buffer);
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Fp64 => {
                let mut result_data = UninitVec::<f64>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_typed::<f64>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = backend.asum_f64(&slice_buffer);
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Fp16 => {
                let mut result_data = UninitVec::<half::f16>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_f16(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = half::f16::from_f32(backend.asum_f16(&slice_buffer));
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Bf16 => {
                let mut result_data = UninitVec::<half::bf16>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_bf16(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = half::bf16::from_f32(backend.asum_bf16(&slice_buffer));
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            // Integer types - use backend asum for acceleration
            DType::Int8 => {
                let mut result_data = UninitVec::<i8>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_int::<i8>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = backend.asum_i8(&slice_buffer);
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Int16 => {
                let mut result_data = UninitVec::<i16>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_int::<i16>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = backend.asum_i16(&slice_buffer);
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Int32 => {
                let mut result_data = UninitVec::<i32>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_int::<i32>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = backend.asum_i32(&slice_buffer);
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Int64 => {
                let mut result_data = UninitVec::<i64>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_int::<i64>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = backend.asum_i64(&slice_buffer);
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Uint8 => {
                let mut result_data = UninitVec::<u8>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_int::<u8>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = backend.asum_u8(&slice_buffer);
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Uint16 => {
                let mut result_data = UninitVec::<u16>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_int::<u16>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = backend.asum_u16(&slice_buffer);
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Uint32 => {
                let mut result_data = UninitVec::<u32>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_int::<u32>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = backend.asum_u32(&slice_buffer);
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Uint64 => {
                let mut result_data = UninitVec::<u64>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_int::<u64>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = backend.asum_u64(&slice_buffer);
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            _ => anyhow::bail!("Unsupported dtype for norm1: {:?}", self.dtype),
        }
    }

    #[inline(always)]
    fn norm1_non_contiguous(&self, dim_idx: usize, keepdim: bool) -> Result<Tensor> {
        let output_shape = self._compute_output_shape(dim_idx, keepdim)?;
        let slice_size = self._compute_slice_size(dim_idx);
        let dim_size = self.shape[dim_idx];

        match self.dtype {
            DType::Fp32 => {
                let mut result_data = UninitVec::<f32>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_typed::<f32>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = slice_buffer
                        .iter()
                        .fold(0.0f32, |acc, &val| acc + val.abs());
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Fp64 => {
                let mut result_data = UninitVec::<f64>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_typed::<f64>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = slice_buffer
                        .iter()
                        .fold(0.0f64, |acc, &val| acc + val.abs());
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Fp16 => {
                let mut result_data = UninitVec::<half::f16>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_f16(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = half::f16::from_f32(
                        slice_buffer
                            .iter()
                            .fold(0.0f32, |acc, &val| acc + val.to_f32().abs()),
                    );
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Bf16 => {
                let mut result_data = UninitVec::<half::bf16>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_bf16(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = half::bf16::from_f32(
                        slice_buffer
                            .iter()
                            .fold(0.0f32, |acc, &val| acc + val.to_f32().abs()),
                    );
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            // Integer types - manual computation for non-contiguous
            DType::Int8 => {
                let mut result_data = UninitVec::<i8>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_int::<i8>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = slice_buffer.iter().fold(0i8, |acc, &val| acc + val.abs());
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Int16 => {
                let mut result_data = UninitVec::<i16>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_int::<i16>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = slice_buffer.iter().fold(0i16, |acc, &val| acc + val.abs());
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Int32 => {
                let mut result_data = UninitVec::<i32>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_int::<i32>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = slice_buffer.iter().fold(0i32, |acc, &val| acc + val.abs());
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Int64 => {
                let mut result_data = UninitVec::<i64>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_int::<i64>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = slice_buffer.iter().fold(0i64, |acc, &val| acc + val.abs());
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Uint8 => {
                let mut result_data = UninitVec::<u8>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Preallocate slice buffer to avoid repeated allocations
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer, do not clear capacity

                    self._fill_slice_values_int::<u8>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = slice_buffer.iter().sum::<u8>();
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Uint16 => {
                let mut result_data = UninitVec::<u16>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_int::<u16>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = slice_buffer.iter().sum::<u16>();
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Uint32 => {
                let mut result_data = UninitVec::<u32>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_int::<u32>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = slice_buffer.iter().sum::<u32>();
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Uint64 => {
                let mut result_data = UninitVec::<u64>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_int::<u64>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = slice_buffer.iter().sum::<u64>();
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            _ => anyhow::bail!("Unsupported dtype for norm1: {:?}", self.dtype),
        }
    }

    // ========== L2 Norm Implementation ==========

    #[inline(always)]
    fn norm2_contiguous(&self, dim_idx: usize, keepdim: bool) -> Result<Tensor> {
        let output_shape = self._compute_output_shape(dim_idx, keepdim)?;
        let slice_size = self._compute_slice_size(dim_idx);
        let dim_size = self.shape[dim_idx];
        let backend = global_backend();

        match self.dtype {
            DType::Fp32 => {
                let mut result_data = UninitVec::<f32>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_typed::<f32>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = backend.nrm2_f32(&slice_buffer) as f32;
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Fp64 => {
                let mut result_data = UninitVec::<f64>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_typed::<f64>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = backend.nrm2_f64(&slice_buffer);
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Fp16 => {
                let mut result_data = UninitVec::<half::f16>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_f16(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = half::f16::from_f64(backend.nrm2_f16(&slice_buffer));
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Bf16 => {
                let mut result_data = UninitVec::<half::bf16>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_bf16(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to the pre-allocated result array
                    *dst = half::bf16::from_f64(backend.nrm2_bf16(&slice_buffer));
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            _ => anyhow::bail!("Unsupported dtype for norm2: {:?}", self.dtype),
        }
    }

    #[inline(always)]
    fn norm2_non_contiguous(&self, dim_idx: usize, keepdim: bool) -> Result<Tensor> {
        let output_shape = self._compute_output_shape(dim_idx, keepdim)?;
        let slice_size = self._compute_slice_size(dim_idx);
        let dim_size = self.shape[dim_idx];

        match self.dtype {
            DType::Fp32 => {
                let mut result_data = UninitVec::<f32>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_typed::<f32>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to pre-allocated result array
                    let sum_squares: f32 = slice_buffer
                        .iter()
                        .fold(0.0f32, |acc, &val| acc + val * val);
                    *dst = sum_squares.sqrt();
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Fp64 => {
                let mut result_data = UninitVec::<f64>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_typed::<f64>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to pre-allocated result array
                    let sum_squares: f64 = slice_buffer
                        .iter()
                        .fold(0.0f64, |acc, &val| acc + val * val);
                    *dst = sum_squares.sqrt();
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Fp16 => {
                let mut result_data = UninitVec::<half::f16>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_f16(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to pre-allocated result array
                    let sum_squares: f32 = slice_buffer
                        .iter()
                        .fold(0.0f32, |acc, &val| acc + val.to_f32() * val.to_f32());
                    *dst = half::f16::from_f32(sum_squares.sqrt());
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Bf16 => {
                let mut result_data = UninitVec::<half::bf16>::new(slice_size);
                let dst_to_set = result_data.as_mut_slice();

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_bf16(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to pre-allocated result array
                    let sum_squares: f32 = slice_buffer
                        .iter()
                        .fold(0.0f32, |acc, &val| acc + val.to_f32() * val.to_f32());
                    *dst = half::bf16::from_f32(sum_squares.sqrt());
                }

                let result_data = unsafe { result_data.finalize() };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            _ => anyhow::bail!("Unsupported dtype for norm2: {:?}", self.dtype),
        }
    }

    // ========== Lp Norm Implementation ==========

    #[inline(always)]
    fn normp_contiguous(&self, dim_idx: usize, p: f32, keepdim: bool) -> Result<Tensor> {
        let output_shape = self._compute_output_shape(dim_idx, keepdim)?;
        let slice_size = self._compute_slice_size(dim_idx);
        let dim_size = self.shape[dim_idx];

        match self.dtype {
            DType::Fp32 => {
                // Pre-allocate result array
                let mut result_data = Vec::with_capacity(slice_size);
                let dst_to_set = result_data.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<f32>] as *mut [f32])
                };

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_typed::<f32>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to pre-allocated result array
                    let sum_powers: f32 = slice_buffer
                        .iter()
                        .fold(0.0f32, |acc, &val| acc + val.to_f32().abs().powf(p));
                    *dst = sum_powers.powf(1.0 / p);
                }

                unsafe { result_data.set_len(slice_size) };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Fp64 => {
                // Pre-allocate result array
                let mut result_data = Vec::with_capacity(slice_size);
                let dst_to_set = result_data.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<f64>] as *mut [f64])
                };

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_typed::<f64>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to pre-allocated result array
                    let sum_powers: f64 = slice_buffer
                        .iter()
                        .fold(0.0f64, |acc, &val| acc + val.abs().powf(p as f64));
                    *dst = sum_powers.powf(1.0 / p as f64);
                }

                unsafe { result_data.set_len(slice_size) };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Fp16 => {
                // Pre-allocate result array
                let mut result_data = Vec::with_capacity(slice_size);
                let dst_to_set = result_data.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<half::f16>]
                        as *mut [half::f16])
                };

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_f16(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to pre-allocated result array
                    let sum_powers: f32 = slice_buffer
                        .iter()
                        .fold(0.0f32, |acc, &val| acc + val.to_f32().abs().powf(p));
                    *dst = half::f16::from_f32(sum_powers.powf(1.0 / p));
                }

                unsafe { result_data.set_len(slice_size) };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Bf16 => {
                // Pre-allocate result array
                let mut result_data = Vec::with_capacity(slice_size);
                let dst_to_set = result_data.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<half::bf16>]
                        as *mut [half::bf16])
                };

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_bf16(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to pre-allocated result array
                    let sum_powers: f32 = slice_buffer
                        .iter()
                        .fold(0.0f32, |acc, &val| acc + val.to_f32().abs().powf(p));
                    *dst = half::bf16::from_f32(sum_powers.powf(1.0 / p));
                }

                unsafe { result_data.set_len(slice_size) };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            _ => anyhow::bail!("Unsupported dtype for normp: {:?}", self.dtype),
        }
    }

    #[inline(always)]
    fn normp_non_contiguous(&self, dim_idx: usize, p: f32, keepdim: bool) -> Result<Tensor> {
        let output_shape = self._compute_output_shape(dim_idx, keepdim)?;
        let slice_size = self._compute_slice_size(dim_idx);
        let dim_size = self.shape[dim_idx];

        match self.dtype {
            DType::Fp32 => {
                // Pre-allocate result array
                let mut result_data = Vec::with_capacity(slice_size);
                let dst_to_set = result_data.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<f32>] as *mut [f32])
                };

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_typed::<f32>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to pre-allocated result array
                    let sum_powers: f32 = slice_buffer
                        .iter()
                        .fold(0.0f32, |acc, &val| acc + val.to_f32().abs().powf(p));
                    *dst = sum_powers.powf(1.0 / p);
                }

                unsafe { result_data.set_len(slice_size) };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Fp64 => {
                // Pre-allocate result array
                let mut result_data = Vec::with_capacity(slice_size);
                let dst_to_set = result_data.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<f64>] as *mut [f64])
                };

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_typed::<f64>(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to pre-allocated result array
                    let sum_powers: f64 = slice_buffer
                        .iter()
                        .fold(0.0f64, |acc, &val| acc + val.abs().powf(p as f64));
                    *dst = sum_powers.powf(1.0 / p as f64);
                }

                unsafe { result_data.set_len(slice_size) };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Fp16 => {
                // Pre-allocate result array
                let mut result_data = Vec::with_capacity(slice_size);
                let dst_to_set = result_data.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<half::f16>]
                        as *mut [half::f16])
                };

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_f16(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to pre-allocated result array
                    let sum_powers: f32 = slice_buffer
                        .iter()
                        .fold(0.0f32, |acc, &val| acc + val.to_f32().abs().powf(p));
                    *dst = half::f16::from_f32(sum_powers.powf(1.0 / p));
                }

                unsafe { result_data.set_len(slice_size) };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            DType::Bf16 => {
                // Pre-allocate result array
                let mut result_data = Vec::with_capacity(slice_size);
                let dst_to_set = result_data.spare_capacity_mut();
                let dst_to_set = unsafe {
                    &mut *(dst_to_set as *mut [std::mem::MaybeUninit<half::bf16>]
                        as *mut [half::bf16])
                };

                // Pre-allocate slice buffer to avoid repeated allocation
                let mut slice_buffer = Vec::with_capacity(dim_size);

                for (slice_idx, dst) in dst_to_set.iter_mut().enumerate() {
                    // Reuse buffer without clearing capacity

                    self._fill_slice_values_bf16(dim_idx, slice_idx, &mut slice_buffer)?;
                    unsafe { slice_buffer.set_len(dim_size) };

                    // Directly write to pre-allocated result array
                    let sum_powers: f32 = slice_buffer
                        .iter()
                        .fold(0.0f32, |acc, &val| acc + val.to_f32().abs().powf(p));
                    *dst = half::bf16::from_f32(sum_powers.powf(1.0 / p));
                }

                unsafe { result_data.set_len(slice_size) };
                Ok(Tensor::from_vec(result_data, output_shape)?)
            }
            _ => anyhow::bail!("Unsupported dtype for normp: {:?}", self.dtype),
        }
    }

    // ========== Helper Functions ==========

    /// Generic slice value filling function, supports all types
    #[inline(always)]
    fn _fill_slice_values_generic<T: TensorElement>(
        &self,
        dim_idx: usize,
        slice_idx: usize,
        slice_buffer: &mut Vec<T>,
    ) -> Result<()> {
        let rank = self.rank();
        let shape = self.shape.as_slice();
        let dim_size = shape[dim_idx];

        // Pre-allocate memory and get writable slice
        slice_buffer.clear();
        let dst_to_set = slice_buffer.spare_capacity_mut();
        let dst_to_set =
            unsafe { &mut *(dst_to_set as *mut [std::mem::MaybeUninit<T>] as *mut [T]) };

        // Calculate slice indices - simplified version
        let mut slice_indices = vec![0; rank - 1];
        let mut remaining = slice_idx;
        let mut slice_pos = 0;

        // Calculate all dimension indices except dim_idx from right to left
        for i in (0..rank).rev() {
            if i != dim_idx {
                slice_indices[slice_pos] = remaining % shape[i];
                remaining /= shape[i];
                slice_pos += 1;
            }
        }

        // Fill data
        for (dim_val, dst) in (0..dim_size).zip(dst_to_set.iter_mut()) {
            let mut full_indices = [0usize; 8];
            let mut slice_pos = 0;

            // Build complete index
            for (i, _item) in full_indices.iter_mut().enumerate().take(rank) {
                if i == dim_idx {
                    *_item = dim_val;
                } else {
                    *_item = slice_indices[slice_pos];
                    slice_pos += 1;
                }
            }

            let val = self.at::<T>(&full_indices[..rank]);
            *dst = val;
        }

        // Set actual length
        unsafe { slice_buffer.set_len(dim_size) };
        Ok(())
    }

    /// Type-specialized slice value filling function - using generic version
    #[inline(always)]
    fn _fill_slice_values_typed<T: TensorElement + Float>(
        &self,
        dim_idx: usize,
        slice_idx: usize,
        slice_buffer: &mut Vec<T>,
    ) -> Result<()> {
        self._fill_slice_values_generic(dim_idx, slice_idx, slice_buffer)
    }

    #[inline(always)]
    fn _fill_slice_values_int<T: TensorElement>(
        &self,
        dim_idx: usize,
        slice_idx: usize,
        slice_buffer: &mut Vec<T>,
    ) -> Result<()> {
        self._fill_slice_values_generic(dim_idx, slice_idx, slice_buffer)
    }

    #[inline(always)]
    fn _fill_slice_values_f16(
        &self,
        dim_idx: usize,
        slice_idx: usize,
        slice_buffer: &mut Vec<half::f16>,
    ) -> Result<()> {
        self._fill_slice_values_generic(dim_idx, slice_idx, slice_buffer)
    }

    #[inline(always)]
    fn _fill_slice_values_bf16(
        &self,
        dim_idx: usize,
        slice_idx: usize,
        slice_buffer: &mut Vec<half::bf16>,
    ) -> Result<()> {
        self._fill_slice_values_generic(dim_idx, slice_idx, slice_buffer)
    }

    #[inline(always)]
    fn _compute_output_shape(&self, dim_idx: usize, keepdim: bool) -> Result<Vec<usize>> {
        let rank = self.rank();
        let shape = self.shape.as_slice();

        let mut output_shape = Vec::with_capacity(rank);
        for (i, &dim_size) in shape.iter().enumerate().take(rank) {
            if i == dim_idx {
                if keepdim {
                    output_shape.push(1);
                }
            } else {
                output_shape.push(dim_size);
            }
        }

        if rank == 1 {
            output_shape = vec![];
        }

        Ok(output_shape)
    }

    #[inline(always)]
    fn _compute_slice_size(&self, dim_idx: usize) -> usize {
        let rank = self.rank();
        let shape = self.shape.as_slice();

        let mut slice_size = 1;
        for (i, &dim_size) in shape.iter().enumerate().take(rank) {
            if i != dim_idx {
                slice_size *= dim_size;
            }
        }

        slice_size
    }
}
