use anyhow::Result;

use crate::{DType, Dim, Shape, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    /// Stacks a sequence of tensors along a new dimension.
    ///
    /// This function creates a new tensor by stacking the input tensors along a new dimension.
    /// All input tensors must have the same shape and data type. The resulting tensor will have
    /// one more dimension than the input tensors.
    ///
    /// # Arguments
    ///
    /// * `tensors` - A slice of tensor references to stack. Must not be empty.
    /// * `dim` - The dimension along which to stack. Can be in range `[0, rank]` where `rank`
    ///   is the rank of the input tensors. A value of `rank` will insert the new dimension at the end.
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A new tensor with the stacked data
    /// - `Err`: If tensors have different shapes/dtypes, or if the dimension is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use slsl::Tensor;
    ///
    /// // Stack 1D tensors
    /// let tensor1 = Tensor::from_vec(vec![1, 2, 3], [3])?;
    /// let tensor2 = Tensor::from_vec(vec![4, 5, 6], [3])?;
    /// let tensor3 = Tensor::from_vec(vec![7, 8, 9], [3])?;
    /// let tensors = vec![tensor1, tensor2, tensor3];
    ///
    /// // Stack along dimension 0 (beginning)
    /// let stacked = Tensor::stack(&tensors, 0)?;
    /// assert_eq!(stacked.dims(), [3, 3]);
    /// assert_eq!(stacked.to_flat_vec::<i32>()?, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    ///
    /// // Stack along dimension 1 (end)
    /// let stacked = Tensor::stack(&tensors, 1)?;
    /// assert_eq!(stacked.dims(), [3, 3]);
    /// assert_eq!(stacked.to_flat_vec::<i32>()?, vec![1, 4, 7, 2, 5, 8, 3, 6, 9]);
    ///
    /// // Stack 2D tensors
    /// let tensor_2d1 = Tensor::from_vec(vec![1, 2, 3, 4], [2, 2])?;
    /// let tensor_2d2 = Tensor::from_vec(vec![5, 6, 7, 8], [2, 2])?;
    /// let tensors_2d = vec![tensor_2d1, tensor_2d2];
    ///
    /// let stacked_2d = Tensor::stack(&tensors_2d, 0)?;
    /// assert_eq!(stacked_2d.dims(), [2, 2, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - All input tensors must have identical shapes and data types
    /// - The new dimension size equals the number of input tensors
    /// - The function follows PyTorch's `stack` behavior
    /// - For out-of-bounds dimensions, the new dimension is inserted at the end
    ///
    pub fn stack<D: Dim>(tensors: &[Self], dim: D) -> Result<Tensor> {
        Self::stack_impl(tensors, dim)
    }

    fn stack_impl<D: Dim>(tensors: &[Self], dim: D) -> Result<Tensor> {
        if tensors.is_empty() {
            anyhow::bail!("Cannot stack empty tensor list");
        }

        // For stack operation, we allow the dimension to be beyond the current rank
        // because stack creates a new dimension
        let current_rank = tensors[0].rank();
        let dim_idx = if dim.to_dim(current_rank).is_ok() {
            dim.to_dim(current_rank)?
        } else {
            // If the dimension is beyond current rank, it's valid for stack
            // We'll insert the new dimension at the end
            current_rank
        };
        let first_tensor = &tensors[0];
        let first_shape = first_tensor.shape;
        let first_dtype = first_tensor.dtype;

        // Check that all tensors have the same shape and dtype
        for (i, tensor) in tensors.iter().enumerate() {
            if tensor.shape != first_shape {
                anyhow::bail!(
                    "Tensor {} has different shape {:?} than first tensor {:?}",
                    i,
                    tensor.shape,
                    first_shape
                );
            }
            if tensor.dtype != first_dtype {
                anyhow::bail!(
                    "Tensor {} has different dtype {:?} than first tensor {:?}",
                    i,
                    tensor.dtype,
                    first_dtype
                );
            }
        }

        // Calculate new shape using Shape
        let rank = first_shape.len();
        let mut new_shape = Shape::empty().with_len(rank + 1);
        let mut out_i = 0;
        for i in 0..=rank {
            if i == dim_idx {
                new_shape[out_i] = tensors.len();
                out_i += 1;
            }
            if i < rank {
                new_shape[out_i] = first_shape[i];
                out_i += 1;
            }
        }

        Self::stack_dispatch(tensors, &new_shape, dim_idx, first_dtype)
    }

    fn stack_dispatch(
        tensors: &[Self],
        new_shape: &Shape,
        dim_idx: usize,
        dtype: DType,
    ) -> Result<Tensor> {
        let total_elements = new_shape.numel();

        // Check if we can use contiguous optimizations
        let can_use_contiguous = Self::can_stack_contiguous(tensors);

        macro_rules! dispatch_stack {
            ($ty:ty) => {{
                let out = UninitVec::<$ty>::new(total_elements).init_with(|dst| {
                    if can_use_contiguous {
                        Self::stack_contiguous::<$ty>(tensors, new_shape, dim_idx, dst)
                    } else {
                        Self::stack_general::<$ty>(tensors, new_shape, dim_idx, dst)
                    }
                });
                Tensor::from_vec(out, new_shape.clone())
            }};
        }

        match dtype {
            DType::Bool => dispatch_stack!(bool),
            DType::Int8 => dispatch_stack!(i8),
            DType::Int16 => dispatch_stack!(i16),
            DType::Int32 => dispatch_stack!(i32),
            DType::Int64 => dispatch_stack!(i64),
            DType::Uint8 => dispatch_stack!(u8),
            DType::Uint16 => dispatch_stack!(u16),
            DType::Uint32 => dispatch_stack!(u32),
            DType::Uint64 => dispatch_stack!(u64),
            DType::Fp16 => dispatch_stack!(half::f16),
            DType::Fp32 => dispatch_stack!(f32),
            DType::Fp64 => dispatch_stack!(f64),
            DType::Bf16 => dispatch_stack!(half::bf16),
            _ => anyhow::bail!("stack function not supported for dtype: {:?}", dtype),
        }
    }

    /// Check if tensors can use contiguous memory optimizations
    fn can_stack_contiguous(tensors: &[Self]) -> bool {
        if tensors.is_empty() {
            return false;
        }

        let rank = tensors[0].rank();
        if rank == 0 {
            return true;
        }

        // Calculate expected contiguous strides
        let mut expected_strides = Shape::empty().with_len(rank);
        expected_strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            expected_strides[i] = expected_strides[i + 1] * tensors[0].shape[i + 1];
        }

        // Check if all tensors are contiguous with expected strides
        for tensor in tensors {
            for i in 0..rank {
                let expected = expected_strides[i];
                let actual = if i < tensor.strides.len() {
                    tensor.strides[i]
                } else {
                    0
                };
                if actual != expected {
                    return false;
                }
            }
        }

        true
    }

    fn stack_contiguous<T: crate::TensorElement>(
        tensors: &[Self],
        new_shape: &Shape,
        dim_idx: usize,
        dst: &mut [T],
    ) {
        let out_rank = new_shape.len();
        let in_rank = out_rank - 1;

        // Fast path: if stacking along new leading dim (dim_idx == 0) and all inputs are contiguous,
        // we can bulk copy blocks of size numel(input) directly.
        if dim_idx == 0 {
            let elem_count_per = tensors[0].numel();
            let mut offset = 0usize;
            for t in tensors.iter() {
                let src_ptr = t.as_ptr() as *const T;
                let dst_ptr = unsafe { dst.as_mut_ptr().add(offset) };
                unsafe {
                    core::ptr::copy_nonoverlapping(src_ptr, dst_ptr, elem_count_per);
                }
                offset += elem_count_per;
            }
            return;
        }

        // Fast path: if stacking along the last dimension and all inputs are contiguous,
        if dim_idx == out_rank - 1 {
            let outer_size = if in_rank > 0 {
                (0..in_rank).map(|i| tensors[0].shape[i]).product::<usize>()
            } else {
                1
            };
            let num_tensors = tensors.len();

            for outer_idx in 0..outer_size {
                let dst_base = outer_idx * num_tensors;
                for (tensor_idx, tensor) in tensors.iter().enumerate() {
                    let src_ptr = tensor.as_ptr() as *const T;
                    let src_offset = outer_idx;
                    let dst_offset = dst_base + tensor_idx;

                    unsafe {
                        let val = core::ptr::read(src_ptr.add(src_offset));
                        *dst.get_unchecked_mut(dst_offset) = val;
                    }
                }
            }
            return;
        }

        // Fast path: if stacking along middle dimensions and all inputs are contiguous,
        let outer_size = (0..dim_idx).map(|i| tensors[0].shape[i]).product::<usize>();
        let inner_size = (dim_idx..in_rank)
            .map(|i| tensors[0].shape[i])
            .product::<usize>();
        let num_tensors = tensors.len();

        for outer_idx in 0..outer_size {
            for (tensor_idx, tensor) in tensors.iter().enumerate() {
                let src_base = outer_idx * inner_size;
                let dst_base = outer_idx * num_tensors * inner_size + tensor_idx * inner_size;

                let src_ptr = tensor.as_ptr() as *const T;
                let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_base) };

                unsafe {
                    core::ptr::copy_nonoverlapping(src_ptr.add(src_base), dst_ptr, inner_size);
                }
            }
        }
    }

    /// General stack implementation for non-contiguous tensors
    fn stack_general<T: crate::TensorElement>(
        tensors: &[Self],
        new_shape: &Shape,
        dim_idx: usize,
        dst: &mut [T],
    ) {
        let out_rank = new_shape.len();
        let in_rank = out_rank - 1;

        // Cache per-tensor base pointer and strides
        struct Src<'a> {
            base: *const u8,
            strides: Shape,
            _p: core::marker::PhantomData<&'a ()>,
        }
        let mut sources: Vec<Src> = Vec::with_capacity(tensors.len());
        for t in tensors {
            let mut s = Shape::empty().with_len(in_rank);
            for i in 0..in_rank {
                s[i] = if i < t.strides.len() { t.strides[i] } else { 0 };
            }
            sources.push(Src {
                base: t.as_ptr(),
                strides: s,
                _p: core::marker::PhantomData,
            });
        }

        let total = new_shape.numel();
        let chunk_size = 1024.min(total); // Process in chunks for better cache locality
        let mut out_indices = Shape::empty().with_len(out_rank);

        for chunk_start in (0..total).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(total);

            for pos in chunk_start..chunk_end {
                let slot = unsafe { dst.get_unchecked_mut(pos) };

                // Decode linear index of output
                let mut rem = pos;
                for i in (0..out_rank).rev() {
                    let dim = new_shape[i];
                    out_indices[i] = rem % dim;
                    rem /= dim;
                }

                let tensor_idx = out_indices[dim_idx];
                let src = unsafe { sources.get_unchecked(tensor_idx) };

                // Map output indices to input indices (skip stacked dim)
                let mut offset = 0usize;
                let mut in_i = 0;
                for i in 0..out_rank {
                    if i == dim_idx {
                        continue;
                    }
                    let idx = out_indices[i];
                    offset += idx * src.strides[in_i];
                    in_i += 1;
                }

                let src_typed = src.base as *const T;
                let val = unsafe { core::ptr::read(src_typed.add(offset)) };
                *slot = val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_1d_dim0_i32() {
        let a = Tensor::from_vec(vec![1i32, 2, 3], [3]).unwrap();
        let b = Tensor::from_vec(vec![4i32, 5, 6], [3]).unwrap();
        let out = Tensor::stack(&[a, b], 0).unwrap();
        assert_eq!(out.dims(), [2, 3]);
        assert_eq!(out.to_flat_vec::<i32>().unwrap(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn stack_1d_dim1_i32() {
        let a = Tensor::from_vec(vec![1i32, 2, 3], [3]).unwrap();
        let b = Tensor::from_vec(vec![4i32, 5, 6], [3]).unwrap();
        let out = Tensor::stack(&[a, b], 1).unwrap();
        assert_eq!(out.dims(), [3, 2]);
        assert_eq!(out.to_flat_vec::<i32>().unwrap(), vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn stack_2d_dim0_f32() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], [2, 2]).unwrap();
        let out = Tensor::stack(&[a, b], 0).unwrap();
        assert_eq!(out.dims(), [2, 2, 2]);
        assert_eq!(
            out.to_flat_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }
}
