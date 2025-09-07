use anyhow::Result;

use crate::{DType, Dim, Shape, StorageTrait, Stride, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    /// Concatenates a sequence of tensors along an existing dimension.
    ///
    /// This function creates a new tensor by concatenating the input tensors along the specified
    /// dimension. All input tensors must have the same rank and data type, and all dimensions
    /// except the concatenation dimension must have the same size.
    ///
    /// # Arguments
    ///
    /// * `tensors` - A slice of tensor references to concatenate. Must not be empty.
    /// * `dim` - The dimension along which to concatenate. Must be a valid dimension index
    ///   for the input tensors.
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A new tensor with the concatenated data
    /// - `Err`: If tensors have different ranks/shapes/dtypes, or if the dimension is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use slsl::Tensor;
    ///
    /// // Concatenate 1D tensors
    /// let tensor1 = Tensor::from_vec(vec![1, 2, 3], [3])?;
    /// let tensor2 = Tensor::from_vec(vec![4, 5, 6], [3])?;
    /// let tensor3 = Tensor::from_vec(vec![7, 8, 9], [3])?;
    /// let tensors = vec![&tensor1, &tensor2, &tensor3];
    ///
    /// // Concatenate along dimension 0
    /// let concatenated = Tensor::cat(&tensors, 0)?;
    /// assert_eq!(concatenated.dims(), [9]);
    /// assert_eq!(concatenated.to_flat_vec::<i32>()?, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    ///
    /// // Concatenate 2D tensors along dimension 0
    /// let tensor_2d1 = Tensor::from_vec(vec![1, 2, 3, 4], [2, 2])?;
    /// let tensor_2d2 = Tensor::from_vec(vec![5, 6, 7, 8], [2, 2])?;
    /// let tensors_2d = vec![&tensor_2d1, &tensor_2d2];
    ///
    /// let concatenated_2d = Tensor::cat(&tensors_2d, 0)?;
    /// assert_eq!(concatenated_2d.dims(), [4, 2]);
    ///
    /// // Concatenate along dimension 1
    /// let concatenated_2d_dim1 = Tensor::cat(&tensors_2d, 1)?;
    /// assert_eq!(concatenated_2d_dim1.dims(), [2, 4]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - All input tensors must have identical ranks and data types
    /// - All dimensions except the concatenation dimension must have the same size
    /// - The concatenation dimension size equals the sum of all input tensor sizes in that dimension
    /// - The function follows PyTorch's `cat` behavior
    /// - For out-of-bounds dimensions, the function will return an error
    ///
    /// Concatenate tensors with any storage type (Tensor or TensorView)
    pub fn cat<D: Dim>(tensors: &[&Self], dim: D) -> Result<Tensor> {
        Self::cat_any::<S, D>(tensors, dim)
    }
    fn cat_any<TS: StorageTrait, D: Dim>(tensors: &[&TensorBase<TS>], dim: D) -> Result<Tensor> {
        if tensors.is_empty() {
            anyhow::bail!("Cannot concatenate empty tensor list");
        }

        let dim_idx = dim.to_dim(tensors[0].rank())?;
        let first_tensor = tensors[0];
        let first_shape = first_tensor.shape;
        let first_dtype = first_tensor.dtype;

        // Check that all tensors have the same rank and dtype
        for (i, tensor) in tensors.iter().enumerate() {
            if tensor.rank() != first_shape.len() {
                anyhow::bail!(
                    "Tensor {} has different rank {} than first tensor {}",
                    i,
                    tensor.rank(),
                    first_shape.len()
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

        // Check that all dimensions except the concatenation dimension are the same
        for (i, tensor) in tensors.iter().enumerate() {
            for d in 0..first_shape.len() {
                if d != dim_idx && tensor.shape[d] != first_shape[d] {
                    anyhow::bail!(
                        "Tensor {} has different size {} in dimension {} than first tensor {}",
                        i,
                        tensor.shape[d],
                        d,
                        first_shape[d]
                    );
                }
            }
        }

        // Calculate new shape using Shape: sum the sizes along the concatenation dimension
        let rank = first_shape.len();
        let mut new_shape = Shape::empty().with_len(rank);
        for d in 0..rank {
            new_shape[d] = if d == dim_idx {
                tensors.iter().map(|t| t.shape[d]).sum()
            } else {
                first_shape[d]
            };
        }

        let total_elements: usize = new_shape.numel();

        // Handle different data types
        macro_rules! dispatch_cat {
            ($ty:ty) => {{
                let out = UninitVec::<$ty>::new(total_elements).init_with(|dst| {
                    Self::_cat_fill_for::<TS, $ty>(tensors, &new_shape, dim_idx, dst)
                });
                Tensor::from_vec(out, new_shape)
            }};
        }

        match first_dtype {
            DType::Bool => dispatch_cat!(bool),
            DType::Int8 => dispatch_cat!(i8),
            DType::Int16 => dispatch_cat!(i16),
            DType::Int32 => dispatch_cat!(i32),
            DType::Int64 => dispatch_cat!(i64),
            DType::Uint8 => dispatch_cat!(u8),
            DType::Uint16 => dispatch_cat!(u16),
            DType::Uint32 => dispatch_cat!(u32),
            DType::Uint64 => dispatch_cat!(u64),
            DType::Fp16 => dispatch_cat!(half::f16),
            DType::Fp32 => dispatch_cat!(f32),
            DType::Fp64 => dispatch_cat!(f64),
            DType::Bf16 => dispatch_cat!(half::bf16),
            _ => anyhow::bail!("cat function not supported for dtype: {:?}", first_dtype),
        }
    }

    /// Fill destination slice for cat implementation without intermediate Vec allocations
    fn _cat_fill_for<TS: StorageTrait, T: crate::TensorElement>(
        tensors: &[&TensorBase<TS>],
        new_shape: &Shape,
        dim_idx: usize,
        dst: &mut [T],
    ) {
        let rank = new_shape.len();

        // Build prefix sizes along concat dim for fast tensor selection
        let mut prefix = Vec::with_capacity(tensors.len() + 1);
        prefix.push(0usize);
        for t in tensors {
            let last = *prefix.last().unwrap();
            prefix.push(last + t.shape[dim_idx]);
        }

        let total_elements: usize = new_shape.numel();

        // Precompute strides for each tensor to avoid virtual calls inside loop
        // and reuse typed base pointers for fast reads
        struct Src<'a> {
            base: *const u8,
            strides: Stride,
            _phantom: core::marker::PhantomData<&'a ()>,
        }

        let mut sources: Vec<Src> = Vec::with_capacity(tensors.len());
        for t in tensors {
            let mut s = Shape::empty().with_len(rank);
            for i in 0..rank {
                s[i] = if i < t.strides.len() { t.strides[i] } else { 0 };
            }
            sources.push(Src {
                base: t.as_ptr(),
                strides: s,
                _phantom: core::marker::PhantomData,
            });
        }

        let mut indices = Shape::empty().with_len(rank);

        for (pos, slot) in dst.iter_mut().enumerate().take(total_elements) {
            // decode linear index into multi-dimensional indices (row-major)
            let mut rem = pos;
            for i in (0..rank).rev() {
                let dim = new_shape[i];
                indices[i] = rem % dim;
                rem /= dim;
            }

            // locate source tensor by prefix sums (linear scan; can be optimized to binary search)
            let target = indices[dim_idx];
            // Binary search over prefix (monotonic ascending)
            let mut lo = 0usize;
            let mut hi = prefix.len() - 1; // last is total
            while lo + 1 < hi {
                let mid = (lo + hi) >> 1;
                if target < prefix[mid] {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let tensor_idx = lo;

            let base_offset_in_dim = prefix[tensor_idx];
            let src = &sources[tensor_idx];

            // compute source element offset (in elements) using strides and adjusted indices
            let mut offset_elems = 0usize;
            for i in 0..rank {
                let idx = if i == dim_idx {
                    indices[i] - base_offset_in_dim
                } else {
                    indices[i]
                };
                offset_elems += idx * src.strides[i];
            }

            // read value
            let src_typed = src.base as *const T;
            let val = unsafe { core::ptr::read(src_typed.add(offset_elems)) };

            // write into destination
            *slot = val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cat_1d_int32_basic() {
        let a = Tensor::from_vec(vec![1i32, 2, 3], [3]).unwrap();
        let b = Tensor::from_vec(vec![4i32, 5, 6], [3]).unwrap();
        let out = Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(out.dims(), [6]);
        assert_eq!(out.to_flat_vec::<i32>().unwrap(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn cat_2d_dim0_f32() {
        let a_src = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let b_src = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], [2, 2]).unwrap();
        let a = a_src.view();
        let b = b_src.view();
        let out = Tensor::cat_any(&[&a, &b], 0).unwrap();
        assert_eq!(out.dims(), [4, 2]);
        assert_eq!(
            out.to_flat_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn cat_2d_dim1_f32() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], [2, 2]).unwrap();
        let out = Tensor::cat(&[&a, &b], 1).unwrap();
        assert_eq!(out.dims(), [2, 4]);
        assert_eq!(
            out.to_flat_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );
    }

    #[test]
    fn cat_mismatched_shapes_error() {
        let a = Tensor::from_vec(vec![1i32, 2, 3, 4], [2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5i32, 6, 7, 8, 9, 10], [3, 2]).unwrap();
        let err = Tensor::cat(&[&a, &b], 1).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("different size") || msg.contains("different rank"));
    }
}
