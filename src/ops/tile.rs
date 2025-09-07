use anyhow::Result;

use crate::{DType, Shape, StorageTrait, Tensor, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    /// Constructs a tensor by repeating the elements of `self` according to the specified pattern.
    ///
    /// This function creates a new tensor where each element of the input tensor is repeated
    /// according to the `repeats` argument. The `repeats` argument specifies the number of
    /// repetitions in each dimension.
    ///
    /// # Arguments
    ///
    /// * `repeats` - An array specifying the number of repetitions for each dimension.
    ///   Can be converted into `Shape` (e.g., `[2, 3]`, `vec![2, 3]`, etc.)
    ///
    /// # Behavior
    ///
    /// - **Fewer dimensions in repeats**: If `repeats` has fewer dimensions than `self`,
    ///   ones are prepended to `repeats` until all dimensions are specified.
    ///   - Example: `self` shape `(8, 6, 4, 2)`, `repeats` `[2, 2]` → treated as `[1, 1, 2, 2]`
    ///
    /// - **More dimensions in repeats**: If `self` has fewer dimensions than `repeats`,
    ///   `self` is treated as if it were unsqueezed at dimension zero until it has as many
    ///   dimensions as `repeats` specifies.
    ///   - Example: `self` shape `(4, 2)`, `repeats` `[3, 3, 2, 2]` → `self` treated as `(1, 1, 4, 2)`
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A new tensor with the repeated data
    /// - `Err`: If there's an error during the tiling operation
    ///
    /// # Examples
    ///
    /// ```
    /// use slsl::Tensor;
    ///
    /// // 1D tensor
    /// let tensor = Tensor::from_vec(vec![1, 2, 3], [3])?;
    ///
    /// // Repeat 2 times
    /// let tiled = tensor.tile([2])?;
    /// assert_eq!(tiled.dims(), [6]);
    /// assert_eq!(tiled.to_flat_vec::<i32>()?, vec![1, 2, 3, 1, 2, 3]);
    ///
    /// // 2D tensor
    /// let tensor_2d = Tensor::from_vec(vec![1, 2, 3, 4], [2, 2])?;
    ///
    /// // Repeat 2 times in each dimension
    /// let tiled = tensor_2d.tile([2, 2])?;
    /// assert_eq!(tiled.dims(), [4, 4]);
    ///
    /// // Repeat with different counts
    /// let tiled = tensor_2d.tile([3, 1])?;
    /// assert_eq!(tiled.dims(), [6, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - This operation creates a new tensor with copied data (not a view)
    /// - The resulting tensor size is the element-wise product of `self.shape` and `repeats`
    /// - The function follows PyTorch's `tile` behavior
    /// - All supported data types are handled automatically
    ///
    /// # See Also
    ///
    /// - Note: `repeat` function is not yet implemented in this library
    /// - [PyTorch tile][]: <https://pytorch.org/docs/stable/generated/torch.tile.html>
    ///
    /// [PyTorch tile]: https://pytorch.org/docs/stable/generated/torch.tile.html
    pub fn tile<D: Into<Shape>>(&self, repeats: D) -> Result<Tensor> {
        let repeats = repeats.into();
        let self_rank = self.rank();
        let repeats_len = repeats.len();

        // Determine the target rank (maximum of self rank and repeats length)
        let target_rank = self_rank.max(repeats_len);

        // Create expanded shape and repeats arrays using Shape
        let mut expanded_shape = Shape::empty().with_len(target_rank);
        let mut expanded_repeats = Shape::empty().with_len(target_rank);

        // repeats (prepend ones if needed)
        let rep_pad = target_rank - repeats_len;
        for i in 0..rep_pad {
            expanded_repeats[i] = 1;
        }
        for i in 0..repeats_len {
            expanded_repeats[rep_pad + i] = repeats[i];
        }

        // shape (prepend ones to self if needed)
        let shp_pad = target_rank - self_rank;
        for i in 0..shp_pad {
            expanded_shape[i] = 1;
        }
        for i in 0..self_rank {
            expanded_shape[shp_pad + i] = self.shape[i];
        }

        // Calculate the final shape after tiling
        let mut final_shape = Shape::empty().with_len(target_rank);
        for i in 0..target_rank {
            final_shape[i] = expanded_shape[i] * expanded_repeats[i];
        }

        // Calculate the total number of elements in the tiled tensor
        let total_elements = final_shape.numel();

        // We need to handle different data types generically
        // Let's use a match statement to handle all supported DType cases
        match self.dtype {
            DType::Bool => {
                let out = UninitVec::<bool>::new(total_elements).init_with(|dst| {
                    Self::_tile_fill::<bool>(self, &expanded_shape, &final_shape, target_rank, dst)
                });
                Tensor::from_vec(out, final_shape)
            }
            DType::Int8 => {
                let out = UninitVec::<i8>::new(total_elements).init_with(|dst| {
                    Self::_tile_fill::<i8>(self, &expanded_shape, &final_shape, target_rank, dst)
                });
                Tensor::from_vec(out, final_shape)
            }
            DType::Int16 => {
                let out = UninitVec::<i16>::new(total_elements).init_with(|dst| {
                    Self::_tile_fill::<i16>(self, &expanded_shape, &final_shape, target_rank, dst)
                });
                Tensor::from_vec(out, final_shape)
            }
            DType::Int32 => {
                let out = UninitVec::<i32>::new(total_elements).init_with(|dst| {
                    Self::_tile_fill::<i32>(self, &expanded_shape, &final_shape, target_rank, dst)
                });
                Tensor::from_vec(out, final_shape)
            }
            DType::Int64 => {
                let out = UninitVec::<i64>::new(total_elements).init_with(|dst| {
                    Self::_tile_fill::<i64>(self, &expanded_shape, &final_shape, target_rank, dst)
                });
                Tensor::from_vec(out, final_shape)
            }
            DType::Uint8 => {
                let out = UninitVec::<u8>::new(total_elements).init_with(|dst| {
                    Self::_tile_fill::<u8>(self, &expanded_shape, &final_shape, target_rank, dst)
                });
                Tensor::from_vec(out, final_shape)
            }
            DType::Uint16 => {
                let out = UninitVec::<u16>::new(total_elements).init_with(|dst| {
                    Self::_tile_fill::<u16>(self, &expanded_shape, &final_shape, target_rank, dst)
                });
                Tensor::from_vec(out, final_shape)
            }
            DType::Uint32 => {
                let out = UninitVec::<u32>::new(total_elements).init_with(|dst| {
                    Self::_tile_fill::<u32>(self, &expanded_shape, &final_shape, target_rank, dst)
                });
                Tensor::from_vec(out, final_shape)
            }
            DType::Uint64 => {
                let out = UninitVec::<u64>::new(total_elements).init_with(|dst| {
                    Self::_tile_fill::<u64>(self, &expanded_shape, &final_shape, target_rank, dst)
                });
                Tensor::from_vec(out, final_shape)
            }
            DType::Fp16 => {
                let out = UninitVec::<half::f16>::new(total_elements).init_with(|dst| {
                    Self::_tile_fill::<half::f16>(
                        self,
                        &expanded_shape,
                        &final_shape,
                        target_rank,
                        dst,
                    )
                });
                Tensor::from_vec(out, final_shape)
            }
            DType::Fp32 => {
                let out = UninitVec::<f32>::new(total_elements).init_with(|dst| {
                    Self::_tile_fill::<f32>(self, &expanded_shape, &final_shape, target_rank, dst)
                });
                Tensor::from_vec(out, final_shape)
            }
            DType::Fp64 => {
                let out = UninitVec::<f64>::new(total_elements).init_with(|dst| {
                    Self::_tile_fill::<f64>(self, &expanded_shape, &final_shape, target_rank, dst)
                });
                Tensor::from_vec(out, final_shape)
            }
            DType::Bf16 => {
                let out = UninitVec::<half::bf16>::new(total_elements).init_with(|dst| {
                    Self::_tile_fill::<half::bf16>(
                        self,
                        &expanded_shape,
                        &final_shape,
                        target_rank,
                        dst,
                    )
                });
                Tensor::from_vec(out, final_shape)
            }
            _ => {
                anyhow::bail!("tile function not supported for Auto dtype")
            }
        }
    }

    /// Fast filler for tile using pointer reads and expanded strides
    fn _tile_fill<T: crate::TensorElement>(
        &self,
        expanded_shape: &Shape,
        final_shape: &Shape,
        target_rank: usize,
        dst: &mut [T],
    ) {
        let self_rank = self.rank();
        let shp_pad = target_rank - self_rank;

        // Build expanded strides that correspond to expanded_shape
        let mut exp_strides = Shape::empty().with_len(target_rank);
        for i in 0..target_rank {
            if i < shp_pad {
                exp_strides[i] = 0;
            } else {
                exp_strides[i] = if (i - shp_pad) < self.strides.len() {
                    self.strides[i - shp_pad]
                } else {
                    0
                };
            }
        }

        let base = self.as_ptr() as *const T;
        let total = final_shape.numel();

        let mut idx = Shape::empty().with_len(target_rank);

        for (pos, slot) in dst.iter_mut().enumerate().take(total) {
            let mut rem = pos;
            for i in (0..target_rank).rev() {
                let dim = final_shape[i];
                idx[i] = rem % dim;
                rem /= dim;
            }
            // Reduce indices by repeats to original indices
            let mut offset_elems = 0usize;
            for i in 0..target_rank {
                let dim_sz = expanded_shape[i];
                let orig = if dim_sz == 0 { 0 } else { idx[i] % dim_sz };
                offset_elems += orig * exp_strides[i];
            }
            let val = unsafe { core::ptr::read(base.add(offset_elems)) };
            *slot = val;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn tile_1d_basic() {
        let t = Tensor::from_vec(vec![1i32, 2, 3], [3]).unwrap();
        let out = t.tile([2]).unwrap();
        assert_eq!(out.dims(), [6]);
        assert_eq!(out.to_flat_vec::<i32>().unwrap(), vec![1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn tile_2d_symmetric() {
        let t = Tensor::from_vec(vec![1i32, 2, 3, 4], [2, 2]).unwrap();
        let out = t.tile([2, 2]).unwrap();
        assert_eq!(out.dims(), [4, 4]);
    }

    #[test]
    fn tile_prepended_repeats() {
        let t = Tensor::from_vec(vec![1i32, 2, 3, 4], [2, 2]).unwrap();
        let out = t.tile([3]).unwrap();
        assert_eq!(out.dims(), [2, 6]);
    }

    #[test]
    fn tile_expand_rank() {
        let t = Tensor::from_vec(vec![1i32, 2, 3], [3]).unwrap();
        let out = t.tile([2, 2]).unwrap();
        assert_eq!(out.dims(), [2, 6]);
    }
}
