use anyhow::Result;

use crate::{Dim, Shape, StorageTrait, TensorBase, TensorView};

impl<S: StorageTrait> TensorBase<S> {
    /// Flattens a range of dimensions in the tensor into a single dimension.
    ///
    /// This function creates a new view of the tensor where the specified range of dimensions
    /// is merged into a single dimension. The returned tensor shares the same underlying data
    /// as the original tensor, making this a zero-copy operation.
    ///
    /// # Arguments
    ///
    /// * `start_dim` - The starting dimension to flatten (inclusive)
    /// * `end_dim` - The ending dimension to flatten (inclusive)
    ///
    /// Both dimensions must be valid indices for this tensor, and `start_dim <= end_dim`.
    ///
    /// # Returns
    ///
    /// A `Result<TensorView>` containing:
    /// - `Ok(TensorView)`: A new view with the specified dimensions flattened
    /// - `Err`: If the dimension indices are invalid or out of bounds
    ///
    /// # Examples
    ///
    /// ```
    /// use slsl::Tensor;
    ///
    /// // 3D tensor
    /// let tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])?;
    ///
    /// // Flatten dimensions 0 and 1
    /// let flattened = tensor.flatten(0, 1)?;
    /// assert_eq!(flattened.dims(), [4, 2]);
    ///
    /// // Flatten dimensions 1 and 2
    /// let flattened = tensor.flatten(1, 2)?;
    /// assert_eq!(flattened.dims(), [2, 4]);
    ///
    /// // Flatten all dimensions
    /// let flattened = tensor.flatten_all()?;
    /// assert_eq!(flattened.dims(), [8]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - This operation is memory-efficient as it returns a view rather than copying data
    /// - The flattened dimension size is the product of all dimensions in the range
    /// - Strides are recalculated to maintain correct memory access patterns
    /// - The function follows PyTorch's `flatten` behavior
    /// - For invalid dimension ranges, the function will return an error
    ///
    /// # See Also
    ///
    /// - [`Self::flatten_all`]: Flatten all dimensions into a single dimension
    /// - [`TensorView`]: The view type returned by this function
    ///
    /// [PyTorch flatten]: https://pytorch.org/docs/stable/generated/torch.flatten.html
    pub fn flatten<D1: Dim, D2: Dim>(&self, start_dim: D1, end_dim: D2) -> Result<TensorView<'_>> {
        let rank = self.rank();
        let start_idx = start_dim.to_dim(rank)?;
        let end_idx = end_dim.to_dim(rank)?;

        // Validate dimension range
        if start_idx > end_idx {
            anyhow::bail!(
                "Start dimension {} must be <= end dimension {}",
                start_idx,
                end_idx
            );
        }

        if start_idx >= rank || end_idx >= rank {
            anyhow::bail!(
                "Dimension indices {} and {} are out of bounds for tensor with {} dimensions",
                start_idx,
                end_idx,
                rank
            );
        }

        // Calculate the size of the flattened dimension
        let mut flattened_size = 1;
        for i in start_idx..=end_idx {
            flattened_size *= self.shape[i];
        }

        // Build new shape: [dims_before, flattened_size, dims_after]
        let new_rank = rank - (end_idx - start_idx);
        let mut new_shape_array = Shape::empty().with_len(new_rank);
        let mut out_i = 0;
        for i in 0..start_idx {
            new_shape_array[out_i] = self.shape[i];
            out_i += 1;
        }
        new_shape_array[out_i] = flattened_size;
        out_i += 1;
        for i in (end_idx + 1)..rank {
            new_shape_array[out_i] = self.shape[i];
            out_i += 1;
        }

        // Recalculate strides for contiguous layout of the new view
        let new_strides = Self::compute_contiguous_strides(&new_shape_array);

        Ok(TensorView {
            storage: self.storage.as_storage(),
            ptr: self.ptr,
            dtype: self.dtype,
            shape: new_shape_array,
            strides: new_strides,
            offset_bytes: self.offset_bytes,
        })
    }

    /// Flattens all dimensions of the tensor into a single dimension.
    ///
    /// This is a convenience function that flattens the entire tensor into a 1D tensor.
    /// It's equivalent to calling `flatten(0, self.rank() - 1)`.
    ///
    /// # Returns
    ///
    /// A `Result<TensorView>` containing:
    /// - `Ok(TensorView)`: A new 1D view with all dimensions flattened
    /// - `Err`: If there's an error during the flatten operation
    ///
    /// # Examples
    ///
    /// ```
    /// use slsl::Tensor;
    ///
    /// // 3D tensor
    /// let tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])?;
    ///
    /// // Flatten all dimensions
    /// let flattened = tensor.flatten_all()?;
    /// assert_eq!(flattened.dims(), [8]);
    /// assert_eq!(flattened.to_flat_vec::<i32>()?, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    ///
    /// // 2D tensor
    /// let tensor_2d = Tensor::from_vec(vec![1, 2, 3, 4], [2, 2])?;
    /// let flattened = tensor_2d.flatten_all()?;
    /// assert_eq!(flattened.dims(), [4]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - This operation is memory-efficient as it returns a view rather than copying data
    /// - The resulting tensor will always have exactly one dimension
    /// - This is equivalent to `self.flatten(0, self.rank() - 1)`
    ///
    /// # See Also
    ///
    /// - [`Self::flatten`]: Flatten a specific range of dimensions
    /// - [`TensorView`]: The view type returned by this function
    pub fn flatten_all(&self) -> Result<TensorView<'_>> {
        self.flatten(0, self.rank() - 1)
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn flatten_range_basic() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]).unwrap();
        let v = t.flatten(0, 1).unwrap();
        assert_eq!(v.dims(), [4, 2]);
        let v2 = t.flatten(1, 2).unwrap();
        assert_eq!(v2.dims(), [2, 4]);
    }

    #[test]
    fn flatten_all_basic() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4], [2, 2]).unwrap();
        let v = t.flatten_all().unwrap();
        assert_eq!(v.dims(), [4]);
    }

    #[test]
    fn flatten_errors() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4], [2, 2]).unwrap();
        assert!(t.flatten(1, 0).is_err());
        assert!(t.flatten(0, 3).is_err());
    }
}
