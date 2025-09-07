use anyhow::Result;

use crate::{Dim, Dims, Shape, StorageTrait, TensorBase, TensorView};

impl<S: StorageTrait> TensorBase<S> {
    /// Returns a new tensor with dimensions of size one removed from the specified positions.
    ///
    /// This function creates a new view of the tensor with dimensions of size 1 removed
    /// from the specified positions. The returned tensor shares the same underlying data
    /// as the original tensor, making this a zero-copy operation.
    ///
    /// # Arguments
    ///
    /// * `dims` - The dimensions to squeeze. Must be valid dimension indices for this tensor.
    ///   - Can be a single dimension index, a slice of indices, or a range
    ///   - Only dimensions of size 1 will be removed
    ///   - Dimensions of size greater than 1 will be ignored
    ///
    /// # Returns
    ///
    /// A `Result<TensorView>` containing:
    /// - `Ok(TensorView)`: A new view with the specified dimensions removed
    /// - `Err`: If any dimension index is out of bounds
    ///
    /// # Examples
    ///
    /// ```
    /// use slsl::Tensor;
    ///
    /// // 3D tensor with some dimensions of size 1
    /// let tensor = Tensor::from_vec(vec![1, 2, 3, 4], [1, 4, 1])?;
    ///
    /// // Squeeze specific dimensions
    /// let squeezed = tensor.squeeze(0)?;  // Remove dimension 0
    /// assert_eq!(squeezed.dims(), [4, 1]);
    ///
    /// let squeezed = tensor.squeeze([0, 2])?;  // Remove dimensions 0 and 2
    /// assert_eq!(squeezed.dims(), [4]);
    ///
    /// // Squeeze all dimensions of size 1
    /// let squeezed = tensor.squeeze_all()?;
    /// assert_eq!(squeezed.dims(), [4]);
    ///
    /// // 2D tensor with no dimensions of size 1
    /// let tensor_2d = Tensor::from_vec(vec![1, 2, 3, 4], [2, 2])?;
    /// let squeezed = tensor_2d.squeeze(0)?;  // No effect
    /// assert_eq!(squeezed.dims(), [2, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - This operation is memory-efficient as it returns a view rather than copying data
    /// - Only dimensions of size 1 are removed; larger dimensions are preserved
    /// - If all dimensions are removed, a scalar tensor (shape `[1]`) is returned
    /// - The function follows PyTorch's `squeeze` behavior
    /// - For out-of-bounds dimensions, the function will return an error
    ///
    /// # See Also
    ///
    /// - [`Self::unsqueeze`]: Add dimensions of size 1
    /// - [`Self::squeeze_all`]: Remove all dimensions of size 1
    /// - [`TensorView`]: The view type returned by this function
    ///
    /// [PyTorch squeeze]: https://docs.pytorch.org/docs/stable/generated/torch.squeeze.html
    pub fn squeeze<D: Dims>(&self, dims: D) -> Result<TensorView<'_>> {
        let dims = dims.to_dims(self.rank())?;

        // Squeeze only the specified dimensions (if they are of size 1)
        let mut new_shape_array = Shape::empty();
        let mut new_strides_array = Shape::empty();
        let mut count = 0usize;

        for i in 0..self.rank() {
            let should_squeeze = dims.iter().any(|&d_idx| d_idx == i);

            if !should_squeeze || self.shape[i] != 1 {
                // safe because count < rank <= 8
                unsafe {
                    new_shape_array.set_len(count + 1);
                    new_strides_array.set_len(count + 1);
                }
                new_shape_array[count] = self.shape[i];
                new_strides_array[count] = self.strides[i];
                count += 1;
            }
        }

        if count == 0 {
            // If all dimensions were 1, create a scalar tensor
            unsafe {
                new_shape_array.set_len(1);
                new_strides_array.set_len(1);
            }
            new_shape_array[0] = 1;
            new_strides_array[0] = 0;
        }

        // new_shape_array/new_strides_array already populated

        Ok(TensorView {
            storage: self.storage.as_storage(),
            ptr: self.ptr,
            dtype: self.dtype,
            shape: new_shape_array,
            strides: new_strides_array,
            offset_bytes: self.offset_bytes,
        })
    }

    /// Returns a new tensor with all dimensions of size one removed.
    ///
    /// This is a convenience function that removes all dimensions of size 1 from the tensor.
    /// It's equivalent to calling `squeeze` with all dimension indices.
    ///
    /// # Returns
    ///
    /// A `Result<TensorView>` containing:
    /// - `Ok(TensorView)`: A new view with all size-1 dimensions removed
    /// - `Err`: If there's an error during the squeeze operation
    ///
    /// # Examples
    ///
    /// ```
    /// use slsl::Tensor;
    ///
    /// // Tensor with multiple dimensions of size 1
    /// let tensor = Tensor::from_vec(vec![1, 2, 3, 4], [1, 4, 1, 1])?;
    ///
    /// // Remove all dimensions of size 1
    /// let squeezed = tensor.squeeze_all()?;
    /// assert_eq!(squeezed.dims(), [4]);
    ///
    /// // Tensor with no dimensions of size 1
    /// let tensor_2d = Tensor::from_vec(vec![1, 2, 3, 4], [2, 2])?;
    /// let squeezed = tensor_2d.squeeze_all()?;
    /// assert_eq!(squeezed.dims(), [2, 2]);  // No change
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - This operation is memory-efficient as it returns a view rather than copying data
    /// - If all dimensions are of size 1, a scalar tensor (shape `[1]`) is returned
    /// - This is equivalent to `self.squeeze(0..self.rank())`
    ///
    /// # See Also
    ///
    /// - [`Self::squeeze`]: Remove specific dimensions of size 1
    /// - [`Self::unsqueeze`]: Add dimensions of size 1
    pub fn squeeze_all(&self) -> Result<TensorView<'_>> {
        // Create a range of all dimensions
        let all_dims: Vec<usize> = (0..self.rank()).collect();
        self.squeeze(&*all_dims)
    }

    /// Returns a new tensor with a dimension of size one inserted at the specified position.
    ///
    /// This function creates a new view of the tensor with an additional dimension of size 1
    /// inserted at the specified position. The returned tensor shares the same underlying data
    /// as the original tensor, making this a zero-copy operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The position at which to insert the new dimension. Must be in the range `[-rank-1, rank]`.
    ///   - For a 1D tensor `[4]`, valid values are `[-2, 1]`
    ///   - For a 2D tensor `[2, 2]`, valid values are `[-3, 2]`
    ///   - Negative indices count from the end: `-1` means the last position, `-2` means the second-to-last, etc.
    ///
    /// # Returns
    ///
    /// A `Result<TensorView>` containing:
    /// - `Ok(TensorView)`: A new view with the inserted dimension
    /// - `Err`: If the dimension index is out of bounds
    ///
    /// # Examples
    ///
    /// ```
    /// use slsl::Tensor;
    ///
    /// // 1D tensor
    /// let tensor = Tensor::from_vec(vec![1, 2, 3, 4], [4])?;
    ///
    /// // Insert at beginning (dimension 0)
    /// let unsqueezed = tensor.unsqueeze(0)?;
    /// assert_eq!(unsqueezed.dims(), [1, 4]);
    ///
    /// // Insert at end (dimension 1)
    /// let unsqueezed = tensor.unsqueeze(1)?;
    /// assert_eq!(unsqueezed.dims(), [4, 1]);
    ///
    /// // Using negative indices
    /// let unsqueezed = tensor.unsqueeze(-1)?;  // Same as unsqueeze(1)
    /// assert_eq!(unsqueezed.dims(), [4, 1]);
    ///
    /// let unsqueezed = tensor.unsqueeze(-2)?;  // Same as unsqueeze(0)
    /// assert_eq!(unsqueezed.dims(), [1, 4]);
    ///
    /// // 2D tensor
    /// let tensor_2d = Tensor::from_vec(vec![1, 2, 3, 4], [2, 2])?;
    /// let unsqueezed = tensor_2d.unsqueeze(1)?;
    /// assert_eq!(unsqueezed.dims(), [2, 1, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - This operation is memory-efficient as it returns a view rather than copying data
    /// - The stride for the new dimension is set to 0 since its size is 1
    /// - The function follows PyTorch's `unsqueeze` behavior for dimension indexing
    /// - For out-of-bounds dimensions, the function will return an error rather than silently
    ///   inserting at the end, ensuring user intent is clear
    ///
    /// # See Also
    ///
    /// - [`Self::squeeze`]: Remove dimensions of size 1
    /// - [`TensorView`]: The view type returned by this function
    ///
    /// [PyTorch unsqueeze]: https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html
    pub fn unsqueeze<D: Dim>(&self, dim: D) -> Result<TensorView<'_>> {
        // For unsqueeze, dimension index can be in range [-rank-1, rank]
        // This allows inserting at any position including at the end
        let current_rank = self.rank();
        let dim_idx = dim.to_dim(current_rank + 1)?;

        // Create new shape/strides using Shape with size 1 inserted at the specified dimension
        let mut new_shape_array = Shape::empty().with_len(current_rank + 1);
        let mut new_strides_array = Shape::empty().with_len(current_rank + 1);

        // Insert dimensions before the specified position
        for i in 0..dim_idx {
            new_shape_array[i] = self.shape[i];
            new_strides_array[i] = self.strides[i];
        }

        // Insert the new dimension of size 1
        new_shape_array[dim_idx] = 1;
        new_strides_array[dim_idx] = 0; // Stride 0 for size 1 dimension

        // Insert dimensions after the specified position
        for i in dim_idx..current_rank {
            new_shape_array[i + 1] = self.shape[i];
            new_strides_array[i + 1] = self.strides[i];
        }

        Ok(TensorView {
            storage: self.storage.as_storage(),
            ptr: self.ptr,
            dtype: self.dtype,
            shape: new_shape_array,
            strides: new_strides_array,
            offset_bytes: self.offset_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn squeeze_specific_dims() {
        let t = Tensor::from_vec(vec![1u8, 2, 3, 4], [1, 4, 1]).unwrap();
        let v = t.squeeze([0, 2]).unwrap();
        assert_eq!(v.dims(), [4]);
    }

    #[test]
    fn squeeze_all_dims() {
        let t = Tensor::from_vec(vec![1u8, 2, 3, 4], [1, 4, 1]).unwrap();
        let v = t.squeeze_all().unwrap();
        assert_eq!(v.dims(), [4]);

        let t = Tensor::rand(0., 10., [1, 2, 1, 3, 4, 1, 2, 1]).unwrap();
        let v = t.squeeze_all().unwrap();
        assert_eq!(v.dims(), [2, 3, 4, 2]);
    }

    #[test]
    fn squeeze_no_effect() {
        let t = Tensor::from_vec(vec![1u8, 2, 3, 4], [2, 2]).unwrap();
        let v = t.squeeze(0).unwrap();
        assert_eq!(v.dims(), [2, 2]);
    }

    #[test]
    fn unsqueeze_all_dims() {
        let t = Tensor::rand(0., 10., [2, 3, 4]).unwrap();
        let v = t.unsqueeze(0).unwrap();
        assert_eq!(v.dims(), [1, 2, 3, 4]);

        let v = t.unsqueeze(1).unwrap();
        assert_eq!(v.dims(), [2, 1, 3, 4]);

        let v = t.unsqueeze(2).unwrap();
        assert_eq!(v.dims(), [2, 3, 1, 4]);

        let v = t.unsqueeze(3).unwrap();
        assert_eq!(v.dims(), [2, 3, 4, 1]);

        let v = t.unsqueeze(-1).unwrap();
        assert_eq!(v.dims(), [2, 3, 4, 1]);

        let v = t.unsqueeze(-2).unwrap();
        assert_eq!(v.dims(), [2, 3, 1, 4]);
    }
}
