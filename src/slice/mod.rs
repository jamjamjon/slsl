pub mod elem;
#[macro_use]
mod r#macro;

pub use elem::*;

use crate::{Shape, Stride, Tensor, TensorView};

/// Implementation for [`TensorView`] - reuse [`Tensor`] logic
impl<'a> TensorView<'a> {
    /// Creates a slice view of this tensor view.
    ///
    /// This method provides zero-copy slicing capabilities for tensor views.
    ///
    /// # Parameters
    ///
    /// * `specs` - The slice specification (indices, ranges, tuples, etc.)
    ///
    /// # Returns
    ///
    /// A new [`TensorView`] representing the sliced portion.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::{s, Tensor};
    ///
    /// let tensor = Tensor::zeros::<f32>([4, 6])?;
    /// let view = tensor.view();
    ///
    /// // Single index
    /// let row = view.slice(1);
    /// assert_eq!(row.shape().as_slice(), &[6]);
    ///
    /// // Range slice using s! macro
    /// let rows = view.slice(s![1..3]);
    /// assert_eq!(rows.shape().as_slice(), &[2, 6]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn slice<S: IntoSliceElem + Copy>(&self, specs: S) -> TensorView<'a> {
        let slice = specs.into_slice();
        let (new_shape, new_strides, offset) =
            Tensor::compute_slice(&self.shape, &self.strides, slice);

        TensorView {
            storage: self.storage,
            ptr: self.ptr,
            shape: new_shape,
            strides: new_strides,
            offset_bytes: self.offset_bytes + offset * self.dtype.size_in_bytes(),
            dtype: self.dtype,
        }
    }
}

impl Tensor {
    /// Creates a slice view of this tensor with optimized performance.
    ///
    /// This method provides efficient tensor slicing with fast paths for
    /// common patterns. Creates a zero-copy view without memory allocation.
    ///
    /// # Parameters
    ///
    /// * `specs` - The slice specification (index, range with s! macro, tuple, etc.)
    ///
    /// # Returns
    ///
    /// A [`TensorView`] representing the sliced portion of the tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::{s, Tensor};
    ///
    /// let tensor = Tensor::zeros::<f32>([4, 6])?;
    ///
    /// // Single index
    /// let row = tensor.slice(1);
    /// assert_eq!(row.shape().as_slice(), &[6]);
    ///
    /// // Range slice using s! macro
    /// let rows = tensor.slice(s![1..3]);
    /// assert_eq!(rows.shape().as_slice(), &[2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn slice<S: IntoSliceElem + Copy>(&self, specs: S) -> TensorView<'_> {
        let slice = specs.into_slice();

        // Specialized paths for common patterns
        match slice.len() {
            1 => match slice[0] {
                SliceElem::Index(idx) => self.slice_index(idx),
                SliceElem::Range {
                    start,
                    end,
                    step: 1,
                } => self.slice_range_1d(start, end),
                _ => self.slice_general(slice),
            },
            2 if self.shape.len() >= 2 => match (slice[0], slice[1]) {
                (SliceElem::Index(idx1), SliceElem::Index(idx2)) => self.slice_index_2d(idx1, idx2),
                (
                    SliceElem::Range {
                        start: s1,
                        end: e1,
                        step: 1,
                    },
                    SliceElem::Range {
                        start: s2,
                        end: e2,
                        step: 1,
                    },
                ) => self.slice_range_2d(s1, e1, s2, e2),
                (
                    SliceElem::Index(idx),
                    SliceElem::Range {
                        start,
                        end,
                        step: 1,
                    },
                ) => self.slice_mixed(idx, start, end, true),
                (
                    SliceElem::Range {
                        start,
                        end,
                        step: 1,
                    },
                    SliceElem::Index(idx),
                ) => self.slice_mixed(idx, start, end, false),
                _ => self.slice_general(slice),
            },
            _ => self.slice_general(slice),
        }
    }

    #[inline]
    fn slice_index(&self, idx: isize) -> TensorView<'_> {
        let resolved_idx = self.resolve_index(idx, 0);
        let offset_bytes = self.offset_bytes
            + resolved_idx
                .saturating_mul(self.strides[0])
                .saturating_mul(self.dtype.size_in_bytes());

        if self.shape.len() == 1 {
            // Scalar result
            TensorView {
                storage: &self.storage,
                ptr: self.ptr,
                shape: Shape::empty(),
                strides: Stride::empty(),
                offset_bytes,
                dtype: self.dtype,
            }
        } else {
            // Remove first dimension
            let mut new_shape = Shape::empty();
            let mut new_strides = Stride::empty();

            for i in 1..self.shape.len() {
                new_shape.push(self.shape[i]);
                new_strides.push(self.strides[i]);
            }

            TensorView {
                storage: &self.storage,
                ptr: self.ptr,
                shape: new_shape,
                strides: new_strides,
                offset_bytes,
                dtype: self.dtype,
            }
        }
    }

    #[inline]
    fn slice_index_2d(&self, idx1: isize, idx2: isize) -> TensorView<'_> {
        let resolved_idx1 = self.resolve_index(idx1, 0);
        let resolved_idx2 = self.resolve_index(idx2, 1);

        let offset_bytes = self.offset_bytes
            + (resolved_idx1 * self.strides[0] + resolved_idx2 * self.strides[1])
                * self.dtype.size_in_bytes();

        if self.shape.len() == 2 {
            // Scalar result
            TensorView {
                storage: &self.storage,
                ptr: self.ptr,
                shape: Shape::empty(),
                strides: Stride::empty(),
                offset_bytes,
                dtype: self.dtype,
            }
        } else {
            // Remove first two dimensions
            let mut new_shape = Shape::empty();
            let mut new_strides = Stride::empty();

            for i in 2..self.shape.len() {
                new_shape.push(self.shape[i]);
                new_strides.push(self.strides[i]);
            }

            TensorView {
                storage: &self.storage,
                ptr: self.ptr,
                shape: new_shape,
                strides: new_strides,
                offset_bytes,
                dtype: self.dtype,
            }
        }
    }

    #[inline]
    fn slice_range_1d(&self, start: isize, end: Option<isize>) -> TensorView<'_> {
        let resolved_start = if start >= 0 {
            start as usize
        } else {
            self.shape[0].wrapping_sub((-start) as usize)
        };
        let resolved_end = end.map_or(self.shape[0], |e| self.resolve_range_bound(e, 0));

        let new_size = resolved_end.saturating_sub(resolved_start);
        let offset_bytes = self.offset_bytes
            + resolved_start
                .saturating_mul(self.strides[0])
                .saturating_mul(self.dtype.size_in_bytes());

        let mut new_shape = Shape::empty();
        new_shape.push(new_size);

        let mut new_strides = Stride::empty();
        new_strides.push(self.strides[0]);

        TensorView {
            storage: &self.storage,
            ptr: self.ptr,
            shape: new_shape,
            strides: new_strides,
            offset_bytes,
            dtype: self.dtype,
        }
    }

    #[inline]
    fn slice_range_2d(
        &self,
        start1: isize,
        end1: Option<isize>,
        start2: isize,
        end2: Option<isize>,
    ) -> TensorView<'_> {
        let resolved_start1 = if start1 >= 0 {
            start1 as usize
        } else {
            self.shape[0].wrapping_sub((-start1) as usize)
        };
        let resolved_end1 = end1.map_or(self.shape[0], |e| self.resolve_range_bound(e, 0));
        let resolved_start2 = if start2 >= 0 {
            start2 as usize
        } else {
            self.shape[1].wrapping_sub((-start2) as usize)
        };
        let resolved_end2 = end2.map_or(self.shape[1], |e| self.resolve_range_bound(e, 1));

        let new_size1 = resolved_end1.saturating_sub(resolved_start1);
        let new_size2 = resolved_end2.saturating_sub(resolved_start2);

        let offset_bytes = self.offset_bytes
            + (resolved_start1 * self.strides[0] + resolved_start2 * self.strides[1])
                * self.dtype.size_in_bytes();

        let mut new_shape = Shape::empty();
        new_shape.push(new_size1);
        new_shape.push(new_size2);

        let mut new_strides = Stride::empty();
        new_strides.push(self.strides[0]);
        new_strides.push(self.strides[1]);

        TensorView {
            storage: &self.storage,
            ptr: self.ptr,
            shape: new_shape,
            strides: new_strides,
            offset_bytes,
            dtype: self.dtype,
        }
    }

    #[inline]
    fn slice_mixed(
        &self,
        idx: isize,
        start: isize,
        end: Option<isize>,
        index_first: bool,
    ) -> TensorView<'_> {
        let resolved_idx = if index_first {
            self.resolve_index(idx, 0)
        } else {
            self.resolve_index(idx, 1)
        };

        let (dim_size, range_stride) = if index_first {
            (self.shape[1], self.strides[1])
        } else {
            (self.shape[0], self.strides[0])
        };

        let resolved_start = if start >= 0 {
            start as usize
        } else {
            dim_size.wrapping_sub((-start) as usize)
        };
        let resolved_end = end.map_or(dim_size, |e| {
            if e >= 0 {
                (e as usize).min(dim_size)
            } else {
                dim_size.wrapping_sub((-e) as usize)
            }
        });

        let new_range_size = resolved_end.saturating_sub(resolved_start);

        let offset_bytes = if index_first {
            self.offset_bytes
                + (resolved_idx * self.strides[0] + resolved_start * self.strides[1])
                    * self.dtype.size_in_bytes()
        } else {
            self.offset_bytes
                + (resolved_start * self.strides[0] + resolved_idx * self.strides[1])
                    * self.dtype.size_in_bytes()
        };

        let mut new_shape = Shape::empty();
        new_shape.push(new_range_size);
        let mut new_strides = Stride::empty();
        new_strides.push(range_stride);

        TensorView {
            storage: &self.storage,
            ptr: self.ptr,
            shape: new_shape,
            strides: new_strides,
            offset_bytes,
            dtype: self.dtype,
        }
    }

    #[inline]
    fn resolve_index(&self, idx: isize, axis: usize) -> usize {
        if idx >= 0 {
            (idx as usize).min(self.shape[axis].saturating_sub(1))
        } else {
            self.shape[axis].wrapping_sub((-idx) as usize)
        }
    }

    /// Resolve range boundary (different from index resolution)
    /// Range end can be equal to shape[axis] (exclusive upper bound)
    #[inline]
    fn resolve_range_bound(&self, bound: isize, axis: usize) -> usize {
        if bound >= 0 {
            (bound as usize).min(self.shape[axis])
        } else {
            self.shape[axis].wrapping_sub((-bound) as usize)
        }
    }

    /// General slicing for complex patterns
    #[inline]
    fn slice_general(&self, slice: SliceSpecs) -> TensorView<'_> {
        let (new_shape, new_strides, offset) =
            Self::compute_slice(&self.shape, &self.strides, slice);

        TensorView {
            storage: &self.storage,
            ptr: self.ptr,
            shape: new_shape,
            strides: new_strides,
            offset_bytes: self.offset_bytes + offset * self.dtype.size_in_bytes(),
            dtype: self.dtype,
        }
    }

    #[inline]
    pub fn compute_slice(
        shape: &Shape,
        strides: &Stride,
        slice: SliceSpecs,
    ) -> (Shape, Stride, usize) {
        let mut new_shape = Shape::empty();
        let mut new_strides = Stride::empty();
        let mut offset_bytes = 0;
        let mut dim_idx = 0;

        for i in 0..slice.len() {
            if dim_idx >= shape.len() {
                break;
            }

            let slice_elem = slice[i];
            let dim_size = shape[dim_idx];
            let stride = strides[dim_idx];

            match slice_elem {
                SliceElem::Index(idx) => {
                    let resolved_idx = Self::resolve_index_static(idx, dim_size);
                    offset_bytes += resolved_idx * stride;
                    dim_idx += 1;
                }

                SliceElem::Range { start, end, step } => {
                    let (new_dim_size, new_stride, start_offset) =
                        Self::compute_range_slice(dim_size, stride, start, end, step);

                    new_shape.push(new_dim_size);
                    new_strides.push(new_stride);
                    offset_bytes += start_offset;
                    dim_idx += 1;
                }

                SliceElem::NewAxis => {
                    new_shape.push(1);
                    new_strides.push(0);
                }
            }
        }

        // Copy remaining dimensions
        while dim_idx < shape.len() {
            new_shape.push(shape[dim_idx]);
            new_strides.push(strides[dim_idx]);
            dim_idx += 1;
        }

        (new_shape, new_strides, offset_bytes)
    }

    #[inline]
    fn resolve_index_static(idx: isize, dim_size: usize) -> usize {
        if idx >= 0 {
            (idx as usize).min(dim_size.saturating_sub(1))
        } else {
            dim_size.wrapping_sub((-idx) as usize)
        }
    }

    /// Static version of resolve_range_bound
    #[inline]
    fn resolve_range_bound_static(bound: isize, dim_size: usize) -> usize {
        if bound >= 0 {
            (bound as usize).min(dim_size)
        } else {
            dim_size.wrapping_sub((-bound) as usize)
        }
    }

    #[inline]
    fn compute_range_slice(
        dim_size: usize,
        stride: usize,
        start: isize,
        end: Option<isize>,
        step: isize,
    ) -> (usize, usize, usize) {
        let resolved_start = if start >= 0 {
            start as usize
        } else {
            dim_size.wrapping_sub((-start) as usize)
        };
        // For range end, interpret as exclusive upper bound. Allow end == dim_size.
        let resolved_end = if let Some(end_val) = end {
            Self::resolve_range_bound_static(end_val, dim_size)
        } else {
            dim_size
        };

        if step == 1 {
            let new_dim_size = resolved_end.saturating_sub(resolved_start);
            (new_dim_size, stride, resolved_start * stride)
        } else if step > 0 {
            let step_usize = step as usize;
            let new_dim_size = if resolved_start < resolved_end {
                (resolved_end - resolved_start).div_ceil(step_usize)
            } else {
                0
            };
            (new_dim_size, stride * step_usize, resolved_start * stride)
        } else {
            let step_abs = (-step) as usize;
            let new_dim_size = if resolved_start > resolved_end {
                (resolved_start - resolved_end).div_ceil(step_abs)
            } else {
                0
            };
            (new_dim_size, stride * step_abs, resolved_start * stride)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{s, Tensor};

    #[test]
    fn test_slice_boundary_bug_regression_internal() {
        let tensor = Tensor::zeros::<f32>([8400, 84]).unwrap();

        // Range from index to end
        let slice1 = tensor.slice(s![.., 4..]);
        assert_eq!(slice1.shape().as_slice(), &[8400, 80]);

        // Explicit end at boundary
        let slice2 = tensor.slice(s![.., 4..84]);
        assert_eq!(slice2.shape().as_slice(), &[8400, 80]);

        // End beyond boundary should clamp
        let slice3 = tensor.slice(s![.., 4..85]);
        assert_eq!(slice3.shape().as_slice(), &[8400, 80]);

        let slice4 = tensor.slice(s![.., 4..100]);
        assert_eq!(slice4.shape().as_slice(), &[8400, 80]);

        // Different starting points
        let slice5 = tensor.slice(s![.., 10..90]);
        assert_eq!(slice5.shape().as_slice(), &[8400, 74]);

        // Edge case near boundary
        let slice6 = tensor.slice(s![.., 83..85]);
        assert_eq!(slice6.shape().as_slice(), &[8400, 1]);

        // Empty at boundary
        let slice7 = tensor.slice(s![.., 84..85]);
        assert_eq!(slice7.shape().as_slice(), &[8400, 0]);

        let slice8 = tensor.slice(s![.., 1..2]);
        assert_eq!(slice8.shape().as_slice(), &[8400, 1]);

        let slice9 = tensor.slice(s![.., 1]);
        assert_eq!(slice9.shape().as_slice(), &[8400]);
    }

    #[test]
    fn test_slice_inclusive_ranges_internal() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, [10]).unwrap();

        // Inclusive 1D range
        let inclusive = tensor.slice(s![2..=5]);
        assert_eq!(inclusive.shape().as_slice(), &[4]);

        // Inclusive to end
        let to_end_inclusive = tensor.slice(s![..=4]);
        assert_eq!(to_end_inclusive.shape().as_slice(), &[5]);

        // 2D inclusive ranges
        let data_2d: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let tensor_2d = Tensor::from_vec(data_2d, [4, 5]).unwrap();
        let inclusive_2d = tensor_2d.slice(s![1..=2, 1..=3]);
        let shape = inclusive_2d.shape().as_slice();
        assert_eq!(shape.len(), 2);
        assert!(shape[0] > 0 && shape[1] > 0);
    }

    #[test]
    fn test_slice_negative_indices_internal() {
        // Build 5x4 matrix with 0..20
        let mut data = vec![];
        for i in 0..5 {
            for j in 0..4 {
                data.push((i * 4 + j) as f32);
            }
        }
        let tensor = Tensor::from_vec(data, [5, 4]).unwrap();

        // Last row using negative index
        let last_row = tensor.slice(s![-1]);
        assert_eq!(last_row.shape().as_slice(), &[4]);

        // Last two columns
        let last_cols = tensor.slice(s![.., -2..]);
        assert_eq!(last_cols.shape().as_slice(), &[5, 2]);

        // Mixed negative indices
        let mixed = tensor.slice(s![-2, -3..]);
        assert_eq!(mixed.shape().as_slice(), &[3]);
    }

    #[test]
    fn test_slice_non_contiguous_internal() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, [4, 6]).unwrap();

        // Make non-contiguous by permuting axes
        let transposed = tensor.permute([1, 0]).unwrap();
        assert_eq!(transposed.shape().as_slice(), &[6, 4]);

        let slice = transposed.slice(s![2..5, 1..3]);
        assert_eq!(slice.shape().as_slice(), &[3, 2]);

        // Ensure data is accessible and size is correct
        let flat = slice.to_flat_vec::<f32>().unwrap();
        assert_eq!(flat.len(), 6);
    }

    #[test]
    fn test_slice_view_consistency_internal() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, [4, 6]).unwrap();

        let view = tensor.view();
        let from_view = view.slice(s![1..3, 2..5]);
        let from_tensor = tensor.slice(s![1..3, 2..5]);

        assert_eq!(from_view.shape().as_slice(), from_tensor.shape().as_slice());
        assert_eq!(
            from_view.to_flat_vec::<f32>().unwrap(),
            from_tensor.to_flat_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn test_slice_3d_comprehensive_internal() {
        let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, [3, 4, 5]).unwrap();

        // 3D slice with all dimensions
        let slice_3d = tensor.slice(s![1..3, 2..4, 1..4]);
        assert_eq!(slice_3d.shape().as_slice(), &[2, 2, 3]);

        // 3D slice with mixed indices and ranges
        let mixed_3d = tensor.slice(s![1, 1..3, 2..]);
        assert_eq!(mixed_3d.shape().as_slice(), &[2, 3]);

        // Single element from 3D
        let element_3d = tensor.slice(s![1, 2, 3]);
        assert_eq!(element_3d.shape().as_slice(), &[] as &[usize]);

        // Verify data accessibility
        let flat = slice_3d.to_flat_vec::<f32>().unwrap();
        assert_eq!(flat.len(), 12); // 2 * 2 * 3
    }

    #[test]
    fn test_slice_stepped_ranges_internal() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, [20]).unwrap();

        // Step by 2
        let stepped = tensor.slice(s![..;2]);
        assert_eq!(stepped.shape().as_slice(), &[10]);

        // Step by 3 with range
        let stepped_range = tensor.slice(s![2..17;3]);
        assert_eq!(stepped_range.shape().as_slice(), &[5]); // indices 2, 5, 8, 11, 14

        // 2D stepped slicing (using supported pattern)
        let data_2d: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor_2d = Tensor::from_vec(data_2d, [6, 4]).unwrap();
        let stepped_2d = tensor_2d.slice(s![..;2, 1..3]);
        assert_eq!(stepped_2d.shape().as_slice(), &[3, 2]);
    }

    #[test]
    fn test_slice_edge_cases_internal() {
        // Empty tensor slicing
        let empty_tensor = Tensor::zeros::<f32>([0, 5]).unwrap();
        let empty_slice = empty_tensor.slice(s![.., 1..3]);
        assert_eq!(empty_slice.shape().as_slice(), &[0, 2]);

        // Single element tensor
        let single = Tensor::from_vec(vec![42.0f32], [1]).unwrap();
        let single_slice = single.slice(s![..]);
        assert_eq!(single_slice.shape().as_slice(), &[1]);
        assert_eq!(single_slice.to_vec::<f32>().unwrap(), vec![42.0]);

        // Zero-width slice
        let tensor = Tensor::zeros::<f32>([5, 5]).unwrap();
        let zero_width = tensor.slice(s![2..2, ..]);
        assert_eq!(zero_width.shape().as_slice(), &[0, 5]);

        // Boundary slicing
        let boundary_tensor = Tensor::zeros::<f32>([3, 3]).unwrap();
        let at_boundary = boundary_tensor.slice(s![2.., 2..]);
        assert_eq!(at_boundary.shape().as_slice(), &[1, 1]);
    }

    #[test]
    fn test_slice_different_dtypes_internal() {
        // Test with i32
        let data_i32: Vec<i32> = (0..12).collect();
        let tensor_i32 = Tensor::from_vec(data_i32, [3, 4]).unwrap();
        let slice_i32 = tensor_i32.slice(s![1..3, 1..3]);
        assert_eq!(slice_i32.shape().as_slice(), &[2, 2]);

        // Test with u8
        let data_u8: Vec<u8> = (0..16).map(|i| i as u8).collect();
        let tensor_u8 = Tensor::from_vec(data_u8, [4, 4]).unwrap();
        let slice_u8 = tensor_u8.slice(s![1..3, 1..3]);
        assert_eq!(slice_u8.shape().as_slice(), &[2, 2]);

        // Test with f64
        let data_f64: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let tensor_f64 = Tensor::from_vec(data_f64, [3, 3]).unwrap();
        let slice_f64 = tensor_f64.slice(s![1.., 1..]);
        assert_eq!(slice_f64.shape().as_slice(), &[2, 2]);
    }

    #[test]
    fn test_slice_chaining_internal() {
        let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, [3, 4, 5]).unwrap();

        // Chain multiple slices
        let first_slice = tensor.slice(s![1.., 1.., 1..]);
        assert_eq!(first_slice.shape().as_slice(), &[2, 3, 4]);

        let second_slice = first_slice.slice(s![1.., 1.., 1..]);
        assert_eq!(second_slice.shape().as_slice(), &[1, 2, 3]);

        let third_slice = second_slice.slice(s![0, 1.., 1..]);
        assert_eq!(third_slice.shape().as_slice(), &[1, 2]);

        // Verify final result is accessible
        let final_flat = third_slice.to_flat_vec::<f32>().unwrap();
        assert_eq!(final_flat.len(), 2);
    }

    #[test]
    fn test_slice_large_tensor_internal() {
        // Test with larger tensor to check performance and correctness
        let size = 1000;
        let data: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, [size, size]).unwrap();

        // Large slice
        let large_slice = tensor.slice(s![100..900, 100..900]);
        assert_eq!(large_slice.shape().as_slice(), &[800, 800]);

        // Corner slice
        let corner = tensor.slice(s![990.., 990..]);
        assert_eq!(corner.shape().as_slice(), &[10, 10]);

        // Strided large slice (using supported 2D pattern)
        let strided_large = tensor.slice(s![..;10, 0..1000]);
        assert_eq!(strided_large.shape().as_slice(), &[100, 1000]);
    }

    #[test]
    fn test_slice_mixed_operations_internal() {
        let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, [4, 5, 6]).unwrap();

        // Mix of index, range, and full slice
        let mixed1 = tensor.slice(s![1, .., 2..5]);
        assert_eq!(mixed1.shape().as_slice(), &[5, 3]);

        let mixed2 = tensor.slice(s![1..3, 2, ..]);
        assert_eq!(mixed2.shape().as_slice(), &[2, 6]);

        let mixed3 = tensor.slice(s![.., 1..4, 3]);
        assert_eq!(mixed3.shape().as_slice(), &[4, 3]);

        // Verify all results are accessible
        assert!(mixed1.to_flat_vec::<f32>().is_ok());
        assert!(mixed2.to_flat_vec::<f32>().is_ok());
        assert!(mixed3.to_flat_vec::<f32>().is_ok());
    }

    #[test]
    fn test_slice_boundary_conditions_internal() {
        let tensor = Tensor::zeros::<f32>([10, 10]).unwrap();

        // Test all boundary combinations
        let top_left = tensor.slice(s![..1, ..1]);
        assert_eq!(top_left.shape().as_slice(), &[1, 1]);

        let top_right = tensor.slice(s![..1, 9..]);
        assert_eq!(top_right.shape().as_slice(), &[1, 1]);

        let bottom_left = tensor.slice(s![9.., ..1]);
        assert_eq!(bottom_left.shape().as_slice(), &[1, 1]);

        let bottom_right = tensor.slice(s![9.., 9..]);
        assert_eq!(bottom_right.shape().as_slice(), &[1, 1]);

        // Test middle strips
        let horizontal_strip = tensor.slice(s![4..6, ..]);
        assert_eq!(horizontal_strip.shape().as_slice(), &[2, 10]);

        let vertical_strip = tensor.slice(s![.., 4..6]);
        assert_eq!(vertical_strip.shape().as_slice(), &[10, 2]);
    }

    #[test]
    fn test_slice_stepped_patterns_internal() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, [4, 5]).unwrap();

        // Test basic stepped slicing with supported 2D patterns
        let stepped_rows = tensor.slice(s![..;2, 0..5]);
        assert_eq!(stepped_rows.shape().as_slice(), &[2, 5]);

        // Test with explicit ranges and steps (1D)
        let range_step = tensor.slice(s![0..4;2]);
        assert_eq!(range_step.shape().as_slice(), &[2, 5]);

        // Test step from start
        let step_from_start = tensor.slice(s![1..;2]);
        assert_eq!(step_from_start.shape().as_slice(), &[2, 5]); // indices 1, 3
    }

    #[test]
    fn test_slice_performance_patterns_internal() {
        let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, [100, 100]).unwrap();

        // Common performance-critical patterns

        // Row extraction
        let single_row = tensor.slice(s![50, ..]);
        assert_eq!(single_row.shape().as_slice(), &[100]);

        // Column extraction
        let single_col = tensor.slice(s![.., 50]);
        assert_eq!(single_col.shape().as_slice(), &[100]);

        // Block extraction
        let block = tensor.slice(s![25..75, 25..75]);
        assert_eq!(block.shape().as_slice(), &[50, 50]);

        // Diagonal-like pattern
        let diagonal_region = tensor.slice(s![10..90, 10..90]);
        assert_eq!(diagonal_region.shape().as_slice(), &[80, 80]);

        // Strided sampling (using supported pattern)
        let sampled = tensor.slice(s![..;5, 0..100]);
        assert_eq!(sampled.shape().as_slice(), &[20, 100]);
    }
}
