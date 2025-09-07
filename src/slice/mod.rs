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
        let offset_bytes =
            self.offset_bytes + resolved_idx * self.strides[0] * self.dtype.size_in_bytes();

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
        let resolved_start = self.resolve_index(start, 0);
        let resolved_end = end.map_or(self.shape[0], |e| self.resolve_index(e, 0));

        let new_size = resolved_end.saturating_sub(resolved_start);
        let offset_bytes =
            self.offset_bytes + resolved_start * self.strides[0] * self.dtype.size_in_bytes();

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
        let resolved_start1 = self.resolve_index(start1, 0);
        let resolved_end1 = end1.map_or(self.shape[0], |e| self.resolve_index(e, 0));
        let resolved_start2 = self.resolve_index(start2, 1);
        let resolved_end2 = end2.map_or(self.shape[1], |e| self.resolve_index(e, 1));

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
                e as usize
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

    #[inline]
    fn compute_range_slice(
        dim_size: usize,
        stride: usize,
        start: isize,
        end: Option<isize>,
        step: isize,
    ) -> (usize, usize, usize) {
        let resolved_start = Self::resolve_index_static(start, dim_size);
        let resolved_end = if let Some(end_val) = end {
            Self::resolve_index_static(end_val, dim_size).min(dim_size)
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
