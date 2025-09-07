use anyhow::{bail, Result};

use crate::{Dim, Shape, StorageTrait, TensorBase, TensorView};

impl<S: StorageTrait> TensorBase<S> {
    /// Split the tensor along `dim` at `index`, returning (left, right).
    ///
    /// Rules:
    /// - `index` must be > 0 and < size_of(dim). If `index` == 0 or `index` == size, this returns an error.
    /// - Does not allocate; returns two views.
    pub fn split_at<D: Dim>(
        &self,
        dim: D,
        index: usize,
    ) -> Result<(TensorView<'_>, TensorView<'_>)> {
        let dim_idx = dim.to_dim(self.rank())?;
        let dim_size = self.shape[dim_idx];

        // Disallow producing empty views by forbidding index at boundaries
        if index == 0 || index >= dim_size {
            bail!(
                "Index must be > 0 and < {} for dimension {} (got {})",
                dim_size,
                dim_idx,
                index
            );
        }

        // Left view: same offset, adjust shape along split dim
        let mut left_shape: Shape = self.shape;
        left_shape[dim_idx] = index;
        let left_strides = self.strides;
        let left = unsafe {
            TensorView::from_raw_parts(
                self.storage.as_storage(),
                self.ptr,
                left_shape,
                left_strides,
                self.offset_bytes,
                self.dtype,
            )
        };

        // Right view: advance offset by index along dim, adjust shape along split dim
        let mut right_shape: Shape = self.shape;
        right_shape[dim_idx] = dim_size - index;
        let right_strides = self.strides;
        let step_bytes = self.strides[dim_idx] * self.dtype.size_in_bytes();
        let right_offset = self.offset_bytes + index * step_bytes;
        let right = unsafe {
            TensorView::from_raw_parts(
                self.storage.as_storage(),
                self.ptr,
                right_shape,
                right_strides,
                right_offset,
                self.dtype,
            )
        };

        Ok((left, right))
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn split_at_basic_1d() {
        let t = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], [4]).unwrap();
        let (l, r) = t.split_at(0, 2).unwrap();
        assert_eq!(l.shape().as_slice(), &[2]);
        assert_eq!(r.shape().as_slice(), &[2]);
        assert_eq!(l.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
        assert_eq!(r.as_slice::<f64>().unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn split_at_forbid_boundaries() {
        let t = Tensor::from_vec(vec![1i32, 2, 3], [3]).unwrap();
        assert!(t.split_at(0, 0).is_err());
        assert!(t.split_at(0, 3).is_err());
    }

    #[test]
    fn split_at_2d_dim1() {
        let t = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let (l, r) = t.split_at(1, 2).unwrap();
        assert_eq!(l.shape().as_slice(), &[2, 2]);
        assert_eq!(r.shape().as_slice(), &[2, 1]);
        let l_c = l.to_contiguous().unwrap();
        let r_c = r.to_contiguous().unwrap();
        assert_eq!(l_c.as_slice::<f64>().unwrap(), &[1.0, 2.0, 4.0, 5.0]);
        assert_eq!(r_c.as_slice::<f64>().unwrap(), &[3.0, 6.0]);
    }
}
