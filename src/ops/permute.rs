use anyhow::Result;

use crate::{Dims, Shape, StorageTrait, TensorBase, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    /// Permute tensor dimensions according to the given order
    pub fn permute<D: Dims>(self, dims: D) -> Result<TensorBase<S>> {
        let rank = self.rank();
        let perm_dims = dims.to_dims(rank)?;

        // Validate permutation dimensions
        if perm_dims.len() != rank {
            anyhow::bail!(
                "Permutation dimensions length {} must match tensor rank {}",
                perm_dims.len(),
                rank
            );
        }

        // Check that all dimensions are unique and valid
        // Use UninitVec for better performance - avoid initializing to false
        let mut used = UninitVec::<bool>::new(rank).full(false);
        for &dim in perm_dims.as_slice() {
            if dim >= rank {
                anyhow::bail!(
                    "Dimension {} is out of bounds for tensor with {} dimensions",
                    dim,
                    rank
                );
            }
            if used[dim] {
                anyhow::bail!("Dimension {} appears multiple times in permutation", dim);
            }
            used[dim] = true;
        }

        // Create new shape and strides based on permutation
        let mut new_shape = Shape::empty().with_len(rank);
        let mut new_strides = Shape::empty().with_len(rank);
        for (i, &dim) in perm_dims.as_slice().iter().enumerate() {
            new_shape[i] = self.shape[dim];
            new_strides[i] = self.strides[dim];
        }

        Ok(TensorBase {
            storage: self.storage,
            ptr: self.ptr,
            dtype: self.dtype,
            shape: new_shape,
            strides: new_strides,
            offset_bytes: self.offset_bytes,
        })
    }

    /// Flip tensor dimensions, reversing the order of dimensions
    /// For example: shape [1, 2, 3, 4] becomes [4, 3, 2, 1]
    pub fn flip_dims(self) -> Result<TensorBase<S>> {
        let rank = self.rank();
        if rank == 0 {
            // Scalar tensor, return as-is
            return Ok(TensorBase {
                storage: self.storage,
                ptr: self.ptr,
                dtype: self.dtype,
                shape: self.shape,
                strides: self.strides,
                offset_bytes: self.offset_bytes,
            });
        }

        // Create reversed dimension indices: [n-1, n-2, ..., 1, 0]
        // Use UninitVec for better performance - avoid repeated push operations
        let flipped_dims = UninitVec::<usize>::new(rank).init_with(|dims| {
            for (i, dim) in dims.iter_mut().enumerate() {
                *dim = rank - 1 - i;
            }
        });

        // Use existing permute logic for efficiency
        self.permute(flipped_dims)
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn permute_basic() {
        let t = Tensor::from_vec(vec![1u8, 2, 3, 4, 3, 4, 5, 6, 1, 4, 6, 7], [1, 3, 4]).unwrap();
        let v = t.permute([1, 0, 2]).unwrap();
        assert_eq!(v.dims(), [3, 1, 4]);
        assert_eq!(v.strides.as_slice().len(), 3);
    }

    #[test]
    fn permute_invalid_dims() {
        let t = Tensor::from_vec(vec![1u8, 2, 3, 4], [2, 2]).unwrap();
        assert!(t.permute([0, 0]).is_err());
    }

    #[test]
    fn flip_dims_basic() {
        // Test 4D tensor: [1, 2, 3, 4] -> [4, 3, 2, 1]
        let t = Tensor::from_vec((0..24).collect::<Vec<u8>>(), [1, 2, 3, 4]).unwrap();
        let flipped = t.flip_dims().unwrap();
        assert_eq!(flipped.dims(), [4, 3, 2, 1]);
    }

    #[test]
    fn flip_dims_2d() {
        // Test 2D tensor: [3, 4] -> [4, 3]
        let t = Tensor::from_vec(vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 4]).unwrap();
        let flipped = t.flip_dims().unwrap();
        assert_eq!(flipped.dims(), [4, 3]);
    }

    #[test]
    fn flip_dims_1d() {
        // Test 1D tensor: [5] -> [5] (should remain the same)
        let t = Tensor::from_vec(vec![1u8, 2, 3, 4, 5], [5]).unwrap();
        let flipped = t.flip_dims().unwrap();
        assert_eq!(flipped.dims(), [5]);
    }

    #[test]
    fn flip_dims_scalar() {
        // Test scalar tensor: [] -> [] (should remain the same)
        let t = Tensor::from_vec(vec![42u8], []).unwrap();
        let flipped = t.flip_dims().unwrap();
        assert_eq!(flipped.dims(), &[] as &[usize]);
    }
}
