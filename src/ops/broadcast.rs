use anyhow::Result;

use crate::{Shape, StorageTrait, TensorBase, TensorView};

impl<S: StorageTrait> TensorBase<S> {
    /// Broadcast tensor to new shape
    pub fn broadcast_to<D: Into<Shape>>(&self, target_shape: D) -> Result<TensorView<'_>> {
        let new_shape = target_shape.into();

        // Check if broadcasting is possible
        if new_shape.len() < self.rank() {
            anyhow::bail!(
                "Cannot broadcast shape {:?} to smaller shape {:?}",
                self.shape.as_slice(),
                new_shape.as_slice()
            );
        }

        // Fast path: shapes identical → return a simple view
        if new_shape.as_slice() == self.shape.as_slice() {
            return Ok(TensorView {
                storage: self.storage.as_storage(),
                ptr: self.ptr,
                dtype: self.dtype,
                shape: new_shape,
                strides: self.strides, // preserve existing strides
                offset_bytes: self.offset_bytes,
            });
        }

        // Calculate new strides for broadcasting using Shape (no heap alloc)
        let new_len = new_shape.len();
        let rank = self.rank();
        let offset = new_len - rank;

        let mut new_strides = Shape::full(0, new_len);

        // Copy existing strides with broadcasting rules
        for i in 0..rank {
            let old_dim = self.shape[i];
            let new_dim = new_shape[i + offset];

            new_strides[i + offset] = if old_dim == 1 {
                // Broadcasting: stride becomes 0
                0
            } else if old_dim == new_dim {
                // Same size: keep original stride
                if i < self.strides.len() {
                    self.strides[i]
                } else {
                    0
                }
            } else {
                anyhow::bail!(
                    "Cannot broadcast dimension {} from size {} to size {}",
                    i,
                    old_dim,
                    new_dim
                );
            };
        }

        Ok(TensorView {
            storage: self.storage.as_storage(),
            ptr: self.ptr,
            dtype: self.dtype,
            shape: new_shape,
            strides: new_strides,
            offset_bytes: self.offset_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn broadcast_same_shape_fast_path() {
        let t: Tensor = Tensor::full([2, 3], 1.0f32).unwrap();
        let view = t.broadcast_to([2, 3]).unwrap();
        // shape unchanged
        assert_eq!(view.dims(), &[2, 3]);
        // strides should be preserved
        assert_eq!(view.strides.as_slice(), t.strides.as_slice());
        // storage/pointer should match (view over same buffer)
        assert_eq!(view.ptr, t.ptr);
    }

    #[test]
    fn broadcast_add_leading_dim() {
        let t: Tensor = Tensor::full([2, 3], 1.0f32).unwrap();
        let view = t.broadcast_to([1, 2, 3]).unwrap();
        // shape updated
        assert_eq!(view.dims(), &[1, 2, 3]);
        // leading new dim should have stride 0
        assert_eq!(view.strides[0], 0);
        // tail strides should match original
        assert_eq!(view.strides[1], t.strides[0]);
        assert_eq!(view.strides[2], t.strides[1]);

        let view = view.broadcast_to([6, 2, 3]).unwrap();
        assert_eq!(view.dims(), &[6, 2, 3]);
    }

    #[test]
    fn broadcast_expand_singleton_dim() {
        let t: Tensor = Tensor::full([1, 3], 2.0f32).unwrap();
        let view = t.broadcast_to([4, 3]).unwrap();
        assert_eq!(view.dims(), &[4, 3]);
        // expanded singleton dimension must have stride 0
        assert_eq!(view.strides[0], 0);
        // second dim stride equals original second dim stride
        assert_eq!(view.strides[1], t.strides[1]);
    }

    #[test]
    fn broadcast_incompatible_dimension() {
        let t: Tensor = Tensor::full([2, 3], 3.0f32).unwrap();
        // Mismatch: trying to expand 2 to 4 is invalid
        let err = t.broadcast_to([4, 3]).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("Cannot broadcast dimension") || msg.contains("Cannot broadcast shape")
        );
    }
}
