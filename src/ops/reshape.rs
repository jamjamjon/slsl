use anyhow::Result;

use crate::{Shape, StorageTrait, TensorBase};

impl<S: StorageTrait> TensorBase<S> {
    pub fn reshape<D: Into<Shape>>(mut self, new_shape: D) -> Result<TensorBase<S>> {
        if !self.is_contiguous() {
            anyhow::bail!("Cannot reshape non-contiguous tensor");
        }
        let new_shape = new_shape.into();
        let current_numel = self.numel();
        let new_numel = new_shape.numel();
        if current_numel != new_numel {
            anyhow::bail!(
                "Cannot reshape tensor of size {} to size {}",
                current_numel,
                new_numel
            );
        }
        let new_strides = Self::compute_contiguous_strides(&new_shape);
        self.shape = new_shape;
        self.strides = new_strides;

        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn reshape_contiguous_same_numel() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4, 3, 4, 5, 6, 1, 4, 6, 7], [2, 2, 3]).unwrap();
        let v = t.reshape([4, 3, 1]).unwrap();
        assert_eq!(v.dims(), [4, 3, 1]);
    }

    #[test]
    fn reshape_tensorview() {
        let t = Tensor::rand(1.0, 10.0, [2, 2, 3]).unwrap();
        let view = t.view();
        let v = view.reshape([4, 3, 1]).unwrap();
        assert_eq!(v.dims(), [4, 3, 1]);
    }

    #[test]
    fn reshape_invalid_numel() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4], [2, 2]).unwrap();
        assert!(t.reshape([3]).is_err());
    }
}
