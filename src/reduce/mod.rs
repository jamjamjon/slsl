use crate::{Shape, Stride};

pub fn reduce_shape_stride(shape: Shape, dims: &[usize], keepdim: bool) -> (Shape, Stride) {
    let ndim = shape.len();

    // Calculate new shape length first
    let new_len = if keepdim { ndim } else { ndim - dims.len() };

    // Create Shape directly without Vec allocation or idx variable
    let mut new_shape = Shape::empty().with_len(new_len);

    // Use fold to build new_shape in one pass
    let _ = shape.iter().enumerate().fold(0, |out_idx, (i, &size)| {
        if dims.contains(&i) {
            if keepdim {
                new_shape[out_idx] = 1;
                out_idx + 1
            } else {
                out_idx
            }
        } else {
            new_shape[out_idx] = size;
            out_idx + 1
        }
    });

    // Create stride directly
    let mut new_stride = Stride::empty().with_len(new_len);
    if new_len > 0 {
        new_stride[new_len - 1] = 1;
        for i in (0..new_len - 1).rev() {
            new_stride[i] = new_stride[i + 1] * new_shape[i + 1];
        }
    }

    (new_shape, new_stride)
}

mod argmax;
mod argmin;
mod argmin_argmax;
mod max;
mod max_argmax;
mod mean;
mod min;
mod min_argmin;
mod min_max;
mod min_max_argmin_argmax;
mod sum;
