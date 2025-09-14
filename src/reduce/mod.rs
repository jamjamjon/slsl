use crate::{Shape, StorageTrait, Stride, TensorBase};

impl<S: StorageTrait> TensorBase<S> {
    /// Check if we can reduce over dimensions efficiently using contiguous memory access
    pub fn can_reduce_over_last_dims(&self, dim_indices: &[usize]) -> bool {
        if dim_indices.is_empty() {
            return false;
        }
        let shape = self.shape();

        // For dimension-agnostic optimization, we can use contiguous path for:
        // 1. Single dimension reductions (any dimension)
        // 2. Multiple consecutive dimensions ending at the last dimension
        // 3. Any reduction that can benefit from SIMD optimization
        if dim_indices.len() == 1 {
            // Single dimension reduction - always use contiguous path for better performance
            let reduce_size = shape[dim_indices[0]];
            return reduce_size >= 4; // SIMD threshold
        }

        // For multiple dimensions, check if they form a consecutive block
        let mut sorted_dims = dim_indices.to_vec();
        sorted_dims.sort_unstable();

        // Check if all dimensions are consecutive
        let mut consecutive = true;
        for i in 1..sorted_dims.len() {
            if sorted_dims[i] != sorted_dims[i - 1] + 1 {
                consecutive = false;
                break;
            }
        }

        if consecutive {
            // For consecutive dimensions, prefer contiguous path for better cache locality
            let reduce_size: usize = sorted_dims.iter().map(|&dim| shape[dim]).product();
            return reduce_size >= 4;
        }

        // Even non-consecutive dimensions can benefit from contiguous optimization
        // if the total reduction size is large enough
        let total_reduce_size: usize = dim_indices.iter().map(|&dim| shape[dim]).product();
        total_reduce_size >= 16 // Higher threshold for non-consecutive dims
    }
}

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
