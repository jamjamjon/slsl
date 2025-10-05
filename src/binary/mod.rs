use crate::{Shape, MAX_DIM};
use anyhow::Result;

pub mod add;
pub mod div;
pub mod mul;
pub mod sub;

/// Compute the broadcast shape for two tensor shapes
///
/// This function implements NumPy-style broadcasting rules:
/// - Dimensions are aligned from the right
/// - A dimension of size 1 can be broadcast to any size
/// - Dimensions of the same size are compatible
/// - Missing dimensions are treated as size 1
pub(crate) fn compute_broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Shape> {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len1.max(len2);

    if max_len > MAX_DIM {
        anyhow::bail!(
            "Broadcast shape would exceed maximum dimensions: {} > {}",
            max_len,
            MAX_DIM
        );
    }

    // First pass: validate compatibility
    for i in 0..max_len {
        let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };

        if dim1 != 1 && dim2 != 1 && dim1 != dim2 {
            anyhow::bail!(
                "Cannot broadcast shapes {:?} and {:?}: dimension {} has incompatible sizes {} and {}",
                shape1, shape2, max_len - 1 - i, dim1, dim2
            );
        }
    }

    // Create result shape directly
    let mut result = Shape::empty().with_len(max_len);

    // Compute the result
    for i in 0..max_len {
        let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };

        let broadcast_dim = if dim1 == 1 {
            dim2
        } else {
            dim1 // dim2 == 1 or dim1 == dim2
        };

        result[i] = broadcast_dim;
    }

    // Reverse the result to get the correct order
    for i in 0..max_len / 2 {
        let temp = result[i];
        result[i] = result[max_len - 1 - i];
        result[max_len - 1 - i] = temp;
    }

    Ok(result)
}
