//! Shape and stride definitions for tensor dimensions.
//!
//! This module provides type aliases for working with tensor shapes and memory strides.
//! Both [`Shape`] and [`Stride`] are built on top of [`ArrayN`] with a maximum
//! capacity of [`MAX_DIM`] dimensions.

use crate::{ArrayN, MAX_DIM};

/// Type alias for representing tensor shapes.
///
/// A [`Shape`] stores the size of each dimension in a tensor, using a fixed-capacity
/// array that can hold up to [`MAX_DIM`] dimensions. This provides efficient
/// stack-allocated storage for dimension information.
///
/// # Examples
///
/// ```rust
/// use slsl::Shape;
///
/// // Create a 2D shape (3x4 matrix)
/// let shape = Shape::from_slice(&[3, 4]);
/// assert_eq!(shape.len(), 2);
/// assert_eq!(shape[0], 3);
/// assert_eq!(shape[1], 4);
/// assert_eq!(shape.numel(), 12); // 3 * 4 = 12 elements
/// ```
pub type Shape = ArrayN<usize, { MAX_DIM }>;

/// Type alias for representing memory strides.
///
/// A [`Stride`] stores the byte offset between consecutive elements along
/// each dimension of a tensor. This is used for efficient memory layout
/// and indexing calculations.
///
/// # Examples
///
/// ```rust
/// use slsl::Stride;
///
/// // Create strides for a 2D array stored in row-major order
/// let stride = Stride::from_slice(&[4, 1]); // 4 bytes per row, 1 byte per element
/// assert_eq!(stride.len(), 2);
/// assert_eq!(stride[0], 4); // Row stride
/// assert_eq!(stride[1], 1); // Column stride
/// ```
pub type Stride = ArrayN<usize, { MAX_DIM }>;

impl Shape {
    /// Returns the total number of elements in the tensor.
    ///
    /// This method calculates the product of all dimensions in the shape,
    /// which gives the total number of elements that would be stored in
    /// a tensor with this shape.
    ///
    /// # Returns
    ///
    /// The total number of elements as a `usize`. Returns 1 for empty shapes
    /// (0-dimensional tensors/scalars).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Shape;
    ///
    /// // 2D shape: 3x4 matrix
    /// let shape = Shape::from_slice(&[3, 4]);
    /// assert_eq!(shape.numel(), 12); // 3 * 4
    ///
    /// // 1D shape: vector of 5 elements
    /// let shape_1d = Shape::from_slice(&[5]);
    /// assert_eq!(shape_1d.numel(), 5);
    ///
    /// // Empty shape: scalar (0-dimensional)
    /// let shape_scalar = Shape::empty();
    /// assert_eq!(shape_scalar.numel(), 1);
    /// ```
    #[inline(always)]
    pub fn numel(&self) -> usize {
        self.as_slice().iter().product()
    }
}

// #[repr(transparent)]
// #[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
// pub struct Shape(pub(crate) ArrayN<usize, { MAX_DIM }>);

// impl std::ops::Deref for Shape {
//     type Target = ArrayN<usize, { MAX_DIM }>;
//     fn deref(&self) -> &Self::Target {
//         &self.0
//     }
// }

// impl std::ops::DerefMut for Shape {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.0
//     }
// }
