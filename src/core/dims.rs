//! Tensor dimension indexing utilities.
//!
//! This module provides traits for converting various types to tensor dimension indices,
//! with support for negative indexing and multi-dimensional specifications.

use anyhow::Result;

use crate::Shape;

/// Trait for types that can be converted to a single dimension index.
///
/// This trait is used to convert various numeric types to tensor dimension indices,
/// with support for negative indexing (counting from the end).
pub trait Dim {
    /// Converts this value to a dimension index for a tensor with the given rank.
    ///
    /// # Parameters
    ///
    /// * `rank` - The number of dimensions in the tensor
    ///
    /// # Returns
    ///
    /// The resolved dimension index as a `usize`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor is 0-dimensional (scalar)
    /// - The index is out of bounds for the tensor rank
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::{Dim, Shape};
    ///
    /// // Positive indexing
    /// assert_eq!(1usize.to_dim(3)?, 1);
    ///
    /// // Negative indexing (counting from the end)
    /// assert_eq!((-1isize).to_dim(3)?, 2);
    /// assert_eq!((-2isize).to_dim(3)?, 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn to_dim(&self, rank: usize) -> Result<usize>;

    /// Converts this dimension to a single-element shape.
    ///
    /// This is a convenience method that wraps the dimension in a `Shape`.
    ///
    /// # Parameters
    ///
    /// * `rank` - The number of dimensions in the tensor
    ///
    /// # Returns
    ///
    /// A `Shape` containing the single dimension index.
    fn to_dims(&self, rank: usize) -> Result<Shape> {
        let dim = self.to_dim(rank)?;
        Ok(Shape::from_slice(&[dim]))
    }
}

/// Trait for types that can be converted to multiple dimension indices.
///
/// This trait allows various collection types (arrays, tuples, vectors) to be
/// used as dimension specifications for tensor operations.
pub trait Dims: Sized {
    /// Converts this value to a `Shape` containing multiple dimension indices.
    ///
    /// # Parameters
    ///
    /// * `rank` - The number of dimensions in the tensor
    ///
    /// # Returns
    ///
    /// A `Shape` containing the resolved dimension indices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::{Dims, Shape};
    ///
    /// // From tuple
    /// let dims = (0, 2).to_dims(3)?;
    /// assert_eq!(dims.len(), 2);
    ///
    /// // From array
    /// let dims = [0, 1].to_dims(3)?;
    /// assert_eq!(dims.len(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn to_dims(self, rank: usize) -> Result<Shape>;
}

impl Dim for isize {
    /// Converts an `isize` to a dimension index with support for negative indexing.
    ///
    /// Negative values count backward from the end of the tensor dimensions.
    /// For a tensor with rank N:
    /// - `-1` refers to dimension `N-1` (last dimension)
    /// - `-2` refers to dimension `N-2`, and so on
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::{Dim, Shape};
    ///
    /// // For a 3D tensor (rank = 3)
    /// assert_eq!(0isize.to_dim(3)?, 0);   // First dimension
    /// assert_eq!((-1isize).to_dim(3)?, 2); // Last dimension
    /// assert_eq!((-3isize).to_dim(3)?, 0); // First dimension (via negative indexing)
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn to_dim(&self, rank: usize) -> Result<usize> {
        if rank == 0 {
            anyhow::bail!("Cannot index into a 0-dimensional tensor");
        }
        let rank_isize = rank as isize;
        let idx = if *self < 0 { rank_isize + *self } else { *self };
        if idx < 0 || idx >= rank_isize {
            anyhow::bail!(
                "Dimension index {} is out of bounds for tensor with {} dimensions",
                self,
                rank
            );
        }
        Ok(idx as usize)
    }
}

impl Dim for i32 {
    /// Converts an `i32` to a dimension index by delegating to the `isize` implementation.
    fn to_dim(&self, rank: usize) -> Result<usize> {
        (*self as isize).to_dim(rank)
    }
}

impl Dim for usize {
    /// Converts a `usize` to a dimension index with bounds checking.
    ///
    /// Only positive indexing is supported. The index must be less than the tensor rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::{Dim, Shape};
    ///
    /// assert_eq!(0usize.to_dim(3)?, 0);
    /// assert_eq!(2usize.to_dim(3)?, 2);
    ///
    /// // This would error: 3usize.to_dim(3) - index out of bounds
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn to_dim(&self, rank: usize) -> Result<usize> {
        if rank == 0 {
            anyhow::bail!("Cannot index into a 0-dimensional tensor");
        }
        if *self >= rank {
            anyhow::bail!(
                "Dimension index {} is out of bounds for tensor with {} dimensions",
                self,
                rank
            );
        }
        Ok(*self)
    }
}

impl Dim for u32 {
    /// Converts a `u32` to a dimension index by delegating to the `usize` implementation.
    fn to_dim(&self, rank: usize) -> Result<usize> {
        (*self as usize).to_dim(rank)
    }
}

impl Dims for () {
    /// Returns an empty `Shape` for the unit type.
    ///
    /// This is used when no specific dimensions are specified.
    fn to_dims(self, _rank: usize) -> Result<Shape> {
        Ok(Shape::empty())
    }
}

impl<D: Dim> Dims for D {
    /// Converts a single `Dim` to a `Shape` containing one dimension.
    ///
    /// This allows any type implementing `Dim` to be used where `Dims` is expected.
    fn to_dims(self, rank: usize) -> Result<Shape> {
        let dim = self.to_dim(rank)?;
        Ok(Shape::from_slice(&[dim]))
    }
}

impl<D: Dim, const N: usize> Dims for [D; N] {
    /// Converts a fixed-size array of dimensions to a `Shape`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::{Dims, Shape};
    ///
    /// let dims = [0usize, 2usize].to_dims(3)?;
    /// assert_eq!(dims.len(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn to_dims(self, rank: usize) -> Result<Shape> {
        let mut dims = Vec::with_capacity(N);
        for dim in self {
            dims.push(dim.to_dim(rank)?);
        }
        Ok(Shape::from_slice(&dims))
    }
}

impl<D: Dim> Dims for Vec<D> {
    /// Converts a vector of dimensions to a `Shape`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::{Dims, Shape};
    ///
    /// let dims = vec![0usize, 1usize, 2usize].to_dims(3)?;
    /// assert_eq!(dims.len(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn to_dims(self, rank: usize) -> Result<Shape> {
        let mut dims = Vec::with_capacity(self.len());
        for dim in self {
            dims.push(dim.to_dim(rank)?);
        }
        Ok(Shape::from_slice(&dims))
    }
}

impl<D: Dim> Dims for &[D] {
    /// Converts a slice of dimensions to a `Shape`.
    ///
    /// This allows using slices for dynamic dimension specification.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::{Dims, Shape};
    ///
    /// let dims_array = [0usize, 2usize];
    /// let dims = dims_array.as_slice().to_dims(3)?;
    /// assert_eq!(dims.len(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn to_dims(self, rank: usize) -> Result<Shape> {
        let mut dims = Vec::with_capacity(self.len());
        for dim in self {
            dims.push(dim.to_dim(rank)?);
        }
        Ok(Shape::from_slice(&dims))
    }
}

impl<D: Dim + Copy, const N: usize> Dims for &[D; N] {
    /// Converts a reference to a fixed-size array of dimensions to a `Shape`.
    ///
    /// This provides an alternative way to use arrays without moving them.
    fn to_dims(self, rank: usize) -> Result<Shape> {
        let mut dims = Vec::with_capacity(N);
        for &dim in self {
            dims.push(dim.to_dim(rank)?);
        }
        Ok(Shape::from_slice(&dims))
    }
}

impl<D1: Dim, D2: Dim> Dims for (D1, D2) {
    /// Converts a 2-tuple of dimensions to a `Shape`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::{Dims, Shape};
    ///
    /// let dims = (0usize, 2usize).to_dims(3)?;
    /// assert_eq!(dims.len(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn to_dims(self, rank: usize) -> Result<Shape> {
        Ok(Shape::from_slice(&[
            self.0.to_dim(rank)?,
            self.1.to_dim(rank)?,
        ]))
    }
}

impl<D1: Dim, D2: Dim, D3: Dim> Dims for (D1, D2, D3) {
    /// Converts a 3-tuple of dimensions to a `Shape`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::{Dims, Shape};
    ///
    /// let dims = (0usize, 1usize, 2usize).to_dims(3)?;
    /// assert_eq!(dims.len(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn to_dims(self, rank: usize) -> Result<Shape> {
        Ok(Shape::from_slice(&[
            self.0.to_dim(rank)?,
            self.1.to_dim(rank)?,
            self.2.to_dim(rank)?,
        ]))
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim> Dims for (D1, D2, D3, D4) {
    /// Converts a 4-tuple of dimensions to a `Shape`.
    ///
    /// Useful for specifying 4D tensor operations like batch operations on 3D data.
    fn to_dims(self, rank: usize) -> Result<Shape> {
        Ok(Shape::from_slice(&[
            self.0.to_dim(rank)?,
            self.1.to_dim(rank)?,
            self.2.to_dim(rank)?,
            self.3.to_dim(rank)?,
        ]))
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim> Dims for (D1, D2, D3, D4, D5) {
    /// Converts a 5-tuple of dimensions to a `Shape`.
    ///
    /// Supports up to 5-dimensional tensor operations.
    fn to_dims(self, rank: usize) -> Result<Shape> {
        Ok(Shape::from_slice(&[
            self.0.to_dim(rank)?,
            self.1.to_dim(rank)?,
            self.2.to_dim(rank)?,
            self.3.to_dim(rank)?,
            self.4.to_dim(rank)?,
        ]))
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, D6: Dim> Dims for (D1, D2, D3, D4, D5, D6) {
    /// Converts a 6-tuple of dimensions to a `Shape`.
    ///
    /// Supports up to 6-dimensional tensor operations.
    fn to_dims(self, rank: usize) -> Result<Shape> {
        Ok(Shape::from_slice(&[
            self.0.to_dim(rank)?,
            self.1.to_dim(rank)?,
            self.2.to_dim(rank)?,
            self.3.to_dim(rank)?,
            self.4.to_dim(rank)?,
            self.5.to_dim(rank)?,
        ]))
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, D6: Dim, D7: Dim> Dims
    for (D1, D2, D3, D4, D5, D6, D7)
{
    /// Converts a 7-tuple of dimensions to a `Shape`.
    ///
    /// Supports up to 7-dimensional tensor operations.
    fn to_dims(self, rank: usize) -> Result<Shape> {
        Ok(Shape::from_slice(&[
            self.0.to_dim(rank)?,
            self.1.to_dim(rank)?,
            self.2.to_dim(rank)?,
            self.3.to_dim(rank)?,
            self.4.to_dim(rank)?,
            self.5.to_dim(rank)?,
            self.6.to_dim(rank)?,
        ]))
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, D6: Dim, D7: Dim, D8: Dim> Dims
    for (D1, D2, D3, D4, D5, D6, D7, D8)
{
    /// Converts an 8-tuple of dimensions to a `Shape`.
    ///
    /// Supports up to 8-dimensional tensor operations (maximum supported).
    fn to_dims(self, rank: usize) -> Result<Shape> {
        Ok(Shape::from_slice(&[
            self.0.to_dim(rank)?,
            self.1.to_dim(rank)?,
            self.2.to_dim(rank)?,
            self.3.to_dim(rank)?,
            self.4.to_dim(rank)?,
            self.5.to_dim(rank)?,
            self.6.to_dim(rank)?,
            self.7.to_dim(rank)?,
        ]))
    }
}
