use anyhow::Result;

use crate::{StorageTrait, TensorBase, TensorElement};

impl<S: StorageTrait> TensorBase<S> {
    /// Extract scalar value from a 0-dimensional tensor
    ///
    /// # Performance
    /// This method is optimized for zero-cost scalar extraction with compile-time checks.
    ///
    /// # Errors
    /// Returns error if tensor is not 0-dimensional or dtype mismatch
    #[inline(always)]
    pub fn to_scalar<T: TensorElement + Copy>(&self) -> Result<T> {
        // Validate tensor is scalar (0-dimensional)
        if !self.dims().is_empty() {
            return Err(anyhow::anyhow!(
                "Cannot convert {}-dimensional tensor to scalar",
                self.dims().len()
            ));
        }

        // Validate dtype compatibility
        debug_assert_eq!(
            self.dtype(),
            T::DTYPE,
            "DType mismatch: tensor has {:?}, requested {:?}",
            self.dtype(),
            T::DTYPE
        );

        // Direct memory access for maximum performance
        unsafe {
            let ptr = self.as_ptr().add(self.offset_bytes) as *const T;
            Ok(*ptr)
        }
    }

    /// Convert 1D tensor to `Vec<T>`
    ///
    /// # Performance
    /// - Contiguous tensors: Direct memory copy (memcpy-like performance)
    /// - Non-contiguous tensors: Optimized iteration
    ///
    /// # Errors
    /// Returns error if tensor is not 1D or dtype mismatch
    pub fn to_vec<T: TensorElement + Copy>(&self) -> Result<Vec<T>> {
        // Validate tensor is 1D
        if self.dims().len() != 1 {
            return Err(anyhow::anyhow!(
                "Cannot convert {}-dimensional tensor to 1D Vec",
                self.dims().len()
            ));
        }

        // Validate dtype compatibility
        debug_assert_eq!(
            self.dtype(),
            T::DTYPE,
            "DType mismatch: tensor has {:?}, requested {:?}",
            self.dtype(),
            T::DTYPE
        );

        let len = self.shape()[0];
        let mut result = Vec::with_capacity(len);

        // Fast path: contiguous memory
        if self.is_contiguous() {
            unsafe {
                let src_ptr = self.as_ptr().add(self.offset_bytes) as *const T;
                result.set_len(len);
                std::ptr::copy_nonoverlapping(src_ptr, result.as_mut_ptr(), len);
            }
        } else {
            // Slow path: non-contiguous memory
            for i in 0..len {
                result.push(self.at::<T>([i]));
            }
        }

        Ok(result)
    }

    /// Convert tensor to flat `Vec<T>` (any dimensionality)
    ///
    /// # Performance
    /// - Contiguous tensors: Direct memory copy
    /// - Non-contiguous tensors: Optimized recursive iteration
    ///
    /// # Errors
    /// Returns error if dtype mismatch
    pub fn to_flat_vec<T: TensorElement + Copy>(&self) -> Result<Vec<T>> {
        // Validate dtype compatibility
        debug_assert_eq!(
            self.dtype(),
            T::DTYPE,
            "DType mismatch: tensor has {:?}, requested {:?}",
            self.dtype(),
            T::DTYPE
        );

        let numel = self.numel();
        let mut result = Vec::with_capacity(numel);

        // Fast path: contiguous memory
        if self.is_contiguous() {
            unsafe {
                let src_ptr = self.as_ptr().add(self.offset_bytes) as *const T;
                result.set_len(numel);
                std::ptr::copy_nonoverlapping(src_ptr, result.as_mut_ptr(), numel);
            }
        } else {
            // Slow path: iterate through all elements
            match self.dims().len() {
                0 => {
                    // Scalar tensor
                    result.push(self.to_scalar::<T>()?);
                }
                1 => {
                    // 1D tensor: direct element access
                    for i in 0..self.shape()[0] {
                        result.push(self.at::<T>([i]));
                    }
                }
                2 => {
                    // 2D tensor: row-major iteration
                    for i in 0..self.shape()[0] {
                        for j in 0..self.shape()[1] {
                            result.push(self.at::<T>([i, j]));
                        }
                    }
                }
                3 => {
                    // 3D tensor: optimized nested iteration
                    for i in 0..self.shape()[0] {
                        for j in 0..self.shape()[1] {
                            for k in 0..self.shape()[2] {
                                result.push(self.at::<T>([i, j, k]));
                            }
                        }
                    }
                }
                _ => {
                    // General case: use recursive indexing
                    self.to_vec_recursive::<T>(&mut result, &mut vec![0; self.dims().len()], 0)?;
                }
            }
        }

        Ok(result)
    }

    /// Convert 2D tensor to `Vec<Vec<T>>`
    pub fn to_vec2<T: TensorElement + Copy>(&self) -> Result<Vec<Vec<T>>> {
        if self.dims().len() != 2 {
            return Err(anyhow::anyhow!(
                "Cannot convert {}-dimensional tensor to 2D Vec",
                self.dims().len()
            ));
        }

        debug_assert_eq!(self.dtype(), T::DTYPE);

        let [rows, cols] = [self.shape()[0], self.shape()[1]];
        let mut result = Vec::with_capacity(rows);

        for i in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for j in 0..cols {
                row.push(self.at::<T>([i, j]));
            }
            result.push(row);
        }

        Ok(result)
    }

    /// Convert 3D tensor to `Vec<Vec<Vec<T>>>`
    pub fn to_vec3<T: TensorElement + Copy>(&self) -> Result<Vec<Vec<Vec<T>>>> {
        if self.dims().len() != 3 {
            return Err(anyhow::anyhow!(
                "Cannot convert {}-dimensional tensor to 3D Vec",
                self.dims().len()
            ));
        }

        debug_assert_eq!(self.dtype(), T::DTYPE);

        let [dim0, dim1, dim2] = [self.shape()[0], self.shape()[1], self.shape()[2]];
        let mut result = Vec::with_capacity(dim0);

        for i in 0..dim0 {
            let mut plane = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let mut row = Vec::with_capacity(dim2);
                for k in 0..dim2 {
                    row.push(self.at::<T>([i, j, k]));
                }
                plane.push(row);
            }
            result.push(plane);
        }

        Ok(result)
    }

    /// Helper method for recursive tensor traversal
    fn to_vec_recursive<T: TensorElement + Copy>(
        &self,
        result: &mut Vec<T>,
        indices: &mut Vec<usize>,
        dim: usize,
    ) -> Result<()> {
        if dim == self.dims().len() {
            // Leaf: extract element
            let array_indices = crate::Shape::from_slice(indices);
            result.push(self.at::<T>(array_indices));
            return Ok(());
        }

        // Recurse through current dimension
        for i in 0..self.shape()[dim] {
            indices[dim] = i;
            self.to_vec_recursive::<T>(result, indices, dim + 1)?;
        }

        Ok(())
    }
}
