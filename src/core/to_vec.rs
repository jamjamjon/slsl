use anyhow::Result;

use crate::{StorageTrait, TensorBase, TensorElement, UninitVec};

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

        debug_assert_eq!(
            self.dtype(),
            T::DTYPE,
            "DType mismatch: tensor has {:?}, requested {:?}",
            self.dtype(),
            T::DTYPE
        );

        unsafe {
            let ptr = self.as_ptr().add(self.offset_bytes) as *const T;
            Ok(*ptr)
        }
    }

    /// Convert tensor to flat `Vec<T>` (any dimensionality)
    ///
    /// # Performance
    /// - Contiguous tensors: Direct memory copy
    /// - Non-contiguous tensors: Using tensor iterator
    ///
    /// # Errors
    /// Returns error if dtype mismatch
    pub fn to_flat_vec<T: TensorElement + Copy>(&self) -> Result<Vec<T>> {
        debug_assert_eq!(
            self.dtype(),
            T::DTYPE,
            "DType mismatch: tensor has {:?}, requested {:?}",
            self.dtype(),
            T::DTYPE
        );
        let numel = self.numel();
        if numel == 0 {
            return Ok(Vec::new());
        }

        let result = if self.is_contiguous() {
            UninitVec::new(numel).init_with(|slice| unsafe {
                let src_ptr = self.as_ptr().add(self.offset_bytes) as *const T;
                std::ptr::copy_nonoverlapping(src_ptr, slice.as_mut_ptr(), numel);
            })
        } else {
            UninitVec::new(numel).init_with(|slice| {
                for (i, elem) in self.iter().enumerate() {
                    unsafe {
                        let ptr = elem.as_ptr(self.as_ptr()) as *const T;
                        slice[i] = *ptr;
                    }
                }
            })
        };

        Ok(result)
    }

    /// Convert 1D tensor to `Vec<T>`
    ///
    /// # Performance
    /// - Contiguous tensors: Direct memory copy (memcpy-like performance)
    /// - Non-contiguous tensors: Using tensor iterator
    ///
    /// # Errors
    /// Returns error if tensor is not 1D or dtype mismatch
    pub fn to_vec<T: TensorElement + Copy>(&self) -> Result<Vec<T>> {
        if self.rank() != 1 {
            anyhow::bail!(
                "Cannot convert {}-dimensional tensor to 1D Vec. Try using `.to_flat_vec()` instead.",
                self.rank()
            );
        }
        self.to_flat_vec()
    }

    /// Convert 2D tensor to `Vec<Vec<T>>`
    pub fn to_vec2<T: TensorElement + Copy>(&self) -> Result<Vec<Vec<T>>> {
        if self.rank() != 2 {
            anyhow::bail!(
                "Cannot convert {}-dimensional tensor to 2D Vec. Try using `.to_flat_vec()` instead.",
                self.rank()
            );
        }
        debug_assert_eq!(
            self.dtype(),
            T::DTYPE,
            "DType mismatch: tensor has {:?}, requested {:?}",
            self.dtype(),
            T::DTYPE
        );

        let [rows, cols] = [self.shape()[0], self.shape()[1]];
        let mut result = Vec::with_capacity(rows);
        let mut iter = self.iter();
        for i in 0..rows {
            let row = UninitVec::new(cols).init_with(|row_slice| {
                for (j, row_elem) in row_slice.iter_mut().enumerate().take(cols) {
                    if let Some(elem) = iter.next() {
                        unsafe {
                            let ptr = elem.as_ptr(self.as_ptr()) as *const T;
                            *row_elem = *ptr;
                        }
                    } else {
                        panic!("Iterator exhausted unexpectedly at row {}, col {}", i, j);
                    }
                }
            });
            result.push(row);
        }

        Ok(result)
    }

    /// Convert 3D tensor to `Vec<Vec<Vec<T>>>`
    pub fn to_vec3<T: TensorElement + Copy>(&self) -> Result<Vec<Vec<Vec<T>>>> {
        if self.rank() != 3 {
            anyhow::bail!(
                "Cannot convert {}-dimensional tensor to 3D Vec. Try using `.to_flat_vec()` instead.",
                self.rank()
            );
        }

        debug_assert_eq!(
            self.dtype(),
            T::DTYPE,
            "DType mismatch: tensor has {:?}, requested {:?}",
            self.dtype(),
            T::DTYPE
        );

        let [dim0, dim1, dim2] = [self.shape()[0], self.shape()[1], self.shape()[2]];
        let mut result: Vec<Vec<Vec<T>>> = Vec::with_capacity(dim0);
        let mut iter = self.iter();
        for i in 0..dim0 {
            let mut plane: Vec<Vec<T>> = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let mut row: Vec<T> = Vec::with_capacity(dim2);
                for k in 0..dim2 {
                    if let Some(elem) = iter.next() {
                        unsafe {
                            let ptr = elem.as_ptr(self.as_ptr()) as *const T;
                            row.push(*ptr);
                        }
                    } else {
                        panic!("Iterator exhausted unexpectedly at [{}, {}, {}]", i, j, k);
                    }
                }
                plane.push(row);
            }
            result.push(plane);
        }

        Ok(result)
    }
}
