//! Core tensor implementation and type definitions.
//!
//! This module provides the fundamental tensor structure that supports various
//! storage backends and tensor operations.

use anyhow::Result;
use half::{bf16, f16};
use std::ptr::NonNull;

use crate::{DType, Shape, Storage, StorageTrait, Stride, TensorElement};

/// Core tensor structure that can work with different storage backends.
///
/// `TensorBase` is a generic tensor implementation that abstracts over different
/// storage types through the `StorageTrait`. This allows for both owned tensors
/// and tensor views with zero-cost abstractions.
///
/// # Type Parameters
///
/// * `S` - Storage type that implements `StorageTrait`
///
/// # Fields
///
/// * `storage` - The underlying storage backend
/// * `ptr` - Raw pointer to the tensor data for fast access
/// * `dtype` - Data type of tensor elements
/// * `shape` - Dimensions of the tensor
/// * `strides` - Memory layout information for indexing
/// * `offset_bytes` - Byte offset into the storage
#[derive(Clone)]
pub struct TensorBase<S: StorageTrait> {
    /// The underlying storage that holds the tensor data
    pub(crate) storage: S,
    /// Raw pointer to tensor data for efficient access
    pub(crate) ptr: NonNull<u8>,
    /// Data type of the tensor elements
    pub(crate) dtype: DType,
    /// Shape (dimensions) of the tensor
    pub(crate) shape: Shape,
    /// Strides for memory layout and indexing
    pub(crate) strides: Stride,
    /// Byte offset into the storage buffer
    pub(crate) offset_bytes: usize,
}

/// Type alias for owned tensors with `Storage` backend.
///
/// This is the most common tensor type for owned data that manages
/// its own memory allocation and cleanup.
pub type Tensor = TensorBase<Storage>;

/// Type alias for tensor views that borrow from existing storage.
///
/// `TensorView` provides a lightweight way to create tensor slices
/// and views without copying data, using borrowed storage.
pub type TensorView<'a> = TensorBase<&'a Storage>;

unsafe impl<S: StorageTrait + Send> Send for TensorBase<S> {}
unsafe impl<S: StorageTrait + Sync> Sync for TensorBase<S> {}

impl Tensor {
    /// Creates a view of this tensor without copying data.
    ///
    /// A tensor view provides a lightweight way to access tensor data
    /// without taking ownership. The view borrows from the original tensor's storage.
    ///
    /// # Returns
    ///
    /// A [`TensorView`] that references the same data as this tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::zeros::<f32>([2, 3])?;
    /// let view = tensor.view();
    /// assert_eq!(view.shape(), tensor.shape());
    /// assert_eq!(view.dtype(), tensor.dtype());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline(always)]
    pub fn view(&self) -> TensorView<'_> {
        TensorBase {
            storage: &self.storage,
            ptr: self.ptr,
            shape: self.shape,
            strides: self.strides,
            dtype: self.dtype,
            offset_bytes: self.offset_bytes,
        }
    }
}

impl<S: StorageTrait> TensorBase<S> {
    /// Returns the reference count for the underlying storage.
    ///
    /// This shows how many tensor instances are sharing the same storage.
    /// Useful for memory management and debugging.
    ///
    /// # Returns
    ///
    /// The number of references to the underlying storage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor1 = Tensor::zeros::<f32>([2, 3])?;
    /// assert_eq!(tensor1.strong_count(), 1);
    ///
    /// let tensor2 = tensor1.clone();
    /// assert_eq!(tensor1.strong_count(), 2);
    /// assert_eq!(tensor2.strong_count(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline(always)]
    pub fn strong_count(&self) -> usize {
        self.storage.as_storage().strong_count()
    }

    /// Returns a reference to the tensor's strides.
    ///
    /// Strides define how to traverse the tensor data in memory.
    /// Each stride represents the number of bytes to skip to move
    /// to the next element along that dimension.
    ///
    /// # Returns
    ///
    /// A reference to the [`Stride`] array.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::zeros::<f32>([2, 3])?;
    /// let strides = tensor.strides();
    /// assert_eq!(strides.len(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline(always)]
    pub fn strides(&self) -> &Stride {
        &self.strides
    }

    /// Returns a reference to the tensor's shape.
    ///
    /// The shape defines the size of each dimension of the tensor.
    ///
    /// # Returns
    ///
    /// A reference to the [`Shape`] containing dimension sizes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::zeros::<f32>([2, 3, 4])?;
    /// let shape = tensor.shape();
    /// assert_eq!(shape.len(), 3);
    /// assert_eq!(shape[0], 2);
    /// assert_eq!(shape[1], 3);
    /// assert_eq!(shape[2], 4);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline(always)]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the tensor dimensions as a slice.
    ///
    /// This provides a convenient way to access the shape dimensions
    /// as a standard Rust slice.
    ///
    /// # Returns
    ///
    /// A slice containing the size of each dimension.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::zeros::<f32>([2, 3, 4])?;
    /// let dims = tensor.dims();
    /// assert_eq!(dims, &[2, 3, 4]);
    /// assert_eq!(dims.len(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline(always)]
    pub fn dims(&self) -> &[usize] {
        self.shape.as_slice()
    }

    /// Returns the number of dimensions (rank) of the tensor.
    ///
    /// A scalar has rank 0, a vector has rank 1, a matrix has rank 2, etc.
    ///
    /// # Returns
    ///
    /// The number of dimensions as a `usize`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// // Scalar (rank 0)
    /// let scalar = Tensor::from_scalar(3.14f32)?;
    /// assert_eq!(scalar.rank(), 0);
    ///
    /// // Vector (rank 1)
    /// let vector = Tensor::zeros::<f32>([5])?;
    /// assert_eq!(vector.rank(), 1);
    ///
    /// // Matrix (rank 2)
    /// let matrix = Tensor::zeros::<f32>([3, 4])?;
    /// assert_eq!(matrix.rank(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline(always)]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns the size of a specific dimension.
    ///
    /// # Parameters
    ///
    /// * `n` - The dimension index (0-based)
    ///
    /// # Returns
    ///
    /// The size of the specified dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::zeros::<f32>([2, 3, 4])?;
    ///
    /// assert_eq!(tensor.ndim(0)?, 2); // First dimension
    /// assert_eq!(tensor.ndim(1)?, 3); // Second dimension
    /// assert_eq!(tensor.ndim(2)?, 4); // Third dimension
    ///
    /// // This would return an error:
    /// // tensor.ndim(3) // Index out of bounds
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline(always)]
    pub fn ndim(&self, n: usize) -> Result<usize> {
        if n >= self.rank() {
            anyhow::bail!("Dim {} >= {}", n, self.rank());
        }
        Ok(self.shape[n])
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// This is the product of all dimension sizes.
    ///
    /// # Returns
    ///
    /// The total number of elements as a `usize`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// // Scalar has 1 element
    /// let scalar = Tensor::from_scalar(3.14f32)?;
    /// assert_eq!(scalar.numel(), 1);
    ///
    /// // 2x3 matrix has 6 elements
    /// let matrix = Tensor::zeros::<f32>([2, 3])?;
    /// assert_eq!(matrix.numel(), 6);
    ///
    /// // 2x3x4 tensor has 24 elements
    /// let tensor3d = Tensor::zeros::<f32>([2, 3, 4])?;
    /// assert_eq!(tensor3d.numel(), 24);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline(always)]
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    /// Returns the data type of the tensor elements.
    ///
    /// # Returns
    ///
    /// The [`DType`] enum value representing the element type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::{Tensor, DType};
    ///
    /// let f32_tensor = Tensor::zeros::<f32>([2, 3])?;
    /// assert_eq!(f32_tensor.dtype(), DType::Fp32);
    ///
    /// let i32_tensor = Tensor::zeros::<i32>([2, 3])?;
    /// assert_eq!(i32_tensor.dtype(), DType::Int32);
    ///
    /// let bool_tensor = Tensor::zeros::<bool>([2, 3])?;
    /// assert_eq!(bool_tensor.dtype(), DType::Bool);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline(always)]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the byte offset of this tensor in the underlying storage.
    ///
    /// This is useful for tensor views and slices that point to a subset
    /// of the original tensor's data.
    ///
    /// # Returns
    ///
    /// The byte offset as a `usize`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::zeros::<f32>([4, 4])?;
    /// assert_eq!(tensor.offset_bytes(), 0); // Original tensor has no offset
    ///
    /// // Sliced tensors may have non-zero offsets
    /// // let slice = tensor.slice(...); // Slicing would create offsets
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline(always)]
    pub fn offset_bytes(&self) -> usize {
        self.offset_bytes
    }

    /// Get tensor data as slice (only for contiguous tensors)
    #[inline(always)]
    pub fn as_slice<T: TensorElement + Copy>(&self) -> Result<&[T]> {
        if !self.is_contiguous() {
            anyhow::bail!(
                "as_slice() only works with contiguous tensors. Use to_contiguous() first."
            );
        }

        debug_assert_eq!(
            self.dtype(),
            T::DTYPE,
            "DType mismatch: tensor has {:?}, requested {:?}",
            self.dtype(),
            T::DTYPE
        );

        let numel = self.numel();
        unsafe {
            let ptr = self.as_ptr() as *const T;
            Ok(std::slice::from_raw_parts(ptr, numel))
        }
    }

    /// Get mutable tensor data as slice (only for contiguous tensors)
    #[inline(always)]
    pub fn as_mut_slice<T: TensorElement + Copy>(&mut self) -> Result<&mut [T]> {
        if !self.is_contiguous() {
            anyhow::bail!(
                "as_mut_slice() only works with contiguous tensors. Use to_contiguous() first."
            );
        }

        debug_assert_eq!(
            self.dtype(),
            T::DTYPE,
            "DType mismatch: tensor has {:?}, requested {:?}",
            self.dtype(),
            T::DTYPE
        );

        let numel = self.numel();
        unsafe {
            let ptr = self.as_mut_ptr() as *mut T;
            Ok(std::slice::from_raw_parts_mut(ptr, numel))
        }
    }

    /// Get data pointer
    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        unsafe { self.storage.ptr().as_ptr().add(self.offset_bytes) }
    }

    /// Get a mutable pointer to the tensor data
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        unsafe { self.storage.ptr().as_ptr().add(self.offset_bytes) }
    }

    /// Create a new TensorView from tensor components
    ///
    /// # Safety
    /// Caller must ensure all parameters are valid and consistent
    #[inline]
    pub unsafe fn from_raw_parts(
        storage: S,
        ptr: NonNull<u8>,
        shape: Shape,
        strides: Shape,
        offset_bytes: usize,
        dtype: DType,
    ) -> Self {
        Self {
            storage,
            ptr,
            shape,
            strides,
            offset_bytes,
            dtype,
        }
    }

    /// Calculate strides for row-major (C-style) layout
    #[inline]
    pub(crate) fn compute_contiguous_strides(shape: &Shape) -> Stride {
        if shape.is_empty() {
            return Stride::empty();
        }

        let mut stride = 1;
        let mut strides_vec = vec![0; shape.len()];

        // Calculate strides from right to left (row-major)
        for i in (0..shape.len()).rev() {
            strides_vec[i] = stride;
            stride *= if i < shape.len() { shape[i] } else { 1 };
        }

        Stride::from_slice(&strides_vec)
    }

    /// Checks if the tensor's memory layout is contiguous(C-style (row-major) or Fortran-style (column-major)).
    #[inline(always)]
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }

        let mut expected_stride = 1;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }
        true
    }

    /// Get a single element from the tensor at the specified indices
    ///
    /// This method provides fast, bounds-checked access to individual tensor elements.
    /// It calculates the memory offset based on the indices and strides, then
    /// returns the value at that location.
    ///
    /// # Arguments
    /// * `indices` - The indices for each dimension. Can be:
    ///   - A single value for 1D tensors (e.g., `i`, `(i,)`, or `[i]`)
    ///   - An array for multi-dimensional tensors (e.g., `[i, j]` for 2D, `[i, j, k]` for 3D)
    ///   - Any type that implements `Into<Shape>`
    ///
    /// # Returns
    /// The value at the specified indices
    ///
    /// # Safety
    /// This method performs bounds checking in debug mode. In release mode,
    /// bounds checking is disabled for maximum performance.
    ///
    /// # Examples
    /// ```
    /// use slsl::Tensor;
    ///
    /// // 1D tensor - multiple formats work
    /// let tensor_1d = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]).unwrap();
    /// let value1 = tensor_1d.at::<f32>(1);      // Single value (most convenient)
    /// let value2 = tensor_1d.at::<f32>((1,));   // Tuple format
    /// let value3 = tensor_1d.at::<f32>([1]);    // Array format
    ///
    /// // Debug: let's see what we get
    /// println!("value1: {}, value2: {}, value3: {}", value1, value2, value3);
    ///
    /// // 2D tensor
    /// let tensor_2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    /// let value = tensor_2d.at::<f32>([1, 0]); // Returns 3.0 (second row, first column)
    /// ```
    #[inline(always)]
    pub fn at<T: TensorElement + Copy>(&self, indices: impl Into<Shape>) -> T {
        let indices = indices.into();
        debug_assert_eq!(indices.len(), self.shape.len(), "Index dimension mismatch");

        let mut offset = self.offset_bytes;
        for (i, &idx) in indices.iter().enumerate() {
            debug_assert!(
                idx < self.shape[i],
                "Index {} out of bounds for dimension {} of size {}",
                idx,
                i,
                self.shape[i]
            );
            // Convert element stride to byte stride
            offset += idx * self.strides[i] * self.dtype.size_in_bytes();
        }
        unsafe {
            let ptr = self.storage.ptr().as_ptr().add(offset) as *const T;
            *ptr
        }
    }

    /// Creates a contiguous copy of the tensor.
    ///
    /// This is an alias for [`Self::to_contiguous`] for convenience.
    /// If the tensor is already contiguous and has no offset, it returns
    /// a clone without copying data.
    ///
    /// # Returns
    ///
    /// A contiguous [`Tensor`] with the same data.
    ///
    /// # Errors
    ///
    /// Returns an error if memory allocation fails or if the data type
    /// is not supported.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::zeros::<f32>([2, 3])?;
    /// let copy = tensor.clone_or_copy()?;
    ///
    /// assert_eq!(tensor.shape(), copy.shape());
    /// assert_eq!(tensor.dtype(), copy.dtype());
    /// assert_eq!(tensor.numel(), copy.numel());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline(always)]
    pub fn clone_or_copy(&self) -> Result<Tensor> {
        self.to_contiguous()
    }

    /// Creates a contiguous copy of the tensor if necessary.
    ///
    /// If the tensor is already contiguous with no offset, this method
    /// returns a clone without copying data. Otherwise, it creates a new
    /// tensor with contiguous memory layout.
    ///
    /// # Returns
    ///
    /// A contiguous [`Tensor`] with the same data and shape.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails
    /// - The tensor's data type is not supported
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// // Create a tensor
    /// let tensor = Tensor::zeros::<f32>([2, 3])?;
    ///
    /// // Get a contiguous version
    /// let contiguous = tensor.to_contiguous()?;
    ///
    /// assert_eq!(tensor.shape(), contiguous.shape());
    /// assert_eq!(tensor.dtype(), contiguous.dtype());
    /// assert!(contiguous.is_contiguous());
    ///
    /// // For already contiguous tensors, this is very efficient
    /// let contiguous2 = contiguous.to_contiguous()?;
    /// assert_eq!(contiguous.strong_count(), contiguous2.strong_count());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline(always)]
    pub fn to_contiguous(&self) -> Result<Tensor> {
        // Fast path: already contiguous with no offset
        if self.is_contiguous() && self.offset_bytes == 0 {
            return Ok(Tensor {
                storage: self.storage.as_storage().clone(),
                ptr: self.ptr,
                dtype: self.dtype,
                shape: self.shape,
                strides: self.strides,
                offset_bytes: self.offset_bytes,
            });
        }

        // Need to create a contiguous copy
        let total_elements = self.numel();
        if total_elements == 0 {
            // Empty tensor case
            let new_storage = Storage::new(0, self.dtype.size_in_bytes())?;
            let new_ptr = new_storage.ptr();
            let new_strides = Self::compute_contiguous_strides(&self.shape);

            return Ok(Tensor {
                storage: new_storage,
                ptr: new_ptr,
                dtype: self.dtype,
                shape: self.shape,
                strides: new_strides,
                offset_bytes: 0,
            });
        }

        // Dispatch to type-specific implementation for maximum performance
        match self.dtype {
            DType::Fp32 => self.make_contiguous_typed::<f32>(),
            DType::Fp64 => self.make_contiguous_typed::<f64>(),
            DType::Int32 => self.make_contiguous_typed::<i32>(),
            DType::Int64 => self.make_contiguous_typed::<i64>(),
            DType::Uint8 => self.make_contiguous_typed::<u8>(),
            DType::Uint16 => self.make_contiguous_typed::<u16>(),
            DType::Uint32 => self.make_contiguous_typed::<u32>(),
            DType::Uint64 => self.make_contiguous_typed::<u64>(),
            DType::Int8 => self.make_contiguous_typed::<i8>(),
            DType::Int16 => self.make_contiguous_typed::<i16>(),
            DType::Bool => self.make_contiguous_typed::<bool>(),
            _ => anyhow::bail!("Unsupported dtype for to_contiguous: {:?}", self.dtype),
        }
    }

    /// Type-specific contiguous copy implementation
    #[inline(always)]
    fn make_contiguous_typed<T: TensorElement + Copy>(&self) -> Result<Tensor> {
        let total_elements = self.numel();
        let total_bytes = total_elements * std::mem::size_of::<T>();

        // Create new storage
        let new_storage = Storage::new(total_bytes, std::mem::align_of::<T>())?;
        let new_ptr = new_storage.ptr();
        let dst_ptr = new_ptr.as_ptr() as *mut T;

        // Analyze copy strategy for optimal performance
        if self.is_contiguous() {
            // Contiguous but with offset - use fast memcpy
            unsafe {
                let src_ptr = self.as_ptr() as *const T;
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, total_elements);
            }
        } else {
            // Non-contiguous - use optimized strided copy
            self.copy_strided::<T>(dst_ptr)?;
        }

        // Create contiguous strides
        let new_strides = Self::compute_contiguous_strides(&self.shape);

        Ok(Tensor {
            storage: new_storage,
            ptr: new_ptr,
            dtype: self.dtype,
            shape: self.shape,
            strides: new_strides,
            offset_bytes: 0,
        })
    }

    /// Optimized strided copy for non-contiguous tensors
    /// Uses block-wise copying when possible for better cache performance
    #[inline(always)]
    fn copy_strided<T: TensorElement + Copy>(&self, dst_ptr: *mut T) -> Result<()> {
        let dims = self.shape.as_slice();

        if dims.is_empty() {
            // Scalar case
            unsafe {
                let src_ptr = self.as_ptr() as *const T;
                *dst_ptr = *src_ptr;
            }
            return Ok(());
        }

        // Use dimension-specific optimizations
        match dims.len() {
            1 => self.copy_1d::<T>(dst_ptr)?,
            2 => self.copy_2d::<T>(dst_ptr)?,
            3 => self.copy_3d::<T>(dst_ptr)?,
            4 => self.copy_4d::<T>(dst_ptr)?,
            5 => self.copy_5d::<T>(dst_ptr)?,
            6 => self.copy_6d::<T>(dst_ptr)?,
            _ => {
                // For higher dimensions, use the general block-based approach
                let (block_dims, block_size) = self.find_contiguous_block();

                if block_size >= 16 {
                    // Use block-wise copy for better performance
                    self.copy_blocks::<T>(dst_ptr, block_dims, block_size)?;
                } else {
                    // Fall back to element-wise copy with optimized indexing
                    self.copy_elements::<T>(dst_ptr)?;
                }
            }
        }

        Ok(())
    }

    /// Optimized 1D copy - direct memory copy with potential SIMD
    #[inline(always)]
    fn copy_1d<T: TensorElement + Copy>(&self, dst_ptr: *mut T) -> Result<()> {
        let total_elements = self.numel();
        if total_elements == 0 {
            return Ok(());
        }

        unsafe {
            let src_ptr = self.as_ptr() as *const T;
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, total_elements);
        }

        Ok(())
    }

    /// Optimized 2D copy - cache-friendly row-wise copy
    #[inline(always)]
    fn copy_2d<T: TensorElement + Copy>(&self, dst_ptr: *mut T) -> Result<()> {
        let dims = self.shape.as_slice();
        let strides = self.strides.as_slice();
        let rows = dims[0];
        let cols = dims[1];
        let row_stride = strides[0];
        let col_stride = strides[1];

        if rows == 0 || cols == 0 {
            return Ok(());
        }

        unsafe {
            let src_base_ptr = self.storage.ptr().as_ptr().add(self.offset_bytes) as *const T;

            // Cache-friendly: copy row by row
            for row in 0..rows {
                let src_row_start = src_base_ptr.add(row * row_stride);
                let dst_row_start = dst_ptr.add(row * cols);

                // Copy entire row at once if possible
                if col_stride == 1 {
                    // Contiguous row - fast copy
                    std::ptr::copy_nonoverlapping(src_row_start, dst_row_start, cols);
                } else {
                    // Strided row - element by element
                    for col in 0..cols {
                        *(dst_row_start.add(col)) = *(src_row_start.add(col * col_stride));
                    }
                }
            }
        }

        Ok(())
    }

    /// Optimized 3D copy - cache-friendly plane-wise copy
    #[inline(always)]
    fn copy_3d<T: TensorElement + Copy>(&self, dst_ptr: *mut T) -> Result<()> {
        let dims = self.shape.as_slice();
        let strides = self.strides.as_slice();
        let depth = dims[0];
        let rows = dims[1];
        let cols = dims[2];
        let depth_stride = strides[0];
        let row_stride = strides[1];
        let col_stride = strides[2];

        if depth == 0 || rows == 0 || cols == 0 {
            return Ok(());
        }

        unsafe {
            let src_base_ptr = self.storage.ptr().as_ptr().add(self.offset_bytes) as *const T;

            // Cache-friendly: copy plane by plane, row by row
            for d in 0..depth {
                for row in 0..rows {
                    let src_plane_start = src_base_ptr.add(d * depth_stride + row * row_stride);
                    let dst_plane_start = dst_ptr.add(d * rows * cols + row * cols);

                    // Copy entire row if possible
                    if col_stride == 1 {
                        std::ptr::copy_nonoverlapping(src_plane_start, dst_plane_start, cols);
                    } else {
                        for col in 0..cols {
                            *(dst_plane_start.add(col)) = *(src_plane_start.add(col * col_stride));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Optimized 4D copy - optimized for common 4D layouts
    #[inline(always)]
    fn copy_4d<T: TensorElement + Copy>(&self, dst_ptr: *mut T) -> Result<()> {
        let dims = self.shape.as_slice();
        let strides = self.strides.as_slice();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];

        let batch_stride = strides[0];
        let channel_stride = strides[1];
        let height_stride = strides[2];
        let width_stride = strides[3];

        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Ok(());
        }

        unsafe {
            let src_base_ptr = self.storage.ptr().as_ptr().add(self.offset_bytes) as *const T;

            // Optimized for common 4D layouts (batch, channels, height, width)
            for b in 0..batch {
                for c in 0..channels {
                    for h in 0..height {
                        let src_start = src_base_ptr
                            .add(b * batch_stride + c * channel_stride + h * height_stride);
                        let dst_start = dst_ptr
                            .add(b * channels * height * width + c * height * width + h * width);

                        // Copy entire row if possible
                        if width_stride == 1 {
                            std::ptr::copy_nonoverlapping(src_start, dst_start, width);
                        } else {
                            for w in 0..width {
                                *(dst_start.add(w)) = *(src_start.add(w * width_stride));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Optimized 5D copy - optimized for common 5D layouts
    #[inline(always)]
    fn copy_5d<T: TensorElement + Copy>(&self, dst_ptr: *mut T) -> Result<()> {
        let dims = self.shape.as_slice();
        let strides = self.strides.as_slice();
        let dim0 = dims[0];
        let dim1 = dims[1];
        let dim2 = dims[2];
        let dim3 = dims[3];
        let dim4 = dims[4];

        let stride0 = strides[0];
        let stride1 = strides[1];
        let stride2 = strides[2];
        let stride3 = strides[3];
        let stride4 = strides[4];

        if dim0 == 0 || dim1 == 0 || dim2 == 0 || dim3 == 0 || dim4 == 0 {
            return Ok(());
        }

        unsafe {
            let src_base_ptr = self.storage.ptr().as_ptr().add(self.offset_bytes) as *const T;

            // Use nested loops with optimized innermost dimension
            for i0 in 0..dim0 {
                for i1 in 0..dim1 {
                    for i2 in 0..dim2 {
                        for i3 in 0..dim3 {
                            let src_start = src_base_ptr
                                .add(i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3);
                            let dst_start = dst_ptr.add(
                                i0 * dim1 * dim2 * dim3 * dim4
                                    + i1 * dim2 * dim3 * dim4
                                    + i2 * dim3 * dim4
                                    + i3 * dim4,
                            );

                            // Copy innermost dimension
                            if stride4 == 1 {
                                std::ptr::copy_nonoverlapping(src_start, dst_start, dim4);
                            } else {
                                for i4 in 0..dim4 {
                                    *(dst_start.add(i4)) = *(src_start.add(i4 * stride4));
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Optimized 6D copy - optimized for common 6D layouts
    #[inline(always)]
    fn copy_6d<T: TensorElement + Copy>(&self, dst_ptr: *mut T) -> Result<()> {
        let dims = self.shape.as_slice();
        let strides = self.strides.as_slice();
        let dim0 = dims[0];
        let dim1 = dims[1];
        let dim2 = dims[2];
        let dim3 = dims[3];
        let dim4 = dims[4];
        let dim5 = dims[5];

        let stride0 = strides[0];
        let stride1 = strides[1];
        let stride2 = strides[2];
        let stride3 = strides[3];
        let stride4 = strides[4];
        let stride5 = strides[5];

        if dim0 == 0 || dim1 == 0 || dim2 == 0 || dim3 == 0 || dim4 == 0 || dim5 == 0 {
            return Ok(());
        }

        unsafe {
            let src_base_ptr = self.storage.ptr().as_ptr().add(self.offset_bytes) as *const T;

            // Use nested loops with optimized innermost dimensions
            for i0 in 0..dim0 {
                for i1 in 0..dim1 {
                    for i2 in 0..dim2 {
                        for i3 in 0..dim3 {
                            for i4 in 0..dim4 {
                                let src_start = src_base_ptr.add(
                                    i0 * stride0
                                        + i1 * stride1
                                        + i2 * stride2
                                        + i3 * stride3
                                        + i4 * stride4,
                                );
                                let dst_start = dst_ptr.add(
                                    i0 * dim1 * dim2 * dim3 * dim4 * dim5
                                        + i1 * dim2 * dim3 * dim4 * dim5
                                        + i2 * dim3 * dim4 * dim5
                                        + i3 * dim4 * dim5
                                        + i4 * dim5,
                                );

                                // Copy innermost dimension
                                if stride5 == 1 {
                                    std::ptr::copy_nonoverlapping(src_start, dst_start, dim5);
                                } else {
                                    for i5 in 0..dim5 {
                                        *(dst_start.add(i5)) = *(src_start.add(i5 * stride5));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Find the largest contiguous memory block for optimization
    /// Returns (number of contiguous dimensions, size of contiguous block)
    /// Enhanced with cache-line analysis for better performance
    #[inline(always)]
    fn find_contiguous_block(&self) -> (usize, usize) {
        let dims = self.shape.as_slice();
        let strides = self.strides.as_slice();

        if dims.is_empty() {
            return (0, 1);
        }

        // Find rightmost contiguous dimensions
        let mut block_size = 1;
        let mut contiguous_dims = 0;

        for i in (0..dims.len()).rev() {
            if strides[i] == block_size {
                block_size *= dims[i];
                contiguous_dims += 1;
            } else {
                break;
            }
        }

        // Optimize block size for cache-line alignment
        if block_size > 1 {
            // Ensure block size is cache-line friendly
            const CACHE_LINE_SIZE: usize = 64;
            let element_size = self.dtype.size_in_bytes();

            if element_size > 0 {
                let elements_per_cache_line = CACHE_LINE_SIZE / element_size;
                if elements_per_cache_line > 1 {
                    // Round down to cache-line aligned size
                    let aligned_size =
                        (block_size / elements_per_cache_line) * elements_per_cache_line;
                    if aligned_size >= elements_per_cache_line {
                        block_size = aligned_size;
                    }
                }
            }
        }

        (contiguous_dims, block_size)
    }

    /// Block-wise copy for tensors with large contiguous blocks
    /// Uses cache-line aligned copying for optimal performance
    #[inline(always)]
    fn copy_blocks<T: TensorElement + Copy>(
        &self,
        dst_ptr: *mut T,
        block_dims: usize,
        block_size: usize,
    ) -> Result<()> {
        let dims = self.shape.as_slice();
        let strides = self.strides.as_slice();
        let outer_dims = dims.len() - block_dims;

        if outer_dims == 0 {
            // Entire tensor is one contiguous block
            unsafe {
                let src_ptr = self.as_ptr() as *const T;
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, block_size);
            }
            return Ok(());
        }

        // Pre-allocate indices array to avoid repeated allocations
        let mut indices = vec![0; outer_dims];
        let mut dst_offset = 0;

        // Copy blocks with cache-line alignment
        loop {
            // Calculate source offset for current block
            let mut src_offset = 0;
            for (i, &idx) in indices.iter().enumerate() {
                src_offset += idx * strides[i];
            }

            // Copy the contiguous block with potential SIMD optimization
            unsafe {
                let base_ptr = self.storage.ptr().as_ptr().add(self.offset_bytes) as *const T;
                let src_ptr = base_ptr.add(src_offset);
                let dst_block_ptr = dst_ptr.add(dst_offset);

                // Use SIMD-optimized copy for large blocks
                if block_size >= 128 && std::mem::size_of::<T>() == 4 {
                    // For f32/i32, try to use SIMD
                    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
                        let src_f32 = src_ptr as *const f32;
                        let dst_f32 = dst_block_ptr as *mut f32;
                        self.copy_block(src_f32, dst_f32, block_size);
                    } else {
                        std::ptr::copy_nonoverlapping(src_ptr, dst_block_ptr, block_size);
                    }
                } else {
                    std::ptr::copy_nonoverlapping(src_ptr, dst_block_ptr, block_size);
                }
            }

            dst_offset += block_size;

            // Advance to next block using optimized index increment
            let mut carry = 1;
            for i in (0..outer_dims).rev() {
                indices[i] += carry;
                if indices[i] < dims[i] {
                    carry = 0;
                    break;
                } else {
                    indices[i] = 0;
                }
            }

            if carry == 1 {
                break; // All blocks processed
            }
        }

        Ok(())
    }

    /// SIMD-optimized block copy for f32/i32 types
    #[inline(always)]
    unsafe fn copy_block(&self, src_ptr: *const f32, dst_ptr: *mut f32, block_size: usize) {
        // Use chunks of 16 elements for potential SIMD optimization
        const CHUNK_SIZE: usize = 16;
        let chunks = block_size / CHUNK_SIZE;
        let remainder = block_size % CHUNK_SIZE;

        // Copy chunks with manual unrolling
        for i in 0..chunks {
            let src_chunk = src_ptr.add(i * CHUNK_SIZE);
            let dst_chunk = dst_ptr.add(i * CHUNK_SIZE);

            // Manual unrolling for 16 elements
            *dst_chunk = *src_chunk;
            *(dst_chunk.add(1)) = *(src_chunk.add(1));
            *(dst_chunk.add(2)) = *(src_chunk.add(2));
            *(dst_chunk.add(3)) = *(src_chunk.add(3));
            *(dst_chunk.add(4)) = *(src_chunk.add(4));
            *(dst_chunk.add(5)) = *(src_chunk.add(5));
            *(dst_chunk.add(6)) = *(src_chunk.add(6));
            *(dst_chunk.add(7)) = *(src_chunk.add(7));
            *(dst_chunk.add(8)) = *(src_chunk.add(8));
            *(dst_chunk.add(9)) = *(src_chunk.add(9));
            *(dst_chunk.add(10)) = *(src_chunk.add(10));
            *(dst_chunk.add(11)) = *(src_chunk.add(11));
            *(dst_chunk.add(12)) = *(src_chunk.add(12));
            *(dst_chunk.add(13)) = *(src_chunk.add(13));
            *(dst_chunk.add(14)) = *(src_chunk.add(14));
            *(dst_chunk.add(15)) = *(src_chunk.add(15));
        }

        // Copy remainder
        let remainder_start = chunks * CHUNK_SIZE;
        for i in 0..remainder {
            *(dst_ptr.add(remainder_start + i)) = *(src_ptr.add(remainder_start + i));
        }
    }

    /// Element-wise copy for highly fragmented layouts
    /// Optimized to minimize cache misses and reduce allocations
    #[inline(always)]
    fn copy_elements<T: TensorElement + Copy>(&self, dst_ptr: *mut T) -> Result<()> {
        let total_elements = self.numel();
        let dims = self.shape.as_slice();
        let strides = self.strides.as_slice();

        if total_elements == 0 {
            return Ok(());
        }

        // Use stack allocation for small dimensions to avoid heap allocation
        if dims.len() <= 8 {
            self.copy_elements_small_dims::<T, 8>(dst_ptr, dims, strides)?;
        } else {
            // Fall back to heap allocation for larger dimensions
            let mut indices = vec![0; dims.len()];
            self.copy_elements_with_indices::<T>(dst_ptr, dims, strides, &mut indices)?;
        }

        Ok(())
    }

    /// Optimized element copy for small dimensions (≤8) using stack allocation
    #[inline(always)]
    fn copy_elements_small_dims<T: TensorElement + Copy, const MAX_DIMS: usize>(
        &self,
        dst_ptr: *mut T,
        dims: &[usize],
        strides: &[usize],
    ) -> Result<()> {
        let total_elements = self.numel();
        let mut indices = [0; MAX_DIMS];

        // Initialize indices for actual dimensions
        for index in indices.iter_mut().take(dims.len()) {
            *index = 0;
        }

        unsafe {
            let base_ptr = self.storage.ptr().as_ptr().add(self.offset_bytes) as *const T;

            for dst_idx in 0..total_elements {
                // Calculate source offset using strides
                let mut src_offset = 0;
                for (i, &idx) in indices.iter().take(dims.len()).enumerate() {
                    src_offset += idx * strides[i];
                }

                // Copy element
                let src_ptr = base_ptr.add(src_offset);
                let dst_element_ptr = dst_ptr.add(dst_idx);
                *dst_element_ptr = *src_ptr;

                // Advance indices (row-major order) - optimized for small dimensions
                let mut carry = true;
                for dim in (0..dims.len()).rev() {
                    if carry {
                        indices[dim] += 1;
                        if indices[dim] < dims[dim] {
                            carry = false;
                        } else {
                            indices[dim] = 0;
                        }
                    }
                }

                if carry {
                    break; // All dimensions exhausted
                }
            }
        }

        Ok(())
    }

    /// Element copy using heap-allocated indices for larger dimensions
    #[inline(always)]
    fn copy_elements_with_indices<T: TensorElement + Copy>(
        &self,
        dst_ptr: *mut T,
        dims: &[usize],
        strides: &[usize],
        indices: &mut [usize],
    ) -> Result<()> {
        let total_elements = self.numel();

        unsafe {
            let base_ptr = self.storage.ptr().as_ptr().add(self.offset_bytes) as *const T;

            for dst_idx in 0..total_elements {
                // Calculate source offset using strides
                let mut src_offset = 0;
                for (i, &idx) in indices.iter().enumerate() {
                    src_offset += idx * strides[i];
                }

                // Copy element
                let src_ptr = base_ptr.add(src_offset);
                let dst_element_ptr = dst_ptr.add(dst_idx);
                *dst_element_ptr = *src_ptr;

                // Advance indices (row-major order)
                let mut carry = true;
                for dim in (0..dims.len()).rev() {
                    if carry {
                        indices[dim] += 1;
                        if indices[dim] < dims[dim] {
                            carry = false;
                        } else {
                            indices[dim] = 0;
                        }
                    }
                }

                if carry {
                    break; // All dimensions exhausted
                }
            }
        }

        Ok(())
    }

    /// Deep clone - creates new storage with copied data
    ///
    /// This method creates a completely independent copy of the tensor data,
    /// regardless of whether the original is contiguous or not.
    ///
    /// # Performance
    /// - Fast path for contiguous tensors using optimized memcpy
    /// - Efficient strided copy for non-contiguous tensors
    /// - Type-specific optimizations for different data types
    ///
    /// # Returns
    /// * `Result<Tensor>` - A new owned tensor with copied data
    #[inline(always)]
    pub fn to_owned(&self) -> Result<Tensor> {
        let total_elements = self.numel();
        let total_bytes = total_elements * self.dtype.size_in_bytes();

        if total_elements == 0 {
            // Empty tensor case
            let new_storage = Storage::new(0, self.dtype.size_in_bytes())?;
            let new_ptr = new_storage.ptr();
            let new_strides = Shape::empty();

            return Ok(Tensor {
                storage: new_storage,
                ptr: new_ptr,
                dtype: self.dtype,
                shape: self.shape,
                strides: new_strides,
                offset_bytes: 0,
            });
        }

        // Create new storage
        let new_storage = Storage::new(total_bytes, self.dtype.size_in_bytes())?;
        let new_ptr = new_storage.ptr();

        // Copy data based on data type with optimized paths
        match self.dtype {
            DType::Fp32 => self.copy_data_to_storage::<f32>(new_ptr.as_ptr())?,
            DType::Fp64 => self.copy_data_to_storage::<f64>(new_ptr.as_ptr())?,
            DType::Fp16 => self.copy_data_to_storage::<f16>(new_ptr.as_ptr())?,
            DType::Bf16 => self.copy_data_to_storage::<bf16>(new_ptr.as_ptr())?,
            DType::Int8 => self.copy_data_to_storage::<i8>(new_ptr.as_ptr())?,
            DType::Int16 => self.copy_data_to_storage::<i16>(new_ptr.as_ptr())?,
            DType::Int32 => self.copy_data_to_storage::<i32>(new_ptr.as_ptr())?,
            DType::Int64 => self.copy_data_to_storage::<i64>(new_ptr.as_ptr())?,
            DType::Uint8 => self.copy_data_to_storage::<u8>(new_ptr.as_ptr())?,
            DType::Uint16 => self.copy_data_to_storage::<u16>(new_ptr.as_ptr())?,
            DType::Uint32 => self.copy_data_to_storage::<u32>(new_ptr.as_ptr())?,
            DType::Uint64 => self.copy_data_to_storage::<u64>(new_ptr.as_ptr())?,
            DType::Bool => self.copy_data_to_storage::<bool>(new_ptr.as_ptr())?,
            _ => anyhow::bail!("Unsupported dtype for to_owned: {:?}", self.dtype),
        }

        // Create contiguous strides
        let new_strides = Self::compute_contiguous_strides(&self.shape);

        Ok(Tensor {
            storage: new_storage,
            ptr: new_ptr,
            dtype: self.dtype,
            shape: self.shape,
            strides: new_strides,
            offset_bytes: 0,
        })
    }

    /// Optimized data copy to new storage
    ///
    /// This method provides the fastest possible data copying by:
    /// - Using memcpy for contiguous tensors
    /// - Leveraging existing optimized strided copy for non-contiguous tensors
    /// - Avoiding redundant memory allocations
    #[inline(always)]
    fn copy_data_to_storage<T: TensorElement + Copy>(&self, dst_ptr: *mut u8) -> Result<()> {
        let dst_ptr = dst_ptr as *mut T;
        let total_elements = self.numel();

        if self.is_contiguous() {
            // Fast path: contiguous memory - use optimized memcpy
            unsafe {
                let src_ptr = self.as_ptr() as *const T;
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, total_elements);
            }
        } else {
            // Non-contiguous: use existing optimized strided copy
            // This reuses the highly optimized copy logic from to_contiguous
            self.copy_strided::<T>(dst_ptr)?;
        }

        Ok(())
    }

    /// Generic map function that applies a closure to each element
    /// Input and output types are the same by default
    ///
    /// # Arguments
    /// * `f` - Closure that takes a reference to an element and returns a new value
    ///
    /// # Returns
    /// * `Result<Tensor>` - New tensor with mapped values
    ///
    /// # Examples
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// // Same type operation (most common case)
    /// let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3]).unwrap();
    /// let result = tensor.map::<f32>(|x| x.abs()).unwrap();
    ///
    /// // For i32 tensors
    /// let tensor = Tensor::from_vec(vec![1i32, -2, 3], [3]).unwrap();
    /// let result = tensor.map::<i32>(|x| x.abs()).unwrap();
    /// ```
    #[inline(always)]
    pub fn map<T>(&self, f: impl FnMut(&T) -> T) -> Result<Tensor>
    where
        T: Copy + 'static + TensorElement,
    {
        if self.is_contiguous() {
            self.map_contiguous::<T>(f)
        } else {
            self.map_non_contiguous::<T>(f)
        }
    }

    /// Applies a function `f` to each element of a contiguous tensor.
    ///
    /// # Arguments
    ///
    /// * `f` - A mutable closure that takes a reference to an element of type `T` and returns a new element of type `T`.
    ///
    /// # Returns
    ///
    /// A `Result` containing a new `Tensor` with the mapped elements, or an error if the operation fails.
    #[inline(always)]
    pub fn map_contiguous<T>(&self, f: impl FnMut(&T) -> T) -> Result<Tensor>
    where
        T: Copy + 'static + TensorElement,
    {
        let input_data = self.as_slice::<T>()?;
        let output: Vec<T> = input_data.iter().map(f).collect();
        Tensor::from_vec(output, self.shape)
    }

    /// Applies a function `f` to each element of a non-contiguous tensor.
    /// This method iterates through the tensor's elements based on its strides and applies the given function.
    ///
    /// # Arguments
    ///
    /// * `f` - A mutable closure that takes a reference to an element of type `T` and returns a new element of type `T`.
    ///
    /// # Returns
    ///
    /// A `Result` containing a new `Tensor` with the mapped elements, or an error if the operation fails.
    #[inline(always)]
    pub fn map_non_contiguous<T>(&self, mut f: impl FnMut(&T) -> T) -> Result<Tensor>
    where
        T: Copy + 'static + TensorElement,
    {
        let output: Vec<T> = self
            .iter_with_meta::<T>()
            .map(|item| f(item.value))
            .collect();

        Tensor::from_vec(output, self.shape)
    }

    /// Parallel version of map() that applies a closure to each element using Rayon
    ///
    /// This method provides parallel processing for large tensors where the overhead
    /// of parallelization is justified by the computation complexity.
    ///
    /// # Arguments
    /// * `f` - Closure that takes a reference to an element and returns a new value
    ///
    /// # Returns
    /// * `Result<Tensor>` - New tensor with mapped values
    ///
    /// # Examples
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    /// let result = tensor.par_map::<f32>(|x| x * 2.0).unwrap();
    /// ```
    #[cfg(feature = "rayon")]
    #[inline(always)]
    pub fn par_map<T>(&self, f: impl Fn(&T) -> T + Send + Sync) -> Result<Tensor>
    where
        T: Copy + 'static + TensorElement + Send + Sync,
        S: Send + Sync,
    {
        if self.is_contiguous() {
            self.par_map_contiguous::<T>(f)
        } else {
            self.par_map_non_contiguous::<T>(f)
        }
    }

    /// Parallel version of map_contiguous() using Rayon
    ///
    /// # Arguments
    /// * `f` - Closure that takes a reference to an element and returns a new value
    ///
    /// # Returns
    /// * `Result<Tensor>` - New tensor with mapped values
    #[cfg(feature = "rayon")]
    #[inline(always)]
    pub fn par_map_contiguous<T>(&self, f: impl Fn(&T) -> T + Send + Sync) -> Result<Tensor>
    where
        T: Copy + 'static + TensorElement + Send + Sync,
        S: Send + Sync,
    {
        use rayon::prelude::*;

        let input_data = self.as_slice::<T>()?;
        let output: Vec<T> = input_data.par_iter().map(f).collect();
        Tensor::from_vec(output, self.shape)
    }

    /// Parallel version of map_non_contiguous() using Rayon
    ///
    /// # Arguments
    /// * `f` - Closure that takes a reference to an element and returns a new value
    ///
    /// # Returns
    /// * `Result<Tensor>` - New tensor with mapped values
    #[cfg(feature = "rayon")]
    #[inline(always)]
    pub fn par_map_non_contiguous<T>(&self, f: impl Fn(&T) -> T + Send + Sync) -> Result<Tensor>
    where
        T: Copy + 'static + TensorElement + Send + Sync,
        S: Send + Sync,
    {
        use rayon::prelude::*;

        let output: Vec<T> = self.par_iter::<T>().map(f).collect();

        Tensor::from_vec(output, self.shape)
    }

    /// Applies a closure to each element of the tensor for side effects.
    ///
    /// This method iterates through all elements of the tensor and applies
    /// the given closure to each element. Unlike `map()`, this method does
    /// not return a new tensor but is used for side effects like printing,
    /// counting, or validation.
    ///
    /// # Arguments
    /// * `f` - Closure that takes a reference to an element
    ///
    /// # Examples
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3]).unwrap();
    ///
    /// // Count positive values
    /// let mut count = 0;
    /// tensor.for_each::<f32>(|x| {
    ///     if *x > 0.0 { count += 1; }
    /// });
    /// assert_eq!(count, 3);
    /// ```
    #[inline(always)]
    pub fn for_each<T>(&self, f: impl FnMut(&T))
    where
        T: Copy + 'static + TensorElement,
    {
        if self.is_contiguous() {
            self.for_each_contiguous::<T>(f)
        } else {
            self.for_each_non_contiguous::<T>(f)
        }
    }

    /// Applies a closure to each element of a contiguous tensor for side effects.
    ///
    /// # Arguments
    /// * `f` - Closure that takes a reference to an element
    #[inline(always)]
    fn for_each_contiguous<T>(&self, mut f: impl FnMut(&T))
    where
        T: Copy + 'static + TensorElement,
    {
        if let Ok(data) = self.as_slice::<T>() {
            data.iter().for_each(&mut f);
        }
    }

    /// Applies a closure to each element of a non-contiguous tensor for side effects.
    ///
    /// # Arguments
    /// * `f` - Closure that takes a reference to an element
    #[inline(always)]
    fn for_each_non_contiguous<T>(&self, mut f: impl FnMut(&T))
    where
        T: Copy + 'static + TensorElement,
    {
        self.iter::<T>().for_each(&mut f);
    }

    /// Parallel version of for_each() that applies a closure to each element using Rayon
    ///
    /// This method provides parallel processing for side effects on large tensors.
    /// The closure is applied to each element in parallel, which can be significantly
    /// faster for CPU-intensive operations.
    ///
    /// # Arguments
    /// * `f` - Closure that takes a reference to an element
    ///
    /// # Examples
    /// ```rust
    /// use slsl::Tensor;
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    ///
    /// // Count values greater than 2.0 in parallel
    /// let counter = std::sync::Arc::new(AtomicUsize::new(0));
    /// let counter_clone = counter.clone();
    /// tensor.par_for_each::<f32>(move |x| {
    ///     if *x > 2.0 {
    ///         counter_clone.fetch_add(1, Ordering::Relaxed);
    ///     }
    /// });
    /// assert_eq!(counter.load(Ordering::Relaxed), 2);
    /// ```
    #[cfg(feature = "rayon")]
    #[inline(always)]
    pub fn par_for_each<T>(&self, f: impl Fn(&T) + Send + Sync)
    where
        T: Copy + 'static + TensorElement + Send + Sync,
        S: Send + Sync,
    {
        if self.is_contiguous() {
            self.par_for_each_contiguous::<T>(f)
        } else {
            self.par_for_each_non_contiguous::<T>(f)
        }
    }

    /// Parallel version of for_each_contiguous() using Rayon
    ///
    /// # Arguments
    /// * `f` - Closure that takes a reference to an element
    #[cfg(feature = "rayon")]
    #[inline(always)]
    fn par_for_each_contiguous<T>(&self, f: impl Fn(&T) + Send + Sync)
    where
        T: Copy + 'static + TensorElement + Send + Sync,
        S: Send + Sync,
    {
        use rayon::prelude::*;

        if let Ok(data) = self.as_slice::<T>() {
            data.par_iter().for_each(f);
        }
    }

    /// Parallel version of for_each_non_contiguous() using Rayon
    ///
    /// # Arguments
    /// * `f` - Closure that takes a reference to an element
    #[cfg(feature = "rayon")]
    #[inline(always)]
    fn par_for_each_non_contiguous<T>(&self, f: impl Fn(&T) + Send + Sync)
    where
        T: Copy + 'static + TensorElement + Send + Sync,
        S: Send + Sync,
    {
        use rayon::prelude::*;

        self.par_iter::<T>().for_each(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_owned_on_tensor() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, [2, 2]).unwrap();

        let owned = tensor.to_owned().unwrap();

        // Verify it's a deep copy
        assert_eq!(tensor.shape(), owned.shape());
        assert_eq!(tensor.dtype(), owned.dtype());
        assert_eq!(tensor.numel(), owned.numel());

        // Verify data is copied
        let original_slice = tensor.as_slice::<f32>().unwrap();
        let owned_slice = owned.as_slice::<f32>().unwrap();
        assert_eq!(original_slice, owned_slice);

        // Verify they have different storage (deep copy)
        // Note: strong_count might be the same if there are no other references
        // but the storage should be different
        assert_ne!(tensor.as_ptr(), owned.as_ptr());
    }

    #[test]
    fn test_to_owned_on_tensor_view() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3]).unwrap();

        // Create a view
        let view = tensor.view();

        let owned = view.to_owned().unwrap();

        // Verify it's a deep copy
        assert_eq!(view.shape(), owned.shape());
        assert_eq!(view.dtype(), owned.dtype());
        assert_eq!(view.numel(), owned.numel());

        // Verify data is copied
        let view_slice = view.as_slice::<f32>().unwrap();
        let owned_slice = owned.as_slice::<f32>().unwrap();
        assert_eq!(view_slice, owned_slice);

        // Verify they have different storage (deep copy)
        assert_ne!(view.as_ptr(), owned.as_ptr());
    }

    #[test]
    fn test_to_owned_empty_tensor() {
        // Create an empty tensor with shape [0] instead of []
        let tensor = Tensor::from_vec(vec![0u8; 0], [0]).unwrap();
        let owned = tensor.to_owned().unwrap();

        assert_eq!(tensor.shape(), owned.shape());
        assert_eq!(tensor.dtype(), owned.dtype());
        assert_eq!(tensor.numel(), owned.numel());
        assert_eq!(owned.numel(), 0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_map_contiguous() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3]).unwrap();

        let result = tensor.par_map::<f32>(|x| x * 2.0).unwrap();

        assert_eq!(result.shape(), tensor.shape());
        assert_eq!(result.dtype(), tensor.dtype());

        let result_data = result.as_slice::<f32>().unwrap();
        assert_eq!(result_data, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_map_non_contiguous() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3]).unwrap();

        // Create a non-contiguous view by transposing
        let transposed = tensor.permute([1, 0]).unwrap();
        assert!(!transposed.is_contiguous());

        let result = transposed.par_map::<f32>(|x| x * 3.0).unwrap();

        assert_eq!(result.shape(), transposed.shape());
        assert_eq!(result.dtype(), transposed.dtype());

        // The result should be contiguous after mapping
        assert!(result.is_contiguous());
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_map_vs_map_consistency() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, [10, 100]).unwrap();

        // Test with contiguous tensor
        let map_result = tensor.map::<f32>(|x| x.sin()).unwrap();
        let par_map_result = tensor.par_map::<f32>(|x| x.sin()).unwrap();

        assert_eq!(map_result.shape(), par_map_result.shape());
        assert_eq!(map_result.dtype(), par_map_result.dtype());

        let map_data = map_result.as_slice::<f32>().unwrap();
        let par_map_data = par_map_result.as_slice::<f32>().unwrap();

        // Results should be identical (within floating point precision)
        for (a, b) in map_data.iter().zip(par_map_data.iter()) {
            assert!((a - b).abs() < 1e-6, "Results differ: {} vs {}", a, b);
        }
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_map_different_types() {
        // Test with i32
        let data_i32 = vec![1i32, -2, 3, -4, 5];
        let tensor_i32 = Tensor::from_vec(data_i32, [5]).unwrap();
        let result_i32 = tensor_i32.par_map::<i32>(|x| x.abs()).unwrap();
        let result_data_i32 = result_i32.as_slice::<i32>().unwrap();
        assert_eq!(result_data_i32, &[1, 2, 3, 4, 5]);

        // Test with f64
        let data_f64 = vec![1.0f64, 2.0, 3.0];
        let tensor_f64 = Tensor::from_vec(data_f64, [3]).unwrap();
        let result_f64 = tensor_f64.par_map::<f64>(|x| x.sqrt()).unwrap();
        let result_data_f64 = result_f64.as_slice::<f64>().unwrap();
        assert_eq!(
            result_data_f64,
            &[1.0, std::f64::consts::SQRT_2, 1.7320508075688772]
        );
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_map_empty_tensor() {
        let tensor = Tensor::from_vec(vec![0u8; 0], [0]).unwrap();
        let result = tensor.par_map::<u8>(|x| x + 1).unwrap();

        assert_eq!(result.shape(), tensor.shape());
        assert_eq!(result.dtype(), tensor.dtype());
        assert_eq!(result.numel(), 0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_map_comprehensive_contiguous() {
        // Test 1D contiguous tensor
        let data_1d = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor_1d = Tensor::from_vec(data_1d, [5]).unwrap();
        assert!(tensor_1d.is_contiguous());

        let result_1d = tensor_1d.par_map::<f32>(|x| x * 2.0).unwrap();
        let expected_1d = vec![2.0f32, 4.0, 6.0, 8.0, 10.0];
        assert_eq!(result_1d.as_slice::<f32>().unwrap(), &expected_1d);

        // Test 2D contiguous tensor
        let data_2d = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor_2d = Tensor::from_vec(data_2d, [2, 3]).unwrap();
        assert!(tensor_2d.is_contiguous());

        let result_2d = tensor_2d.par_map::<f32>(|x| x + 1.0).unwrap();
        let expected_2d = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(result_2d.as_slice::<f32>().unwrap(), &expected_2d);

        // Test 3D contiguous tensor
        let data_3d: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor_3d = Tensor::from_vec(data_3d, [2, 3, 4]).unwrap();
        assert!(tensor_3d.is_contiguous());

        let result_3d = tensor_3d.par_map::<f32>(|x| x.sqrt()).unwrap();
        let expected_3d: Vec<f32> = (0..24).map(|i| (i as f32).sqrt()).collect();
        let result_slice = result_3d.as_slice::<f32>().unwrap();

        for (i, (actual, expected)) in result_slice.iter().zip(expected_3d.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mismatch at index {}: actual={}, expected={}",
                i,
                actual,
                expected
            );
        }
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_map_comprehensive_non_contiguous() {
        // Test 1: Transposed 2D tensor
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let original = Tensor::from_vec(data, [2, 3]).unwrap();
        let transposed = original.permute([1, 0]).unwrap();
        assert!(!transposed.is_contiguous());

        // Verify the transposed data layout
        // Original: [[1, 2, 3], [4, 5, 6]]
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        let result = transposed.par_map::<f32>(|x| x * 2.0).unwrap();
        assert!(result.is_contiguous()); // Result should be contiguous

        // The result should be [2.0, 8.0, 4.0, 10.0, 6.0, 12.0] in contiguous layout
        let result_slice = result.as_slice::<f32>().unwrap();
        let expected = vec![2.0f32, 8.0, 4.0, 10.0, 6.0, 12.0];
        assert_eq!(result_slice, &expected);

        // Test 2: Sliced tensor (if slice operation exists)
        // For now, let's test with a different permutation
        let data_3d: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensor_3d = Tensor::from_vec(data_3d, [2, 2, 3]).unwrap();
        let permuted = tensor_3d.permute([2, 0, 1]).unwrap();
        assert!(!permuted.is_contiguous());

        let result_3d = permuted.par_map::<f32>(|x| x + 10.0).unwrap();
        assert!(result_3d.is_contiguous());

        // Verify the result has the correct shape and values
        assert_eq!(result_3d.shape(), permuted.shape());
        assert_eq!(result_3d.dtype(), permuted.dtype());

        // Test 3: Complex non-contiguous layout
        let data_4d: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tensor_4d = Tensor::from_vec(data_4d, [2, 2, 2, 2]).unwrap();
        let complex_permuted = tensor_4d.permute([3, 1, 0, 2]).unwrap();
        assert!(!complex_permuted.is_contiguous());

        let result_4d = complex_permuted.par_map::<f32>(|x| x * x).unwrap();
        assert!(result_4d.is_contiguous());
        assert_eq!(result_4d.shape(), complex_permuted.shape());
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_map_vs_map_consistency_detailed() {
        // Test 1D tensor
        let data_1d: Vec<f32> = (0..10).map(|i| (i as f32) * 0.1).collect();
        let tensor_1d = Tensor::from_vec(data_1d, [10]).unwrap();

        let map_result_1d = tensor_1d.map::<f32>(|x| x * 2.0).unwrap();
        let par_map_result_1d = tensor_1d.par_map::<f32>(|x| x * 2.0).unwrap();

        assert_eq!(map_result_1d.shape(), par_map_result_1d.shape());
        assert_eq!(map_result_1d.dtype(), par_map_result_1d.dtype());

        let map_data_1d = map_result_1d.as_slice::<f32>().unwrap();
        let par_map_data_1d = par_map_result_1d.as_slice::<f32>().unwrap();

        for (i, (a, b)) in map_data_1d.iter().zip(par_map_data_1d.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "1D mismatch at index {}: {} vs {}",
                i,
                a,
                b
            );
        }

        // Test 2D tensor
        let data_2d: Vec<f32> = (0..20).map(|i| (i as f32) * 0.1).collect();
        let tensor_2d = Tensor::from_vec(data_2d, [5, 4]).unwrap();

        let map_result_2d = tensor_2d.map::<f32>(|x| x * x).unwrap();
        let par_map_result_2d = tensor_2d.par_map::<f32>(|x| x * x).unwrap();

        assert_eq!(map_result_2d.shape(), par_map_result_2d.shape());
        assert_eq!(map_result_2d.dtype(), par_map_result_2d.dtype());

        let map_data_2d = map_result_2d.as_slice::<f32>().unwrap();
        let par_map_data_2d = par_map_result_2d.as_slice::<f32>().unwrap();

        for (i, (a, b)) in map_data_2d.iter().zip(par_map_data_2d.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "2D mismatch at index {}: {} vs {}",
                i,
                a,
                b
            );
        }

        // Test 3D tensor
        let data_3d: Vec<f32> = (0..27).map(|i| (i as f32) * 0.1).collect();
        let tensor_3d = Tensor::from_vec(data_3d, [3, 3, 3]).unwrap();

        let map_result_3d = tensor_3d.map::<f32>(|x| x.sin()).unwrap();
        let par_map_result_3d = tensor_3d.par_map::<f32>(|x| x.sin()).unwrap();

        assert_eq!(map_result_3d.shape(), par_map_result_3d.shape());
        assert_eq!(map_result_3d.dtype(), par_map_result_3d.dtype());

        let map_data_3d = map_result_3d.as_slice::<f32>().unwrap();
        let par_map_data_3d = par_map_result_3d.as_slice::<f32>().unwrap();

        for (i, (a, b)) in map_data_3d.iter().zip(par_map_data_3d.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "3D mismatch at index {}: {} vs {}",
                i,
                a,
                b
            );
        }

        // Test non-contiguous tensor (2D transposed)
        let data_2d_nc: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1).collect();
        let tensor_2d_nc = Tensor::from_vec(data_2d_nc, [3, 4]).unwrap();
        let transposed = tensor_2d_nc.permute([1, 0]).unwrap();
        assert!(!transposed.is_contiguous());

        let map_result_nc = transposed.map::<f32>(|x| x.abs()).unwrap();
        let par_map_result_nc = transposed.par_map::<f32>(|x| x.abs()).unwrap();

        assert_eq!(map_result_nc.shape(), par_map_result_nc.shape());
        assert_eq!(map_result_nc.dtype(), par_map_result_nc.dtype());

        let map_data_nc = map_result_nc.as_slice::<f32>().unwrap();
        let par_map_data_nc = par_map_result_nc.as_slice::<f32>().unwrap();

        for (i, (a, b)) in map_data_nc.iter().zip(par_map_data_nc.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Non-contiguous mismatch at index {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_map_different_data_types() {
        // Test i32
        let data_i32 = vec![1i32, -2, 3, -4, 5, -6];
        let tensor_i32 = Tensor::from_vec(data_i32, [2, 3]).unwrap();
        let result_i32 = tensor_i32.par_map::<i32>(|x| x.abs()).unwrap();
        let expected_i32 = vec![1i32, 2, 3, 4, 5, 6];
        assert_eq!(result_i32.as_slice::<i32>().unwrap(), &expected_i32);

        // Test f64
        let data_f64 = vec![1.0f64, 4.0, 9.0, 16.0];
        let tensor_f64 = Tensor::from_vec(data_f64, [2, 2]).unwrap();
        let result_f64 = tensor_f64.par_map::<f64>(|x| x.sqrt()).unwrap();
        let expected_f64 = [1.0f64, 2.0, 3.0, 4.0];
        let result_slice_f64 = result_f64.as_slice::<f64>().unwrap();
        for (i, (actual, expected)) in result_slice_f64.iter().zip(expected_f64.iter()).enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-10,
                "f64 mismatch at index {}: actual={}, expected={}",
                i,
                actual,
                expected
            );
        }

        // Test u8
        let data_u8 = vec![1u8, 2, 3, 4, 5];
        let tensor_u8 = Tensor::from_vec(data_u8, [5]).unwrap();
        let result_u8 = tensor_u8.par_map::<u8>(|x| x * 2).unwrap();
        let expected_u8 = vec![2u8, 4, 6, 8, 10];
        assert_eq!(result_u8.as_slice::<u8>().unwrap(), &expected_u8);
    }

    #[test]
    fn test_for_each_basic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, [5]).unwrap();

        // Test counting positive values
        let mut count = 0;
        tensor.for_each::<f32>(|x| {
            if *x > 0.0 {
                count += 1;
            }
        });
        assert_eq!(count, 5);

        // Test finding maximum value
        let mut max_val = f32::NEG_INFINITY;
        tensor.for_each::<f32>(|x| {
            if *x > max_val {
                max_val = *x;
            }
        });
        assert_eq!(max_val, 5.0);
    }

    #[test]
    fn test_for_each_contiguous() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3]).unwrap();
        assert!(tensor.is_contiguous());

        let mut sum = 0.0f32;
        tensor.for_each::<f32>(|x| {
            sum += *x;
        });
        assert_eq!(sum, 21.0);
    }

    #[test]
    fn test_for_each_non_contiguous() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3]).unwrap();
        let transposed = tensor.permute([1, 0]).unwrap();
        assert!(!transposed.is_contiguous());

        let mut sum = 0.0f32;
        transposed.for_each::<f32>(|x| {
            sum += *x;
        });
        assert_eq!(sum, 21.0); // Same sum as original
    }

    #[test]
    fn test_for_each_different_types() {
        // Test with i32
        let data_i32 = vec![1i32, -2, 3, -4, 5];
        let tensor_i32 = Tensor::from_vec(data_i32, [5]).unwrap();
        let mut count_positive = 0;
        tensor_i32.for_each::<i32>(|x| {
            if *x > 0 {
                count_positive += 1;
            }
        });
        assert_eq!(count_positive, 3);

        // Test with u8
        let data_u8 = vec![1u8, 2, 3, 4, 5];
        let tensor_u8 = Tensor::from_vec(data_u8, [5]).unwrap();
        let mut sum_u8 = 0u8;
        tensor_u8.for_each::<u8>(|x| {
            sum_u8 += *x;
        });
        assert_eq!(sum_u8, 15);
    }

    #[test]
    fn test_for_each_empty_tensor() {
        let tensor = Tensor::from_vec(vec![0u8; 0], [0]).unwrap();
        let mut count = 0;
        tensor.for_each::<u8>(|_| {
            count += 1;
        });
        assert_eq!(count, 0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_for_each_basic() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, [5]).unwrap();

        // Test counting values greater than 2.0
        let counter = std::sync::Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        tensor.par_for_each::<f32>(move |x| {
            if *x > 2.0 {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_for_each_contiguous() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3]).unwrap();
        assert!(tensor.is_contiguous());

        // Test counting values greater than 3.0
        let counter = std::sync::Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        tensor.par_for_each::<f32>(move |x| {
            if *x > 3.0 {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(counter.load(Ordering::Relaxed), 3); // 4.0, 5.0, 6.0
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_for_each_non_contiguous() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, [2, 3]).unwrap();
        let transposed = tensor.permute([1, 0]).unwrap();
        assert!(!transposed.is_contiguous());

        // Test counting even values
        let counter = std::sync::Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        transposed.par_for_each::<f32>(move |x| {
            if (*x as i32) % 2 == 0 {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(counter.load(Ordering::Relaxed), 3); // 2, 4, 6
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_for_each_vs_for_each_consistency() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, [1000]).unwrap();

        // Sequential count
        let mut seq_count = 0;
        tensor.for_each::<f32>(|x| {
            if *x > 500.0 {
                seq_count += 1;
            }
        });

        // Parallel count
        let par_counter = std::sync::Arc::new(AtomicUsize::new(0));
        let par_counter_clone = par_counter.clone();
        tensor.par_for_each::<f32>(move |x| {
            if *x > 500.0 {
                par_counter_clone.fetch_add(1, Ordering::Relaxed);
            }
        });
        let par_count = par_counter.load(Ordering::Relaxed);

        assert_eq!(seq_count, par_count);
        assert_eq!(seq_count, 499); // Values 501-999
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_for_each_different_types() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Test with i32
        let data_i32: Vec<i32> = (0..100).collect();
        let tensor_i32 = Tensor::from_vec(data_i32, [100]).unwrap();
        let counter_i32 = std::sync::Arc::new(AtomicUsize::new(0));
        let counter_i32_clone = counter_i32.clone();
        tensor_i32.par_for_each::<i32>(move |x| {
            if *x % 2 == 0 {
                counter_i32_clone.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(counter_i32.load(Ordering::Relaxed), 50);

        // Test with u8
        let data_u8: Vec<u8> = (0..50).map(|i| i as u8).collect();
        let tensor_u8 = Tensor::from_vec(data_u8, [50]).unwrap();
        let counter_u8 = std::sync::Arc::new(AtomicUsize::new(0));
        let counter_u8_clone = counter_u8.clone();
        tensor_u8.par_for_each::<u8>(move |x| {
            if *x < 25 {
                counter_u8_clone.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(counter_u8.load(Ordering::Relaxed), 25);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_for_each_empty_tensor() {
        let tensor = Tensor::from_vec(vec![0u8; 0], [0]).unwrap();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter_clone = counter.clone();
        tensor.par_for_each::<u8>(move |_| {
            counter_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
}
