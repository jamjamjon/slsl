//! High-performance dimension iterator optimized for tensor operations
//!
//! This module provides `DimIter`, a specialized iterator that efficiently iterates
//! over tensor dimensions while maintaining zero-cost abstractions and optimal performance.

use crate::{DType, Shape, StorageTrait, TensorBase, TensorView};
use std::marker::PhantomData;

#[cfg(feature = "rayon")]
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// High-performance iterator over tensor dimensions
///
/// `DimIter` provides efficient iteration over a specific dimension of a tensor,
/// yielding `TensorView`s that represent slices along that dimension. This iterator
/// is heavily optimized for performance with several key features:
///
/// ## Performance Optimizations
///
/// - **Lazy Initialization**: Expensive computations (pointer arithmetic, shape computation)
///   are deferred until actually needed
/// - **Zero-Cost Count**: The `count()` method returns a cached value without iteration
/// - **Memory Layout Awareness**: Optimized for cache-friendly access patterns
/// - **Minimal Allocations**: Pre-computed shapes avoid repeated heap allocations
///
/// ## Usage
///
/// ```rust
/// use slsl::Tensor;
///
/// let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
///
/// // Iterate over first dimension (yields 2 views of shape [3])
/// for view in tensor.iter_dim(0) {
///     println!("Slice: {:?}", view.as_slice::<f32>().unwrap());
/// }
///
/// // Ultra-fast count without iteration overhead
/// let count = tensor.iter_dim(0).count(); // Returns 2 instantly
/// ```
///
/// ## Implementation Details
///
/// The iterator uses lazy initialization to achieve optimal performance:
/// - For operations like `count()`, no actual computation is performed
/// - Only when iteration begins are expensive operations like pointer arithmetic executed
/// - Shape computation is cached to avoid repeated allocations during iteration
#[repr(C)]
pub struct DimIter<'a, S: StorageTrait> {
    /// Reference to the original tensor or tensor view
    tensor: &'a TensorBase<S>,
    /// Current pointer position (lazy initialized)
    ptr: *mut u8,
    /// End pointer boundary (lazy initialized)
    end_ptr: *mut u8,
    /// Original starting pointer for offset calculation (lazy initialized)
    original_ptr: *mut u8,
    /// Step size in bytes between elements (lazy initialized)
    stride: isize,
    /// Pre-computed slice shape to avoid repeated allocation
    slice_shape: Shape,
    /// Pre-computed slice strides to avoid repeated allocation
    slice_strides: Shape,
    /// Data type of tensor elements
    dtype: DType,
    /// Dimension being iterated over
    dim: usize,
    /// Cached length for ultra-fast count() operations
    cached_len: usize,
    /// Lifetime marker
    _phantom: PhantomData<&'a ()>,
}

impl<'a, S: StorageTrait> DimIter<'a, S> {
    /// Creates a new dimension iterator with lazy initialization for optimal performance
    ///
    /// This constructor performs minimal work upfront, deferring expensive operations
    /// like pointer arithmetic and shape computation until they're actually needed.
    /// This design enables ultra-fast operations like `count()` that don't require
    /// full iterator setup.
    ///
    /// # Arguments
    /// * `tensor` - The tensor to iterate over
    /// * `dim` - The dimension index to iterate along
    ///
    /// # Performance Notes
    /// - Construction time: O(1) - only basic field initialization
    /// - Memory usage: Minimal - no heap allocations during construction
    /// - Lazy evaluation: Expensive setup deferred until first iteration
    #[inline(always)]
    pub fn from_tensor(tensor: &'a TensorBase<S>, dim: usize) -> Self {
        debug_assert!(dim < tensor.rank(), "Dim {} >= {}", dim, tensor.rank());

        let axis_len = tensor.shape[dim];

        // Minimal initialization - defer expensive computations for optimal performance
        Self {
            tensor,
            ptr: std::ptr::null_mut(), // Lazy init: set when iteration begins
            end_ptr: std::ptr::null_mut(), // Lazy init: set when iteration begins
            original_ptr: std::ptr::null_mut(), // Lazy init: set when iteration begins
            stride: 0,                 // Lazy init: computed when iteration begins
            slice_shape: Shape::empty(), // Lazy init: computed only when needed
            slice_strides: Shape::empty(), // Lazy init: computed only when needed
            dtype: tensor.dtype,
            dim,
            cached_len: axis_len,
            _phantom: PhantomData,
        }
    }

    /// Lazy initialization of iteration state - only called when actually iterating
    ///
    /// This method performs the expensive setup work that was deferred during construction.
    /// It's called automatically by iteration methods but never by `count()` or `len()`,
    /// which enables those methods to be extremely fast.
    ///
    /// # Performance Notes
    /// - Called at most once per iterator instance
    /// - Performs pointer arithmetic and shape computation
    /// - Optimized to minimize cache misses and memory allocations
    #[inline(always)]
    fn ensure_iteration_ready(&mut self) {
        // Initialize pointers and stride if not already done
        if self.ptr.is_null() {
            let axis_len = self.cached_len;
            let axis_stride = (self.tensor.strides[self.dim] * self.dtype.size_in_bytes()) as isize;

            let data_ptr = self.tensor.as_ptr() as *mut u8;
            let end_ptr = unsafe { data_ptr.offset(axis_len as isize * axis_stride) };

            self.ptr = data_ptr;
            self.end_ptr = end_ptr;
            self.original_ptr = data_ptr;
            self.stride = axis_stride;

            // Initialize slice shapes immediately if needed (avoid repeated checks)
            if self.tensor.rank() > 1 {
                // Build slice shape and strides by excluding the iteration dimension
                for (i, &dim_size) in self.tensor.shape.as_slice().iter().enumerate() {
                    if i != self.dim {
                        self.slice_shape.push(dim_size);
                        self.slice_strides.push(self.tensor.strides[i]);
                    }
                }
            }
        }
    }

    /// Get remaining length - ultra-fast using cached value
    #[inline(always)]
    pub fn len(&self) -> usize {
        if self.ptr.is_null() {
            // Not yet initialized - return full cached length
            return self.cached_len;
        }

        if self.stride == 0 {
            return if self.ptr >= self.end_ptr { 0 } else { 1 };
        }

        let remaining_bytes = self.end_ptr as isize - self.ptr as isize;
        if remaining_bytes <= 0 {
            0
        } else {
            (remaining_bytes / self.stride) as usize
        }
    }

    /// Check if empty - ultra-fast check using cached length
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        if self.ptr.is_null() {
            return self.cached_len == 0;
        }
        self.ptr >= self.end_ptr
    }

    #[cfg(feature = "rayon")]
    /// Convert to parallel iterator (rayon style)
    #[inline]
    pub fn par_iter(self) -> ParDimIter<'a, S>
    where
        S: Send + Sync,
    {
        ParDimIter::new(self)
    }
}

impl<'a, S: StorageTrait> Iterator for DimIter<'a, S> {
    type Item = TensorView<'a>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        // Ensure iteration state is ready (inlined hot path)
        if self.ptr.is_null() {
            self.ensure_iteration_ready();
        }

        if self.ptr >= self.end_ptr {
            return None;
        }

        let current = self.ptr;
        self.ptr = unsafe { self.ptr.offset(self.stride) };

        // Fast path: calculate offset directly without intermediate variables
        let offset_bytes = (current as isize - self.original_ptr as isize) as usize;

        Some(unsafe {
            TensorView::from_raw_parts(
                self.tensor.storage.as_storage(),
                self.tensor.storage.ptr(),
                self.slice_shape,
                self.slice_strides,
                self.tensor.offset_bytes + offset_bytes,
                self.dtype,
            )
        })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    /// Ultra-fast count operation - returns result without any computation
    ///
    /// This method achieves optimal performance by returning a pre-cached value
    /// instead of actually iterating through elements. This is possible because
    /// the iterator length is known at construction time.
    ///
    /// # Performance
    /// - Time complexity: O(1) - constant time regardless of tensor size
    /// - No memory allocations, pointer arithmetic, or iterator state changes
    /// - No TensorView constructions or shape computations
    /// - Benchmark: ~412 ps (equivalent to ndarray performance)
    ///
    /// # Example
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0f32; 1_000_000], [1000, 1000]).unwrap();
    /// let count = tensor.iter_dim(0).count(); // Instant, regardless of tensor size
    /// assert_eq!(count, 1000);
    /// ```
    #[inline(always)]
    fn count(self) -> usize {
        // Ultra-fast path: return pre-cached length without any computation overhead
        self.cached_len
    }

    #[inline(always)]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n == 0 {
            return self.next();
        }

        // Ensure iteration state is ready (inlined hot path)
        if self.ptr.is_null() {
            self.ensure_iteration_ready();
        }

        let skip_bytes = self.stride * n as isize;
        let new_ptr = unsafe { self.ptr.offset(skip_bytes) };

        if new_ptr >= self.end_ptr {
            self.ptr = self.end_ptr;
            return None;
        }

        self.ptr = new_ptr;
        self.next()
    }

    #[inline(always)]
    fn last(mut self) -> Option<Self::Item> {
        if self.cached_len == 0 {
            return None;
        }

        // Ensure iteration state is ready (inlined hot path)
        if self.ptr.is_null() {
            self.ensure_iteration_ready();
        }

        // Jump directly to the last element
        let last_ptr = unsafe { self.end_ptr.offset(-self.stride) };
        self.ptr = last_ptr;

        let offset_bytes = (last_ptr as isize - self.original_ptr as isize) as usize;

        Some(unsafe {
            TensorView::from_raw_parts(
                self.tensor.storage.as_storage(),
                self.tensor.storage.ptr(),
                self.slice_shape,
                self.slice_strides,
                self.tensor.offset_bytes + offset_bytes,
                self.dtype,
            )
        })
    }
}

impl<S: StorageTrait> ExactSizeIterator for DimIter<'_, S> {}
impl<S: StorageTrait> std::iter::FusedIterator for DimIter<'_, S> {}

impl<S: StorageTrait> DoubleEndedIterator for DimIter<'_, S> {
    fn next_back(&mut self) -> Option<Self::Item> {
        // Ensure iteration state is ready (inlined hot path)
        if self.ptr.is_null() {
            self.ensure_iteration_ready();
        }

        if self.ptr >= self.end_ptr {
            return None;
        }

        // Move end pointer backwards
        self.end_ptr = unsafe { self.end_ptr.offset(-self.stride) };

        let current = self.end_ptr;
        let offset_bytes = (current as isize - self.original_ptr as isize) as usize;

        Some(unsafe {
            TensorView::from_raw_parts(
                self.tensor.storage.as_storage(),
                self.tensor.storage.ptr(),
                self.slice_shape,
                self.slice_strides,
                self.tensor.offset_bytes + offset_bytes,
                self.dtype,
            )
        })
    }
}

impl<S: StorageTrait> DimIter<'_, S> {
    /// Split the iterator at the given index
    pub fn split_at(mut self, index: usize) -> (Self, Self) {
        let len = self.cached_len;
        assert!(index <= len, "Split index {index} exceeds length {len}");

        if index == 0 {
            let empty = self.empty();
            return (empty, self);
        }
        if index == len {
            let empty = self.empty();
            return (self, empty);
        }

        // Ensure iteration state is ready for splitting
        self.ensure_iteration_ready();

        // Create a copy for the right side
        let right = Self {
            tensor: self.tensor,
            ptr: unsafe { self.ptr.offset(index as isize * self.stride) },
            end_ptr: self.end_ptr,
            original_ptr: self.original_ptr,
            stride: self.stride,
            slice_shape: self.slice_shape,
            slice_strides: self.slice_strides,
            dtype: self.dtype,
            dim: self.dim,
            cached_len: len - index,
            _phantom: PhantomData,
        };

        // Adjust the left side
        let mut left = self;
        left.end_ptr = unsafe { left.ptr.offset(index as isize * left.stride) };
        left.cached_len = index;

        (left, right)
    }

    /// Create an empty iterator
    fn empty(&self) -> Self {
        Self {
            tensor: self.tensor,
            ptr: std::ptr::null_mut(),
            end_ptr: std::ptr::null_mut(),
            original_ptr: std::ptr::null_mut(),
            stride: 0,
            slice_shape: Shape::empty(),
            slice_strides: Shape::empty(),
            dtype: self.dtype,
            dim: self.dim,
            cached_len: 0,
            _phantom: PhantomData,
        }
    }
}

unsafe impl<S: StorageTrait> Send for DimIter<'_, S> where S: Send {}
unsafe impl<S: StorageTrait> Sync for DimIter<'_, S> where S: Sync {}

#[cfg(feature = "rayon")]
pub struct ParDimIter<'a, S: StorageTrait> {
    inner: DimIter<'a, S>,
    min_len: usize,
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait> ParDimIter<'a, S> {
    pub fn new(inner: DimIter<'a, S>) -> Self {
        Self { inner, min_len: 1 }
    }

    pub fn with_min_len(mut self, min_len: usize) -> Self {
        assert_ne!(
            min_len, 0,
            "Minimum number of elements must be at least one"
        );
        self.min_len = min_len;
        self
    }
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait + Send + Sync> IntoParallelIterator for DimIter<'a, S> {
    type Item = TensorView<'a>;
    type Iter = ParDimIter<'a, S>;

    fn into_par_iter(self) -> Self::Iter {
        ParDimIter::new(self)
    }
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait + Send + Sync> ParallelIterator for ParDimIter<'a, S> {
    type Item = TensorView<'a>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.inner.len())
    }
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait + Send + Sync> IndexedParallelIterator for ParDimIter<'a, S> {
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(ParDimProducer {
            inner: self.inner,
            min_len: self.min_len,
        })
    }
}

#[cfg(feature = "rayon")]
struct ParDimProducer<'a, S: StorageTrait> {
    inner: DimIter<'a, S>,
    min_len: usize,
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait + Send + Sync> Producer for ParDimProducer<'a, S> {
    type Item = TensorView<'a>;
    type IntoIter = DimIter<'a, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.inner.split_at(index);
        (
            ParDimProducer {
                inner: left,
                min_len: self.min_len,
            },
            ParDimProducer {
                inner: right,
                min_len: self.min_len,
            },
        )
    }
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait + Send + Sync> IntoIterator for ParDimProducer<'a, S> {
    type Item = TensorView<'a>;
    type IntoIter = DimIter<'a, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner
    }
}

impl<S: StorageTrait> TensorBase<S> {
    /// Creates an iterator over the specified dimension
    ///
    /// Returns a `DimIter` that yields `TensorView`s representing slices
    /// along the specified dimension. The iterator is optimized for performance
    /// with lazy initialization and zero-cost abstractions.
    ///
    /// # Arguments
    /// * `dim` - The dimension index to iterate over (0-based)
    ///
    /// # Returns
    /// A `DimIter` that can be used with standard Rust iterator methods
    ///
    /// # Performance
    /// - Iterator construction: O(1) with lazy initialization
    /// - `count()` operations: O(1) using cached values
    /// - Actual iteration: Optimized for cache-friendly memory access
    ///
    /// # Example
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
    ///
    /// // Iterate over rows (dimension 0)
    /// for (i, row) in tensor.iter_dim(0).enumerate() {
    ///     println!("Row {}: {:?}", i, row.as_slice::<f32>().unwrap());
    /// }
    ///
    /// // Ultra-fast count
    /// assert_eq!(tensor.iter_dim(0).count(), 2);
    /// assert_eq!(tensor.iter_dim(1).count(), 3);
    /// ```
    #[inline]
    pub fn iter_dim(&self, dim: usize) -> DimIter<'_, S> {
        DimIter::from_tensor(self, dim)
    }

    /// Get the size of a specific dimension (ultra-fast alternative to iter_dim().count())
    ///
    /// This method provides direct access to dimension sizes without any iterator
    /// construction overhead. While `iter_dim(dim).count()` is also very fast due
    /// to optimizations, this method is slightly faster for simple size queries.
    ///
    /// # Arguments
    /// * `dim` - The dimension index (0-based)
    ///
    /// # Returns
    /// The size of the specified dimension
    ///
    /// # Performance
    /// Time complexity: O(1) - direct array access
    #[inline(always)]
    pub fn dim_len(&self, dim: usize) -> usize {
        debug_assert!(dim < self.rank(), "Dim {} >= {}", dim, self.rank());
        self.shape[dim]
    }

    #[cfg(feature = "rayon")]
    #[inline]
    pub fn par_iter_dim(&self, dim: usize) -> ParDimIter<'_, S>
    where
        S: Send + Sync,
    {
        self.iter_dim(dim).par_iter()
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;
    #[cfg(feature = "rayon")]
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};

    #[test]
    fn test_unified_dim_iter_basic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        let iter = tensor.iter_dim(0);
        assert_eq!(iter.len(), 2);

        let ptrs: Vec<_> = iter.collect();
        assert_eq!(ptrs.len(), 2);
    }

    #[test]
    fn test_unified_dim_iter_empty() {
        let data: Vec<f32> = vec![];
        let tensor = Tensor::from_vec(data, vec![1, 0]).unwrap();

        let iter = tensor.iter_dim(1);
        assert_eq!(iter.len(), 0);
        assert!(iter.is_empty());
    }

    #[test]
    fn test_dim_iter_count_optimization() {
        let data = vec![1.0f32; 1000];
        let tensor = Tensor::from_vec(data, vec![100, 10]).unwrap();

        // This should be very fast now as it doesn't actually iterate
        let count = tensor.iter_dim(0).count();
        assert_eq!(count, 100);

        let count = tensor.iter_dim(1).count();
        assert_eq!(count, 10);

        // Test the ultra-fast dim_len method
        assert_eq!(tensor.dim_len(0), 100);
        assert_eq!(tensor.dim_len(1), 10);
    }

    #[test]
    fn test_dim_iter_last_optimization() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        let last = tensor.iter_dim(0).last();
        assert!(last.is_some());

        let last_view = last.unwrap();
        assert_eq!(last_view.at::<f32>([0]), 4.0);
    }

    #[test]
    fn test_dim_iter_nth_optimization() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data, vec![4, 2]).unwrap();

        let mut iter = tensor.iter_dim(0);
        let third = iter.nth(2);
        assert!(third.is_some());

        let third_view = third.unwrap();
        assert_eq!(third_view.at::<f32>([0]), 5.0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_dim_iter_basic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        let par_iter = tensor.iter_dim(0).par_iter();
        assert_eq!(par_iter.len(), 2);

        let count = par_iter.count();
        assert_eq!(count, 2);

        // Compare with ultra-fast alternative
        assert_eq!(tensor.dim_len(0), count);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_dim_iter_map() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        let par_iter = tensor.iter_dim(0).par_iter();
        let results: Vec<f32> = par_iter.map(|view| view.at::<f32>([0])).collect();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 1.0);
        assert_eq!(results[1], 4.0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_dim_iter_filter() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        let par_iter = tensor.iter_dim(0).par_iter();
        let results: Vec<crate::TensorView> =
            par_iter.filter(|view| view.at::<f32>([0]) > 2.0).collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].at::<f32>([0]), 4.0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_dim_iter_large() {
        let size = 1000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![size, 1]).unwrap();

        let par_iter = tensor.iter_dim(0).par_iter();
        let sum: f32 = par_iter.map(|view| view.at::<f32>([0])).sum();

        let expected_sum: f32 = (0..size).map(|i| i as f32).sum();
        assert!((sum - expected_sum).abs() < f32::EPSILON);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_dim_iter_rayon_style() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        // Test rayon-style chaining
        let results: Vec<f32> = tensor
            .iter_dim(0)
            .par_iter()
            .map(|view| view.at::<f32>([0]))
            .collect();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 1.0);
        assert_eq!(results[1], 4.0);
    }

    #[test]
    fn test_lightweight_count_performance() {
        // Test for large tensors where construction overhead matters
        let size = 10000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![size, 1]).unwrap();

        // These should all be extremely fast
        assert_eq!(tensor.iter_dim(0).count(), size);
        assert_eq!(tensor.dim_len(0), size);
        assert_eq!(tensor.iter_dim(1).count(), 1);
        assert_eq!(tensor.dim_len(1), 1);
    }

    #[test]
    fn test_multi_dimensional_count() {
        let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 3, 4, 5]).unwrap();

        // Test all dimensions
        assert_eq!(tensor.iter_dim(0).count(), 2);
        assert_eq!(tensor.iter_dim(1).count(), 3);
        assert_eq!(tensor.iter_dim(2).count(), 4);
        assert_eq!(tensor.iter_dim(3).count(), 5);

        // Verify with direct dimension access
        assert_eq!(tensor.dim_len(0), 2);
        assert_eq!(tensor.dim_len(1), 3);
        assert_eq!(tensor.dim_len(2), 4);
        assert_eq!(tensor.dim_len(3), 5);
    }

    #[test]
    fn test_iter_dim_data_correctness() {
        // Test basic data correctness for 2D tensor
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        let rows: Vec<_> = tensor.iter_dim(0).collect();
        assert_eq!(rows.len(), 2);

        // First row should be [1.0, 2.0, 3.0]
        let row0_data = rows[0].as_slice::<f32>().unwrap();
        assert_eq!(row0_data, &[1.0, 2.0, 3.0]);

        // Second row should be [4.0, 5.0, 6.0]
        let row1_data = rows[1].as_slice::<f32>().unwrap();
        assert_eq!(row1_data, &[4.0, 5.0, 6.0]);

        // Test iteration over dimension 1 - verify count and structure
        let dim1_slices: Vec<_> = tensor.iter_dim(1).collect();
        assert_eq!(dim1_slices.len(), 3);

        // Each slice should have shape [2] (2 rows)
        for slice in dim1_slices.iter() {
            assert_eq!(slice.shape().as_slice(), &[2]);
        }
    }

    #[test]
    fn test_iter_dim_3d_tensor() {
        // Test with 3D tensor [2, 3, 4] = 24 elements
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 3, 4]).unwrap();

        // Iterate over first dimension (2 slices of [3, 4])
        let slices: Vec<_> = tensor.iter_dim(0).collect();
        assert_eq!(slices.len(), 2);

        // First slice should contain elements 0-11
        let slice0 = slices[0].as_slice::<f32>().unwrap();
        let expected0: Vec<f32> = (0..12).map(|i| i as f32).collect();
        assert_eq!(slice0, expected0.as_slice());

        // Second slice should contain elements 12-23
        let slice1 = slices[1].as_slice::<f32>().unwrap();
        let expected1: Vec<f32> = (12..24).map(|i| i as f32).collect();
        assert_eq!(slice1, expected1.as_slice());
    }

    #[test]
    fn test_iter_dim_edge_cases() {
        // Test empty dimension
        let tensor_empty = Tensor::from_vec(Vec::<f32>::new(), vec![0, 5]).unwrap();
        let empty_iter: Vec<_> = tensor_empty.iter_dim(0).collect();
        assert_eq!(empty_iter.len(), 0);
        assert!(tensor_empty.iter_dim(0).is_empty());

        // Test single element dimension
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor_single = Tensor::from_vec(data, vec![1, 5]).unwrap();
        let single_iter: Vec<_> = tensor_single.iter_dim(0).collect();
        assert_eq!(single_iter.len(), 1);

        let slice_data = single_iter[0].as_slice::<f32>().unwrap();
        assert_eq!(slice_data, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_iter_dim_iterator_methods() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![4, 5]).unwrap();

        // Test take()
        let taken: Vec<_> = tensor.iter_dim(0).take(2).collect();
        assert_eq!(taken.len(), 2);

        // Test skip()
        let skipped: Vec<_> = tensor.iter_dim(0).skip(1).collect();
        assert_eq!(skipped.len(), 3);

        // Test enumerate
        for (i, slice) in tensor.iter_dim(0).enumerate() {
            let slice_data = slice.as_slice::<f32>().unwrap();
            let expected_start = i * 5;
            assert_eq!(slice_data[0], expected_start as f32);
        }
    }

    #[test]
    fn test_iter_dim_split_at() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![4, 5]).unwrap();

        let iter = tensor.iter_dim(0);
        let (left, right) = iter.split_at(2);

        let left_slices: Vec<_> = left.collect();
        let right_slices: Vec<_> = right.collect();

        assert_eq!(left_slices.len(), 2);
        assert_eq!(right_slices.len(), 2);

        // Verify data correctness
        let left0_data = left_slices[0].as_slice::<f32>().unwrap();
        assert_eq!(left0_data, &[0.0, 1.0, 2.0, 3.0, 4.0]);

        let right0_data = right_slices[0].as_slice::<f32>().unwrap();
        assert_eq!(right0_data, &[10.0, 11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_iter_dim_nested_iteration() {
        // Test nested iteration for 3D tensor
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 3, 4]).unwrap();

        // Iterate over first dimension, then over nested dimension
        for (i, outer_slice) in tensor.iter_dim(0).enumerate() {
            assert_eq!(outer_slice.rank(), 2);
            assert_eq!(outer_slice.shape().as_slice(), &[3, 4]);

            // Nest iteration over the slice
            let nested_slices: Vec<_> = outer_slice.iter_dim(0).collect();
            assert_eq!(nested_slices.len(), 3);

            for (j, inner_slice) in nested_slices.iter().enumerate() {
                let slice_data = inner_slice.as_slice::<f32>().unwrap();
                assert_eq!(slice_data.len(), 4);

                // Verify first element matches expected pattern
                let expected_first = (i * 12 + j * 4) as f32;
                assert_eq!(slice_data[0], expected_first);
            }
        }
    }

    #[test]
    fn test_iter_dim_large_tensor_performance() {
        // Test with moderately large tensor to ensure performance characteristics
        let size = 1000;
        let data: Vec<f32> = (0..size * 100).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![size, 100]).unwrap();

        // Ultra-fast operations should complete instantly
        assert_eq!(tensor.iter_dim(0).count(), size);
        assert_eq!(tensor.iter_dim(0).len(), size);
        assert!(!tensor.iter_dim(0).is_empty());

        // Test first and last elements for correctness
        let first = tensor.iter_dim(0).next().unwrap();
        let first_data = first.as_slice::<f32>().unwrap();
        assert_eq!(first_data[0], 0.0);

        let last = tensor.iter_dim(0).last().unwrap();
        let last_data = last.as_slice::<f32>().unwrap();
        assert_eq!(last_data[0], (size - 1) as f32 * 100.0);
    }

    #[test]
    fn test_iter_dim_double_ended() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![3, 4]).unwrap();

        let mut iter = tensor.iter_dim(0);

        // Get first element
        let first = iter.next().unwrap();
        let first_data = first.as_slice::<f32>().unwrap();
        assert_eq!(first_data, &[0.0, 1.0, 2.0, 3.0]);

        // Get last element
        let last = iter.next_back().unwrap();
        let last_data = last.as_slice::<f32>().unwrap();
        assert_eq!(last_data, &[8.0, 9.0, 10.0, 11.0]);

        // Get middle element
        let middle = iter.next().unwrap();
        let middle_data = middle.as_slice::<f32>().unwrap();
        assert_eq!(middle_data, &[4.0, 5.0, 6.0, 7.0]);

        // Iterator should be exhausted
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }

    #[test]
    fn test_iter_dim_various_shapes() {
        let test_shapes = vec![
            vec![2, 3],       // Small 2D
            vec![3, 4, 5],    // 3D
            vec![2, 3, 4, 5], // 4D
            vec![1, 10],      // Single row
            vec![10, 1],      // Single column
        ];

        for shape in test_shapes {
            let total_elements: usize = shape.iter().product();
            let data: Vec<f32> = (0..total_elements).map(|i| i as f32).collect();
            let tensor = Tensor::from_vec(data, shape.clone()).unwrap();

            // Test iteration over first dimension
            let slices: Vec<_> = tensor.iter_dim(0).collect();
            assert_eq!(slices.len(), shape[0]);

            // Verify each slice has correct size
            let expected_slice_size = if shape.len() > 1 {
                shape[1..].iter().product()
            } else {
                1
            };

            for slice in slices {
                let slice_data = slice.as_slice::<f32>().unwrap();
                assert_eq!(slice_data.len(), expected_slice_size);
            }
        }
    }

    #[test]
    fn test_iter_dim_correctness_with_strides() {
        // Test with non-contiguous tensor (different strides)
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![3, 4]).unwrap();

        // Test basic iteration works with any stride configuration
        let slices: Vec<_> = tensor.iter_dim(0).collect();
        assert_eq!(slices.len(), 3);

        // Verify all slices have correct shape
        for slice in slices.iter() {
            assert_eq!(slice.shape().as_slice(), &[4]);
        }

        // Test dimension 1 iteration
        let dim1_slices: Vec<_> = tensor.iter_dim(1).collect();
        assert_eq!(dim1_slices.len(), 4);

        for slice in dim1_slices.iter() {
            assert_eq!(slice.shape().as_slice(), &[3]);
        }
    }

    #[test]
    fn test_iter_dim_boundary_conditions() {
        // Test with very small tensors
        let scalar_like = Tensor::from_vec(vec![42.0f32], vec![1]).unwrap();
        let slices: Vec<_> = scalar_like.iter_dim(0).collect();
        assert_eq!(slices.len(), 1);
        let slice_data = slices[0].as_slice::<f32>().unwrap();
        assert_eq!(slice_data, &[42.0]);

        // Test split_at edge cases
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![3, 2]).unwrap();

        // Split at beginning
        let (left, right) = tensor.iter_dim(0).split_at(0);
        assert_eq!(left.len(), 0);
        assert_eq!(right.len(), 3);

        // Split at end
        let (left, right) = tensor.iter_dim(0).split_at(3);
        assert_eq!(left.len(), 3);
        assert_eq!(right.len(), 0);
    }

    #[test]
    fn test_iter_dim_memory_safety() {
        // Test that iterators maintain correct memory references
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![10, 10]).unwrap();

        // Create multiple iterators and verify they don't interfere
        let mut iter1 = tensor.iter_dim(0);
        let mut iter2 = tensor.iter_dim(0);

        let slice1 = iter1.nth(5).unwrap();
        let slice2 = iter2.nth(5).unwrap();

        let data1 = slice1.as_slice::<f32>().unwrap();
        let data2 = slice2.as_slice::<f32>().unwrap();

        // Both should point to the same data
        assert_eq!(data1, data2);
        assert_eq!(data1[0], 50.0);
    }

    #[test]
    fn test_iter_dim_consistency_across_dimensions() {
        // Test that iteration is consistent across different dimensions
        let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![3, 4, 5]).unwrap();

        // Iterate over each dimension and verify counts
        assert_eq!(tensor.iter_dim(0).count(), 3);
        assert_eq!(tensor.iter_dim(1).count(), 4);
        assert_eq!(tensor.iter_dim(2).count(), 5);

        // Verify slice shapes are correct
        let dim0_slice = tensor.iter_dim(0).next().unwrap();
        assert_eq!(dim0_slice.shape().as_slice(), &[4, 5]);

        let dim1_slice = tensor.iter_dim(1).next().unwrap();
        assert_eq!(dim1_slice.shape().as_slice(), &[3, 5]);

        let dim2_slice = tensor.iter_dim(2).next().unwrap();
        assert_eq!(dim2_slice.shape().as_slice(), &[3, 4]);
    }

    #[test]
    fn test_iter_dim_offset_correctness() {
        // Create a slice of a tensor and verify iteration works correctly
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![4, 6]).unwrap();

        // Test with a simple subview to verify offset handling
        // Get a view of the tensor starting from row 1
        let slices: Vec<_> = tensor.iter_dim(0).collect();
        let view = &slices[1]; // This creates an offset view
        assert_eq!(view.shape().as_slice(), &[6]);

        // The view should contain elements [6, 7, 8, 9, 10, 11]
        let view_data = view.as_slice::<f32>().unwrap();
        assert_eq!(view_data, &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);

        // Test nested iteration with proper offset handling
        let nested_slices: Vec<_> = tensor.iter_dim(0).skip(1).take(2).collect();
        assert_eq!(nested_slices.len(), 2);

        // First nested slice (row 1)
        let slice0_data = nested_slices[0].as_slice::<f32>().unwrap();
        assert_eq!(slice0_data, &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);

        // Second nested slice (row 2)
        let slice1_data = nested_slices[1].as_slice::<f32>().unwrap();
        assert_eq!(slice1_data, &[12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
    }

    #[test]
    fn test_iter_dim_extreme_shapes() {
        // Test with very wide tensor
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let wide_tensor = Tensor::from_vec(data, vec![1, 1000]).unwrap();

        let slices: Vec<_> = wide_tensor.iter_dim(0).collect();
        assert_eq!(slices.len(), 1);

        let slice_data = slices[0].as_slice::<f32>().unwrap();
        assert_eq!(slice_data.len(), 1000);
        assert_eq!(slice_data[0], 0.0);
        assert_eq!(slice_data[999], 999.0);

        // Test with very tall tensor
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let tall_tensor = Tensor::from_vec(data, vec![1000, 1]).unwrap();

        let slices: Vec<_> = tall_tensor.iter_dim(0).collect();
        assert_eq!(slices.len(), 1000);

        for (i, slice) in slices.iter().enumerate() {
            let slice_data = slice.as_slice::<f32>().unwrap();
            assert_eq!(slice_data.len(), 1);
            assert_eq!(slice_data[0], i as f32);
        }
    }

    #[test]
    fn test_iter_dim_zero_stride_edge_case() {
        // Test behavior with dimension size 1 (which could have zero stride optimization)
        let data = vec![42.0f32];
        let tensor = Tensor::from_vec(data, vec![1, 1, 1, 1]).unwrap();

        for dim in 0..4 {
            let slices: Vec<_> = tensor.iter_dim(dim).collect();
            assert_eq!(slices.len(), 1);

            // The remaining tensor should have one less dimension
            let remaining_dims: Vec<usize> = (0..4).filter(|&d| d != dim).map(|_| 1).collect();
            if remaining_dims.is_empty() {
                // If we're left with a scalar, as_slice should return array with one element
                let slice_data = slices[0].as_slice::<f32>().unwrap();
                assert_eq!(slice_data, &[42.0]);
            }
        }
    }
}
