//! High-performance dimension iterator optimized for tensor operations
//!
//! This module provides `DimIter`, a specialized iterator that efficiently iterates
//! over tensor dimensions while maintaining zero-cost abstractions and optimal performance.

use crate::{DType, Shape, StorageTrait, TensorBase, TensorView};
use std::marker::PhantomData;
use std::ptr::NonNull;

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

        // Optimized TensorView construction with minimal overhead
        Some(unsafe {
            TensorView::from_raw_parts(
                self.tensor.storage.as_storage(),
                NonNull::new_unchecked(self.original_ptr),
                self.slice_shape,
                self.slice_strides,
                offset_bytes,
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

    /// Optimized last() implementation
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
                NonNull::new_unchecked(self.original_ptr),
                self.slice_shape,
                self.slice_strides,
                offset_bytes,
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
                NonNull::new_unchecked(self.original_ptr),
                self.slice_shape,
                self.slice_strides,
                offset_bytes,
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
}
