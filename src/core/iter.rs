#[cfg(feature = "rayon")]
use rayon::iter::{
    plumbing::{bridge, Consumer, Producer, ProducerCallback},
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
};
use std::marker::PhantomData;

use crate::{ArrayUsize8, DType, StorageTrait, TensorBase};

/// Thread-safe element reference for parallel iteration
#[derive(Clone, Copy)]
pub struct TensorIterElement {
    /// Indices of the element
    pub indices: ArrayUsize8,
    /// Offset in bytes from the tensor's base pointer
    pub offset: usize,
}

impl TensorIterElement {
    /// Get a pointer to the element data given the tensor's base pointer
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The base_ptr is valid and points to the tensor's data
    /// - The offset is within the tensor's bounds
    /// - The returned pointer is used safely
    #[inline]
    pub unsafe fn as_ptr(&self, base_ptr: *const u8) -> *const u8 {
        base_ptr.add(self.offset)
    }

    /// Get a mutable pointer to the element data given the tensor's base pointer
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The base_ptr is valid and points to the tensor's data
    /// - The offset is within the tensor's bounds
    /// - The returned pointer is used safely
    /// - No other references to this data exist
    #[inline]
    pub unsafe fn as_mut_ptr(&self, base_ptr: *mut u8) -> *mut u8 {
        base_ptr.add(self.offset)
    }
}

/// Iterator over all elements in a tensor
#[repr(C)]
pub struct TensorIter<'a, S: StorageTrait> {
    /// Reference to the original tensor
    tensor: &'a TensorBase<S>,
    /// Current pointer position
    ptr: *mut u8,
    /// End pointer boundary
    end_ptr: *mut u8,
    /// Original starting pointer (for offset calculation)
    original_ptr: *mut u8,
    /// Current indices for multi-dimensional access
    current_indices: ArrayUsize8,
    /// Data type
    dtype: DType,
    /// Maximum elements to yield (used only for split iterators)
    max_elements: Option<usize>,
    /// Elements yielded so far (used only for split iterators)
    elements_yielded: usize,
    /// Lifetime marker
    _phantom: PhantomData<&'a ()>,
}

// Safety: TensorIterElement is Send because it only contains indices and offset
unsafe impl Send for TensorIterElement {}
unsafe impl Sync for TensorIterElement {}

impl<'a, S: StorageTrait> TensorIter<'a, S> {
    /// Create iterator from Tensor
    #[inline]
    pub fn from_tensor(tensor: &'a TensorBase<S>) -> Self {
        let data_ptr = tensor.as_ptr() as *mut u8;
        let numel = tensor.numel();
        let element_size = tensor.dtype.size_in_bytes();
        let end_ptr = unsafe { data_ptr.add(numel * element_size) };

        // Initialize current indices to zeros
        let mut current_indices = ArrayUsize8::empty();
        for _ in 0..tensor.rank() {
            current_indices.push(0);
        }

        Self {
            tensor,
            ptr: data_ptr,
            end_ptr,
            original_ptr: data_ptr,
            current_indices,
            dtype: tensor.dtype,
            max_elements: None,
            elements_yielded: 0,
            _phantom: PhantomData,
        }
    }

    /// Get remaining length
    #[inline(always)]
    pub fn len(&self) -> usize {
        // For split iterators, use the limited count
        if let Some(max) = self.max_elements {
            return max.saturating_sub(self.elements_yielded);
        }

        // For scalar tensors
        if self.tensor.rank() == 0 {
            return if self.ptr >= self.end_ptr { 0 } else { 1 };
        }

        // Calculate remaining elements based on current indices
        if self.current_indices[0] >= self.tensor.shape[0] {
            return 0;
        }

        // Calculate remaining elements efficiently using mathematical formula
        let mut remaining = 0;
        let rank = self.tensor.rank();

        // Calculate the linear index of current position
        let mut current_linear_index = 0;
        let mut multiplier = 1;

        for i in (0..rank).rev() {
            current_linear_index += self.current_indices[i] * multiplier;
            multiplier *= self.tensor.shape[i];
        }

        // Total elements minus current position gives remaining elements
        let total_elements = self.tensor.numel();
        if current_linear_index < total_elements {
            remaining = total_elements - current_linear_index;
        }

        remaining
    }

    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.ptr >= self.end_ptr
    }

    /// Get current indices
    #[inline]
    pub fn current_indices(&self) -> &ArrayUsize8 {
        &self.current_indices
    }

    #[cfg(feature = "rayon")]
    /// Convert to parallel iterator (rayon style)
    #[inline]
    pub fn par_iter(self) -> ParTensorIter<'a, S>
    where
        S: Send + Sync,
    {
        ParTensorIter::new(self)
    }
}

impl<S: StorageTrait> Iterator for TensorIter<'_, S> {
    type Item = TensorIterElement;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've reached the maximum elements for split iterators
        if let Some(max) = self.max_elements {
            if self.elements_yielded >= max {
                return None;
            }
        }

        // For scalar tensors
        if self.tensor.rank() == 0 {
            if self.ptr >= self.end_ptr {
                return None;
            }
            self.ptr = self.end_ptr; // Mark as exhausted
            self.elements_yielded += 1;
            return Some(TensorIterElement {
                indices: self.current_indices,
                offset: 0,
            });
        }

        // Check if we've exhausted all elements by checking if first dimension overflowed
        if self.tensor.rank() > 0 && self.current_indices[0] >= self.tensor.shape[0] {
            return None;
        }

        let current_indices = self.current_indices;

        // Calculate offset using strides for non-contiguous tensors
        let mut offset_bytes = 0;
        let element_size = self.dtype.size_in_bytes();
        for (dim_idx, &index) in current_indices
            .as_slice()
            .iter()
            .enumerate()
            .take(self.tensor.rank())
        {
            // Only add to offset if stride is non-zero (for broadcasting)
            if self.tensor.strides[dim_idx] != 0 {
                // Strides are in elements, convert to bytes
                offset_bytes += index * self.tensor.strides[dim_idx] * element_size;
            }
        }

        let offset = offset_bytes;

        // Update indices for next iteration
        self.update_indices();
        self.elements_yielded += 1;

        Some(TensorIterElement {
            indices: current_indices,
            offset,
        })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline(always)]
    fn count(self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let element_size = self.dtype.size_in_bytes();
        let skip_bytes = element_size * n;
        self.ptr = unsafe { self.ptr.add(skip_bytes) };

        // Update indices accordingly
        for _ in 0..n {
            self.update_indices();
        }

        self.next()
    }
}

impl<S: StorageTrait> ExactSizeIterator for TensorIter<'_, S> {}
impl<S: StorageTrait> std::iter::FusedIterator for TensorIter<'_, S> {}

// Implement DoubleEndedIterator for rayon compatibility
impl<S: StorageTrait> DoubleEndedIterator for TensorIter<'_, S> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.ptr >= self.end_ptr {
            return None;
        }

        // Move end pointer backwards
        let element_size = self.dtype.size_in_bytes();
        self.end_ptr = unsafe { self.end_ptr.sub(element_size) };

        let current_ptr = self.end_ptr;
        let offset = (current_ptr as isize - self.original_ptr as isize) as usize;

        // Calculate indices for this position
        let element_index = offset / element_size;
        let indices = self.compute_indices_at(element_index);

        Some(TensorIterElement { indices, offset })
    }
}

impl<S: StorageTrait> TensorIter<'_, S> {
    /// Update current indices to point to the next element
    #[inline]
    fn update_indices(&mut self) {
        if self.tensor.rank() == 0 {
            return;
        }

        // Start from the rightmost dimension
        for i in (0..self.tensor.rank()).rev() {
            self.current_indices[i] += 1;
            if self.current_indices[i] < self.tensor.shape[i] {
                return; // No carry needed
            }
            // Carry to next dimension
            self.current_indices[i] = 0;
        }

        // If we reach here, all dimensions have overflowed
        // Set the first dimension to shape[0] to signal completion
        if self.tensor.rank() > 0 {
            self.current_indices[0] = self.tensor.shape[0];
        }
    }

    /// Split the iterator at the given index
    pub fn split_at(self, index: usize) -> (Self, Self) {
        let len = self.len();
        assert!(index <= len, "Split index {index} exceeds length {len}");

        if index == 0 {
            let empty = self.empty();
            return (empty, self);
        }
        if index == len {
            let empty = self.empty();
            return (self, empty);
        }

        // Create left iterator with limited elements
        let mut left = self.clone();
        left.max_elements = Some(index);
        left.elements_yielded = 0;

        // Create right iterator by skipping first `index` elements
        let mut right = self;
        for _ in 0..index {
            if right.next().is_none() {
                break;
            }
        }
        // Reset elements_yielded for right iterator
        right.elements_yielded = 0;

        (left, right)
    }

    /// Compute indices at a given offset
    fn compute_indices_at(&self, offset: usize) -> ArrayUsize8 {
        let mut indices = ArrayUsize8::empty();
        let mut remaining = offset;

        // Convert from rightmost (fastest changing) to leftmost dimension
        for i in (0..self.tensor.rank()).rev() {
            let dim_size = self.tensor.shape[i];
            indices.push(remaining % dim_size);
            remaining /= dim_size;
        }

        // Reverse to get correct order (leftmost dimension first)
        let mut result = ArrayUsize8::empty();
        for i in (0..indices.len()).rev() {
            result.push(indices[i]);
        }

        result
    }

    /// Create an empty iterator
    fn empty(&self) -> Self {
        Self {
            tensor: self.tensor,
            ptr: self.original_ptr,
            end_ptr: self.original_ptr,
            original_ptr: self.original_ptr,
            current_indices: ArrayUsize8::empty(),
            dtype: self.dtype,
            max_elements: Some(0),
            elements_yielded: 0,
            _phantom: PhantomData,
        }
    }
}

impl<S: StorageTrait> Clone for TensorIter<'_, S> {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor,
            ptr: self.ptr,
            end_ptr: self.end_ptr,
            original_ptr: self.original_ptr,
            current_indices: self.current_indices,
            dtype: self.dtype,
            max_elements: self.max_elements,
            elements_yielded: self.elements_yielded,
            _phantom: PhantomData,
        }
    }
}

unsafe impl<S: StorageTrait> Send for TensorIter<'_, S> where S: Send {}
unsafe impl<S: StorageTrait> Sync for TensorIter<'_, S> where S: Sync {}

#[cfg(feature = "rayon")]
pub struct ParTensorIter<'a, S: StorageTrait> {
    inner: TensorIter<'a, S>,
    min_len: usize,
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait> ParTensorIter<'a, S> {
    pub fn new(inner: TensorIter<'a, S>) -> Self {
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
impl<'a, S: StorageTrait + Send + Sync> IntoParallelIterator for TensorIter<'a, S> {
    type Item = TensorIterElement;
    type Iter = ParTensorIter<'a, S>;

    fn into_par_iter(self) -> Self::Iter {
        ParTensorIter::new(self)
    }
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait + Send + Sync> ParallelIterator for ParTensorIter<'a, S> {
    type Item = TensorIterElement;

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
impl<'a, S: StorageTrait + Send + Sync> IndexedParallelIterator for ParTensorIter<'a, S> {
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
        callback.callback(ParTensorProducer {
            inner: self.inner,
            min_len: self.min_len,
        })
    }
}

#[cfg(feature = "rayon")]
struct ParTensorProducer<'a, S: StorageTrait> {
    inner: TensorIter<'a, S>,
    min_len: usize,
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait + Send + Sync> Producer for ParTensorProducer<'a, S> {
    type Item = TensorIterElement;
    type IntoIter = TensorIter<'a, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.inner.split_at(index);
        (
            ParTensorProducer {
                inner: left,
                min_len: self.min_len,
            },
            ParTensorProducer {
                inner: right,
                min_len: self.min_len,
            },
        )
    }
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait + Send + Sync> IntoIterator for ParTensorProducer<'a, S> {
    type Item = TensorIterElement;
    type IntoIter = TensorIter<'a, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner
    }
}

impl<S: StorageTrait> TensorBase<S> {
    /// Create an iterator over all elements in the tensor
    #[inline]
    pub fn iter(&self) -> TensorIter<'_, S> {
        TensorIter::from_tensor(self)
    }

    #[cfg(feature = "rayon")]
    /// Create a parallel iterator over all elements in the tensor
    #[inline]
    pub fn par_iter(&self) -> ParTensorIter<'_, S>
    where
        S: Send + Sync,
    {
        self.iter().par_iter()
    }
}

// Implement IntoIterator for &TensorBase to support for _ in &tensor syntax
impl<'a, S: StorageTrait> IntoIterator for &'a TensorBase<S> {
    type Item = TensorIterElement;
    type IntoIter = TensorIter<'a, S>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;
    #[cfg(feature = "rayon")]
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};

    #[test]
    fn test_tensor_iter_basic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, vec![2, 2]).unwrap();

        let iter = tensor.iter();
        assert_eq!(iter.len(), 4);

        let elements: Vec<_> = iter.collect();
        assert_eq!(elements.len(), 4);
    }

    #[test]
    fn test_tensor_iter_indices() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, vec![2, 2]).unwrap();

        let mut iter = tensor.iter();
        let element = iter.next().unwrap();
        assert_eq!(element.indices.as_slice(), &[0, 0]);

        let element = iter.next().unwrap();
        assert_eq!(element.indices.as_slice(), &[0, 1]);

        let element = iter.next().unwrap();
        assert_eq!(element.indices.as_slice(), &[1, 0]);

        let element = iter.next().unwrap();
        assert_eq!(element.indices.as_slice(), &[1, 1]);
    }

    #[test]
    fn test_tensor_iter_empty() {
        let data: Vec<f32> = vec![];
        let tensor = Tensor::from_vec(data, vec![0, 0]).unwrap();

        let iter = tensor.iter();
        assert_eq!(iter.len(), 0);
        assert!(iter.is_empty());
    }

    #[test]
    fn test_tensor_iter_for_loop() {
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, vec![3]).unwrap();

        let mut count = 0;
        for element in &tensor {
            assert_eq!(element.indices.len(), 1);
            assert_eq!(element.indices[0], count);
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_tensor_iter_1d() {
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, vec![3]).unwrap();

        let iter = tensor.iter();
        assert_eq!(iter.len(), 3);

        let elements: Vec<_> = iter.collect();
        assert_eq!(elements.len(), 3);
    }

    #[test]
    fn test_tensor_iter_3d() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 2, 2]).unwrap();

        let iter = tensor.iter();
        assert_eq!(iter.len(), 8);

        let elements: Vec<_> = iter.collect();
        assert_eq!(elements.len(), 8);
    }

    #[test]
    fn test_tensor_iter_nth() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, vec![2, 2]).unwrap();

        let mut iter = tensor.iter();
        let element = iter.nth(2).unwrap();
        let indices = element.indices;
        assert_eq!(indices.as_slice(), &[1, 0]);
    }

    #[test]
    fn test_tensor_iter_split() {
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        let iter = tensor.iter();
        let (left, right) = iter.split_at(3);

        assert_eq!(left.len(), 3);
        assert_eq!(right.len(), 3);
    }

    #[test]
    fn test_tensor_iter_non_contiguous() {
        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        println!("Original tensor shape: {:?}", tensor.dims());

        // Transpose to make it non-contiguous: [[1, 4], [2, 5], [3, 6]]
        let transposed = tensor.permute([1, 0]).unwrap();
        println!("Transposed shape: {:?}", transposed.dims());
        println!("Transposed strides: {:?}", transposed.strides());
        println!("Is contiguous: {}", transposed.is_contiguous());

        println!("\nIterating over transposed tensor:");
        for (i, elem) in transposed.iter().enumerate() {
            let ptr = unsafe { elem.as_ptr(transposed.as_ptr()) };
            let val = unsafe { *(ptr as *const f32) };
            println!("Element {}: indices={:?}, value={}", i, elem.indices, val);
        }

        // Expected order for transposed tensor should be:
        // [0,0] -> 1.0, [0,1] -> 4.0, [1,0] -> 2.0, [1,1] -> 5.0, [2,0] -> 3.0, [2,1] -> 6.0
        let values: Vec<f32> = transposed
            .iter()
            .map(|elem| {
                let ptr = unsafe { elem.as_ptr(transposed.as_ptr()) };
                unsafe { *(ptr as *const f32) }
            })
            .collect();

        println!("Actual values: {values:?}");
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_tensor_par_iter_basic() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![10, 10]).unwrap();

        let par_iter = tensor.par_iter();
        assert_eq!(par_iter.len(), 100);

        let count = par_iter.count();
        assert_eq!(count, 100);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_tensor_par_iter_map() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![10]).unwrap();

        let par_iter = tensor.par_iter();
        let results: Vec<crate::ArrayUsize8> = par_iter.map(|element| element.indices).collect();

        assert_eq!(results.len(), 10);
        assert_eq!(results[0].as_slice(), &[0]);
        assert_eq!(results[9].as_slice(), &[9]);
    }
}
