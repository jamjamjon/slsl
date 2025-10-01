use std::marker::PhantomData;

use crate::{ArrayUsize8, StorageTrait, TensorBase, TensorElement};

/// Generic iterator over tensor elements of type T
#[repr(C)]
pub struct TensorTypeIter<'a, S: StorageTrait, T> {
    /// Reference to the original tensor
    tensor: &'a TensorBase<S>,
    /// Current pointer position
    ptr: *const T,
    /// End pointer boundary
    end_ptr: *const T,
    /// Cached total length (for fast len() calls)
    cached_len: usize,
    /// Current linear index (for contiguous tensors)
    current_index: usize,
    /// Tail linear index for double-ended iteration (contiguous tensors)
    tail_index: usize,
    /// Whether the tensor is contiguous (optimization flag)
    is_contiguous: bool,
    /// Lifetime marker
    _phantom: PhantomData<&'a ()>,
}

// Safety: TensorTypeIter is Send/Sync because it only contains raw pointers and indices
unsafe impl<S: StorageTrait, T> Send for TensorTypeIter<'_, S, T> {}
unsafe impl<S: StorageTrait, T> Sync for TensorTypeIter<'_, S, T> {}

impl<'a, S: StorageTrait, T: TensorElement> TensorTypeIter<'a, S, T> {
    /// Create iterator from Tensor for specific type T
    #[inline]
    pub fn from_tensor(tensor: &'a TensorBase<S>) -> Self {
        debug_assert_eq!(
            tensor.dtype,
            T::DTYPE,
            "Type mismatch: tensor has dtype {:?}, but requested type has dtype {:?}",
            tensor.dtype,
            T::DTYPE
        );

        let data_ptr = tensor.as_ptr() as *const T;
        let numel = tensor.numel();
        let end_ptr = unsafe { data_ptr.add(numel) };
        let is_contiguous = tensor.is_contiguous();

        Self {
            tensor,
            ptr: data_ptr,
            end_ptr,
            cached_len: numel,
            current_index: 0,
            tail_index: numel,
            is_contiguous,
            _phantom: PhantomData,
        }
    }

    /// Get remaining length
    #[inline(always)]
    pub fn len(&self) -> usize {
        if self.is_contiguous {
            self.cached_len.saturating_sub(self.current_index)
        } else {
            ((self.end_ptr as usize - self.ptr as usize) / std::mem::size_of::<T>()).max(0)
        }
    }

    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        if self.is_contiguous {
            self.current_index >= self.cached_len
        } else {
            self.ptr >= self.end_ptr
        }
    }

    /// Split the iterator at the given index
    #[inline]
    pub fn split_at(self, index: usize) -> (Self, Self) {
        if self.is_contiguous {
            let mid = self.current_index + index;
            let mid = mid.min(self.cached_len);

            let left = TensorTypeIter {
                tensor: self.tensor,
                ptr: self.ptr,
                end_ptr: self.ptr,
                cached_len: mid - self.current_index,
                current_index: 0,
                tail_index: mid - self.current_index,
                is_contiguous: self.is_contiguous,
                _phantom: PhantomData,
            };

            let right = TensorTypeIter {
                tensor: self.tensor,
                ptr: unsafe { self.ptr.add(mid - self.current_index) },
                end_ptr: self.end_ptr,
                cached_len: self.cached_len - mid,
                current_index: 0,
                tail_index: self.cached_len - mid,
                is_contiguous: self.is_contiguous,
                _phantom: PhantomData,
            };

            (left, right)
        } else {
            let mid_ptr = unsafe { self.ptr.add(index) };
            let left = TensorTypeIter {
                tensor: self.tensor,
                ptr: self.ptr,
                end_ptr: mid_ptr,
                cached_len: index,
                current_index: 0,
                tail_index: index,
                is_contiguous: self.is_contiguous,
                _phantom: PhantomData,
            };

            let right = TensorTypeIter {
                tensor: self.tensor,
                ptr: mid_ptr,
                end_ptr: self.end_ptr,
                cached_len: self.cached_len - index,
                current_index: 0,
                tail_index: self.cached_len - index,
                is_contiguous: self.is_contiguous,
                _phantom: PhantomData,
            };

            (left, right)
        }
    }
}

impl<'a, S: StorageTrait, T: TensorElement + 'a> Iterator for TensorTypeIter<'a, S, T> {
    type Item = &'a T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.is_contiguous {
            if self.current_index >= self.cached_len {
                return None;
            }
            let current = self.ptr;
            self.ptr = unsafe { self.ptr.add(1) };
            self.current_index += 1;
            Some(unsafe { &*current })
        } else {
            if self.ptr >= self.end_ptr {
                return None;
            }
            let current = self.ptr;
            self.ptr = unsafe { self.ptr.add(1) };
            Some(unsafe { &*current })
        }
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
        if self.is_contiguous {
            let p = unsafe { self.ptr.add(n) };
            if p >= self.end_ptr {
                return None;
            }
            self.ptr = unsafe { p.add(1) };
            self.current_index += n + 1;
            if self.current_index > self.tail_index {
                self.tail_index = self.current_index;
            }
            Some(unsafe { &*p })
        } else {
            let p = unsafe { self.ptr.add(n) };
            if p >= self.end_ptr {
                return None;
            }
            self.ptr = unsafe { p.add(1) };
            Some(unsafe { &*p })
        }
    }
}

impl<S: StorageTrait, T: TensorElement> ExactSizeIterator for TensorTypeIter<'_, S, T> {}
impl<S: StorageTrait, T: TensorElement> std::iter::FusedIterator for TensorTypeIter<'_, S, T> {}

// Double-ended iteration support
impl<S: StorageTrait, T: TensorElement> DoubleEndedIterator for TensorTypeIter<'_, S, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.is_contiguous {
            if self.current_index >= self.tail_index {
                return None;
            }
            let idx = self.tail_index - 1;
            let offset_bytes = idx * std::mem::size_of::<T>();
            let p = unsafe { self.tensor.as_ptr().add(offset_bytes) as *const T };
            self.tail_index = idx;
            Some(unsafe { &*p })
        } else {
            if self.ptr >= self.end_ptr {
                return None;
            }
            self.end_ptr = unsafe { self.end_ptr.sub(1) };
            Some(unsafe { &*self.end_ptr })
        }
    }
}

/// Parallel iterator over tensor elements of type T
#[cfg(feature = "rayon")]
pub struct ParTensorTypeIter<'a, S: StorageTrait, T> {
    inner: TensorTypeIter<'a, S, T>,
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait, T: TensorElement> ParTensorTypeIter<'a, S, T> {
    pub fn new(iter: TensorTypeIter<'a, S, T>) -> Self {
        Self { inner: iter }
    }
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait, T: TensorElement> IntoIterator for ParTensorTypeIter<'a, S, T> {
    type Item = &'a T;
    type IntoIter = TensorTypeIter<'a, S, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner
    }
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait + Send + Sync, T: TensorElement + Send + Sync>
    rayon::iter::ParallelIterator for ParTensorTypeIter<'a, S, T>
{
    type Item = &'a T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.inner.len())
    }
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait + Send + Sync, T: TensorElement + Send + Sync>
    rayon::iter::IndexedParallelIterator for ParTensorTypeIter<'a, S, T>
{
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::plumbing::ProducerCallback<Self::Item>,
    {
        callback.callback(TensorTypeIterProducer { iter: self.inner })
    }
}

#[cfg(feature = "rayon")]
struct TensorTypeIterProducer<'a, S: StorageTrait, T> {
    iter: TensorTypeIter<'a, S, T>,
}

#[cfg(feature = "rayon")]
impl<'a, S: StorageTrait, T: TensorElement> rayon::iter::plumbing::Producer
    for TensorTypeIterProducer<'a, S, T>
{
    type Item = &'a T;
    type IntoIter = TensorTypeIter<'a, S, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.iter.split_at(index);
        (
            TensorTypeIterProducer { iter: left },
            TensorTypeIterProducer { iter: right },
        )
    }
}

impl<S: StorageTrait> TensorBase<S> {
    /// Create a typed iterator over all elements in the tensor
    #[inline]
    pub fn iter<T: TensorElement>(&self) -> TensorTypeIter<'_, S, T> {
        TensorTypeIter::from_tensor(self)
    }

    /// Internal: Create a typed iterator that yields value together with indices and byte offset
    /// This replaces the need for the untyped iterator when metadata is required.
    #[inline]
    pub(crate) fn iter_with_meta<T: TensorElement>(&self) -> TensorTypeMetaIter<'_, S, T> {
        TensorTypeMetaIter::from_tensor(self)
    }

    /// Public: return indexed typed iterator, produce (indices, &T)
    #[inline]
    pub fn indexed_iter<T: TensorElement>(&self) -> IndexedTensorTypeIter<'_, S, T> {
        IndexedTensorTypeIter {
            inner: self.iter_with_meta::<T>(),
        }
    }

    /// Create a typed parallel iterator over all elements in the tensor
    #[cfg(feature = "rayon")]
    #[inline]
    pub fn par_iter<T>(&self) -> ParTensorTypeIter<'_, S, T>
    where
        S: Send + Sync,
        T: TensorElement + Send + Sync,
    {
        ParTensorTypeIter::new(self.iter::<T>())
    }
}

/// A typed iterator yielding indices, byte offset and value reference
pub struct TensorTypeMetaItem<'a, T> {
    pub indices: ArrayUsize8,
    pub offset: usize,
    pub value: &'a T,
}

/// Typed iterator with metadata (indices and offset)
pub struct TensorTypeMetaIter<'a, S: StorageTrait, T: TensorElement> {
    tensor: &'a TensorBase<S>,
    current_indices: ArrayUsize8,
    linear_index: usize,
    remaining: usize,
    is_contiguous: bool,
    element_size: usize,
    strides_in_bytes: ArrayUsize8,
    _marker: PhantomData<&'a T>,
}

impl<'a, S: StorageTrait, T: TensorElement> TensorTypeMetaIter<'a, S, T> {
    #[inline]
    fn from_tensor(tensor: &'a TensorBase<S>) -> Self {
        debug_assert_eq!(tensor.dtype, T::DTYPE);
        let mut current_indices = ArrayUsize8::empty();
        for _ in 0..tensor.rank() {
            current_indices.push(0);
        }
        let element_size = std::mem::size_of::<T>();
        let mut strides_in_bytes = ArrayUsize8::empty();
        for i in 0..tensor.rank() {
            strides_in_bytes.push(tensor.strides[i] * element_size);
        }
        Self {
            tensor,
            current_indices,
            linear_index: 0,
            remaining: tensor.numel(),
            is_contiguous: tensor.is_contiguous(),
            element_size,
            strides_in_bytes,
            _marker: PhantomData,
        }
    }

    #[inline]
    fn advance_indices(&mut self) {
        if self.tensor.rank() == 0 {
            return;
        }
        for i in (0..self.tensor.rank()).rev() {
            self.current_indices[i] += 1;
            if self.current_indices[i] < self.tensor.shape[i] {
                return;
            }
            self.current_indices[i] = 0;
        }
    }
}

impl<'a, S: StorageTrait, T: TensorElement> Iterator for TensorTypeMetaIter<'a, S, T> {
    type Item = TensorTypeMetaItem<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let offset = if self.is_contiguous {
            self.linear_index * self.element_size
        } else {
            let mut off = 0usize;
            for (dim_idx, &index) in self
                .current_indices
                .as_slice()
                .iter()
                .enumerate()
                .take(self.tensor.rank())
            {
                if self.tensor.strides[dim_idx] != 0 {
                    off += index * self.strides_in_bytes[dim_idx];
                }
            }
            off
        };

        let ptr = unsafe { self.tensor.as_ptr().add(offset) as *const T };
        let value = unsafe { &*ptr };

        let indices = self.current_indices;
        self.remaining -= 1;
        self.linear_index += 1;
        self.advance_indices();

        Some(TensorTypeMetaItem {
            indices,
            offset,
            value,
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<S: StorageTrait, T: TensorElement> ExactSizeIterator for TensorTypeMetaIter<'_, S, T> {}
impl<S: StorageTrait, T: TensorElement> std::iter::FusedIterator for TensorTypeMetaIter<'_, S, T> {}

/// Public indexed iterator adapter yielding (indices, &T)
pub struct IndexedTensorTypeIter<'a, S: StorageTrait, T: TensorElement> {
    inner: TensorTypeMetaIter<'a, S, T>,
}

impl<'a, S: StorageTrait, T: TensorElement> Iterator for IndexedTensorTypeIter<'a, S, T> {
    type Item = (ArrayUsize8, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|m| (m.indices, m.value))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<S: StorageTrait, T: TensorElement> ExactSizeIterator for IndexedTensorTypeIter<'_, S, T> {}
impl<S: StorageTrait, T: TensorElement> std::iter::FusedIterator
    for IndexedTensorTypeIter<'_, S, T>
{
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

        let iter = tensor.iter::<f32>();
        assert_eq!(iter.len(), 4);

        let elements: Vec<_> = tensor.iter::<f32>().collect();
        assert_eq!(elements.len(), 4);
    }

    #[test]
    fn test_tensor_iter_indices() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, vec![2, 2]).unwrap();
        let mut iter = tensor.iter_with_meta::<f32>();
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

        let iter = tensor.iter::<f32>();
        assert_eq!(iter.len(), 0);
        assert!(iter.is_empty());
    }

    #[test]
    fn test_tensor_iter_for_loop() {
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, vec![3]).unwrap();

        let mut count = 0;
        for item in tensor.iter_with_meta::<f32>() {
            assert_eq!(item.indices.len(), 1);
            assert_eq!(item.indices[0], count);
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_tensor_iter_1d() {
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, vec![3]).unwrap();

        let iter = tensor.iter::<f32>();
        assert_eq!(iter.len(), 3);

        let elements: Vec<_> = tensor.iter::<f32>().collect();
        assert_eq!(elements.len(), 3);
    }

    #[test]
    fn test_tensor_iter_3d() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 2, 2]).unwrap();

        let iter = tensor.iter::<f32>();
        assert_eq!(iter.len(), 8);

        let elements: Vec<_> = tensor.iter::<f32>().collect();
        assert_eq!(elements.len(), 8);
    }

    #[test]
    fn test_tensor_iter_nth() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, vec![2, 2]).unwrap();

        let mut iter = tensor.iter::<f32>();
        let v = iter.nth(2).unwrap();
        assert_eq!(*v, 3.0);
    }

    #[test]
    fn test_tensor_iter_split() {
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        let iter = tensor.iter::<f32>();
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
        for (i, item) in transposed.iter_with_meta::<f32>().enumerate() {
            let val = *item.value;
            println!("Element {}: indices={:?}, value={}", i, item.indices, val);
        }

        // Expected order for transposed tensor should be:
        // [0,0] -> 1.0, [0,1] -> 4.0, [1,0] -> 2.0, [1,1] -> 5.0, [2,0] -> 3.0, [2,1] -> 6.0
        let values: Vec<f32> = transposed
            .iter_with_meta::<f32>()
            .map(|item| *item.value)
            .collect();

        println!("Actual values: {values:?}");
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_tensor_par_iter_basic() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![10, 10]).unwrap();

        let par_iter = tensor.par_iter::<f32>();
        assert_eq!(par_iter.len(), 100);

        let count = par_iter.count();
        assert_eq!(count, 100);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_tensor_par_iter_map() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![10]).unwrap();

        let par_iter = tensor.par_iter::<f32>();
        let values: Vec<f32> = par_iter.map(|&v| v).collect();

        assert_eq!(values.len(), 10);
        assert_eq!(values[0], 0.0);
        assert_eq!(values[9], 9.0);
    }
}
