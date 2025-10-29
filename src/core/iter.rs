use crate::{ArrayUsize8, StorageTrait, TensorBase, TensorElement};
use std::marker::PhantomData;

/// Index-based iterator - mirrors the for_at() approach
#[repr(C)]
pub struct TensorTypeIter<'a, S: StorageTrait, T> {
    tensor: &'a TensorBase<S>, // Needed for N-D case
    rank: u8,                  // u8 saves space
    is_contiguous: bool,

    // Core iteration state (hot fields first for cache locality)
    current_index: usize,
    end_index: usize,
    base_ptr: *const u8,
    offset_bytes: usize,

    // Pre-computed strides for fast offset calculation
    stride_bytes: usize,     // 1D
    cols: usize,             // 2D
    row_stride_bytes: usize, // 2D
    col_stride_bytes: usize, // 2D
    _phantom: PhantomData<T>,
}

// TensorTypeIter Send/Sync implementations
//
// Send + Sync are needed when threaded OR rayon feature is enabled
// - Iterator contains references to tensor data
// - Safe when Storage is thread-safe
// - See Shared<T> for detailed safety analysis of Rayon + Rc

#[cfg(any(feature = "threaded", feature = "rayon"))]
unsafe impl<S: StorageTrait, T: Send> Send for TensorTypeIter<'_, S, T> {}

#[cfg(any(feature = "threaded", feature = "rayon"))]
unsafe impl<S: StorageTrait, T: Sync> Sync for TensorTypeIter<'_, S, T> {}

impl<'a, S: StorageTrait, T: TensorElement> TensorTypeIter<'a, S, T> {
    #[inline]
    pub fn from_tensor(tensor: &'a TensorBase<S>) -> Self {
        debug_assert_eq!(tensor.dtype, T::DTYPE);

        let elem_size = std::mem::size_of::<T>();
        let rank = tensor.rank();
        let is_contiguous = tensor.is_contiguous();

        let (stride_bytes, cols, row_stride_bytes, col_stride_bytes) = if rank == 1 {
            (tensor.strides[0] * elem_size, 0, 0, 0)
        } else if rank == 2 {
            (
                0,
                tensor.shape[1],
                tensor.strides[0] * elem_size,
                tensor.strides[1] * elem_size,
            )
        } else {
            (0, 0, 0, 0)
        };

        Self {
            current_index: 0,
            end_index: tensor.numel(),
            base_ptr: tensor.storage.ptr().as_ptr(),
            offset_bytes: tensor.offset_bytes,
            stride_bytes,
            cols,
            row_stride_bytes,
            col_stride_bytes,
            tensor,
            rank: rank as u8,
            is_contiguous,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.end_index - self.current_index
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.current_index >= self.end_index
    }

    /// Calculate offset for current index - mirrors at() logic
    #[inline(always)]
    pub(crate) fn calculate_offset(&self, index: usize) -> usize {
        let elem_size = std::mem::size_of::<T>();

        if self.is_contiguous {
            // Fastest: contiguous case - pure pointer arithmetic
            self.offset_bytes + index * elem_size
        } else if self.rank == 1 {
            // 1D strided - single multiply
            self.offset_bytes + index * self.stride_bytes
        } else if self.rank == 2 {
            // 2D:  row/col calculation
            let row = index / self.cols;
            let col = index % self.cols;
            self.offset_bytes + row * self.row_stride_bytes + col * self.col_stride_bytes
        } else {
            // General N-D case - need tensor for shape/strides
            let mut linear = index;
            let mut offset = self.offset_bytes;
            let rank = self.rank as usize;

            for i in (0..rank).rev() {
                let dim = self.tensor.shape[i];
                let stride = self.tensor.strides[i] * elem_size;
                let idx = linear % dim;
                linear /= dim;
                offset += idx * stride;
            }
            offset
        }
    }

    #[inline]
    pub fn split_at(self, index: usize) -> (Self, Self) {
        let mid = (self.current_index + index).min(self.end_index);

        (
            Self {
                end_index: mid,
                ..self
            },
            Self {
                current_index: mid,
                end_index: self.end_index,
                base_ptr: self.base_ptr,
                offset_bytes: self.offset_bytes,
                stride_bytes: self.stride_bytes,
                cols: self.cols,
                row_stride_bytes: self.row_stride_bytes,
                col_stride_bytes: self.col_stride_bytes,
                tensor: self.tensor,
                rank: self.rank,
                is_contiguous: self.is_contiguous,
                _phantom: PhantomData,
            },
        )
    }
}

impl<'a, S: StorageTrait, T: TensorElement> Iterator for TensorTypeIter<'a, S, T> {
    type Item = &'a T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.end_index {
            // This mirrors exactly what for_at() does!
            let offset = self.calculate_offset(self.current_index);
            let ptr = unsafe { self.base_ptr.add(offset) as *const T };
            self.current_index += 1;
            Some(unsafe { &*ptr })
        } else {
            None
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
        let new_index = self.current_index + n;
        if new_index >= self.end_index {
            self.current_index = self.end_index;
            None
        } else {
            self.current_index = new_index;
            self.next()
        }
    }
}

impl<S: StorageTrait, T: TensorElement> ExactSizeIterator for TensorTypeIter<'_, S, T> {}
impl<S: StorageTrait, T: TensorElement> std::iter::FusedIterator for TensorTypeIter<'_, S, T> {}

impl<S: StorageTrait, T: TensorElement> DoubleEndedIterator for TensorTypeIter<'_, S, T> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_index < self.end_index {
            self.end_index -= 1;
            let offset = self.calculate_offset(self.end_index);
            let ptr = unsafe { self.base_ptr.add(offset) as *const T };
            Some(unsafe { &*ptr })
        } else {
            None
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
impl<S: StorageTrait + Send + Sync, T: TensorElement + Send + Sync>
    rayon::iter::IndexedParallelIterator for ParTensorTypeIter<'_, S, T>
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

/// Typed iterator with metadata - thin wrapper around TensorTypeIter
pub struct TensorTypeMetaIter<'a, S: StorageTrait, T: TensorElement> {
    /// The core iterator - handles all offset calculations
    core: TensorTypeIter<'a, S, T>,
    /// Shape reference for index calculation
    shape: [usize; 8],
    /// Current linear index (tracks separately for index calculation)
    linear_idx: usize,
}

impl<'a, S: StorageTrait, T: TensorElement> TensorTypeMetaIter<'a, S, T> {
    #[inline]
    fn from_tensor(tensor: &'a TensorBase<S>) -> Self {
        let mut shape = [0usize; 8];
        let rank = tensor.rank();
        shape[..rank].copy_from_slice(&tensor.shape.as_slice()[..rank]);

        Self {
            core: TensorTypeIter::from_tensor(tensor),
            shape,
            linear_idx: 0,
        }
    }
}

impl<'a, S: StorageTrait, T: TensorElement> Iterator for TensorTypeMetaIter<'a, S, T> {
    type Item = TensorTypeMetaItem<'a, T>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.linear_idx >= self.core.end_index {
            return None;
        }

        // Reuse core's fast offset calculation
        let offset = self.core.calculate_offset(self.linear_idx);
        let ptr = unsafe { self.core.base_ptr.add(offset) as *const T };
        let value = unsafe { &*ptr };

        // Calculate multi-dimensional indices from linear index
        let rank = self.core.rank;
        let mut indices = ArrayUsize8::empty();

        if rank == 1 {
            indices.push(self.linear_idx);
        } else if rank == 2 {
            indices.push(self.linear_idx / self.shape[1]);
            indices.push(self.linear_idx % self.shape[1]);
        } else {
            // General case for rank > 2
            let mut idx = self.linear_idx;
            let mut temp = [0usize; 8];
            let rank_usize = rank as usize;

            for i in (0..rank_usize).rev() {
                temp[i] = idx % self.shape[i];
                idx /= self.shape[i];
            }

            for &val in temp.iter().take(rank_usize) {
                indices.push(val);
            }
        }

        self.linear_idx += 1;

        Some(TensorTypeMetaItem {
            indices,
            offset,
            value,
        })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.core.end_index - self.linear_idx;
        (remaining, Some(remaining))
    }
}

impl<S: StorageTrait, T: TensorElement> ExactSizeIterator for TensorTypeMetaIter<'_, S, T> {}
impl<S: StorageTrait, T: TensorElement> std::iter::FusedIterator for TensorTypeMetaIter<'_, S, T> {}

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

        // Transpose to make it non-contiguous: [[1, 4], [2, 5], [3, 6]]
        let transposed = tensor.permute([1, 0]).unwrap();

        // Expected order for transposed tensor should be:
        // [0,0] -> 1.0, [0,1] -> 4.0, [1,0] -> 2.0, [1,1] -> 5.0, [2,0] -> 3.0, [2,1] -> 6.0
        let values: Vec<f32> = transposed
            .iter_with_meta::<f32>()
            .map(|item| *item.value)
            .collect();

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

    #[test]
    fn test_iter_permuted_2d_correctness() {
        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        // Transpose to 3x2: [[1, 4], [2, 5], [3, 6]]
        let transposed = tensor.permute([1, 0]).unwrap();

        // Test iteration order
        let values: Vec<f32> = transposed.iter::<f32>().copied().collect();

        // Expected: row-major order of transposed tensor
        // [0,0]=1.0, [0,1]=4.0, [1,0]=2.0, [1,1]=5.0, [2,0]=3.0, [2,1]=6.0
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_iter_sliced_1d_correctness() {
        use crate::s;

        // Create 1D tensor: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![10]).unwrap();

        // Slice with step: [0, 2, 4, 6, 8]
        let sliced = tensor.slice(s![0..10; 2]);

        let values: Vec<f32> = sliced.iter::<f32>().copied().collect();
        assert_eq!(values, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_iter_sliced_2d_correctness() {
        use crate::s;

        // Create 4x4 tensor:
        // [[0,  1,  2,  3],
        //  [4,  5,  6,  7],
        //  [8,  9, 10, 11],
        //  [12, 13, 14, 15]]
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![4, 4]).unwrap();

        // Slice middle 2x2: [[5, 6], [9, 10]]
        let sliced = tensor.slice(s![1..3, 1..3]);

        let values: Vec<f32> = sliced.iter::<f32>().copied().collect();
        assert_eq!(values, vec![5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn test_iter_column_slice_correctness() {
        use crate::s;

        // Create 3x4 tensor:
        // [[0,  1,  2,  3],
        //  [4,  5,  6,  7],
        //  [8,  9, 10, 11]]
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![3, 4]).unwrap();

        // Extract column 2: [2, 6, 10]
        let col = tensor.slice(s![.., 2]);

        let values: Vec<f32> = col.iter::<f32>().copied().collect();
        assert_eq!(values, vec![2.0, 6.0, 10.0]);
    }

    #[test]
    fn test_iter_3d_permuted_correctness() {
        // Create 2x2x2 tensor
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 2, 2]).unwrap();

        // Permute dimensions: [2,2,2] -> [2,2,2] with different order
        let permuted = tensor.permute([2, 0, 1]).unwrap();

        // Get values via iter and via manual indexing
        let iter_values: Vec<f32> = permuted.iter::<f32>().copied().collect();
        let mut manual_values = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    manual_values.push(permuted.at::<f32>([i, j, k]));
                }
            }
        }

        assert_eq!(iter_values, manual_values);
    }

    #[test]
    fn test_iter_vs_at_consistency() {
        // Test that iter() produces same order as at() for all cases
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 3, 4]).unwrap();

        // Test 1: Original tensor (contiguous)
        let iter_vals: Vec<f32> = tensor.iter::<f32>().copied().collect();
        let mut at_vals = Vec::new();
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    at_vals.push(tensor.at::<f32>([i, j, k]));
                }
            }
        }
        assert_eq!(iter_vals, at_vals, "Contiguous: iter vs at mismatch");

        // Test 2: Permuted tensor (non-contiguous)
        let permuted = tensor.permute([1, 2, 0]).unwrap();
        let iter_vals2: Vec<f32> = permuted.iter::<f32>().copied().collect();
        let mut at_vals2 = Vec::new();
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..2 {
                    at_vals2.push(permuted.at::<f32>([i, j, k]));
                }
            }
        }
        assert_eq!(iter_vals2, at_vals2, "Permuted: iter vs at mismatch");
    }
}
