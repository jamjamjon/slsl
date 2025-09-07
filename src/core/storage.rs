//! Memory storage management for tensors.
//!
//! This module provides traits and structures for managing tensor data storage,
//! including both custom allocated memory and Vec-based memory.

use crate::TensorElement;
use anyhow::Result;
use std::alloc::{alloc, Layout};
use std::ptr::NonNull;
use std::sync::Arc;

/// Trait for types that can provide storage functionality.
///
/// This trait defines the interface for accessing storage pointers and
/// converting to storage references.
pub trait StorageTrait {
    /// Returns a non-null pointer to the underlying memory.
    fn ptr(&self) -> NonNull<u8>;

    /// Returns a reference to the storage instance.
    fn as_storage(&self) -> &Storage;
}

/// Internal storage implementation with memory management details.
///
/// This structure handles the actual memory allocation and deallocation,
/// tracking whether the memory comes from a Vec or custom allocation.
#[derive(Debug)]
pub(crate) struct StorageInner {
    /// Non-null pointer to the allocated memory
    ptr: std::ptr::NonNull<u8>,
    /// Memory layout information for proper deallocation
    layout: std::alloc::Layout,
    /// Flag indicating if this memory was allocated from a Vec
    is_vec_memory: bool,
    /// Original Vec capacity for proper deallocation
    vec_capacity: usize,
    /// Size of each element in the original Vec
    vec_element_size: usize,
}

unsafe impl Send for StorageInner {}
unsafe impl Sync for StorageInner {}

impl Drop for StorageInner {
    fn drop(&mut self) {
        unsafe {
            if self.is_vec_memory {
                // For Vec memory, we need to reconstruct the Vec with the correct element type
                // Since we don't know the original element type, we use u8 and adjust capacity
                let byte_capacity = self.vec_capacity * self.vec_element_size;
                let _vec = Vec::from_raw_parts(
                    self.ptr.as_ptr(),
                    0, // length doesn't matter for deallocation
                    byte_capacity,
                );
                // Vec will be dropped automatically here
            } else {
                // Custom allocated memory
                std::alloc::dealloc(self.ptr.as_ptr(), self.layout);
            }
        }
    }
}

/// Thread-safe reference-counted storage for tensor data.
///
/// This structure provides a safe wrapper around memory storage with
/// automatic cleanup and reference counting capabilities.
#[derive(Debug, Clone)]
pub struct Storage {
    /// Reference-counted inner storage
    _inner: Arc<StorageInner>,
    /// Cached pointer for fast access
    cached_ptr: NonNull<u8>,
}

unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl Storage {
    /// Creates a new storage with the specified size and alignment.
    ///
    /// # Parameters
    ///
    /// * `size_bytes` - The number of bytes to allocate
    /// * `align` - The memory alignment requirement
    ///
    /// # Returns
    ///
    /// A new Storage instance or an error if allocation fails.
    pub fn new(size_bytes: usize, align: usize) -> Result<Self> {
        let layout = Layout::from_size_align(size_bytes, align)
            .map_err(|e| anyhow::anyhow!("Invalid layout: {}", e))?;
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            anyhow::bail!("Failed to allocate memory");
        }
        let ptr = NonNull::new(ptr).unwrap();
        let _inner = Arc::new(StorageInner {
            ptr,
            layout,
            is_vec_memory: false,
            vec_capacity: 0,
            vec_element_size: 0,
        });

        Ok(Self {
            _inner,
            cached_ptr: ptr,
        })
    }
    /// Returns the current reference count for this storage.
    ///
    /// This shows how many Storage instances are sharing the same underlying memory.
    pub fn strong_count(&self) -> usize {
        std::sync::Arc::strong_count(&self._inner)
    }

    /// Creates a new storage from an existing Vec.
    ///
    /// Takes ownership of the Vec and manages its memory through the storage system.
    /// The Vec's memory will be properly deallocated when the storage is dropped.
    ///
    /// # Parameters
    ///
    /// * `data` - The Vec to take ownership of
    ///
    /// # Returns
    ///
    /// A new Storage instance wrapping the Vec's memory or an error.
    pub fn from_vec<T: TensorElement>(data: Vec<T>) -> Result<Self> {
        let size_bytes = data.len() * std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        let capacity = data.capacity();
        let layout = Layout::from_size_align(size_bytes, align)
            .map_err(|e| anyhow::anyhow!("Invalid layout: {}", e))?;

        // Extract the raw parts from Vec without dropping it
        // Use ManuallyDrop to prevent Vec from being dropped
        let mut data = std::mem::ManuallyDrop::new(data);
        let ptr = data.as_mut_ptr() as *mut u8;
        let ptr = NonNull::new(ptr).ok_or_else(|| anyhow::anyhow!("Vec pointer is null"))?;
        let _inner = Arc::new(StorageInner {
            ptr,
            layout,
            is_vec_memory: true,
            vec_capacity: capacity,
            vec_element_size: std::mem::size_of::<T>(),
        });

        Ok(Self {
            _inner,
            cached_ptr: ptr,
        })
    }
}

impl StorageTrait for Storage {
    /// Returns the raw memory pointer.
    #[inline]
    fn ptr(&self) -> NonNull<u8> {
        self.cached_ptr
    }

    /// Returns a reference to this storage instance.
    #[inline]
    fn as_storage(&self) -> &Storage {
        self
    }
}

impl StorageTrait for &Storage {
    /// Returns the raw memory pointer from the referenced storage.
    #[inline]
    fn ptr(&self) -> NonNull<u8> {
        self.cached_ptr
    }

    /// Returns a reference to the storage instance.
    #[inline]
    fn as_storage(&self) -> &Storage {
        self
    }
}
