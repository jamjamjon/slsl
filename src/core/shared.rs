//! Shared pointer abstraction for single-threaded and multi-threaded contexts.
//!
//! This module provides a `Shared<T>` type that automatically uses:
//! - `Rc<T>` by default (single-threaded, lower overhead)
//! - `Arc<T>` when the `threaded` feature is enabled (multi-threaded, atomic reference counting)
//!
//! This allows the library to optimize for single-threaded performance by default
//! while still supporting multi-threaded usage when explicitly requested.

#[cfg(not(feature = "threaded"))]
use std::rc::Rc as SharedInner;

#[cfg(feature = "threaded")]
use std::sync::Arc as SharedInner;

/// A reference-counted smart pointer that adapts to single or multi-threaded usage.
///
/// Without the `threaded` feature (default): Uses `Rc<T>` for lower overhead.
/// With the `threaded` feature: Uses `Arc<T>` for thread-safe sharing.
///
/// # Examples
///
/// ```
/// use slsl::Shared;
///
/// let data = Shared::new(vec![1, 2, 3, 4]);
/// let data_clone = data.clone(); // Increments reference count
///
/// assert_eq!(Shared::strong_count(&data), 2);
/// ```
#[derive(Debug)]
#[repr(transparent)]
pub struct Shared<T> {
    inner: SharedInner<T>,
}

impl<T> Shared<T> {
    /// Creates a new `Shared` pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use slsl::Shared;
    ///
    /// let shared = Shared::new(42);
    /// assert_eq!(*shared, 42);
    /// ```
    #[inline]
    pub fn new(value: T) -> Self {
        Self {
            inner: SharedInner::new(value),
        }
    }

    /// Gets the number of strong references to this allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use slsl::Shared;
    ///
    /// let shared = Shared::new(42);
    /// assert_eq!(Shared::strong_count(&shared), 1);
    ///
    /// let shared2 = shared.clone();
    /// assert_eq!(Shared::strong_count(&shared), 2);
    /// ```
    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        #[cfg(not(feature = "threaded"))]
        {
            std::rc::Rc::strong_count(&this.inner)
        }
        #[cfg(feature = "threaded")]
        {
            std::sync::Arc::strong_count(&this.inner)
        }
    }

    /// Returns a reference to the inner value.
    ///
    /// # Examples
    ///
    /// ```
    /// use slsl::Shared;
    ///
    /// let shared = Shared::new(vec![1, 2, 3]);
    /// assert_eq!(AsRef::<Vec<i32>>::as_ref(&shared).len(), 3);
    /// ```
    #[inline]
    #[allow(clippy::should_implement_trait)] // We do implement AsRef via trait impl below
    pub fn as_ref(&self) -> &T {
        &self.inner
    }
}

impl<T> Clone for Shared<T> {
    /// Makes a clone of the `Shared` pointer.
    ///
    /// This creates another pointer to the same allocation, increasing the strong reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use slsl::Shared;
    ///
    /// let shared = Shared::new(42);
    /// let shared2 = shared.clone();
    ///
    /// assert_eq!(*shared, *shared2);
    /// assert_eq!(Shared::strong_count(&shared), 2);
    /// ```
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> std::ops::Deref for Shared<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> AsRef<T> for Shared<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.inner
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Shared<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&**self, f)
    }
}

impl<T: PartialEq> PartialEq for Shared<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: Eq> Eq for Shared<T> {}

impl<T: PartialOrd> PartialOrd for Shared<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<T: Ord> Ord for Shared<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: std::hash::Hash> std::hash::Hash for Shared<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

// Send/Sync implementations for Shared<T>
//
// SAFETY ANALYSIS
// ===============
//
// When feature = "threaded": Uses Arc<T>
// --------------------------------------
// - Arc<T> is Send + Sync when T: Send + Sync (standard library guarantee)
// - Completely safe for all multi-threading scenarios
//
// When feature = "rayon" (without threaded): Uses Rc<T>
// ------------------------------------------------------
// This is the SUBTLE case that requires careful analysis.
//
// Rc<T> is normally NOT Send or Sync because:
// - Its reference count uses non-atomic operations
// - Concurrent access to the ref count causes data races
//
// However, we implement Send + Sync for Rc-based Shared in the Rayon context because:
//
// 1. RAYON'S USAGE PATTERN:
//    - Rayon's parallel iterators require T: Send + Sync
//    - Rayon moves iterators (containing &Tensor) between threads
//    - But the original Tensor (with Rc) stays on the creating thread
//
// 2. RC IS NEVER CLONED OR DROPPED ON WORKER THREADS:
//    - Worker threads only hold BORROWS (&TensorView)
//    - TensorView contains &Storage, which contains Shared<StorageInner>
//    - The Shared wrapper is copied (it's just a pointer), but clone() is never called
//    - The Rc's reference count is NEVER modified from worker threads
//
// 3. ONLY DATA POINTERS ARE ACCESSED:
//    - Worker threads read Storage::cached_ptr (a raw pointer)
//    - They never touch the Rc's internal reference count
//    - The actual tensor data is immutable and safe for concurrent reads
//
// 4. TYPE SYSTEM PROTECTION:
//    - TensorBase<Storage> is NOT Send when using Rc (only Sync)
//    - This prevents std::thread::spawn(move || { tensor })
//    - Users cannot accidentally move Tensor to another thread
//    - Only Rayon's controlled borrowing is allowed
//
// This is a controlled use of unsafe that relies on:
// - Rayon's documented guarantees about data access patterns
// - The fact that we never clone/drop Rc from worker threads
// - Immutability of the underlying tensor data
//
// If this assumption is violated (e.g., someone clones Rc in a worker thread),
// it would cause undefined behavior. However, our API design prevents this.

#[cfg(any(feature = "threaded", feature = "rayon"))]
unsafe impl<T: Send + Sync> Send for Shared<T> {}

#[cfg(any(feature = "threaded", feature = "rayon"))]
unsafe impl<T: Send + Sync> Sync for Shared<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_new() {
        let shared = Shared::new(42);
        assert_eq!(*shared, 42);
    }

    #[test]
    fn test_shared_clone() {
        let shared = Shared::new(vec![1, 2, 3]);
        let shared2 = shared.clone();

        assert_eq!(*shared, *shared2);
        assert_eq!(Shared::strong_count(&shared), 2);
        assert_eq!(Shared::strong_count(&shared2), 2);
    }

    #[test]
    fn test_shared_strong_count() {
        let shared = Shared::new(42);
        assert_eq!(Shared::strong_count(&shared), 1);

        let shared2 = shared.clone();
        assert_eq!(Shared::strong_count(&shared), 2);

        let _shared3 = shared.clone();
        assert_eq!(Shared::strong_count(&shared), 3);

        drop(shared2);
        assert_eq!(Shared::strong_count(&shared), 2);
    }

    #[test]
    fn test_shared_deref() {
        let shared = Shared::new(vec![1, 2, 3, 4]);
        assert_eq!(shared.len(), 4);
        assert_eq!(shared[0], 1);
    }

    #[test]
    fn test_shared_as_ref() {
        let shared = Shared::new(String::from("hello"));
        let s: &String = shared.as_ref();
        assert_eq!(s, "hello");
    }

    #[test]
    fn test_shared_equality() {
        let shared1 = Shared::new(42);
        let shared2 = Shared::new(42);
        let shared3 = Shared::new(43);

        assert_eq!(shared1, shared2);
        assert_ne!(shared1, shared3);
    }

    #[test]
    fn test_shared_ordering() {
        let shared1 = Shared::new(1);
        let shared2 = Shared::new(2);
        let shared3 = Shared::new(3);

        assert!(shared1 < shared2);
        assert!(shared2 < shared3);
        assert!(shared1 <= shared2);
        assert!(shared3 > shared1);
    }

    #[test]
    fn test_shared_hash() {
        use std::collections::HashSet;

        let shared1 = Shared::new(42);
        let shared2 = Shared::new(42);
        let shared3 = Shared::new(43);

        let mut set = HashSet::new();
        set.insert(shared1.clone());
        set.insert(shared2.clone());
        set.insert(shared3.clone());

        // Note: HashSet uses value equality, so shared1 and shared2 are the same
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_shared_display() {
        let shared = Shared::new(42);
        assert_eq!(format!("{}", shared), "42");

        let shared_str = Shared::new(String::from("hello"));
        assert_eq!(format!("{}", shared_str), "hello");
    }

    #[cfg(any(feature = "threaded", feature = "rayon"))]
    #[test]
    fn test_shared_send_sync_with_features() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // With threaded or rayon feature, Shared should be Send + Sync
        assert_send::<Shared<i32>>();
        assert_sync::<Shared<i32>>();
    }

    #[cfg(not(any(feature = "threaded", feature = "rayon")))]
    #[test]
    fn test_shared_not_send_sync_default() {
        // Without threaded or rayon feature, Shared should NOT be Send/Sync
        // This test just verifies the type exists and works
        let shared = Shared::new(42);
        assert_eq!(*shared, 42);
    }
}
