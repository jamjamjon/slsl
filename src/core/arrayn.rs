//! ArrayN - A fixed-capacity array with dynamic length

/// Type alias for 8-element array with usize elements.
///
/// This is a convenience type for working with arrays that typically need to store
/// up to 8 dimension sizes or similar metadata.
pub type ArrayUsize8 = ArrayN<usize, 8>;

/// A fixed-capacity array with dynamic length.
///
/// `ArrayN` provides a stack-allocated array with a compile-time fixed capacity `N`
/// but allows the actual used length to vary at runtime. This is useful for scenarios
/// where you need the performance benefits of stack allocation but don't always use
/// the full capacity.
///
/// # Type Parameters
///
/// * `T` - The element type, must implement `Copy` and `Default`
/// * `N` - The maximum capacity of the array (compile-time constant)
///
/// # Examples
///
/// ```rust
/// use slsl::ArrayN;
///
/// let mut arr = ArrayN::<i32, 4>::empty();
/// arr.push(1);
/// arr.push(2);
/// assert_eq!(arr.len(), 2);
/// assert_eq!(arr.capacity(), 4);
/// ```
#[derive(Clone, Copy)]
pub struct ArrayN<T, const N: usize>
where
    T: Copy + Default,
{
    /// The current number of elements in use
    pub len: usize,
    /// The underlying fixed-size array storage
    pub arr: [T; N],
}

impl<T: Copy + Default, const N: usize> Default for ArrayN<T, N> {
    /// Creates a new `ArrayN` with zero length and default-initialized elements.
    ///
    /// # Returns
    ///
    /// An empty `ArrayN` with all elements initialized to their default values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let arr = ArrayN::<i32, 4>::default();
    /// assert_eq!(arr.len(), 0);
    /// assert!(arr.is_empty());
    /// ```
    fn default() -> Self {
        Self {
            len: 0,
            arr: [T::default(); N],
        }
    }
}

/// Creates an `ArrayN` from a list of values at compile time.
///
/// This macro provides a convenient way to initialize an `ArrayN` with a sequence
/// of values, similar to the `vec!` macro for `Vec`.
///
/// # Examples
///
/// ```rust
/// use slsl::{ArrayN, arrayn};
///
/// let arr: ArrayN<i32, 8> = arrayn![1, 2, 3, 4];
/// assert_eq!(arr.len(), 4);
/// assert_eq!(arr[0], 1);
/// assert_eq!(arr[3], 4);
///
/// // Trailing comma is supported
/// let arr2: ArrayN<i32, 8> = arrayn![5, 6, 7,];
/// assert_eq!(arr2.len(), 3);
/// ```
#[macro_export]
macro_rules! arrayn {
    ($($x:expr),* $(,)?) => {{
        let mut tmp = ArrayN::default();
        let mut i = 0;
        $(
            tmp.arr[i] = $x;
            i += 1;
        )*
        tmp.len = i;
        tmp
    }};
}

impl<T: Copy + Default, const N: usize> std::ops::Deref for ArrayN<T, N> {
    type Target = [T];

    /// Dereferences the `ArrayN` to a slice containing only the active elements.
    ///
    /// This allows `ArrayN` to be used anywhere a slice is expected.
    ///
    /// # Returns
    ///
    /// A slice `&[T]` containing only the elements from index `0` to `len-1`.
    fn deref(&self) -> &Self::Target {
        &self.arr[..self.len]
    }
}

impl<T, const N: usize> ArrayN<T, N>
where
    T: Copy + Default,
{
    /// Creates an `ArrayN` from a slice.
    ///
    /// Copies elements from the provided slice into a new `ArrayN`. The slice length
    /// must not exceed the array's capacity `N`.
    ///
    /// # Parameters
    ///
    /// * `slice` - The source slice to copy elements from
    ///
    /// # Returns
    ///
    /// A new `ArrayN` containing the copied elements with length equal to the slice length.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `slice.len() > N`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let slice = &[1, 2, 3];
    /// let arr = ArrayN::<i32, 5>::from_slice(slice);
    /// assert_eq!(arr.len(), 3);
    /// assert_eq!(arr[0], 1);
    /// ```
    #[inline(always)]
    pub fn from_slice(slice: &[T]) -> Self {
        let len = slice.len();
        debug_assert!(len <= N, "Length {len} exceeds capacity {N}");
        let mut arr: [std::mem::MaybeUninit<T>; N] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), arr.as_mut_ptr() as *mut T, len);
        }
        let arr =
            unsafe { std::mem::transmute_copy::<[std::mem::MaybeUninit<T>; N], [T; N]>(&arr) };

        Self { len, arr }
    }

    /// Creates an empty `ArrayN` with uninitialized storage.
    ///
    /// This is more efficient than `default()` as it doesn't initialize the array elements,
    /// but the elements should not be accessed until they are explicitly set.
    ///
    /// # Returns
    ///
    /// An empty `ArrayN` with zero length and uninitialized elements.
    ///
    /// # Safety
    ///
    /// The returned array has uninitialized elements. Only access elements after
    /// they have been explicitly set (e.g., via `push()` or direct assignment).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let mut arr = ArrayN::<i32, 4>::empty();
    /// assert_eq!(arr.len(), 0);
    /// arr.push(42);
    /// assert_eq!(arr[0], 42);
    /// ```
    #[allow(clippy::uninit_assumed_init)]
    #[inline(always)]
    pub fn empty() -> Self {
        let arr: [T; N] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        Self { len: 0, arr }
    }

    /// Sets the length of the array and returns self.
    ///
    /// This is a builder-style method that allows chaining. The caller must ensure
    /// that all elements up to the new length are properly initialized.
    ///
    /// # Parameters
    ///
    /// * `len` - The new length to set
    ///
    /// # Returns
    ///
    /// Self with the updated length.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `len > N`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let arr = ArrayN::<i32, 4>::default().with_len(2);
    /// assert_eq!(arr.len(), 2);
    /// ```
    pub fn with_len(mut self, len: usize) -> Self {
        debug_assert!(len <= N, "Length {len} exceeds capacity {N}");
        self.len = len;
        self
    }

    /// Creates an `ArrayN` filled with a specific value.
    ///
    /// All array elements are initialized to the provided value, but only the first
    /// `len` elements are considered active.
    ///
    /// # Parameters
    ///
    /// * `val` - The value to fill the array with
    /// * `len` - The active length of the array
    ///
    /// # Returns
    ///
    /// A new `ArrayN` with all elements set to `val` and length set to `len`.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `len > N`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let arr = ArrayN::<i32, 4>::full(42, 3);
    /// assert_eq!(arr.len(), 3);
    /// assert_eq!(arr[0], 42);
    /// assert_eq!(arr[2], 42);
    /// ```
    #[inline(always)]
    pub fn full(val: T, len: usize) -> Self {
        debug_assert!(len <= N, "Length {len} exceeds capacity {N}");
        Self { len, arr: [val; N] }
    }

    /// Set the length of the array (unsafe)
    ///
    /// # Safety
    ///
    /// The caller must ensure that `new_len` is within the capacity bounds
    /// and that all elements up to `new_len` are properly initialized.
    #[inline(always)]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        self.len = new_len.min(N);
    }

    /// Returns the current number of elements in the array.
    ///
    /// # Returns
    ///
    /// The number of active elements (not the total capacity).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let mut arr = ArrayN::<i32, 4>::empty();
    /// assert_eq!(arr.len(), 0);
    /// arr.push(1);
    /// assert_eq!(arr.len(), 1);
    /// ```
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the array contains no elements.
    ///
    /// # Returns
    ///
    /// `true` if the length is 0, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let arr = ArrayN::<i32, 4>::empty();
    /// assert!(arr.is_empty());
    ///
    /// let arr2 = ArrayN::<i32, 4>::full(1, 2);
    /// assert!(!arr2.is_empty());
    /// ```
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the maximum number of elements the array can hold.
    ///
    /// This is the compile-time constant `N` and never changes.
    ///
    /// # Returns
    ///
    /// The maximum capacity `N`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let arr = ArrayN::<i32, 8>::empty();
    /// assert_eq!(arr.capacity(), 8);
    /// ```
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns a slice containing only the active elements.
    ///
    /// This provides a view into the used portion of the array, excluding
    /// any unused capacity.
    ///
    /// # Returns
    ///
    /// A slice `&[T]` of the active elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let arr = ArrayN::<i32, 4>::from_slice(&[1, 2, 3]);
    /// let slice = arr.as_slice();
    /// assert_eq!(slice, &[1, 2, 3]);
    /// ```
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        &self.arr[..self.len]
    }

    /// Returns a mutable slice containing only the active elements.
    ///
    /// This provides mutable access to the used portion of the array.
    ///
    /// # Returns
    ///
    /// A mutable slice `&mut [T]` of the active elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let mut arr = ArrayN::<i32, 4>::from_slice(&[1, 2, 3]);
    /// let slice = arr.as_mut_slice();
    /// slice[0] = 10;
    /// assert_eq!(arr[0], 10);
    /// ```
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.arr[..self.len]
    }

    /// Returns the element at the given index, or `None` if out of bounds.
    ///
    /// This provides safe access to array elements without panicking.
    ///
    /// # Parameters
    ///
    /// * `index` - The index to access
    ///
    /// # Returns
    ///
    /// `Some(element)` if the index is valid, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let arr = ArrayN::<i32, 4>::from_slice(&[1, 2, 3]);
    /// assert_eq!(arr.get(1), Some(2));
    /// assert_eq!(arr.get(5), None);
    /// ```
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<T> {
        if index < self.len {
            Some(self.arr[index])
        } else {
            None
        }
    }

    /// Adds an element to the end of the array.
    ///
    /// Increases the length by 1 and sets the new element to the provided value.
    ///
    /// # Parameters
    ///
    /// * `value` - The element to add
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the array is already at capacity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let mut arr = ArrayN::<i32, 4>::empty();
    /// arr.push(1);
    /// arr.push(2);
    /// assert_eq!(arr.len(), 2);
    /// assert_eq!(arr[1], 2);
    /// ```
    #[inline(always)]
    pub fn push(&mut self, value: T) {
        debug_assert!(self.len < N, "ArrayN capacity exceeded");
        self.arr[self.len] = value;
        self.len += 1;
    }

    /// Removes the element at the specified index.
    ///
    /// All elements after the removed element are shifted left to fill the gap.
    /// The length is decreased by 1.
    ///
    /// # Parameters
    ///
    /// * `index` - The index of the element to remove
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let mut arr = ArrayN::<i32, 4>::from_slice(&[1, 2, 3, 4]);
    /// arr.remove(1);
    /// assert_eq!(arr.as_slice(), &[1, 3, 4]);
    /// ```
    #[inline(always)]
    pub fn remove(&mut self, index: usize) {
        debug_assert!(
            index < self.len,
            "Index {} out of bounds for ArrayN with length {}",
            index,
            self.len
        );
        for i in index..self.len - 1 {
            self.arr[i] = self.arr[i + 1];
        }
        self.len -= 1;
    }

    // pub fn reverse(&mut self) {
    //     self.arr.reverse();
    // }

    // pub fn reversed(&self) -> Self
    // where
    //     T: Clone,
    // {
    //     let mut arr = self.arr;
    //     arr.reverse();
    //     Self { len: self.len, arr }
    // }
}

// From trait implementations
impl<const N: usize> From<()> for ArrayN<usize, N> {
    /// Creates an empty `ArrayN<usize, N>` from the unit type.
    ///
    /// # Returns
    ///
    /// An empty `ArrayN` with zero length.
    #[inline(always)]
    fn from(_: ()) -> Self {
        Self::empty()
    }
}

impl<T, const N: usize, const M: usize> From<[T; M]> for ArrayN<T, N>
where
    T: Copy + Default,
{
    /// Creates an `ArrayN` from a fixed-size array.
    ///
    /// # Parameters
    ///
    /// * `dims` - The source array to copy from
    ///
    /// # Returns
    ///
    /// A new `ArrayN` containing the elements from the source array.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `M > N`.
    #[inline(always)]
    fn from(dims: [T; M]) -> Self {
        Self::from_slice(&dims)
    }
}

impl<T, const N: usize> From<Vec<T>> for ArrayN<T, N>
where
    T: Copy + Default,
{
    /// Creates an `ArrayN` from a `Vec`.
    ///
    /// # Parameters
    ///
    /// * `dims` - The source vector to copy from
    ///
    /// # Returns
    ///
    /// A new `ArrayN` containing the elements from the vector.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `dims.len() > N`.
    #[inline(always)]
    fn from(dims: Vec<T>) -> Self {
        Self::from_slice(&dims)
    }
}

impl<T, const N: usize> From<&[T]> for ArrayN<T, N>
where
    T: Copy + Default,
{
    /// Creates an `ArrayN` from a slice.
    ///
    /// # Parameters
    ///
    /// * `dims` - The source slice to copy from
    ///
    /// # Returns
    ///
    /// A new `ArrayN` containing the elements from the slice.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `dims.len() > N`.
    #[inline(always)]
    fn from(dims: &[T]) -> Self {
        Self::from_slice(dims)
    }
}

// Convenience implementations for usize single/tuple conversions
impl<const N: usize> From<usize> for ArrayN<usize, N> {
    /// Creates an `ArrayN<usize, N>` from a single usize value.
    ///
    /// # Parameters
    ///
    /// * `value` - The value to store as the first element
    ///
    /// # Returns
    ///
    /// A new `ArrayN` with length 1 containing the provided value.
    #[inline(always)]
    fn from(value: usize) -> Self {
        Self::from_slice(&[value])
    }
}

impl<const N: usize> From<(usize,)> for ArrayN<usize, N> {
    /// Creates an `ArrayN<usize, N>` from a single-element tuple.
    ///
    /// # Parameters
    ///
    /// * `dims` - A tuple containing one usize value
    ///
    /// # Returns
    ///
    /// A new `ArrayN` with length 1 containing the tuple's value.
    #[inline(always)]
    fn from(dims: (usize,)) -> Self {
        Self::from_slice(&[dims.0])
    }
}

impl<const N: usize> From<(usize, usize)> for ArrayN<usize, N> {
    /// Creates an `ArrayN<usize, N>` from a two-element tuple.
    ///
    /// # Parameters
    ///
    /// * `dims` - A tuple containing two usize values
    ///
    /// # Returns
    ///
    /// A new `ArrayN` with length 2 containing the tuple's values.
    #[inline(always)]
    fn from(dims: (usize, usize)) -> Self {
        Self::from_slice(&[dims.0, dims.1])
    }
}

impl<const N: usize> From<(usize, usize, usize)> for ArrayN<usize, N> {
    /// Creates an `ArrayN<usize, N>` from a three-element tuple.
    ///
    /// # Parameters
    ///
    /// * `dims` - A tuple containing three usize values
    ///
    /// # Returns
    ///
    /// A new `ArrayN` with length 3 containing the tuple's values.
    #[inline(always)]
    fn from(dims: (usize, usize, usize)) -> Self {
        Self::from_slice(&[dims.0, dims.1, dims.2])
    }
}

impl<const N: usize> From<(usize, usize, usize, usize)> for ArrayN<usize, N> {
    /// Creates an `ArrayN<usize, N>` from a four-element tuple.
    ///
    /// # Parameters
    ///
    /// * `dims` - A tuple containing four usize values
    ///
    /// # Returns
    ///
    /// A new `ArrayN` with length 4 containing the tuple's values.
    #[inline(always)]
    fn from(dims: (usize, usize, usize, usize)) -> Self {
        Self::from_slice(&[dims.0, dims.1, dims.2, dims.3])
    }
}

impl<const N: usize> From<isize> for ArrayN<usize, N> {
    /// Creates an `ArrayN<usize, N>` from an isize value.
    ///
    /// The isize value is cast to usize.
    ///
    /// # Parameters
    ///
    /// * `value` - The isize value to convert and store
    ///
    /// # Returns
    ///
    /// A new `ArrayN` with length 1 containing the converted value.
    #[inline(always)]
    fn from(value: isize) -> Self {
        Self::from_slice(&[value as usize])
    }
}

impl<const N: usize> From<i32> for ArrayN<usize, N> {
    /// Creates an `ArrayN<usize, N>` from an i32 value.
    ///
    /// The i32 value is cast to usize.
    ///
    /// # Parameters
    ///
    /// * `value` - The i32 value to convert and store
    ///
    /// # Returns
    ///
    /// A new `ArrayN` with length 1 containing the converted value.
    #[inline(always)]
    fn from(value: i32) -> Self {
        Self::from_slice(&[value as usize])
    }
}

impl<const N: usize> From<u32> for ArrayN<usize, N> {
    /// Creates an `ArrayN<usize, N>` from a u32 value.
    ///
    /// The u32 value is cast to usize.
    ///
    /// # Parameters
    ///
    /// * `value` - The u32 value to convert and store
    ///
    /// # Returns
    ///
    /// A new `ArrayN` with length 1 containing the converted value.
    #[inline(always)]
    fn from(value: u32) -> Self {
        Self::from_slice(&[value as usize])
    }
}

// Display implementation
impl<T, const N: usize> std::fmt::Display for ArrayN<T, N>
where
    T: Copy + Default + std::fmt::Display,
{
    /// Formats the `ArrayN` as a comma-separated list enclosed in brackets.
    ///
    /// Only the active elements (up to `len`) are displayed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let arr = ArrayN::<i32, 4>::from_slice(&[1, 2, 3]);
    /// assert_eq!(format!("{}", arr), "[1, 2, 3]");
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, item) in self.as_slice().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{item}")?;
        }
        write!(f, "]")
    }
}

impl<T, const N: usize> std::fmt::Debug for ArrayN<T, N>
where
    T: Copy + Default + std::fmt::Display,
{
    /// Formats the `ArrayN` for debugging, showing it as a list of elements.
    ///
    /// Only the active elements (up to `len`) are displayed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, item) in self.as_slice().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{item}")?;
        }
        write!(f, "]")
    }
}

// Custom equality based only on the active portion [..len]
impl<T, const N: usize> PartialEq for ArrayN<T, N>
where
    T: Copy + Default + PartialEq,
{
    /// Compares two `ArrayN` instances for equality.
    ///
    /// Two arrays are equal if they have the same length and all their
    /// active elements are equal. The unused capacity is ignored.
    ///
    /// # Parameters
    ///
    /// * `other` - The other `ArrayN` to compare with
    ///
    /// # Returns
    ///
    /// `true` if the arrays are equal, `false` otherwise.
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        self.as_slice() == other.as_slice()
    }
}

impl<T, const N: usize> Eq for ArrayN<T, N> where T: Copy + Default + Eq {}

impl<T, const N: usize> std::ops::Index<usize> for ArrayN<T, N>
where
    T: Copy + Default,
{
    type Output = T;

    /// Returns a reference to the element at the given index.
    ///
    /// # Parameters
    ///
    /// * `index` - The index to access
    ///
    /// # Returns
    ///
    /// A reference to the element at the specified index.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let arr = ArrayN::<i32, 4>::from_slice(&[1, 2, 3]);
    /// assert_eq!(arr[1], 2);
    /// ```
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(
            index < self.len,
            "index {} out of bounds for ArrayN of len {}",
            index,
            self.len
        );
        &self.arr[index]
    }
}

impl<T, const N: usize> std::ops::IndexMut<usize> for ArrayN<T, N>
where
    T: Copy + Default,
{
    /// Returns a mutable reference to the element at the given index.
    ///
    /// # Parameters
    ///
    /// * `index` - The index to access
    ///
    /// # Returns
    ///
    /// A mutable reference to the element at the specified index.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let mut arr = ArrayN::<i32, 4>::from_slice(&[1, 2, 3]);
    /// arr[1] = 10;
    /// assert_eq!(arr[1], 10);
    /// ```
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(
            index < self.len,
            "index {} out of bounds for ArrayN of len {}",
            index,
            self.len
        );
        &mut self.arr[index]
    }
}

impl<T, const N: usize> ArrayN<T, N>
where
    T: Copy + Default,
{
    /// Returns an iterator over the active elements.
    ///
    /// # Returns
    ///
    /// An iterator yielding references to the active elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let arr = ArrayN::<i32, 4>::from_slice(&[1, 2, 3]);
    /// let sum: i32 = arr.iter().sum();
    /// assert_eq!(sum, 6);
    /// ```
    #[inline(always)]
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.arr[..self.len].iter()
    }

    /// Returns a mutable iterator over the active elements.
    ///
    /// # Returns
    ///
    /// An iterator yielding mutable references to the active elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::ArrayN;
    ///
    /// let mut arr = ArrayN::<i32, 4>::from_slice(&[1, 2, 3]);
    /// for elem in arr.iter_mut() {
    ///     *elem *= 2;
    /// }
    /// assert_eq!(arr.as_slice(), &[2, 4, 6]);
    /// ```
    #[inline(always)]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.arr[..self.len].iter_mut()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a ArrayN<T, N>
where
    T: Copy + Default,
{
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    /// Creates an iterator over references to the active elements.
    ///
    /// # Returns
    ///
    /// An iterator yielding references to the active elements.
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut ArrayN<T, N>
where
    T: Copy + Default,
{
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    /// Creates a mutable iterator over references to the active elements.
    ///
    /// # Returns
    ///
    /// An iterator yielding mutable references to the active elements.
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
