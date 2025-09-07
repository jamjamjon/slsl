//! Tensor slicing macro utilities.
//!
//! This module provides the `s!` macro for creating tensor slice specifications
//! in a convenient and intuitive syntax. The macro supports various slicing patterns
//! including indexing, ranges, stepped ranges, and new axis insertion.

/// Creates slice specifications for tensor slicing operations.
///
/// The `s!` macro provides a convenient syntax for creating [`SliceSpecs`] that can be
/// used with tensor slicing operations. It supports a wide variety of slicing patterns
/// including individual indices, ranges, stepped ranges, inclusive ranges, and new axis
/// insertion for dimension expansion.
///
/// # Syntax Patterns
///
/// ## Basic Indexing
/// - `s!()` - Empty slice specification
/// - `s!(0)` - Single index at position 0
/// - `s!(0, 1)` - Multiple indices for multi-dimensional tensors
///
/// ## Range Slicing
/// - `s!(1..4)` - Range from 1 to 4 (exclusive)
/// - `s!(1..=4)` - Inclusive range from 1 to 4
/// - `s!(1..)` - Range from 1 to end
/// - `s!(..4)` - Range from start to 4
/// - `s!(..)` - Full range (all elements)
///
/// ## Stepped Ranges
/// - `s!(0..10; 2)` - Range with step size 2
/// - `s!(..;3)` - Full range with step size 3
/// - `s!(1..=10;2)` - Inclusive range with step
///
/// ## Multi-dimensional Slicing
/// - `s!(0, 1..3)` - Index in first dimension, range in second
/// - `s!(1..3, 2..5)` - Ranges in multiple dimensions
/// - `s!(.., 0)` - Full range in first dimensions, index in last
///
/// ## New Axis Insertion
/// - `s!(None)` - Insert new axis
/// - `s!(None, 0)` - Insert new axis, then index
/// - `s!(0, None)` - Index, then insert new axis
///
/// # Parameters
///
/// The macro accepts various input patterns:
/// - **Literal integers**: Direct indexing (e.g., `0`, `1`, `-1`)
/// - **Range expressions**: Various range types with optional steps
/// - **None**: For new axis insertion
/// - **General expressions**: Any expression that implements `Into<SliceElem>`
///
/// # Returns
///
/// Returns a [`SliceSpecs`] containing the slice elements that can be used
/// with tensor slicing methods like [`Tensor::slice`] and [`TensorView::slice`].
///
/// # Examples
///
/// ## Basic Indexing
///
/// ```rust
/// use slsl::{s, Tensor};
///
/// let tensor = Tensor::zeros::<f32>([4, 5])?;
///
/// // Single index - selects row 1
/// let row = tensor.slice(s!(1));
/// assert_eq!(row.shape().as_slice(), &[5]);
///
/// // Two indices - selects element at (1, 2)
/// let element = tensor.slice(s!(1, 2));
/// assert_eq!(element.shape().as_slice(), &[]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Range Slicing
///
/// ```rust
/// use slsl::{s, Tensor};
///
/// let tensor = Tensor::zeros::<f32>([6, 8])?;
///
/// // Range slice - rows 1 to 4 (exclusive)  
/// let rows = tensor.slice(s!(1..4));
/// assert_eq!(rows.shape().as_slice(), &[3]); // Takes only the first dimension slice
///
/// // For 2D slicing, need to specify both dimensions
/// let rows_2d = tensor.slice(s!(1..4, ..));
/// assert_eq!(rows_2d.shape().as_slice(), &[3, 8]);
///
/// // Inclusive range
/// let rows_inc = tensor.slice(s!(1..=4));
/// assert_eq!(rows_inc.shape().as_slice(), &[4]);
///
/// // From start to index 3
/// let start = tensor.slice(s!(..3));
/// assert_eq!(start.shape().as_slice(), &[3]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Stepped Ranges
///
/// ```rust
/// use slsl::{s, Tensor};
///
/// let tensor = Tensor::zeros::<f32>([12])?;
///
/// // Every second element from 0 to 8
/// let stepped = tensor.slice(s!(0..8; 2));
/// assert_eq!(stepped.shape().as_slice(), &[4]);
///
/// // Every third element from start to end  
/// let full_stepped = tensor.slice(s!(..;3));
/// assert_eq!(full_stepped.shape().as_slice(), &[4]); // [0, 3, 6, 9]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Multi-dimensional Slicing
///
/// ```rust
/// use slsl::{s, Tensor};
///
/// let tensor = Tensor::zeros::<f32>([4, 6, 8])?;
///
/// // Mixed index and range - selects element 1 from first dim, range from second dim
/// let mixed = tensor.slice(s!(1, 2..5));
/// assert_eq!(mixed.shape().as_slice(), &[3]); // Results in 1D tensor
///
/// // Multiple ranges - keeps specified ranges from first two dimensions
/// let ranges = tensor.slice(s!(1..3, 2..4));
/// assert_eq!(ranges.shape().as_slice(), &[2, 2]); // Results in 2D tensor
///
/// // Full range in first dimension, index in second
/// let partial = tensor.slice(s!(.., 3));
/// assert_eq!(partial.shape().as_slice(), &[4]); // Results in 1D tensor
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## New Axis Insertion
///
/// ```rust
/// use slsl::{s, Tensor};
///
/// let tensor = Tensor::zeros::<f32>([4, 5])?;
///
/// // Insert new axis at the beginning
/// let expanded = tensor.slice(s!(None, .., ..));
/// assert_eq!(expanded.shape().as_slice(), &[1, 4, 5]);
///
/// // Insert new axis in the middle
/// let middle = tensor.slice(s!(.., None, ..));
/// assert_eq!(middle.shape().as_slice(), &[4, 1, 5]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Advanced Patterns
///
/// ```rust
/// use slsl::{s, Tensor};
///
/// let tensor = Tensor::zeros::<f32>([8, 10])?;
///
/// // Complex multi-dimensional slicing
/// let complex = tensor.slice(s!(1..6; 2, 2..8));
/// assert_eq!(complex.shape().as_slice(), &[3, 6]); // Every 2nd row from 1-6, cols 2-8
///
/// // Negative indexing (from end)
/// let from_end = tensor.slice(s!(-2, ..));
/// assert_eq!(from_end.shape().as_slice(), &[10]); // Second-to-last row
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Performance Notes
///
/// - The macro generates efficient slice specifications at compile time
/// - Multiple literal patterns are optimized for common use cases
/// - Complex expressions are handled through the general case conversion
/// - Range operations with step=1 have optimized implementations
///
/// # See Also
///
/// - [`SliceElem`] - Individual slice element types
/// - [`SliceSpecs`] - Collection of slice elements
/// - [`Tensor::slice`] - Apply slice to owned tensor
/// - [`TensorView::slice`] - Apply slice to tensor view
///
/// [`SliceSpecs`]: crate::SliceSpecs
/// [`SliceElem`]: crate::SliceElem
/// [`Tensor::slice`]: crate::Tensor::slice
/// [`TensorView::slice`]: crate::TensorView::slice
/// [`Tensor`]: crate::Tensor
/// [`TensorView`]: crate::TensorView
#[macro_export]
macro_rules! s {
    // Empty slice
    () => {
        $crate::SliceSpecs::empty()
    };

    // Single literal index
    ($idx:literal) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Index($idx));
            specs
        }
    };

    // Two literal indices
    ($idx1:literal, $idx2:literal $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Index($idx1));
            specs.push($crate::SliceElem::Index($idx2));
            specs
        }
    };

    // Three literal indices
    ($idx1:literal, $idx2:literal, $idx3:literal $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Index($idx1));
            specs.push($crate::SliceElem::Index($idx2));
            specs.push($crate::SliceElem::Index($idx3));
            specs
        }
    };

    // Four literal indices
    ($idx1:literal, $idx2:literal, $idx3:literal, $idx4:literal $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Index($idx1));
            specs.push($crate::SliceElem::Index($idx2));
            specs.push($crate::SliceElem::Index($idx3));
            specs.push($crate::SliceElem::Index($idx4));
            specs
        }
    };

    // Range with step: start..end;step
    ($start:literal .. $end:literal ; $step:literal) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: $start,
                end: Some($end),
                step: $step,
            });
            specs
        }
    };

    // Range from start with step: start..;step
    ($start:literal .. ; $step:literal) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: $start,
                end: None,
                step: $step,
            });
            specs
        }
    };

    // Range to end with step: ..end;step
    (.. $end:literal ; $step:literal) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: Some($end),
                step: $step,
            });
            specs
        }
    };

    // Full range with step: ..;step
    (.. ; $step:literal) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: None,
                step: $step,
            });
            specs
        }
    };

    // Simple range patterns without step
    ($start:literal .. $end:literal) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: $start,
                end: Some($end),
                step: 1,
            });
            specs
        }
    };

    ($start:literal ..) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: $start,
                end: None,
                step: 1,
            });
            specs
        }
    };

    (.. $end:literal) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: Some($end),
                step: 1,
            });
            specs
        }
    };

    (..) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: None,
                step: 1,
            });
            specs
        }
    };

    // Inclusive ranges: start..=end
    ($start:literal ..= $end:literal) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: $start,
                end: Some($end + 1),
                step: 1,
            });
            specs
        }
    };

    // Inclusive ranges with step: start..=end;step
    ($start:literal ..= $end:literal ; $step:literal) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: $start,
                end: Some($end + 1),
                step: $step,
            });
            specs
        }
    };

    // Inclusive range to end: ..=end
    (..= $end:literal) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: Some($end + 1),
                step: 1,
            });
            specs
        }
    };

    // Inclusive range to end with step: ..=end;step
    (..= $end:literal ; $step:literal) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: Some($end + 1),
                step: $step,
            });
            specs
        }
    };

    // Mixed 2D patterns: step range with normal range
    (.. ; $step:literal, $start2:literal .. $end2:literal $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: None,
                step: $step,
            });
            specs.push($crate::SliceElem::Range {
                start: $start2,
                end: Some($end2),
                step: 1,
            });
            specs
        }
    };

    ($start1:literal .. ; $step:literal, $start2:literal .. $end2:literal $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: $start1,
                end: None,
                step: $step,
            });
            specs.push($crate::SliceElem::Range {
                start: $start2,
                end: Some($end2),
                step: 1,
            });
            specs
        }
    };

    ($start1:literal .. $end1:literal ; $step:literal, $start2:literal .. $end2:literal $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: $start1,
                end: Some($end1),
                step: $step,
            });
            specs.push($crate::SliceElem::Range {
                start: $start2,
                end: Some($end2),
                step: 1,
            });
            specs
        }
    };

    // Mixed patterns with from ranges and normal ranges
    ($start1:literal .., $start2:literal .. $end2:literal $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: $start1,
                end: None,
                step: 1,
            });
            specs.push($crate::SliceElem::Range {
                start: $start2,
                end: Some($end2),
                step: 1,
            });
            specs
        }
    };

    ($start1:literal .., .. $end2:literal $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: $start1,
                end: None,
                step: 1,
            });
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: Some($end2),
                step: 1,
            });
            specs
        }
    };

    // Index with from range patterns
    ($idx:literal, $start:literal .. $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Index($idx));
            specs.push($crate::SliceElem::Range {
                start: $start,
                end: None,
                step: 1,
            });
            specs
        }
    };

    ($idx:literal, .. $end:literal $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Index($idx));
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: Some($end),
                step: 1,
            });
            specs
        }
    };

    // 2D range slicing common patterns
    ($start1:literal .. $end1:literal, $start2:literal .. $end2:literal $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: $start1,
                end: Some($end1),
                step: 1,
            });
            specs.push($crate::SliceElem::Range {
                start: $start2,
                end: Some($end2),
                step: 1,
            });
            specs
        }
    };

    // Mixed index and range patterns
    ($idx:literal, $start:literal .. $end:literal $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Index($idx));
            specs.push($crate::SliceElem::Range {
                start: $start,
                end: Some($end),
                step: 1,
            });
            specs
        }
    };

    ($start:literal .. $end:literal, $idx:literal $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: $start,
                end: Some($end),
                step: 1,
            });
            specs.push($crate::SliceElem::Index($idx));
            specs
        }
    };

    // Mixed patterns with full range
    ($idx:literal, .. $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Index($idx));
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: None,
                step: 1,
            });
            specs
        }
    };

    (.., $idx:literal $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: None,
                step: 1,
            });
            specs.push($crate::SliceElem::Index($idx));
            specs
        }
    };

    // 2D full slice
    (.., .. $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: None,
                step: 1,
            });
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: None,
                step: 1,
            });
            specs
        }
    };

    // Single expression fallback
    ($expr:expr) => {
        {
            let elem: $crate::SliceElem = $expr.into();
            let mut specs = $crate::SliceSpecs::empty();
            specs.push(elem);
            specs
        }
    };

    // Two expressions
    ($expr1:expr, $expr2:expr $(,)?) => {
        {
            let elem1: $crate::SliceElem = $expr1.into();
            let elem2: $crate::SliceElem = $expr2.into();
            let mut specs = $crate::SliceSpecs::empty();
            specs.push(elem1);
            specs.push(elem2);
            specs
        }
    };

    // Three expressions
    ($expr1:expr, $expr2:expr, $expr3:expr $(,)?) => {
        {
            let elem1: $crate::SliceElem = $expr1.into();
            let elem2: $crate::SliceElem = $expr2.into();
            let elem3: $crate::SliceElem = $expr3.into();
            let mut specs = $crate::SliceSpecs::empty();
            specs.push(elem1);
            specs.push(elem2);
            specs.push(elem3);
            specs
        }
    };

    // Four expressions
    ($expr1:expr, $expr2:expr, $expr3:expr, $expr4:expr $(,)?) => {
        {
            let elem1: $crate::SliceElem = $expr1.into();
            let elem2: $crate::SliceElem = $expr2.into();
            let elem3: $crate::SliceElem = $expr3.into();
            let elem4: $crate::SliceElem = $expr4.into();
            let mut specs = $crate::SliceSpecs::empty();
            specs.push(elem1);
            specs.push(elem2);
            specs.push(elem3);
            specs.push(elem4);
            specs
        }
    };

    // NewAxis support - None for adding new dimensions
    (None) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::NewAxis);
            specs
        }
    };

    // Mixed NewAxis patterns
    (None, $expr:expr $(,)?) => {
        {
            let elem: $crate::SliceElem = $expr.into();
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::NewAxis);
            specs.push(elem);
            specs
        }
    };

    ($expr:expr, None $(,)?) => {
        {
            let elem: $crate::SliceElem = $expr.into();
            let mut specs = $crate::SliceSpecs::empty();
            specs.push(elem);
            specs.push($crate::SliceElem::NewAxis);
            specs
        }
    };

    // NewAxis with literal index
    (None, $idx:literal $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::NewAxis);
            specs.push($crate::SliceElem::Index($idx));
            specs
        }
    };

    ($idx:literal, None $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Index($idx));
            specs.push($crate::SliceElem::NewAxis);
            specs
        }
    };

    // NewAxis with range
    (None, .. $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::NewAxis);
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: None,
                step: 1,
            });
            specs
        }
    };

    (.., None $(,)?) => {
        {
            let mut specs = $crate::SliceSpecs::empty();
            specs.push($crate::SliceElem::Range {
                start: 0,
                end: None,
                step: 1,
            });
            specs.push($crate::SliceElem::NewAxis);
            specs
        }
    };

    // General case for multiple expressions
    ($($expr:expr),+ $(,)?) => {
        {
            let elems = [$($expr.into()),+];
            $crate::SliceSpecs::from_slice(&elems)
        }
    };
}
