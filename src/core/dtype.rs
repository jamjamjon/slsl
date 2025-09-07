//! Data type definitions for tensor elements.
//!
//! This module provides the [`DType`] enum that represents all supported
//! data types in the tensor library, along with utility functions for
//! working with these types.

/// Enumeration of all supported data types for tensor elements.
///
/// This enum represents the various numeric types that can be stored
/// in tensors, including integers, floating-point numbers, and boolean values.
/// Each variant corresponds to a specific Rust primitive type or half-precision type.
///
/// # Examples
///
/// ```rust
/// use slsl::DType;
///
/// let dtype = DType::Fp32;
/// assert_eq!(dtype.size_in_bytes(), 4);
/// assert!(dtype.is_float());
///
/// let int_dtype = DType::Int32;
/// assert_eq!(int_dtype.size_in_bytes(), 4);
/// assert!(!int_dtype.is_float());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DType {
    /// Boolean type (`bool`)
    Bool,
    /// 8-bit signed integer (`i8`)
    Int8,
    /// 16-bit signed integer (`i16`)
    Int16,
    /// 32-bit signed integer (`i32`)
    Int32,
    /// 64-bit signed integer (`i64`)
    Int64,
    /// 8-bit unsigned integer (`u8`)
    Uint8,
    /// 16-bit unsigned integer (`u16`)
    Uint16,
    /// 32-bit unsigned integer (`u32`)
    Uint32,
    /// 64-bit unsigned integer (`u64`)
    Uint64,
    /// 16-bit floating-point number (half-precision)
    Fp16,
    /// 32-bit floating-point number (single-precision)
    Fp32,
    /// 64-bit floating-point number (double-precision)
    Fp64,
    /// 16-bit brain floating-point number
    Bf16,
    /// Placeholder variant - should not be used in practice, use _
    Others,
}

impl DType {
    /// Returns the size in bytes of this data type.
    ///
    /// This method returns the memory footprint of a single element
    /// of this data type, which is useful for memory allocation
    /// and layout calculations.
    ///
    /// # Returns
    ///
    /// The size in bytes as a `usize`.
    ///
    /// # Panics
    ///
    /// Panics if called on [`DType::Others`], as this is a placeholder
    /// variant and does not correspond to a concrete type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::DType;
    ///
    /// assert_eq!(DType::Bool.size_in_bytes(), 1);
    /// assert_eq!(DType::Int32.size_in_bytes(), 4);
    /// assert_eq!(DType::Fp64.size_in_bytes(), 8);
    /// assert_eq!(DType::Fp16.size_in_bytes(), 2);
    /// ```
    #[inline(always)]
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::Bool => std::mem::size_of::<bool>(),
            DType::Int8 => std::mem::size_of::<i8>(),
            DType::Int16 => std::mem::size_of::<i16>(),
            DType::Int32 => std::mem::size_of::<i32>(),
            DType::Int64 => std::mem::size_of::<i64>(),
            DType::Uint8 => std::mem::size_of::<u8>(),
            DType::Uint16 => std::mem::size_of::<u16>(),
            DType::Uint32 => std::mem::size_of::<u32>(),
            DType::Uint64 => std::mem::size_of::<u64>(),
            DType::Fp16 => std::mem::size_of::<half::f16>(),
            DType::Fp32 => std::mem::size_of::<f32>(),
            DType::Fp64 => std::mem::size_of::<f64>(),
            DType::Bf16 => std::mem::size_of::<half::bf16>(),
            _ => panic!("Cannot get size of Auto dtype"),
        }
    }

    /// Checks if this data type represents a floating-point number.
    ///
    /// Returns `true` for all floating-point variants (including half-precision
    /// and brain floating-point types), and `false` for integer and boolean types.
    ///
    /// # Returns
    ///
    /// `true` if the data type is floating-point, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::DType;
    ///
    /// // Floating-point types
    /// assert!(DType::Fp16.is_float());
    /// assert!(DType::Fp32.is_float());
    /// assert!(DType::Fp64.is_float());
    /// assert!(DType::Bf16.is_float());
    ///
    /// // Non-floating-point types
    /// assert!(!DType::Int32.is_float());
    /// assert!(!DType::Bool.is_float());
    /// assert!(!DType::Uint8.is_float());
    /// ```
    #[inline(always)]
    pub fn is_float(self) -> bool {
        matches!(self, DType::Fp16 | DType::Fp32 | DType::Fp64 | DType::Bf16)
    }

    /// Checks if this data type represents an integer number.
    ///
    /// Returns `true` for all signed and unsigned integer variants,
    /// and `false` for floating-point and boolean types.
    ///
    /// # Returns
    ///
    /// `true` if the data type is an integer, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::DType;
    ///
    /// // Integer types
    /// assert!(DType::Int8.is_integer());
    /// assert!(DType::Int32.is_integer());
    /// assert!(DType::Uint16.is_integer());
    /// assert!(DType::Uint64.is_integer());
    ///
    /// // Non-integer types
    /// assert!(!DType::Fp32.is_integer());
    /// assert!(!DType::Bool.is_integer());
    /// ```
    #[inline(always)]
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            DType::Int8
                | DType::Int16
                | DType::Int32
                | DType::Int64
                | DType::Uint8
                | DType::Uint16
                | DType::Uint32
                | DType::Uint64
        )
    }

    /// Checks if this data type represents a signed integer.
    ///
    /// Returns `true` only for signed integer variants.
    ///
    /// # Returns
    ///
    /// `true` if the data type is a signed integer, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::DType;
    ///
    /// // Signed integer types
    /// assert!(DType::Int8.is_signed_integer());
    /// assert!(DType::Int32.is_signed_integer());
    /// assert!(DType::Int64.is_signed_integer());
    ///
    /// // Non-signed integer types
    /// assert!(!DType::Uint32.is_signed_integer());
    /// assert!(!DType::Fp32.is_signed_integer());
    /// assert!(!DType::Bool.is_signed_integer());
    /// ```
    #[inline(always)]
    pub fn is_signed_integer(self) -> bool {
        matches!(
            self,
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
        )
    }

    /// Checks if this data type represents an unsigned integer.
    ///
    /// Returns `true` only for unsigned integer variants.
    ///
    /// # Returns
    ///
    /// `true` if the data type is an unsigned integer, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::DType;
    ///
    /// // Unsigned integer types
    /// assert!(DType::Uint8.is_unsigned_integer());
    /// assert!(DType::Uint32.is_unsigned_integer());
    /// assert!(DType::Uint64.is_unsigned_integer());
    ///
    /// // Non-unsigned integer types
    /// assert!(!DType::Int32.is_unsigned_integer());
    /// assert!(!DType::Fp32.is_unsigned_integer());
    /// assert!(!DType::Bool.is_unsigned_integer());
    /// ```
    #[inline(always)]
    pub fn is_unsigned_integer(self) -> bool {
        matches!(
            self,
            DType::Uint8 | DType::Uint16 | DType::Uint32 | DType::Uint64
        )
    }
}
