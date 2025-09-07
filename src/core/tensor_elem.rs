use half::{bf16, f16};

use crate::DType;

/// Trait for types that can be used as tensor elements
pub trait TensorElement: Clone + Send + Sync + 'static + Copy {
    /// The data type of this tensor element
    const DTYPE: DType;

    /// The zero/additive identity value for this type
    const ZERO: Self;

    /// The one/multiplicative identity value for this type
    const ONE: Self;

    /// The minimum value for this type
    const MIN: Self;

    /// The maximum value for this type
    const MAX: Self;

    /// Convert from f32 (for type casting)
    fn from_f32(val: f32) -> Self;

    /// Convert to f32 (for type casting)
    fn to_f32(self) -> f32;

    /// Convert from f64 (for type casting)
    fn from_f64(val: f64) -> Self;

    /// Convert to f64 (for type casting)
    fn to_f64(self) -> f64;

    /// Check if value is zero
    fn is_zero(self) -> bool;

    /// Zero value for this type (for backward compatibility)
    fn zero() -> Self {
        Self::ZERO
    }

    /// One value for this type (for backward compatibility)
    fn one() -> Self {
        Self::ONE
    }

    /// Get the dtype for this type
    fn dtype() -> DType {
        Self::DTYPE
    }

    /// Infinity value (for division by zero)
    fn infinity() -> Self;

    fn from_u8(val: u8) -> Self;
    fn from_u16(val: u16) -> Self;
    fn from_u32(val: u32) -> Self;
    fn from_u64(val: u64) -> Self;
    fn from_i8(val: i8) -> Self;
    fn from_i16(val: i16) -> Self;
    fn from_i32(val: i32) -> Self;
    fn from_i64(val: i64) -> Self;
    fn from_bool(val: bool) -> Self;
    fn from_f16(val: f16) -> Self;
    fn from_bf16(val: bf16) -> Self;
}

impl TensorElement for f32 {
    const DTYPE: DType = DType::Fp32;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN: Self = f32::MIN;
    const MAX: Self = f32::MAX;

    fn from_f32(val: f32) -> Self {
        val
    }

    fn to_f32(self) -> f32 {
        self
    }

    fn from_f64(val: f64) -> Self {
        val as f32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn is_zero(self) -> bool {
        self == 0.0
    }

    fn infinity() -> Self {
        f32::INFINITY
    }

    fn from_u8(val: u8) -> Self {
        val as f32
    }

    fn from_u16(val: u16) -> Self {
        val as f32
    }

    fn from_u32(val: u32) -> Self {
        val as f32
    }

    fn from_u64(val: u64) -> Self {
        val as f32
    }

    fn from_i8(val: i8) -> Self {
        val as f32
    }

    fn from_i16(val: i16) -> Self {
        val as f32
    }

    fn from_i32(val: i32) -> Self {
        val as f32
    }

    fn from_i64(val: i64) -> Self {
        val as f32
    }

    fn from_bool(val: bool) -> Self {
        val as u8 as f32
    }

    fn from_f16(val: f16) -> Self {
        val.to_f32()
    }

    fn from_bf16(val: bf16) -> Self {
        val.to_f32()
    }
}

impl TensorElement for f64 {
    const DTYPE: DType = DType::Fp64;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN: Self = f64::MIN;
    const MAX: Self = f64::MAX;

    fn from_f32(val: f32) -> Self {
        val as f64
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_f64(val: f64) -> Self {
        val
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn is_zero(self) -> bool {
        self == 0.0
    }

    fn infinity() -> Self {
        f64::INFINITY
    }

    fn from_u8(val: u8) -> Self {
        val as f64
    }

    fn from_u16(val: u16) -> Self {
        val as f64
    }

    fn from_u32(val: u32) -> Self {
        val as f64
    }

    fn from_u64(val: u64) -> Self {
        val as f64
    }

    fn from_i8(val: i8) -> Self {
        val as f64
    }

    fn from_i16(val: i16) -> Self {
        val as f64
    }

    fn from_i32(val: i32) -> Self {
        val as f64
    }

    fn from_i64(val: i64) -> Self {
        val as f64
    }

    fn from_bool(val: bool) -> Self {
        val as u8 as f64
    }

    fn from_f16(val: f16) -> Self {
        val.to_f64()
    }

    fn from_bf16(val: bf16) -> Self {
        val.to_f64()
    }
}

impl TensorElement for f16 {
    const DTYPE: DType = DType::Fp16;
    const ZERO: Self = f16::ZERO;
    const ONE: Self = f16::ONE;
    const MIN: Self = f16::MIN;
    const MAX: Self = f16::MAX;

    fn from_f32(val: f32) -> Self {
        f16::from_f32(val)
    }

    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn is_zero(self) -> bool {
        self.to_f32() == 0.0
    }

    fn from_f64(val: f64) -> Self {
        f16::from_f64(val)
    }

    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    fn infinity() -> Self {
        f16::INFINITY
    }

    fn from_u8(val: u8) -> Self {
        f16::from_f32(val as f32)
    }

    fn from_u16(val: u16) -> Self {
        f16::from_f32(val as f32)
    }

    fn from_u32(val: u32) -> Self {
        f16::from_f32(val as f32)
    }

    fn from_u64(val: u64) -> Self {
        f16::from_f32(val as f32)
    }

    fn from_i8(val: i8) -> Self {
        f16::from_f32(val as f32)
    }

    fn from_i16(val: i16) -> Self {
        f16::from_f32(val as f32)
    }

    fn from_i32(val: i32) -> Self {
        f16::from_f32(val as f32)
    }

    fn from_i64(val: i64) -> Self {
        f16::from_f32(val as f32)
    }

    fn from_bool(val: bool) -> Self {
        f16::from_f32(val as u8 as f32)
    }

    fn from_f16(val: f16) -> Self {
        val
    }

    fn from_bf16(val: bf16) -> Self {
        f16::from_f32(val.to_f32())
    }
}

impl TensorElement for bf16 {
    const DTYPE: DType = DType::Bf16;
    const ZERO: Self = bf16::ZERO;
    const ONE: Self = bf16::ONE;
    const MIN: Self = bf16::MIN;
    const MAX: Self = bf16::MAX;

    fn from_f32(val: f32) -> Self {
        bf16::from_f32(val)
    }

    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn is_zero(self) -> bool {
        self.to_f32() == 0.0
    }

    fn from_f64(val: f64) -> Self {
        bf16::from_f64(val)
    }

    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    fn infinity() -> Self {
        bf16::INFINITY
    }

    fn from_u8(val: u8) -> Self {
        bf16::from_f32(val as f32)
    }

    fn from_u16(val: u16) -> Self {
        bf16::from_f32(val as f32)
    }

    fn from_u32(val: u32) -> Self {
        bf16::from_f32(val as f32)
    }

    fn from_u64(val: u64) -> Self {
        bf16::from_f32(val as f32)
    }

    fn from_i8(val: i8) -> Self {
        bf16::from_f32(val as f32)
    }

    fn from_i16(val: i16) -> Self {
        bf16::from_f32(val as f32)
    }

    fn from_i32(val: i32) -> Self {
        bf16::from_f32(val as f32)
    }

    fn from_i64(val: i64) -> Self {
        bf16::from_f32(val as f32)
    }

    fn from_bool(val: bool) -> Self {
        bf16::from_f32(val as u8 as f32)
    }

    fn from_f16(val: f16) -> Self {
        bf16::from_f32(val.to_f32())
    }

    fn from_bf16(val: bf16) -> Self {
        val
    }
}

impl TensorElement for i32 {
    const DTYPE: DType = DType::Int32;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MIN: Self = i32::MIN;
    const MAX: Self = i32::MAX;

    fn from_f32(val: f32) -> Self {
        val as i32
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_f64(val: f64) -> Self {
        val as i32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn is_zero(self) -> bool {
        self == 0
    }

    fn infinity() -> Self {
        i32::MAX
    }

    fn from_u8(val: u8) -> Self {
        val as i32
    }

    fn from_u16(val: u16) -> Self {
        val as i32
    }

    fn from_u32(val: u32) -> Self {
        val as i32
    }

    fn from_u64(val: u64) -> Self {
        val as i32
    }

    fn from_i8(val: i8) -> Self {
        val as i32
    }

    fn from_i16(val: i16) -> Self {
        val as i32
    }

    fn from_i32(val: i32) -> Self {
        val
    }

    fn from_i64(val: i64) -> Self {
        val as i32
    }

    fn from_bool(val: bool) -> Self {
        val as i32
    }

    fn from_f16(val: f16) -> Self {
        val.to_f32() as i32
    }

    fn from_bf16(val: bf16) -> Self {
        val.to_f32() as i32
    }
}

impl TensorElement for i64 {
    const DTYPE: DType = DType::Int64;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MIN: Self = i64::MIN;
    const MAX: Self = i64::MAX;

    fn from_f32(val: f32) -> Self {
        val as i64
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_f64(val: f64) -> Self {
        val as i64
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn is_zero(self) -> bool {
        self == 0
    }

    fn infinity() -> Self {
        i64::MAX
    }

    fn from_u8(val: u8) -> Self {
        val as i64
    }

    fn from_u16(val: u16) -> Self {
        val as i64
    }

    fn from_u32(val: u32) -> Self {
        val as i64
    }

    fn from_u64(val: u64) -> Self {
        val as i64
    }

    fn from_i8(val: i8) -> Self {
        val as i64
    }

    fn from_i16(val: i16) -> Self {
        val as i64
    }

    fn from_i32(val: i32) -> Self {
        val as i64
    }

    fn from_i64(val: i64) -> Self {
        val
    }

    fn from_bool(val: bool) -> Self {
        val as i64
    }

    fn from_f16(val: f16) -> Self {
        val.to_f32() as i64
    }

    fn from_bf16(val: bf16) -> Self {
        val.to_f32() as i64
    }
}

impl TensorElement for u8 {
    const DTYPE: DType = DType::Uint8;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MIN: Self = u8::MIN;
    const MAX: Self = u8::MAX;

    fn from_f32(val: f32) -> Self {
        val as u8
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_f64(val: f64) -> Self {
        val as u8
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn is_zero(self) -> bool {
        self == 0
    }

    fn infinity() -> Self {
        u8::MAX
    }

    fn from_u8(val: u8) -> Self {
        val
    }

    fn from_u16(val: u16) -> Self {
        val as u8
    }

    fn from_u32(val: u32) -> Self {
        val as u8
    }

    fn from_u64(val: u64) -> Self {
        val as u8
    }

    fn from_i8(val: i8) -> Self {
        val as u8
    }

    fn from_i16(val: i16) -> Self {
        val as u8
    }

    fn from_i32(val: i32) -> Self {
        val as u8
    }

    fn from_i64(val: i64) -> Self {
        val as u8
    }

    fn from_bool(val: bool) -> Self {
        val as u8
    }

    fn from_f16(val: f16) -> Self {
        val.to_f32() as u8
    }

    fn from_bf16(val: bf16) -> Self {
        val.to_f32() as u8
    }
}

impl TensorElement for u16 {
    const DTYPE: DType = DType::Uint16;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MIN: Self = u16::MIN;
    const MAX: Self = u16::MAX;

    fn from_f32(val: f32) -> Self {
        val as u16
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_f64(val: f64) -> Self {
        val as u16
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn is_zero(self) -> bool {
        self == 0
    }

    fn infinity() -> Self {
        u16::MAX
    }

    fn from_u8(val: u8) -> Self {
        val as u16
    }

    fn from_u16(val: u16) -> Self {
        val
    }

    fn from_u32(val: u32) -> Self {
        val as u16
    }

    fn from_u64(val: u64) -> Self {
        val as u16
    }

    fn from_i8(val: i8) -> Self {
        val as u16
    }

    fn from_i16(val: i16) -> Self {
        val as u16
    }

    fn from_i32(val: i32) -> Self {
        val as u16
    }

    fn from_i64(val: i64) -> Self {
        val as u16
    }

    fn from_bool(val: bool) -> Self {
        val as u16
    }

    fn from_f16(val: f16) -> Self {
        val.to_f32() as u16
    }

    fn from_bf16(val: bf16) -> Self {
        val.to_f32() as u16
    }
}

impl TensorElement for u32 {
    const DTYPE: DType = DType::Uint32;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MIN: Self = u32::MIN;
    const MAX: Self = u32::MAX;

    fn from_f32(val: f32) -> Self {
        val as u32
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_f64(val: f64) -> Self {
        val as u32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn is_zero(self) -> bool {
        self == 0
    }

    fn infinity() -> Self {
        u32::MAX
    }

    fn from_u8(val: u8) -> Self {
        val as u32
    }

    fn from_u16(val: u16) -> Self {
        val as u32
    }

    fn from_u32(val: u32) -> Self {
        val
    }

    fn from_u64(val: u64) -> Self {
        val as u32
    }

    fn from_i8(val: i8) -> Self {
        val as u32
    }

    fn from_i16(val: i16) -> Self {
        val as u32
    }

    fn from_i32(val: i32) -> Self {
        val as u32
    }

    fn from_i64(val: i64) -> Self {
        val as u32
    }

    fn from_bool(val: bool) -> Self {
        val as u32
    }

    fn from_f16(val: f16) -> Self {
        val.to_f32() as u32
    }

    fn from_bf16(val: bf16) -> Self {
        val.to_f32() as u32
    }
}

impl TensorElement for u64 {
    const DTYPE: DType = DType::Uint64;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MIN: Self = u64::MIN;
    const MAX: Self = u64::MAX;

    fn from_f32(val: f32) -> Self {
        val as u64
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_f64(val: f64) -> Self {
        val as u64
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn is_zero(self) -> bool {
        self == 0
    }

    fn infinity() -> Self {
        u64::MAX
    }

    fn from_u8(val: u8) -> Self {
        val as u64
    }

    fn from_u16(val: u16) -> Self {
        val as u64
    }

    fn from_u32(val: u32) -> Self {
        val as u64
    }

    fn from_u64(val: u64) -> Self {
        val
    }

    fn from_i8(val: i8) -> Self {
        val as u64
    }

    fn from_i16(val: i16) -> Self {
        val as u64
    }

    fn from_i32(val: i32) -> Self {
        val as u64
    }

    fn from_i64(val: i64) -> Self {
        val as u64
    }

    fn from_bool(val: bool) -> Self {
        val as u64
    }

    fn from_f16(val: f16) -> Self {
        val.to_f32() as u64
    }

    fn from_bf16(val: bf16) -> Self {
        val.to_f32() as u64
    }
}

impl TensorElement for bool {
    const DTYPE: DType = DType::Bool;
    const ZERO: Self = false;
    const ONE: Self = true;
    const MIN: Self = false;
    const MAX: Self = true;

    fn from_f32(val: f32) -> Self {
        val != 0.0
    }

    fn to_f32(self) -> f32 {
        if self {
            1.0
        } else {
            0.0
        }
    }

    fn from_f64(val: f64) -> Self {
        val != 0.0
    }

    fn to_f64(self) -> f64 {
        if self {
            1.0
        } else {
            0.0
        }
    }

    fn is_zero(self) -> bool {
        !self
    }

    fn infinity() -> Self {
        true
    }

    fn from_u8(val: u8) -> Self {
        val != 0
    }

    fn from_u16(val: u16) -> Self {
        val != 0
    }

    fn from_u32(val: u32) -> Self {
        val != 0
    }

    fn from_u64(val: u64) -> Self {
        val != 0
    }

    fn from_i8(val: i8) -> Self {
        val != 0
    }

    fn from_i16(val: i16) -> Self {
        val != 0
    }

    fn from_i32(val: i32) -> Self {
        val != 0
    }

    fn from_i64(val: i64) -> Self {
        val != 0
    }

    fn from_bool(val: bool) -> Self {
        val
    }

    fn from_f16(val: f16) -> Self {
        val.to_f32() != 0.0
    }

    fn from_bf16(val: bf16) -> Self {
        val.to_f32() != 0.0
    }
}

impl TensorElement for i8 {
    const DTYPE: DType = DType::Int8;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MIN: Self = i8::MIN;
    const MAX: Self = i8::MAX;

    fn from_f32(val: f32) -> Self {
        val as i8
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_f64(val: f64) -> Self {
        val as i8
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn is_zero(self) -> bool {
        self == 0
    }

    fn infinity() -> Self {
        i8::MAX
    }

    fn from_u8(val: u8) -> Self {
        val as i8
    }

    fn from_u16(val: u16) -> Self {
        val as i8
    }

    fn from_u32(val: u32) -> Self {
        val as i8
    }

    fn from_u64(val: u64) -> Self {
        val as i8
    }

    fn from_i8(val: i8) -> Self {
        val
    }

    fn from_i16(val: i16) -> Self {
        val as i8
    }

    fn from_i32(val: i32) -> Self {
        val as i8
    }

    fn from_i64(val: i64) -> Self {
        val as i8
    }

    fn from_bool(val: bool) -> Self {
        val as i8
    }

    fn from_f16(val: f16) -> Self {
        val.to_f32() as i8
    }

    fn from_bf16(val: bf16) -> Self {
        val.to_f32() as i8
    }
}

impl TensorElement for i16 {
    const DTYPE: DType = DType::Int16;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MIN: Self = i16::MIN;
    const MAX: Self = i16::MAX;

    fn from_f32(val: f32) -> Self {
        val as i16
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_f64(val: f64) -> Self {
        val as i16
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn is_zero(self) -> bool {
        self == 0
    }

    fn infinity() -> Self {
        i16::MAX
    }

    fn from_u8(val: u8) -> Self {
        val as i16
    }

    fn from_u16(val: u16) -> Self {
        val as i16
    }

    fn from_u32(val: u32) -> Self {
        val as i16
    }

    fn from_u64(val: u64) -> Self {
        val as i16
    }

    fn from_i8(val: i8) -> Self {
        val as i16
    }

    fn from_i16(val: i16) -> Self {
        val
    }

    fn from_i32(val: i32) -> Self {
        val as i16
    }

    fn from_i64(val: i64) -> Self {
        val as i16
    }

    fn from_bool(val: bool) -> Self {
        val as i16
    }

    fn from_f16(val: f16) -> Self {
        val.to_f32() as i16
    }

    fn from_bf16(val: bf16) -> Self {
        val.to_f32() as i16
    }
}
