use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::{ArrayN, MAX_DIM};

pub type SliceSpecs = ArrayN<SliceElem, { MAX_DIM }>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceElem {
    Index(isize),
    Range {
        start: isize,
        end: Option<isize>,
        step: isize,
    },
    NewAxis,
}

impl Default for SliceElem {
    fn default() -> Self {
        SliceElem::Index(0)
    }
}

impl std::fmt::Display for SliceElem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SliceElem::Index(idx) => write!(f, "{idx}"),
            SliceElem::Range { start, end, step } => match (end, *step) {
                (None, 1) => write!(f, "{start}:.."),
                (None, step) => write!(f, "{start}..;{step}"),
                (Some(end), 1) => write!(f, "{start}..{end}"),
                (Some(end), step) => write!(f, "{start}..{end};{step}"),
            },
            SliceElem::NewAxis => write!(f, "NewAxis"),
        }
    }
}

impl From<isize> for SliceElem {
    fn from(index: isize) -> Self {
        SliceElem::Index(index)
    }
}

impl From<usize> for SliceElem {
    fn from(index: usize) -> Self {
        SliceElem::Index(index as isize)
    }
}

impl From<i32> for SliceElem {
    fn from(index: i32) -> Self {
        SliceElem::Index(index as isize)
    }
}

impl From<Range<usize>> for SliceElem {
    fn from(range: Range<usize>) -> Self {
        SliceElem::Range {
            start: range.start as isize,
            end: Some(range.end as isize),
            step: 1,
        }
    }
}

impl From<Range<isize>> for SliceElem {
    fn from(range: Range<isize>) -> Self {
        SliceElem::Range {
            start: range.start,
            end: Some(range.end),
            step: 1,
        }
    }
}

impl From<Range<i32>> for SliceElem {
    fn from(range: Range<i32>) -> Self {
        SliceElem::Range {
            start: range.start as isize,
            end: Some(range.end as isize),
            step: 1,
        }
    }
}

impl From<RangeFrom<usize>> for SliceElem {
    fn from(range: RangeFrom<usize>) -> Self {
        SliceElem::Range {
            start: range.start as isize,
            end: None,
            step: 1,
        }
    }
}

impl From<RangeFrom<isize>> for SliceElem {
    fn from(range: RangeFrom<isize>) -> Self {
        SliceElem::Range {
            start: range.start,
            end: None,
            step: 1,
        }
    }
}

impl From<RangeFrom<i32>> for SliceElem {
    fn from(range: RangeFrom<i32>) -> Self {
        SliceElem::Range {
            start: range.start as isize,
            end: None,
            step: 1,
        }
    }
}

impl From<RangeTo<usize>> for SliceElem {
    fn from(range: RangeTo<usize>) -> Self {
        SliceElem::Range {
            start: 0,
            end: Some(range.end as isize),
            step: 1,
        }
    }
}

impl From<RangeTo<isize>> for SliceElem {
    fn from(range: RangeTo<isize>) -> Self {
        SliceElem::Range {
            start: 0,
            end: Some(range.end),
            step: 1,
        }
    }
}

impl From<RangeTo<i32>> for SliceElem {
    fn from(range: RangeTo<i32>) -> Self {
        SliceElem::Range {
            start: 0,
            end: Some(range.end as isize),
            step: 1,
        }
    }
}

impl From<RangeFull> for SliceElem {
    fn from(_: RangeFull) -> Self {
        SliceElem::Range {
            start: 0,
            end: None,
            step: 1,
        }
    }
}

impl From<RangeInclusive<usize>> for SliceElem {
    fn from(range: RangeInclusive<usize>) -> Self {
        SliceElem::Range {
            start: *range.start() as isize,
            end: Some(*range.end() as isize),
            step: 1,
        }
    }
}

impl From<RangeInclusive<isize>> for SliceElem {
    fn from(range: RangeInclusive<isize>) -> Self {
        SliceElem::Range {
            start: *range.start(),
            end: Some(*range.end()),
            step: 1,
        }
    }
}

impl From<RangeInclusive<i32>> for SliceElem {
    fn from(range: RangeInclusive<i32>) -> Self {
        SliceElem::Range {
            start: *range.start() as isize,
            end: Some(*range.end() as isize),
            step: 1,
        }
    }
}

impl From<RangeToInclusive<usize>> for SliceElem {
    fn from(range: RangeToInclusive<usize>) -> Self {
        SliceElem::Range {
            start: 0,
            end: Some(range.end as isize),
            step: 1,
        }
    }
}

impl From<RangeToInclusive<isize>> for SliceElem {
    fn from(range: RangeToInclusive<isize>) -> Self {
        SliceElem::Range {
            start: 0,
            end: Some(range.end),
            step: 1,
        }
    }
}

impl From<RangeToInclusive<i32>> for SliceElem {
    fn from(range: RangeToInclusive<i32>) -> Self {
        SliceElem::Range {
            start: 0,
            end: Some(range.end as isize),
            step: 1,
        }
    }
}

// Support for None as NewAxis
impl From<Option<()>> for SliceElem {
    fn from(_: Option<()>) -> Self {
        SliceElem::NewAxis
    }
}

// Additional numeric type support for indices
impl From<i8> for SliceElem {
    fn from(index: i8) -> Self {
        SliceElem::Index(index as isize)
    }
}

impl From<i16> for SliceElem {
    fn from(index: i16) -> Self {
        SliceElem::Index(index as isize)
    }
}

impl From<i64> for SliceElem {
    fn from(index: i64) -> Self {
        SliceElem::Index(index as isize)
    }
}

impl From<u8> for SliceElem {
    fn from(index: u8) -> Self {
        SliceElem::Index(index as isize)
    }
}

impl From<u16> for SliceElem {
    fn from(index: u16) -> Self {
        SliceElem::Index(index as isize)
    }
}

impl From<u32> for SliceElem {
    fn from(index: u32) -> Self {
        SliceElem::Index(index as isize)
    }
}

impl From<u64> for SliceElem {
    fn from(index: u64) -> Self {
        SliceElem::Index(index as isize)
    }
}

// Additional range type support
impl From<Range<i8>> for SliceElem {
    fn from(range: Range<i8>) -> Self {
        SliceElem::Range {
            start: range.start as isize,
            end: Some(range.end as isize),
            step: 1,
        }
    }
}

impl From<Range<i16>> for SliceElem {
    fn from(range: Range<i16>) -> Self {
        SliceElem::Range {
            start: range.start as isize,
            end: Some(range.end as isize),
            step: 1,
        }
    }
}

impl From<Range<i64>> for SliceElem {
    fn from(range: Range<i64>) -> Self {
        SliceElem::Range {
            start: range.start as isize,
            end: Some(range.end as isize),
            step: 1,
        }
    }
}

impl From<Range<u8>> for SliceElem {
    fn from(range: Range<u8>) -> Self {
        SliceElem::Range {
            start: range.start as isize,
            end: Some(range.end as isize),
            step: 1,
        }
    }
}

impl From<Range<u16>> for SliceElem {
    fn from(range: Range<u16>) -> Self {
        SliceElem::Range {
            start: range.start as isize,
            end: Some(range.end as isize),
            step: 1,
        }
    }
}

impl From<Range<u32>> for SliceElem {
    fn from(range: Range<u32>) -> Self {
        SliceElem::Range {
            start: range.start as isize,
            end: Some(range.end as isize),
            step: 1,
        }
    }
}

impl From<Range<u64>> for SliceElem {
    fn from(range: Range<u64>) -> Self {
        SliceElem::Range {
            start: range.start as isize,
            end: Some(range.end as isize),
            step: 1,
        }
    }
}

pub trait IntoSliceElem {
    fn into_slice(self) -> SliceSpecs;
}

// Ultra-fast implementations that bypass conversion overhead

macro_rules! impl_into_slice_elem_for_tuple {
    ($($len:expr => ($($T:ident : $v:ident),+)),+) => {
        $(
            impl<$($T),+> IntoSliceElem for ($($T,)+)
            where
                $($T: Into<SliceElem>),+
            {
                fn into_slice(self) -> SliceSpecs {
                    let ($($v,)+) = self;
                    ArrayN::from_slice(&[$($v.into()),+])
                }
            }
        )+
    };
}

impl_into_slice_elem_for_tuple!(
    1 => (T1: t1),
    2 => (T1: t1, T2: t2),
    3 => (T1: t1, T2: t2, T3: t3),
    4 => (T1: t1, T2: t2, T3: t3, T4: t4),
    5 => (T1: t1, T2: t2, T3: t3, T4: t4, T5: t5),
    6 => (T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6),
    7 => (T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7),
    8 => (T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8)
);

// Single element implementations - direct construction
impl IntoSliceElem for SliceElem {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = self;
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for usize {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Index(self as isize);
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for isize {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Index(self);
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for i32 {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Index(self as isize);
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for Range<usize> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: self.start as isize,
            end: Some(self.end as isize),
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for Range<isize> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: self.start,
            end: Some(self.end),
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for Range<i32> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: self.start as isize,
            end: Some(self.end as isize),
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for RangeFrom<usize> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: self.start as isize,
            end: None,
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for RangeFrom<isize> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: self.start,
            end: None,
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for RangeFrom<i32> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: self.start as isize,
            end: None,
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for RangeTo<usize> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: 0,
            end: Some(self.end as isize),
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for RangeTo<isize> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: 0,
            end: Some(self.end),
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for RangeTo<i32> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: 0,
            end: Some(self.end as isize),
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for RangeFull {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: 0,
            end: None,
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for Vec<SliceElem> {
    #[inline]
    fn into_slice(self) -> SliceSpecs {
        SliceSpecs::from_slice(&self)
    }
}

impl IntoSliceElem for &[SliceElem] {
    #[inline]
    fn into_slice(self) -> SliceSpecs {
        SliceSpecs::from_slice(self)
    }
}

impl IntoSliceElem for SliceSpecs {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        self
    }
}

impl IntoSliceElem for RangeInclusive<usize> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: *self.start() as isize,
            end: Some(*self.end() as isize + 1),
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for RangeInclusive<isize> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: *self.start(),
            end: Some(*self.end() + 1),
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for RangeInclusive<i32> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: *self.start() as isize,
            end: Some(*self.end() as isize + 1),
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for RangeToInclusive<usize> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: 0,
            end: Some(self.end as isize + 1),
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for RangeToInclusive<isize> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: 0,
            end: Some(self.end + 1),
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}

impl IntoSliceElem for RangeToInclusive<i32> {
    #[inline(always)]
    fn into_slice(self) -> SliceSpecs {
        let mut specs = SliceSpecs::empty();
        specs.arr[0] = SliceElem::Range {
            start: 0,
            end: Some(self.end as isize + 1),
            step: 1,
        };
        unsafe {
            specs.set_len(1);
        }
        specs
    }
}
