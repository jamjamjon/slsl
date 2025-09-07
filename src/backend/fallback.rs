use crate::OpsTrait;

/// Fallback backend that implements all operations using standard library
/// parallel implementations using rayon for vectorized math functions.
#[derive(Debug)]
pub struct FallbackBackend;

// Do not implement any methods for fallback backend, use default Ops methods!!!
impl OpsTrait for FallbackBackend {}
