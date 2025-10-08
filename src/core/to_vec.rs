use anyhow::Result;

use crate::{StorageTrait, TensorBase, TensorElement, UninitVec};

impl<S: StorageTrait> TensorBase<S> {
    /// Extract scalar value from a 0-dimensional tensor
    ///
    /// # Performance
    /// This method is optimized for zero-cost scalar extraction with compile-time checks.
    ///
    /// # Errors
    /// Returns error if tensor is not 0-dimensional or dtype mismatch
    #[inline(always)]
    pub fn to_scalar<T: TensorElement + Copy>(&self) -> Result<T> {
        // Validate tensor is scalar (0-dimensional)
        if !self.dims().is_empty() {
            return Err(anyhow::anyhow!(
                "Cannot convert {}-dimensional tensor to scalar",
                self.dims().len()
            ));
        }

        debug_assert_eq!(
            self.dtype(),
            T::DTYPE,
            "DType mismatch: tensor has {:?}, requested {:?}",
            self.dtype(),
            T::DTYPE
        );

        unsafe {
            let ptr = self.as_ptr().add(self.offset_bytes) as *const T;
            Ok(*ptr)
        }
    }

    /// Convert tensor to flat `Vec<T>` (any dimensionality)
    ///
    /// # Performance
    /// - Contiguous tensors: Direct memory copy
    /// - Non-contiguous tensors: Using tensor iterator
    ///
    /// # Errors
    /// Returns error if dtype mismatch
    pub fn to_flat_vec<T: TensorElement + Copy>(&self) -> Result<Vec<T>> {
        debug_assert_eq!(
            self.dtype(),
            T::DTYPE,
            "DType mismatch: tensor has {:?}, requested {:?}",
            self.dtype(),
            T::DTYPE
        );
        let numel = self.numel();
        if numel == 0 {
            return Ok(Vec::new());
        }

        let result = if self.is_contiguous() {
            UninitVec::new(numel).init_with(|slice| unsafe {
                let src_ptr = self.as_ptr() as *const T;
                std::ptr::copy_nonoverlapping(src_ptr, slice.as_mut_ptr(), numel);
            })
        } else {
            UninitVec::new(numel).init_with(|slice| {
                for (i, item) in self.iter_with_meta::<T>().enumerate() {
                    slice[i] = *item.value;
                }
            })
        };

        Ok(result)
    }

    /// Convert 1D tensor to `Vec<T>`
    ///
    /// # Performance
    /// - Contiguous tensors: Direct memory copy (memcpy-like performance)
    /// - Non-contiguous tensors: Using tensor iterator
    ///
    /// # Errors
    /// Returns error if tensor is not 1D or dtype mismatch
    pub fn to_vec<T: TensorElement + Copy>(&self) -> Result<Vec<T>> {
        if self.rank() != 1 {
            anyhow::bail!(
                "Cannot convert {}-dimensional tensor to 1D Vec. Try using `.to_flat_vec()` instead.",
                self.rank()
            );
        }
        self.to_flat_vec()
    }

    /// Convert 2D tensor to `Vec<Vec<T>>`
    pub fn to_vec2<T: TensorElement + Copy>(&self) -> Result<Vec<Vec<T>>> {
        if self.rank() != 2 {
            anyhow::bail!(
                "Cannot convert {}-dimensional tensor to 2D Vec. Try using `.to_flat_vec()` instead.",
                self.rank()
            );
        }
        debug_assert_eq!(
            self.dtype(),
            T::DTYPE,
            "DType mismatch: tensor has {:?}, requested {:?}",
            self.dtype(),
            T::DTYPE
        );

        let [rows, cols] = [self.shape()[0], self.shape()[1]];
        let mut result = Vec::with_capacity(rows);
        let mut iter = self.iter_with_meta::<T>();
        for i in 0..rows {
            let row = UninitVec::new(cols).init_with(|row_slice| {
                for (j, row_elem) in row_slice.iter_mut().enumerate().take(cols) {
                    if let Some(item) = iter.next() {
                        *row_elem = *item.value;
                    } else {
                        panic!("Iterator exhausted unexpectedly at row {}, col {}", i, j);
                    }
                }
            });
            result.push(row);
        }

        Ok(result)
    }

    /// Convert 3D tensor to `Vec<Vec<Vec<T>>>`
    pub fn to_vec3<T: TensorElement + Copy>(&self) -> Result<Vec<Vec<Vec<T>>>> {
        if self.rank() != 3 {
            anyhow::bail!(
                "Cannot convert {}-dimensional tensor to 3D Vec. Try using `.to_flat_vec()` instead.",
                self.rank()
            );
        }

        debug_assert_eq!(
            self.dtype(),
            T::DTYPE,
            "DType mismatch: tensor has {:?}, requested {:?}",
            self.dtype(),
            T::DTYPE
        );

        let [dim0, dim1, dim2] = [self.shape()[0], self.shape()[1], self.shape()[2]];
        let mut result: Vec<Vec<Vec<T>>> = Vec::with_capacity(dim0);
        let mut iter = self.iter_with_meta::<T>();
        for i in 0..dim0 {
            let mut plane: Vec<Vec<T>> = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let mut row: Vec<T> = Vec::with_capacity(dim2);
                for k in 0..dim2 {
                    if let Some(item) = iter.next() {
                        row.push(*item.value);
                    } else {
                        panic!("Iterator exhausted unexpectedly at [{}, {}, {}]", i, j, k);
                    }
                }
                plane.push(row);
            }
            result.push(plane);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::{s, Tensor};

    #[test]
    fn test_to_flat_vec_rows_from_real_slices_f32() -> anyhow::Result<()> {
        let rows: [&[f32; 32]; 6] = [
            &[
                -0.573134, 0.440246, -0.305807, -2.028273, 0.232327, -2.410018, 0.769581, 0.383316,
                -3.05553, -0.196661, -0.381331, -1.538175, 0.132812, -0.296569, 0.006016,
                -0.786159, -0.170078, -0.614549, -0.655673, 0.334807, 0.185173, -0.27942,
                -0.602469, 1.092488, -1.670121, -0.89865, 0.425514, 0.94984, 0.662506, -0.204701,
                1.175576, -0.254652,
            ],
            &[
                -1.34039, 0.378945, 0.502759, -2.731842, 0.052494, 0.059023, -0.045204, 0.14522,
                -0.140211, -0.294427, 0.502173, -1.494898, 0.248042, 0.275296, 0.177627, 0.385793,
                -0.211963, 0.054782, 0.137771, 2.180218, -1.541273, 0.43561, -1.450752, 0.714398,
                -0.076198, -0.492589, -0.215164, 0.535234, 0.028169, -0.027559, -0.507422,
                -0.135368,
            ],
            &[
                -0.150777, 0.316537, -0.238364, -2.182105, 0.584001, -3.493148, -0.616415,
                -2.461076, 0.583371, -3.034768, -0.494796, -1.932055, 0.570642, -0.426588,
                -1.24506, -2.323319, 0.740129, -1.467842, -0.321724, -0.685752, 0.367072,
                -1.172034, -1.250113, 1.21934, -2.682786, -0.165661, 0.537876, 1.756091, 0.622274,
                -0.432951, 1.940098, -0.912636,
            ],
            &[
                -0.52418, 0.72207, -0.318727, -1.627795, 0.363655, -1.671963, 0.762139, 0.690708,
                -1.662698, -0.637605, -0.355392, -1.017486, 0.02135, -0.149882, 0.28815, -0.822122,
                -0.304949, -1.37264, -0.732553, 0.256489, 0.358625, -0.304614, -0.411885, 1.110854,
                -1.621181, -0.315292, 0.355361, 1.093259, 0.438233, 0.075833, 0.523163, -0.462546,
            ],
            &[
                -0.072464, 1.100983, 0.067824, -0.926745, 0.582339, 0.54173, 0.395307, -0.623839,
                -0.136779, -1.359123, -0.51776, -0.039643, -0.088844, -0.567864, -0.5437,
                -1.724633, 0.549986, -0.81121, -0.49081, -0.30174, -0.7949, -0.415665, -0.729082,
                2.047881, -2.654069, -0.165292, -0.063843, 1.182565, -1.641462, 0.121879,
                -0.205941, 0.574398,
            ],
            &[
                -0.263864, 0.616768, -0.062717, -1.350574, -0.441162, -1.839529, -0.591859,
                -1.03557, -2.484859, 0.844072, -0.388566, -1.317986, 0.12405, 0.131833, -0.077236,
                -1.296799, -0.136116, -1.766066, -0.459829, -0.268458, -0.260306, -0.669556,
                -0.85604, 1.506129, -2.505321, 0.533659, 0.152177, 1.285206, 0.321787, -0.116884,
                0.445043, -0.480682,
            ],
        ];

        let mut flat: Vec<f32> = Vec::with_capacity(6 * 32);
        for r in &rows {
            flat.extend_from_slice(&r[..]);
        }
        let coefs = Tensor::from_vec(flat, [rows.len(), 32])?;

        for (uid, row) in rows.iter().enumerate() {
            let slice = coefs.slice(s![uid, ..]);
            assert_eq!(slice.shape().as_slice(), &[32]);
            let got_flat = slice.to_flat_vec::<f32>()?;
            let got_vec = slice.to_vec::<f32>()?;
            assert_eq!(got_flat, row.to_vec(), "row {} to_flat_vec mismatch", uid);
            assert_eq!(got_vec, row.to_vec(), "row {} to_vec mismatch", uid);
        }

        Ok(())
    }

    #[test]
    fn test_to_flat_vec_rows_f16_approx_first_row() -> anyhow::Result<()> {
        // Use first row, convert to f16 and verify approximate equality when read back
        let row: [f32; 32] = [
            -0.573134, 0.440246, -0.305807, -2.028273, 0.232327, -2.410018, 0.769581, 0.383316,
            -3.05553, -0.196661, -0.381331, -1.538175, 0.132812, -0.296569, 0.006016, -0.786159,
            -0.170078, -0.614549, -0.655673, 0.334807, 0.185173, -0.27942, -0.602469, 1.092488,
            -1.670121, -0.89865, 0.425514, 0.94984, 0.662506, -0.204701, 1.175576, -0.254652,
        ];
        let row_f16: Vec<half::f16> = row.iter().copied().map(half::f16::from_f32).collect();
        let t = Tensor::from_vec(row_f16, [32])?;
        let got = t.to_flat_vec::<half::f16>()?;
        let got_f32: Vec<f32> = got.iter().map(|&x| x.to_f32()).collect();
        assert_eq!(got_f32.len(), row.len());
        for (a, b) in got_f32.iter().zip(row.iter()) {
            assert!((a - b).abs() < 1e-3, "f16 approx mismatch: {} vs {}", a, b);
        }
        Ok(())
    }

    #[test]
    fn test_to_flat_vec_contiguous_and_transposed_slices() -> anyhow::Result<()> {
        // Basic regression: ensure to_flat_vec respects view offsets both contiguous and non-contiguous
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::from_vec(data, [4, 6])?;

        // Contiguous row slice
        let r2 = t.slice(s![2, ..]);
        assert_eq!(
            r2.to_flat_vec::<f32>()?,
            vec![12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        );

        // Non-contiguous after transpose
        let tr = t.permute([1, 0])?; // [6, 4]
        let c3 = tr.slice(s![3, ..]);
        assert_eq!(c3.to_flat_vec::<f32>()?, vec![3.0, 9.0, 15.0, 21.0]);
        Ok(())
    }
}
