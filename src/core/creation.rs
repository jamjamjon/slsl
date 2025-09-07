use anyhow::Result;
use num_traits::cast::AsPrimitive;
use rand::rngs::SmallRng;
use rand::{rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};

use crate::{DType, Shape, Storage, StorageTrait, Tensor, TensorElement, UninitVec};

impl<T: TensorElement> From<T> for Tensor {
    /// Creates a tensor from a scalar value.
    ///
    /// This conversion creates a 0-dimensional (scalar) tensor containing the given value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor: Tensor = 42.0f32.into();
    /// assert_eq!(tensor.dims(), []);
    /// assert_eq!(tensor.to_scalar::<f32>().unwrap(), 42.0);
    /// ```
    fn from(value: T) -> Self {
        Self::from_scalar(value).unwrap()
    }
}

impl<T: TensorElement> From<&[T]> for Tensor {
    /// Creates a 1D tensor from a slice.
    ///
    /// The tensor will have the same length as the slice and contain copies of all elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let data = [1, 2, 3, 4];
    /// let tensor: Tensor = (&data[..]).into();
    /// assert_eq!(tensor.dims(), [4]);
    /// assert_eq!(tensor.to_vec::<i32>().unwrap(), vec![1, 2, 3, 4]);
    /// ```
    fn from(value: &[T]) -> Self {
        Self::from_slice(value, [value.len()]).unwrap()
    }
}

impl<T: TensorElement> From<&Vec<T>> for Tensor {
    /// Creates a 1D tensor from a vector reference.
    ///
    /// The tensor will have the same length as the vector and contain copies of all elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0];
    /// let tensor: Tensor = (&data).into();
    /// assert_eq!(tensor.dims(), [3]);
    /// assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
    /// ```
    fn from(value: &Vec<T>) -> Self {
        Self::from_slice(value.as_slice(), [value.len()]).unwrap()
    }
}

impl<T: TensorElement, const N: usize> From<[T; N]> for Tensor {
    /// Creates a 1D tensor from a fixed-size array.
    ///
    /// Since `TensorElement` implements `Copy`, the array is efficiently copied.
    /// The tensor will have length N and contain all array elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let data = [1, 2, 3, 4, 5];
    /// let tensor: Tensor = data.into();
    /// assert_eq!(tensor.dims(), [5]);
    /// assert_eq!(tensor.to_vec::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
    ///
    /// // Original array is still available (Copy semantics)
    /// println!("{:?}", data);
    /// ```
    fn from(value: [T; N]) -> Self {
        Self::from_slice(&value, [N]).unwrap()
    }
}

impl<T: TensorElement> From<Vec<T>> for Tensor {
    /// Creates a 1D tensor from a vector by taking ownership.
    ///
    /// This is an efficient conversion that takes ownership of the vector's data
    /// without copying.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let data = vec![1.0f32, 2.0, 3.0, 4.0];
    /// let tensor: Tensor = data.into();
    /// assert_eq!(tensor.dims(), [4]);
    /// assert_eq!(tensor.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    fn from(value: Vec<T>) -> Self {
        let len = value.len();
        Self::from_vec(value, [len]).unwrap()
    }
}

impl Tensor {
    /// Creates a scalar (0-dimensional) tensor from a single value.
    ///
    /// # Parameters
    ///
    /// * `value` - The scalar value to store in the tensor
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A scalar tensor with empty dimensions `[]`
    /// - `Err`: If tensor creation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let scalar = Tensor::from_scalar(42.0f32)?;
    /// assert_eq!(scalar.dims(), []);
    /// assert_eq!(scalar.to_scalar::<f32>()?, 42.0);
    ///
    /// let int_scalar = Tensor::from_scalar(10i32)?;
    /// assert_eq!(int_scalar.to_scalar::<i32>()?, 10);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn from_scalar<T: TensorElement>(value: T) -> Result<Self> {
        let output = UninitVec::<T>::new(1).init_with(|dst| {
            dst[0] = value;
        });
        Self::from_vec(output, [])
    }

    /// Creates a tensor from a vector with the specified shape.
    ///
    /// Takes ownership of the vector data and reshapes it according to the given dimensions.
    /// The total number of elements in the vector must match the product of all dimensions.
    ///
    /// # Parameters
    ///
    /// * `data` - Vector containing the tensor elements
    /// * `shape` - The desired shape/dimensions for the tensor
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A tensor with the specified shape
    /// - `Err`: If the data length doesn't match the expected shape size
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// // Create 2D tensor
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let tensor = Tensor::from_vec(data, [2, 3])?;
    /// assert_eq!(tensor.dims(), [2, 3]);
    ///
    /// // Create 1D tensor
    /// let data = vec![1.0, 2.0, 3.0];
    /// let tensor = Tensor::from_vec(data, [3])?;
    /// assert_eq!(tensor.dims(), [3]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Panics
    ///
    /// Returns an error if `data.len()` doesn't equal the product of shape dimensions.
    #[inline]
    pub fn from_vec<T: TensorElement, S: Into<Shape>>(data: Vec<T>, shape: S) -> Result<Self> {
        let shape = shape.into();
        let expected_len = shape.numel();

        if data.len() != expected_len {
            anyhow::bail!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape.as_slice(),
                expected_len
            );
        }

        let storage = Storage::from_vec(data)?;
        let ptr = storage.ptr();
        let strides = Self::compute_contiguous_strides(&shape);

        Ok(Self {
            storage,
            ptr,
            dtype: T::DTYPE,
            shape,
            strides,
            offset_bytes: 0,
        })
    }

    /// Creates a tensor from a slice with the specified shape.
    ///
    /// Copies data from the slice and creates a tensor with the given dimensions.
    /// The slice length must match the product of all shape dimensions.
    ///
    /// # Parameters
    ///
    /// * `data` - Slice containing the tensor elements
    /// * `shape` - The desired shape/dimensions for the tensor
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A tensor with the specified shape containing copied data
    /// - `Err`: If the slice length doesn't match the expected shape size
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let data = [1, 2, 3, 4];
    /// let tensor = Tensor::from_slice(&data, [2, 2])?;
    /// assert_eq!(tensor.dims(), [2, 2]);
    /// assert_eq!(tensor.at::<i32>([0, 1]), 2);
    ///
    /// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor = Tensor::from_slice(&data, [3, 2])?;
    /// assert_eq!(tensor.dims(), [3, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn from_slice<T: TensorElement, S: Into<Shape>>(data: &[T], shape: S) -> Result<Self> {
        Self::from_vec(data.to_vec(), shape)
    }

    /// Creates a tensor filled with a specific value.
    ///
    /// All elements in the tensor will be set to the provided value.
    ///
    /// # Parameters
    ///
    /// * `shape` - The shape/dimensions of the tensor to create
    /// * `value` - The value to fill all tensor elements with
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A tensor filled with the specified value
    /// - `Err`: If tensor creation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::full([2, 3], 7.5f32)?;
    /// assert_eq!(tensor.dims(), [2, 3]);
    /// assert_eq!(tensor.at::<f32>([0, 0]), 7.5);
    /// assert_eq!(tensor.at::<f32>([1, 2]), 7.5);
    ///
    /// let tensor = Tensor::full([4], -1i32)?;
    /// assert_eq!(tensor.to_vec::<i32>()?, vec![-1, -1, -1, -1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn full<T: TensorElement>(shape: impl Into<Shape>, value: T) -> Result<Self> {
        let shape = shape.into();
        let numel = shape.numel();
        let output = UninitVec::<T>::new(numel).full(value);
        Self::from_vec(output, shape)
    }

    /// Creates a tensor filled with ones.
    ///
    /// All elements in the tensor will be set to the numeric value 1 for the specified type.
    ///
    /// # Parameters
    ///
    /// * `shape` - The shape/dimensions of the tensor to create
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A tensor filled with ones
    /// - `Err`: If tensor creation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::ones::<f32>([2, 2])?;
    /// assert_eq!(tensor.dims(), [2, 2]);
    /// assert_eq!(tensor.to_flat_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0]);
    ///
    /// let tensor = Tensor::ones::<i32>([3])?;
    /// assert_eq!(tensor.to_vec::<i32>()?, vec![1, 1, 1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn ones<T: TensorElement>(shape: impl Into<Shape>) -> Result<Self> {
        Self::full(shape, T::one())
    }

    /// Creates a tensor filled with zeros.
    ///
    /// All elements in the tensor will be set to the numeric value 0 for the specified type.
    /// This operation is optimized for performance.
    ///
    /// # Parameters
    ///
    /// * `shape` - The shape/dimensions of the tensor to create
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A tensor filled with zeros
    /// - `Err`: If tensor creation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::zeros::<f64>([3, 2])?;
    /// assert_eq!(tensor.dims(), [3, 2]);
    /// assert_eq!(tensor.to_flat_vec::<f64>()?, vec![0.0; 6]);
    ///
    /// let tensor = Tensor::zeros::<i32>([4])?;
    /// assert_eq!(tensor.to_vec::<i32>()?, vec![0, 0, 0, 0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn zeros<T: TensorElement>(shape: impl Into<Shape>) -> Result<Self> {
        Self::full(shape, T::zero())
    }

    /// Creates a tensor filled with ones, with the same shape as the input tensor.
    ///
    /// This is a convenience function that creates a new tensor with the same dimensions
    /// as the reference tensor, but filled with ones of the specified type.
    ///
    /// # Parameters
    ///
    /// * `tensor` - The reference tensor whose shape will be copied
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A tensor with the same shape as input, filled with ones
    /// - `Err`: If tensor creation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let original = Tensor::zeros::<f32>([2, 3])?;
    /// let ones_tensor = Tensor::ones_like::<f32>(&original)?;
    /// assert_eq!(ones_tensor.dims(), [2, 3]);
    /// assert_eq!(ones_tensor.to_flat_vec::<f32>()?, vec![1.0; 6]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn ones_like<T: TensorElement>(tensor: &Self) -> Result<Self> {
        Self::ones::<T>(tensor.shape)
    }

    /// Creates a tensor filled with zeros, with the same shape as the input tensor.
    ///
    /// This is a convenience function that creates a new tensor with the same dimensions
    /// as the reference tensor, but filled with zeros of the specified type.
    ///
    /// # Parameters
    ///
    /// * `tensor` - The reference tensor whose shape will be copied
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A tensor with the same shape as input, filled with zeros
    /// - `Err`: If tensor creation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let original = Tensor::ones::<f32>([2, 2])?;
    /// let zeros_tensor = Tensor::zeros_like::<f32>(&original)?;
    /// assert_eq!(zeros_tensor.dims(), [2, 2]);
    /// assert_eq!(zeros_tensor.to_flat_vec::<f32>()?, vec![0.0; 4]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn zeros_like<T: TensorElement>(tensor: &Self) -> Result<Self> {
        Self::zeros::<T>(tensor.shape)
    }

    /// Creates an identity matrix of size n × n.
    ///
    /// An identity matrix has ones on the main diagonal and zeros elsewhere.
    /// This function creates a 2D square tensor with these properties.
    ///
    /// # Parameters
    ///
    /// * `n` - The size of the square matrix (both width and height)
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A 2D tensor of shape `[n, n]` representing the identity matrix
    /// - `Err`: If tensor creation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let eye = Tensor::eye::<f32>(3)?;
    /// assert_eq!(eye.dims(), [3, 3]);
    /// assert_eq!(eye.at::<f32>([0, 0]), 1.0);  // Diagonal elements
    /// assert_eq!(eye.at::<f32>([1, 1]), 1.0);
    /// assert_eq!(eye.at::<f32>([2, 2]), 1.0);
    /// assert_eq!(eye.at::<f32>([0, 1]), 0.0);  // Off-diagonal elements
    /// assert_eq!(eye.at::<f32>([1, 0]), 0.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn eye<T: TensorElement>(n: usize) -> Result<Self> {
        let shape = [n, n];
        let numel = n * n;
        let output = UninitVec::<T>::new(numel).init_with(|dst| {
            dst.fill(T::zero());
            for i in 0..n {
                dst[i * n + i] = T::one();
            }
        });
        Self::from_vec(output, shape)
    }

    /// Creates a 1D tensor with values from start to end (exclusive) with a given step.
    ///
    /// Generates a sequence of values starting from `start`, incrementing by `step`,
    /// and stopping before `end`. Similar to Python's `range()` function.
    ///
    /// # Parameters
    ///
    /// * `start` - The starting value (inclusive)
    /// * `end` - The ending value (exclusive)
    /// * `step` - The increment between consecutive values
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A 1D tensor containing the generated sequence
    /// - `Err`: If step is zero, or if boolean type is used
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// // Basic usage
    /// let tensor = Tensor::arange(0, 5, 1)?;
    /// assert_eq!(tensor.to_vec::<i32>()?, vec![0, 1, 2, 3, 4]);
    ///
    /// // With step > 1
    /// let tensor = Tensor::arange(0.0, 2.0, 0.5)?;
    /// assert_eq!(tensor.to_vec::<f64>()?, vec![0.0, 0.5, 1.0, 1.5]);
    ///
    /// // Negative step
    /// let tensor = Tensor::arange(5, 0, -1)?;
    /// assert_eq!(tensor.to_vec::<i32>()?, vec![5, 4, 3, 2, 1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Panics
    ///
    /// Returns an error if:
    /// - `step` is zero
    /// - The tensor type is boolean
    #[inline(always)]
    pub fn arange<T: TensorElement + PartialOrd + std::ops::Add<Output = T> + AsPrimitive<f32>>(
        start: T,
        end: T,
        step: T,
    ) -> Result<Self> {
        // TODO: optimize
        if T::DTYPE == DType::Bool {
            anyhow::bail!("Boolean tensors do not support arange operation.");
        }
        if step == T::zero() {
            anyhow::bail!("Step cannot be zero for arange operation.");
        }

        let start_f = start.as_();
        let end_f = end.as_();
        let step_f = step.as_();

        let count = ((end_f - start_f) / step_f).ceil() as usize;

        let output = UninitVec::<T>::new(count).init_with(|dst| {
            let mut current = start;
            for item in dst.iter_mut().take(count) {
                *item = current;
                current = current + step;
            }
        });
        Self::from_vec(output, [count])
    }

    /// Creates a 1D tensor with n evenly spaced values from start to end (inclusive).
    ///
    /// Generates `n` values linearly spaced between `start` and `end`, including both endpoints.
    /// Similar to NumPy's `linspace()` function.
    ///
    /// # Parameters
    ///
    /// * `start` - The starting value (inclusive)
    /// * `end` - The ending value (inclusive)
    /// * `n` - The number of values to generate
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A 1D tensor with `n` evenly spaced values
    /// - `Err`: If `n` is zero or if boolean type is used
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// // 5 values from 0 to 10
    /// let tensor = Tensor::linspace(0.0f32, 10.0f32, 5)?;
    /// assert_eq!(tensor.to_vec::<f32>()?, vec![0.0, 2.5, 5.0, 7.5, 10.0]);
    ///
    /// // Single value
    /// let tensor = Tensor::linspace(5.0f32, 10.0f32, 1)?;
    /// assert_eq!(tensor.to_vec::<f32>()?, vec![5.0]);
    ///
    /// // Negative range
    /// let tensor = Tensor::linspace(-1.0f32, 1.0f32, 3)?;
    /// assert_eq!(tensor.to_vec::<f32>()?, vec![-1.0, 0.0, 1.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Panics
    ///
    /// Returns an error if:
    /// - `n` is zero
    /// - The tensor type is boolean
    #[inline]
    pub fn linspace<T>(start: T, end: T, n: usize) -> Result<Self>
    where
        T: TensorElement
            + std::ops::Sub<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Add<Output = T>
            + Copy,
    {
        // TODO: optimize
        if T::DTYPE == DType::Bool {
            anyhow::bail!("Boolean tensors do not support linspace operation.");
        }
        if n == 0 {
            anyhow::bail!("Number of n must be positive");
        }

        let start_f32 = start.to_f32();
        let end_f32 = end.to_f32();

        let output = UninitVec::<T>::new(n).init_with(|dst| {
            if n == 1 {
                dst[0] = start;
            } else {
                let step_f32 = (end_f32 - start_f32) / (n - 1) as f32;
                for (i, item) in dst.iter_mut().enumerate().take(n) {
                    let val_f32 = start_f32 + i as f32 * step_f32;
                    *item = T::from_f32(val_f32);
                }
            }
        });
        Self::from_vec::<T, _>(output, [n])
    }

    /// Creates a tensor with random values from a uniform distribution in the range [low, high).
    ///
    /// Generates random values uniformly distributed between `low` (inclusive) and `high` (exclusive).
    /// The random number generator is automatically seeded.
    ///
    /// # Parameters
    ///
    /// * `low` - The lower bound (inclusive)
    /// * `high` - The upper bound (exclusive)
    /// * `shape` - The shape/dimensions of the tensor to create
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A tensor filled with random values from the uniform distribution
    /// - `Err`: If the distribution parameters are invalid or tensor creation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// // Random floats between 0.0 and 1.0
    /// let tensor = Tensor::rand(0.0f32, 1.0f32, [2, 3])?;
    /// assert_eq!(tensor.dims(), [2, 3]);
    ///
    /// // Random integers between 1 and 10
    /// let tensor = Tensor::rand(1i32, 10i32, [5])?;
    /// assert_eq!(tensor.dims(), [5]);
    /// // All values should be in range [1, 10)
    /// for &val in tensor.to_vec::<i32>()?.iter() {
    ///     assert!(val >= 1 && val < 10);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - Uses a fast random number generator (SmallRng)
    /// - Values are uniformly distributed in the specified range
    /// - The upper bound is exclusive (values will be less than `high`)
    #[inline]
    pub fn rand<T: TensorElement + rand_distr::uniform::SampleUniform>(
        low: T,
        high: T,
        shape: impl Into<Shape>,
    ) -> Result<Self> {
        let shape = shape.into();
        let numel = shape.numel();
        let mut rng = SmallRng::from_rng(&mut rng());
        let uniform = Uniform::new(low, high)?;
        let data: Vec<T> = (0..numel).map(|_| uniform.sample(&mut rng)).collect();

        Self::from_vec(data, shape)
    }

    /// Creates a tensor with random values from a standard normal distribution N(0,1).
    ///
    /// Generates random values from a normal (Gaussian) distribution with mean 0 and
    /// standard deviation 1. This is commonly used for neural network weight initialization.
    ///
    /// # Parameters
    ///
    /// * `shape` - The shape/dimensions of the tensor to create
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A tensor filled with normally distributed random values
    /// - `Err`: If tensor creation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let tensor = Tensor::randn::<f32>([2, 3])?;
    /// assert_eq!(tensor.dims(), [2, 3]);
    ///
    /// // Values should be roughly centered around 0
    /// let values = tensor.to_flat_vec::<f32>()?;
    /// let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
    /// assert!(mean.abs() < 1.0);  // Should be close to 0 for large samples
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - Uses a standard normal distribution (mean=0, std=1)
    /// - Commonly used for initializing neural network weights
    /// - Values follow the bell curve distribution
    #[inline]
    pub fn randn<T: TensorElement + From<f32>>(shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let mut rng = SmallRng::from_rng(&mut rng());
        let normal = Normal::new(0.0f32, 1.0f32)?;
        let data: Vec<T> = (0..shape.numel())
            .map(|_| T::from(normal.sample(&mut rng)))
            .collect();

        Self::from_vec(data, shape)
    }

    /// Extracts the upper triangular part of a matrix (k-th diagonal and above).
    ///
    /// Creates a new tensor containing only the upper triangular elements of the input matrix.
    /// Elements below the k-th diagonal are set to zero.
    ///
    /// # Parameters
    ///
    /// * `matrix` - The input 2D tensor (must be 2-dimensional)
    /// * `k` - Diagonal offset:
    ///   - `k = 0`: Main diagonal and above
    ///   - `k > 0`: Above the main diagonal
    ///   - `k < 0`: Below the main diagonal
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A new tensor with the upper triangular part
    /// - `Err`: If the input is not a 2D matrix
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let matrix = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])?;
    ///
    /// // Main diagonal and above (k=0)
    /// let upper = Tensor::triu::<i32>(&matrix, 0)?;
    /// // Result: [[1, 2, 3],
    /// //          [0, 5, 6],
    /// //          [0, 0, 9]]
    ///
    /// // Above main diagonal (k=1)
    /// let upper = Tensor::triu::<i32>(&matrix, 1)?;
    /// // Result: [[0, 2, 3],
    /// //          [0, 0, 6],
    /// //          [0, 0, 0]]
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Panics
    ///
    /// Returns an error if the input tensor is not 2-dimensional.
    #[inline]
    pub fn triu<T: TensorElement>(matrix: &Tensor, k: i32) -> anyhow::Result<Self> {
        if matrix.dims().len() != 2 {
            anyhow::bail!(
                "triu requires a 2D matrix, got {}D tensor",
                matrix.dims().len()
            );
        }

        let [rows, cols] = [matrix.shape()[0], matrix.shape()[1]];
        let numel = rows * cols;

        let output = UninitVec::<T>::new(numel).init_with(|dst| {
            dst.fill(T::zero());
            for i in 0..rows {
                for j in 0..cols {
                    if (j as i32) >= (i as i32) + k {
                        dst[i * cols + j] = matrix.at::<T>([i, j]);
                    }
                }
            }
        });
        Self::from_vec(output, [rows, cols])
    }

    /// Extracts the lower triangular part of a matrix (k-th diagonal and below).
    ///
    /// Creates a new tensor containing only the lower triangular elements of the input matrix.
    /// Elements above the k-th diagonal are set to zero.
    ///
    /// # Parameters
    ///
    /// * `matrix` - The input 2D tensor (must be 2-dimensional)
    /// * `k` - Diagonal offset:
    ///   - `k = 0`: Main diagonal and below
    ///   - `k > 0`: Above the main diagonal
    ///   - `k < 0`: Below the main diagonal
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A new tensor with the lower triangular part
    /// - `Err`: If the input is not a 2D matrix
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let matrix = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])?;
    ///
    /// // Main diagonal and below (k=0)
    /// let lower = Tensor::tril::<i32>(&matrix, 0)?;
    /// // Result: [[1, 0, 0],
    /// //          [4, 5, 0],
    /// //          [7, 8, 9]]
    ///
    /// // Below main diagonal (k=-1)
    /// let lower = Tensor::tril::<i32>(&matrix, -1)?;
    /// // Result: [[0, 0, 0],
    /// //          [4, 0, 0],
    /// //          [7, 8, 0]]
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Panics
    ///
    /// Returns an error if the input tensor is not 2-dimensional.
    #[inline]
    pub fn tril<T: TensorElement>(matrix: &Tensor, k: i32) -> anyhow::Result<Self> {
        if matrix.dims().len() != 2 {
            anyhow::bail!(
                "tril requires a 2D matrix, got {}D tensor",
                matrix.dims().len()
            );
        }

        let [rows, cols] = [matrix.shape()[0], matrix.shape()[1]];
        let numel = rows * cols;

        let output = UninitVec::<T>::new(numel).init_with(|dst| {
            dst.fill(T::zero());
            for i in 0..rows {
                for j in 0..cols {
                    if (j as i32) <= (i as i32) + k {
                        dst[i * cols + j] = matrix.at::<T>([i, j]);
                    }
                }
            }
        });
        Self::from_vec(output, [rows, cols])
    }

    /// Extracts diagonal elements from a matrix.
    ///
    /// Returns a 1D tensor containing the elements from the specified diagonal of the input matrix.
    /// The k-th diagonal can be the main diagonal (k=0), above it (k>0), or below it (k<0).
    ///
    /// # Parameters
    ///
    /// * `matrix` - The input 2D tensor (must be 2-dimensional)
    /// * `k` - Diagonal offset:
    ///   - `k = 0`: Main diagonal
    ///   - `k > 0`: k-th diagonal above the main diagonal
    ///   - `k < 0`: k-th diagonal below the main diagonal
    ///
    /// # Returns
    ///
    /// A `Result<Tensor>` containing:
    /// - `Ok(Tensor)`: A 1D tensor with the diagonal elements
    /// - `Err`: If the input is not a 2D matrix
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slsl::Tensor;
    ///
    /// let matrix = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])?;
    ///
    /// // Main diagonal (k=0)
    /// let diag = Tensor::diag::<i32>(&matrix, 0)?;
    /// assert_eq!(diag.to_vec::<i32>()?, vec![1, 5, 9]);
    ///
    /// // First super-diagonal (k=1)
    /// let diag = Tensor::diag::<i32>(&matrix, 1)?;
    /// assert_eq!(diag.to_vec::<i32>()?, vec![2, 6]);
    ///
    /// // First sub-diagonal (k=-1)
    /// let diag = Tensor::diag::<i32>(&matrix, -1)?;
    /// assert_eq!(diag.to_vec::<i32>()?, vec![4, 8]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - For non-square matrices, the diagonal length is determined by the matrix dimensions
    /// - If the requested diagonal is out of bounds, an empty tensor is returned
    ///
    /// # Panics
    ///
    /// Returns an error if the input tensor is not 2-dimensional.
    #[inline]
    pub fn diag<T: TensorElement>(matrix: &Tensor, k: i32) -> anyhow::Result<Self> {
        if matrix.dims().len() != 2 {
            anyhow::bail!(
                "diag requires a 2D matrix, got {}D tensor",
                matrix.dims().len()
            );
        }

        let [rows, cols] = [matrix.shape()[0], matrix.shape()[1]];
        let diag_len = if k >= 0 {
            let k = k as usize;
            if k < cols {
                std::cmp::min(rows, cols - k)
            } else {
                0
            }
        } else {
            let k = (-k) as usize;
            if k < rows {
                std::cmp::min(rows - k, cols)
            } else {
                0
            }
        };

        let data = UninitVec::<T>::new(diag_len).init_with(|dst| {
            if k >= 0 {
                let k = k as usize;
                for (i, item) in dst.iter_mut().enumerate().take(diag_len) {
                    *item = matrix.at::<T>([i, i + k]);
                }
            } else {
                let k = (-k) as usize;
                for (i, item) in dst.iter_mut().enumerate().take(diag_len) {
                    *item = matrix.at::<T>([i + k, i]);
                }
            }
        });
        Self::from_vec(data, [diag_len])
    }
}
