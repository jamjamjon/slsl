use slsl::Tensor;

/// Test L1 norm (Manhattan norm)
#[test]
fn test_norm1_1d() {
    let tensor = Tensor::from_vec(vec![3.0f32, -4.0f32], [2]).unwrap();
    let normed = tensor.norm1(0).unwrap();

    // L1 norm: |3| + |-4| = 3 + 4 = 7
    let expected = 7.0f32;
    let actual = normed.at::<f32>(&*vec![]);
    assert!(
        (actual - expected).abs() < 1e-6,
        "Expected {expected}, got {actual}"
    );
}

#[test]
fn test_norm1_2d() {
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], [2, 2]).unwrap();
    let normed = tensor.norm1(0).unwrap();

    // L1 norm along dim 0: [|1|+|3|, |2|+|4|] = [4, 6]
    assert_eq!(normed.shape().as_slice(), &[2]);
    let val1 = normed.at::<f32>(&*vec![0]);
    let val2 = normed.at::<f32>(&*vec![1]);
    assert!((val1 - 4.0f32).abs() < 1e-6, "Expected 4.0, got {val1}");
    assert!((val2 - 6.0f32).abs() < 1e-6, "Expected 6.0, got {val2}");
}

/// Test L2 norm (Euclidean norm)
#[test]
fn test_norm2_1d() {
    let tensor = Tensor::from_vec(vec![3.0f32, 4.0f32], [2]).unwrap();
    let normed = tensor.norm2(0).unwrap();

    // L2 norm: sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    let expected = 5.0f32;
    let actual = normed.at::<f32>(&*vec![]);
    assert!(
        (actual - expected).abs() < 1e-6,
        "Expected {expected}, got {actual}"
    );
}

#[test]
fn test_norm2_2d() {
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], [2, 2]).unwrap();
    let normed = tensor.norm2(0).unwrap();

    // L2 norm along dim 0: [sqrt(1^2+3^2), sqrt(2^2+4^2)] = [sqrt(10), sqrt(20)]
    assert_eq!(normed.shape().as_slice(), &[2]);
    let val1 = normed.at::<f32>(&*vec![0]);
    let val2 = normed.at::<f32>(&*vec![1]);
    assert!(
        (val1 - (10.0f32).sqrt()).abs() < 1e-6,
        "Expected sqrt(10), got {val1}"
    );
    assert!(
        (val2 - (20.0f32).sqrt()).abs() < 1e-6,
        "Expected sqrt(20), got {val2}"
    );
}

/// Test Lp norm
#[test]
fn test_normp_1d() {
    let tensor = Tensor::from_vec(vec![3.0f32, 4.0f32], [2]).unwrap();
    let normed = tensor.normp(0, 3.0).unwrap();

    // L3 norm: (3^3 + 4^3)^(1/3) = (27 + 64)^(1/3) = 91^(1/3)
    let expected = (91.0f32).powf(1.0 / 3.0);
    let actual = normed.at::<f32>(&*vec![]);
    assert!(
        (actual - expected).abs() < 1e-6,
        "Expected {expected}, got {actual}"
    );
}

/// Test keepdim functionality
#[test]
fn test_norm_keepdim() {
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], [2, 2]).unwrap();
    let normed = tensor.norm_keepdim(0, 2.0).unwrap();

    // Should preserve the dimension with size 1
    assert_eq!(normed.shape().as_slice(), &[1, 2]);
    let val1 = normed.at::<f32>(&*vec![0, 0]);
    let val2 = normed.at::<f32>(&*vec![0, 1]);
    assert!(
        (val1 - (10.0f32).sqrt()).abs() < 1e-6,
        "Expected sqrt(10), got {val1}"
    );
    assert!(
        (val2 - (20.0f32).sqrt()).abs() < 1e-6,
        "Expected sqrt(20), got {val2}"
    );
}

/// Test different data types
#[test]
fn test_norm_different_dtypes() {
    // Test f32
    let tensor_f32 = Tensor::from_vec(vec![3.0f32, 4.0f32], [2]).unwrap();
    let normed_f32 = tensor_f32.norm2(0).unwrap();
    assert_eq!(normed_f32.dtype(), slsl::DType::Fp32);

    // Test f64
    let tensor_f64 = Tensor::from_vec(vec![3.0f64, 4.0f64], [2]).unwrap();
    let normed_f64 = tensor_f64.norm2(0).unwrap();
    assert_eq!(normed_f64.dtype(), slsl::DType::Fp64);

    // Test f16
    let tensor_f16 = Tensor::from_vec(
        vec![half::f16::from_f32(3.0f32), half::f16::from_f32(4.0f32)],
        [2],
    )
    .unwrap();
    let normed_f16 = tensor_f16.norm2(0).unwrap();
    assert_eq!(normed_f16.dtype(), slsl::DType::Fp16);

    // Test bf16
    let tensor_bf16 = Tensor::from_vec(
        vec![half::bf16::from_f32(3.0f32), half::bf16::from_f32(4.0f32)],
        [2],
    )
    .unwrap();
    let normed_bf16 = tensor_bf16.norm2(0).unwrap();
    assert_eq!(normed_bf16.dtype(), slsl::DType::Bf16);
}

/// Test error cases
#[test]
fn test_norm_errors() {
    // Test with integer tensor (should fail)
    let int_tensor = Tensor::from_vec(vec![1, 2, 3], [3]).unwrap();
    let result = int_tensor.norm2(0);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("norm2 only supports floating-point types"));

    // Test with invalid dimension (should fail)
    let float_tensor = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32], [3]).unwrap();
    let result = float_tensor.norm2(5);
    assert!(result.is_err());
}

/// Test edge cases
#[test]
fn test_norm_edge_cases() {
    // Test with all zeros
    let zero_tensor = Tensor::from_vec(vec![0.0f32, 0.0f32, 0.0f32], [3]).unwrap();
    let normed = zero_tensor.norm2(0).unwrap();
    let result = normed.at::<f32>(&*vec![]);
    assert!((result - 0.0f32).abs() < 1e-6, "Expected 0.0, got {result}");

    // Test with single element
    let single_tensor = Tensor::from_vec(vec![42.0f32], [1]).unwrap();
    let normed = single_tensor.norm2(0).unwrap();
    let result = normed.at::<f32>(&*vec![]);
    assert!(
        (result - 42.0f32).abs() < 1e-6,
        "Expected 42.0, got {result}"
    );
}

/// Test 3D tensor
#[test]
fn test_norm_3d() {
    let tensor = Tensor::from_vec(
        vec![
            1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32,
        ],
        [2, 2, 2],
    )
    .unwrap();

    // Test norm along dimension 0
    let normed = tensor.norm2(0).unwrap();
    assert_eq!(normed.shape().as_slice(), &[2, 2]);

    // Test norm along dimension 1
    let normed = tensor.norm2(1).unwrap();
    assert_eq!(normed.shape().as_slice(), &[2, 2]);

    // Test norm along dimension 2
    let normed = tensor.norm2(2).unwrap();
    assert_eq!(normed.shape().as_slice(), &[2, 2]);
}
