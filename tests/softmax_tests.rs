use slsl::Tensor;

/// Test softmax on 1D tensors
#[test]
fn test_softmax_1d() {
    // Test 1: Basic 1D tensor
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32], [3]).unwrap();
    let softmaxed = tensor.softmax(0).unwrap();

    // Expected: [0.09003057, 0.24472847, 0.66524096]
    let expected = [0.09003057f32, 0.24472847f32, 0.66524096f32];

    for (i, &expected_val) in expected.iter().enumerate() {
        let actual_val = softmaxed.at::<f32>(&*vec![i]);
        assert!(
            (actual_val - expected_val).abs() < 1e-6,
            "Index {i}: expected {expected_val}, got {actual_val}"
        );
    }

    // Verify sum equals 1.0f32
    let sum: f32 = (0..3).map(|i| softmaxed.at::<f32>(&*vec![i])).sum();
    assert!(
        (sum - 1.0f32).abs() < 1e-6,
        "Sum should be 1.0f32, got {sum}"
    );
}

/// Test softmax on 2D tensors along different dimensions
#[test]
fn test_softmax_2d() {
    // Test 2D tensor: [[1, 2], [3, 4]]
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], [2, 2]).unwrap();

    // Test along dimension 0 (rows)
    let softmaxed_dim0 = tensor.softmax(0).unwrap();

    // Expected for dim 0:
    // Row 0: [0.11920292, 0.11920292] (softmax of [1, 3])
    // Row 1: [0.88079708, 0.88079708] (softmax of [2, 4])
    let expected_dim0 = [
        0.11920292,
        0.11920292, // First row
        0.880_797_1,
        0.880_797_1, // Second row
    ];

    for (i, &expected_val) in expected_dim0.iter().enumerate() {
        let actual_val = softmaxed_dim0.at::<f32>(&*vec![i / 2, i % 2]);
        assert!(
            (actual_val - expected_val).abs() < 1e-6,
            "Index [{}, {}]: expected {}, got {}",
            i / 2,
            i % 2,
            expected_val,
            actual_val
        );
    }

    // Verify sum along dim 0 equals 1.0f32
    for col in 0..2 {
        let sum: f32 = (0..2)
            .map(|row| softmaxed_dim0.at::<f32>(&*vec![row, col]))
            .sum();
        assert!(
            (sum - 1.0f32).abs() < 1e-6,
            "Sum along dim 0, col {col} should be 1.0f32, got {sum}"
        );
    }

    // Test along dimension 1 (columns)
    let softmaxed_dim1 = tensor.softmax(1).unwrap();

    // Expected for dim 1:
    // Row 0: [0.26894142, 0.73105858] (softmax of [1, 2])
    // Row 1: [0.26894142, 0.73105858] (softmax of [3, 4])
    let expected_dim1 = [
        0.26894142,
        0.731_058_6, // First row
        0.26894142,
        0.731_058_6, // Second row
    ];

    for (i, &expected_val) in expected_dim1.iter().enumerate() {
        let actual_val = softmaxed_dim1.at::<f32>(&*vec![i / 2, i % 2]);
        assert!(
            (actual_val - expected_val).abs() < 1e-6,
            "Index [{}, {}]: expected {}, got {}",
            i / 2,
            i % 2,
            expected_val,
            actual_val
        );
    }

    // Verify sum along dim 1 equals 1.0f32
    for row in 0..2 {
        let sum: f32 = (0..2)
            .map(|col| softmaxed_dim1.at::<f32>(&*vec![row, col]))
            .sum();
        assert!(
            (sum - 1.0f32).abs() < 1e-6,
            "Sum along dim 1, row {row} should be 1.0f32, got {sum}"
        );
    }
}

/// Test softmax on 3D tensors
#[test]
fn test_softmax_3d() {
    // Test 3D tensor: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    let tensor = Tensor::from_vec(
        vec![
            1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32,
        ],
        [2, 2, 2],
    )
    .unwrap();

    // Test along dimension 0 (first dimension)
    let softmaxed_dim0 = tensor.softmax(0).unwrap();

    // Verify sum along dim 0 equals 1.0f32 for each position
    for i in 0..2 {
        for j in 0..2 {
            let sum: f32 = (0..2)
                .map(|k| softmaxed_dim0.at::<f32>(&*vec![k, i, j]))
                .sum();
            assert!(
                (sum - 1.0f32).abs() < 1e-6,
                "Sum along dim 0 at [{i}, {j}] should be 1.0f32, got {sum}"
            );
        }
    }

    // Test along dimension 1 (second dimension)
    let softmaxed_dim1 = tensor.softmax(1).unwrap();

    // Verify sum along dim 1 equals 1.0f32 for each slice
    for k in 0..2 {
        for j in 0..2 {
            let sum: f32 = (0..2)
                .map(|i| softmaxed_dim1.at::<f32>(&*vec![k, i, j]))
                .sum();
            assert!(
                (sum - 1.0f32).abs() < 1e-6,
                "Sum along dim 1 at [{k}, {j}] should be 1.0f32, got {sum}"
            );
        }
    }

    // Test along dimension 2 (third dimension)
    let softmaxed_dim2 = tensor.softmax(2).unwrap();

    // Verify sum along dim 2 equals 1.0f32 for each slice
    for k in 0..2 {
        for i in 0..2 {
            let sum: f32 = (0..2)
                .map(|j| softmaxed_dim2.at::<f32>(&*vec![k, i, j]))
                .sum();
            assert!(
                (sum - 1.0f32).abs() < 1e-6,
                "Sum along dim 2 at [{k}, {i}] should be 1.0f32, got {sum}"
            );
        }
    }
}

/// Test softmax with different data types
#[test]
fn test_softmax_different_dtypes() {
    // Test f32
    let tensor_f32 = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32], [3]).unwrap();
    let softmaxed_f32 = tensor_f32.softmax(0).unwrap();
    assert_eq!(softmaxed_f32.dtype(), slsl::DType::Fp32);

    // Test f64
    let tensor_f64 = Tensor::from_vec(vec![1.0f64, 2.0f64, 3.0f64], [3]).unwrap();
    let softmaxed_f64 = tensor_f64.softmax(0).unwrap();
    assert_eq!(softmaxed_f64.dtype(), slsl::DType::Fp64);

    // Test f16
    let tensor_f16 = Tensor::from_vec(
        vec![
            half::f16::from_f32(1.0f32),
            half::f16::from_f32(2.0f32),
            half::f16::from_f32(3.0f32),
        ],
        [3],
    )
    .unwrap();
    let softmaxed_f16 = tensor_f16.softmax(0).unwrap();
    assert_eq!(softmaxed_f16.dtype(), slsl::DType::Fp16);

    // Test bf16
    let tensor_bf16 = Tensor::from_vec(
        vec![
            half::bf16::from_f32(1.0f32),
            half::bf16::from_f32(2.0f32),
            half::bf16::from_f32(3.0f32),
        ],
        [3],
    )
    .unwrap();
    let softmaxed_bf16 = tensor_bf16.softmax(0).unwrap();
    assert_eq!(softmaxed_bf16.dtype(), slsl::DType::Bf16);
}

/// Test softmax error cases
#[test]
fn test_softmax_errors() {
    // Test with integer tensor (should fail)
    let int_tensor = Tensor::from_vec(vec![1, 2, 3], [3]).unwrap();
    let result = int_tensor.softmax(0);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("softmax only supports floating-point types"));

    // Test with invalid dimension (should fail)
    let float_tensor = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32], [3]).unwrap();
    let result = float_tensor.softmax(5);
    assert!(result.is_err());

    // Test with empty tensor (edge case) - skip this test for now
    // let empty_tensor = Tensor::from_vec::<f32, [usize; 0]>(vec![], []).unwrap();
    // let result = empty_tensor.softmax(0);
    // assert!(result.is_err());
}

/// Test softmax with edge cases
#[test]
fn test_softmax_edge_cases() {
    // Test with all zeros
    let zero_tensor = Tensor::from_vec(vec![0.0f32, 0.0f32, 0.0f32], [3]).unwrap();
    let softmaxed = zero_tensor.softmax(0).unwrap();

    // All values should be equal (1/3)
    let expected = 1.0f32 / 3.0f32;
    for i in 0..3 {
        let actual = softmaxed.at::<f32>(&*vec![i]);
        assert!(
            (actual - expected).abs() < 1e-6,
            "Index {i}: expected {expected}, got {actual}"
        );
    }

    // Test with large negative values (numerical stability)
    let large_neg_tensor = Tensor::from_vec(vec![-1000.0f32, -1001.0f32, -1002.0f32], [3]).unwrap();
    let softmaxed = large_neg_tensor.softmax(0).unwrap();

    // Should still sum to 1.0f32
    let sum: f32 = (0..3).map(|i| softmaxed.at::<f32>(&*vec![i])).sum();
    assert!(
        (sum - 1.0f32).abs() < 1e-6,
        "Sum should be 1.0f32, got {sum}"
    );

    // Test with large positive values (numerical stability)
    let large_pos_tensor = Tensor::from_vec(vec![1000.0f32, 1001.0f32, 1002.0f32], [3]).unwrap();
    let softmaxed = large_pos_tensor.softmax(0).unwrap();

    // Should still sum to 1.0f32
    let sum: f32 = (0..3).map(|i| softmaxed.at::<f32>(&*vec![i])).sum();
    assert!(
        (sum - 1.0f32).abs() < 1e-6,
        "Sum should be 1.0f32, got {sum}"
    );
}

/// Test softmax with single element tensors
#[test]
fn test_softmax_single_element() {
    // Single element tensor
    let single_tensor = Tensor::from_vec(vec![42.0f32], [1]).unwrap();
    let softmaxed = single_tensor.softmax(0).unwrap();

    // Should be 1.0f32
    let result = softmaxed.at::<f32>(&*vec![0]);
    assert!(
        (result - 1.0f32).abs() < 1e-6,
        "Single element softmax should be 1.0f32, got {result}"
    );
}

/// Test softmax preserves tensor properties
#[test]
fn test_softmax_preserves_properties() {
    let original = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], [2, 2]).unwrap();
    let softmaxed = original.softmax(1).unwrap();

    // Should preserve shape
    assert_eq!(softmaxed.shape(), original.shape());

    // Should preserve dtype
    assert_eq!(softmaxed.dtype(), original.dtype());

    // Should preserve rank
    assert_eq!(softmaxed.rank(), original.rank());

    // Should preserve numel
    assert_eq!(softmaxed.numel(), original.numel());
}

/// Test softmax with non-contiguous tensors
#[test]
fn test_softmax_non_contiguous() {
    // Create a non-contiguous tensor by permuting
    let original = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], [2, 2]).unwrap();
    let permuted = original.permute([1, 0]).unwrap();

    // Verify it's non-contiguous
    assert!(!permuted.is_contiguous());

    // Apply softmax
    let softmaxed = permuted.softmax(0).unwrap();

    // Should still work correctly
    let sum: f32 = (0..2).map(|i| softmaxed.at::<f32>(&*vec![i, 0])).sum();
    assert!(
        (sum - 1.0f32).abs() < 1e-6,
        "Sum should be 1.0f32, got {sum}"
    );
}

/// Test softmax with very small tensors
#[test]
fn test_softmax_small_tensors() {
    // 2x2 tensor
    let tensor = Tensor::from_vec(vec![0.1f32, 0.2f32, 0.3f32, 0.4f32], [2, 2]).unwrap();
    let softmaxed = tensor.softmax(0).unwrap();

    // Verify properties
    assert_eq!(softmaxed.shape().as_slice(), &[2, 2]);

    // Verify sums along dimension 0
    for col in 0..2 {
        let sum: f32 = (0..2)
            .map(|row| softmaxed.at::<f32>(&*vec![row, col]))
            .sum();
        assert!(
            (sum - 1.0f32).abs() < 1e-6,
            "Sum along dim 0, col {col} should be 1.0f32, got {sum}"
        );
    }
}

/// Test softmax with mixed positive and negative values
#[test]
fn test_softmax_mixed_values() {
    let tensor = Tensor::from_vec(vec![-1.0f32, 0.0f32, 1.0f32], [3]).unwrap();
    let softmaxed = tensor.softmax(0).unwrap();

    // Verify sum equals 1.0f32
    let sum: f32 = (0..3).map(|i| softmaxed.at::<f32>(&*vec![i])).sum();
    assert!(
        (sum - 1.0f32).abs() < 1e-6,
        "Sum should be 1.0f32, got {sum}"
    );

    // Verify all values are positive (softmax always produces positive values)
    for i in 0..3 {
        let val = softmaxed.at::<f32>(&*vec![i]);
        assert!(
            val > 0.0f32,
            "Softmax value at {i} should be positive, got {val}"
        );
    }
}
