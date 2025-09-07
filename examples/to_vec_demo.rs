use slsl::{s, Tensor};

fn main() -> anyhow::Result<()> {
    println!("🎯 SLSL to_vec Series Demo");
    println!("{}", "=".repeat(50));

    // Test to_scalar
    println!("\n📊 Testing to_scalar:");
    let scalar_tensor = Tensor::from_vec(vec![42.5f32], [])?;
    println!("Scalar tensor: {scalar_tensor:?}");
    let scalar_value = scalar_tensor.to_scalar::<f32>()?;
    println!("Extracted scalar: {scalar_value}");

    // Test to_vec with 1D tensor
    println!("\n📊 Testing to_vec with 1D tensor:");
    let vec1d = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let tensor1d = Tensor::from_vec(vec1d.clone(), [5])?;
    println!("1D tensor shape: {:?}", tensor1d.shape().as_slice());
    let extracted_vec = tensor1d.to_vec::<f32>()?;
    println!("Original: {vec1d:?}");
    println!("Extracted: {extracted_vec:?}");
    assert_eq!(vec1d, extracted_vec);
    println!("✅ 1D to_vec test passed!");

    // Test to_flat_vec with 2D tensor (flattened)
    println!("\n📊 Testing to_flat_vec with 2D tensor (flattened):");
    let vec2d_flat = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor2d = Tensor::from_vec(vec2d_flat.clone(), [2, 3])?;
    println!("2D tensor shape: {:?}", tensor2d.shape().as_slice());
    let extracted_flat = tensor2d.to_flat_vec::<f32>()?;
    println!("Original flat: {vec2d_flat:?}");
    println!("Extracted flat: {extracted_flat:?}");
    assert_eq!(vec2d_flat, extracted_flat);
    println!("✅ 2D to_flat_vec test passed!");

    // Test to_vec2 with 2D tensor
    println!("\n📊 Testing to_vec2 with 2D tensor:");
    let tensor2d_for_vec2 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])?;
    println!(
        "2D tensor shape: {:?}",
        tensor2d_for_vec2.shape().as_slice()
    );
    let extracted_vec2 = tensor2d_for_vec2.to_vec2::<f32>()?;
    let expected_vec2 = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    println!("Expected 2D: {expected_vec2:?}");
    println!("Extracted 2D: {extracted_vec2:?}");
    assert_eq!(expected_vec2, extracted_vec2);
    println!("✅ to_vec2 test passed!");

    // Test to_vec3 with 3D tensor
    println!("\n📊 Testing to_vec3 with 3D tensor:");
    let vec3d_data = (0..24).map(|i| i as f32).collect::<Vec<_>>();
    let tensor3d = Tensor::from_vec(vec3d_data, [2, 3, 4])?;
    println!("3D tensor shape: {:?}", tensor3d.shape().as_slice());
    let extracted_vec3 = tensor3d.to_vec3::<f32>()?;

    // Verify structure
    assert_eq!(extracted_vec3.len(), 2); // depth
    assert_eq!(extracted_vec3[0].len(), 3); // rows
    assert_eq!(extracted_vec3[0][0].len(), 4); // cols

    println!(
        "3D structure verified: {} x {} x {}",
        extracted_vec3.len(),
        extracted_vec3[0].len(),
        extracted_vec3[0][0].len()
    );

    // Print first slice
    println!("First slice [0,:,:]:");
    for row in &extracted_vec3[0] {
        println!("  {row:?}");
    }

    println!("✅ to_vec3 test passed!");

    // Performance test with larger tensors
    println!("\n🚀 Performance test with larger tensors:");
    let large_data: Vec<f32> = (0..10000).map(|i| i as f32).collect();
    let large_tensor = Tensor::from_vec(large_data.clone(), [10000])?;

    let start = std::time::Instant::now();
    let _extracted_large = large_tensor.to_vec::<f32>()?;
    let duration = start.elapsed();
    println!("Extracted 10,000 elements in {duration:?}");

    // Test non-contiguous tensor (sliced)
    println!("\n📊 Testing with non-contiguous tensor:");
    let base_tensor = Tensor::from_vec((0..20).map(|i| i as f32).collect(), [4, 5])?;
    let sliced_tensor = base_tensor.slice(s![1..3, 1..4]);
    println!(
        "Sliced tensor shape: {:?}",
        sliced_tensor.shape().as_slice()
    );
    let sliced_vec2 = sliced_tensor.to_vec2::<f32>()?;
    println!("Sliced to_vec2: {sliced_vec2:?}");
    println!("✅ Non-contiguous tensor test passed!");

    println!("\n🎉 All to_vec series tests completed successfully!");
    Ok(())
}
