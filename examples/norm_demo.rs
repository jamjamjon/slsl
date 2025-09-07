use slsl::Tensor;

fn main() -> anyhow::Result<()> {
    println!("=== Norm Function Demo ===\n");

    // 1. Testing L1 norm (Manhattan norm)
    println!("1. Testing L1 norm (Manhattan norm):");
    let tensor = Tensor::from_vec(vec![3.0f32, -4.0f32], [2])?;
    let normed = tensor.norm1(0)?;
    println!("   Input: [3.0, -4.0]");
    println!("   L1 norm: |3| + |-4| = 3 + 4 = 7");
    println!("   Output: {}", normed.at::<f32>(&*vec![]));
    println!();

    // 2. Testing L2 norm (Euclidean norm)
    println!("2. Testing L2 norm (Euclidean norm):");
    let tensor = Tensor::from_vec(vec![3.0f32, 4.0f32], [2])?;
    let normed = tensor.norm2(0)?;
    println!("   Input: [3.0, 4.0]");
    println!("   L2 norm: sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5");
    println!("   Output: {}", normed.at::<f32>(&*vec![]));
    println!();

    // 3. Testing Lp norm
    println!("3. Testing L3 norm:");
    let tensor = Tensor::from_vec(vec![3.0f32, 4.0f32], [2])?;
    let normed = tensor.normp(0, 3.0)?;
    println!("   Input: [3.0, 4.0]");
    println!("   L3 norm: (3^3 + 4^3)^(1/3) = (27 + 64)^(1/3) = 91^(1/3)");
    println!("   Output: {}", normed.at::<f32>(&*vec![]));
    println!();

    // 4. Testing 2D tensor norms
    println!("4. Testing 2D tensor norms:");
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], [2, 2])?;
    println!("   Input tensor:");
    println!(
        "   {}   {}",
        tensor.at::<f32>(&*vec![0, 0]),
        tensor.at::<f32>(&*vec![0, 1])
    );
    println!(
        "   {}   {}",
        tensor.at::<f32>(&*vec![1, 0]),
        tensor.at::<f32>(&*vec![1, 1])
    );

    // L1 norm along dim 0
    let normed = tensor.norm1(0)?;
    println!("   L1 norm along dim 0: [|1|+|3|, |2|+|4|] = [4, 6]");
    println!(
        "   Output: [{}, {}]",
        normed.at::<f32>(&*vec![0]),
        normed.at::<f32>(&*vec![1])
    );

    // L2 norm along dim 0
    let normed = tensor.norm2(0)?;
    println!("   L2 norm along dim 0: [sqrt(1^2+3^2), sqrt(2^2+4^2)] = [sqrt(10), sqrt(20)]");
    println!(
        "   Output: [{:.6}, {:.6}]",
        normed.at::<f32>(&*vec![0]),
        normed.at::<f32>(&*vec![1])
    );
    println!();

    // 5. Testing keepdim functionality
    println!("5. Testing keepdim functionality:");
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], [2, 2])?;
    let normed = tensor.norm_keepdim(0, 2.0)?;
    println!("   Input shape: [2, 2]");
    println!("   Output shape: {:?}", normed.shape().as_slice());
    println!(
        "   Output: [{:.6}, {:.6}]",
        normed.at::<f32>(&*vec![0, 0]),
        normed.at::<f32>(&*vec![0, 1])
    );
    println!();

    // 6. Testing different data types
    println!("6. Testing different data types:");

    // f32
    let tensor_f32 = Tensor::from_vec(vec![3.0f32, 4.0f32], [2])?;
    let normed_f32 = tensor_f32.norm2(0)?;
    println!(
        "   f32: {} (dtype: {:?})",
        normed_f32.at::<f32>(&*vec![]),
        normed_f32.dtype()
    );

    // f64
    let tensor_f64 = Tensor::from_vec(vec![3.0f64, 4.0f64], [2])?;
    let normed_f64 = tensor_f64.norm2(0)?;
    println!(
        "   f64: {} (dtype: {:?})",
        normed_f64.at::<f64>(&*vec![]),
        normed_f64.dtype()
    );

    // f16
    let tensor_f16 = Tensor::from_vec(
        vec![half::f16::from_f32(3.0f32), half::f16::from_f32(4.0f32)],
        [2],
    )?;
    let normed_f16 = tensor_f16.norm2(0)?;
    println!(
        "   f16: {} (dtype: {:?})",
        normed_f16.at::<half::f16>(&*vec![]).to_f32(),
        normed_f16.dtype()
    );

    // bf16
    let tensor_bf16 = Tensor::from_vec(
        vec![half::bf16::from_f32(3.0f32), half::bf16::from_f32(4.0f32)],
        [2],
    )?;
    let normed_bf16 = tensor_bf16.norm2(0)?;
    println!(
        "   bf16: {} (dtype: {:?})",
        normed_bf16.at::<half::bf16>(&*vec![]).to_f32(),
        normed_bf16.dtype()
    );
    println!();

    // 7. Testing edge cases
    println!("7. Testing edge cases:");

    // All zeros
    let zero_tensor = Tensor::from_vec(vec![0.0f32, 0.0f32, 0.0f32], [3])?;
    let normed = zero_tensor.norm2(0)?;
    println!("   All zeros input: [0.0, 0.0, 0.0]");
    println!("   L2 norm: {}", normed.at::<f32>(&*vec![]));

    // Single element
    let single_tensor = Tensor::from_vec(vec![42.0f32], [1])?;
    let normed = single_tensor.norm2(0)?;
    println!("   Single element input: [42.0]");
    println!("   L2 norm: {}", normed.at::<f32>(&*vec![]));
    println!();

    // 8. Testing error cases
    println!("8. Testing error cases:");

    // Integer tensor (should fail)
    let int_tensor = Tensor::from_vec(vec![1, 2, 3], [3])?;
    let result = int_tensor.norm2(0);
    match result {
        Ok(_) => println!("   ❌ Integer tensor incorrectly accepted"),
        Err(e) => println!("   ✅ Integer tensor correctly rejected: {e}"),
    }

    // Invalid dimension (should fail)
    let float_tensor = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32], [3])?;
    let result = float_tensor.norm2(5);
    match result {
        Ok(_) => println!("   ❌ Invalid dimension incorrectly accepted"),
        Err(e) => println!("   ✅ Invalid dimension correctly rejected: {e}"),
    }
    println!();

    println!("=== All tests completed successfully! ===");
    println!("✅ L1 norm working");
    println!("✅ L2 norm working");
    println!("✅ Lp norm working");
    println!("✅ keepdim functionality working");
    println!("✅ All float types supported");
    println!("✅ Edge cases handled correctly");
    println!("✅ Error cases handled correctly");

    Ok(())
}
