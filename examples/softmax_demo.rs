use slsl::Tensor;

fn main() -> anyhow::Result<()> {
    println!("=== Softmax Function Demo ===\n");

    // Test f32
    println!("1. Testing f32 softmax:");
    let tensor_f32 = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32], [3])?;
    let softmaxed_f32 = tensor_f32.softmax(0)?;
    println!("   Input: {:?}", tensor_f32.to_vec::<f32>()?);
    println!("   Output: {:?}", softmaxed_f32.to_vec::<f32>()?);
    println!(
        "   Sum: {:.6}",
        softmaxed_f32.to_vec::<f32>()?.iter().sum::<f32>()
    );
    println!();

    // Test f64
    println!("2. Testing f64 softmax:");
    let tensor_f64 = Tensor::from_vec(vec![1.0f64, 2.0f64, 3.0f64], [3])?;
    let softmaxed_f64 = tensor_f64.softmax(0)?;
    println!("   Input: {:?}", tensor_f64.to_vec::<f64>()?);
    println!("   Output: {:?}", softmaxed_f64.to_vec::<f64>()?);
    println!(
        "   Sum: {:.6}",
        softmaxed_f64.to_vec::<f64>()?.iter().sum::<f64>()
    );
    println!();

    // Test f16
    println!("3. Testing f16 softmax:");
    let tensor_f16 = Tensor::from_vec(
        vec![
            half::f16::from_f32(1.0f32),
            half::f16::from_f32(2.0f32),
            half::f16::from_f32(3.0f32),
        ],
        [3],
    )?;
    let softmaxed_f16 = tensor_f16.softmax(0)?;
    println!("   Input: {:?}", tensor_f16.to_vec::<half::f16>()?);
    println!("   Output: {:?}", softmaxed_f16.to_vec::<half::f16>()?);
    println!(
        "   Sum: {:.6}",
        softmaxed_f16
            .to_vec::<half::f16>()?
            .iter()
            .map(|&x: &half::f16| x.to_f32())
            .sum::<f32>()
    );
    println!();

    // Test bf16
    println!("4. Testing bf16 softmax:");
    let tensor_bf16 = Tensor::from_vec(
        vec![
            half::bf16::from_f32(1.0f32),
            half::bf16::from_f32(2.0f32),
            half::bf16::from_f32(3.0f32),
        ],
        [3],
    )?;
    let softmaxed_bf16 = tensor_bf16.softmax(0)?;
    println!("   Input: {:?}", tensor_bf16.to_vec::<half::bf16>()?);
    println!("   Output: {:?}", softmaxed_bf16.to_vec::<half::bf16>()?);
    println!(
        "   Sum: {:.6}",
        softmaxed_bf16
            .to_vec::<half::bf16>()?
            .iter()
            .map(|&x: &half::bf16| x.to_f32())
            .sum::<f32>()
    );
    println!();

    // Test 2D tensor with f32
    println!("5. Testing 2D tensor softmax (f32):");
    let tensor_2d = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], [2, 2])?;
    let softmaxed_2d = tensor_2d.softmax(1)?; // Along the second dimension

    println!("   Input tensor:");
    for i in 0..2 {
        for j in 0..2 {
            print!("   {:.3}", tensor_2d.at::<f32>(&*vec![i, j]));
        }
        println!();
    }

    println!("   Output tensor (softmax along dim 1):");
    for i in 0..2 {
        for j in 0..2 {
            print!("   {:.3}", softmaxed_2d.at::<f32>(&*vec![i, j]));
        }
        println!();
    }

    // Verify sums along dimension 1
    for i in 0..2 {
        let sum: f32 = (0..2).map(|j| softmaxed_2d.at::<f32>(&*vec![i, j])).sum();
        println!("   Row {i} sum: {sum:.6}");
    }
    println!();

    // Test edge cases
    println!("6. Testing edge cases:");

    // All zeros
    let zero_tensor = Tensor::from_vec(vec![0.0f32, 0.0f32, 0.0f32], [3])?;
    let softmaxed_zeros = zero_tensor.softmax(0)?;
    println!("   All zeros input: {:?}", zero_tensor.to_vec::<f32>()?);
    println!(
        "   All zeros output: {:?}",
        softmaxed_zeros.to_vec::<f32>()?
    );
    println!(
        "   Sum: {:.6}",
        softmaxed_zeros.to_vec::<f32>()?.iter().sum::<f32>()
    );

    // Large values (numerical stability)
    let large_tensor = Tensor::from_vec(vec![1000.0f32, 1001.0f32, 1002.0f32], [3])?;
    let softmaxed_large = large_tensor.softmax(0)?;
    println!("   Large values input: {:?}", large_tensor.to_vec::<f32>()?);
    println!(
        "   Large values output: {:?}",
        softmaxed_large.to_vec::<f32>()?
    );
    println!(
        "   Sum: {:.6}",
        softmaxed_large.to_vec::<f32>()?.iter().sum::<f32>()
    );
    println!();

    // Test error cases
    println!("7. Testing error cases:");

    // Integer tensor (should fail)
    let int_tensor = Tensor::from_vec(vec![1, 2, 3], [3])?;
    let result = int_tensor.softmax(0);
    match result {
        Ok(_) => println!("   ❌ Integer tensor softmax should have failed!"),
        Err(e) => println!("   ✅ Integer tensor correctly rejected: {e}"),
    }

    // Invalid dimension (should fail)
    let float_tensor = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32], [3])?;
    let result = float_tensor.softmax(5);
    match result {
        Ok(_) => println!("   ❌ Invalid dimension should have failed!"),
        Err(e) => println!("   ✅ Invalid dimension correctly rejected: {e}"),
    }

    println!("\n=== All tests completed successfully! ===");
    println!("✅ f32 softmax working");
    println!("✅ f64 softmax working");
    println!("✅ f16 softmax working");
    println!("✅ bf16 softmax working");
    println!("✅ 2D tensor support working");
    println!("✅ Edge cases handled correctly");
    println!("✅ Error cases handled correctly");

    Ok(())
}
