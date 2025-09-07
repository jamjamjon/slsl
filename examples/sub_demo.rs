use slsl::Tensor;

fn main() -> anyhow::Result<()> {
    println!("=== Tensor Subtraction Demo ===\n");

    // Basic subtraction
    println!("1. Basic tensor subtraction:");
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::from_vec(vec![0.5f32, 1.0, 1.5, 2.0], vec![2, 2])?;

    println!("   Tensor A: {:?}", a.as_slice::<f32>()?);
    println!("   Tensor B: {:?}", b.as_slice::<f32>()?);

    let result = a.sub(&b)?;
    println!("   A - B = {:?}", result.as_slice::<f32>()?);
    println!();

    // Scalar subtraction
    println!("2. Scalar subtraction:");
    let a = Tensor::from_vec(vec![5.0f32, 10.0, 15.0, 20.0], vec![2, 2])?;
    println!("   Tensor A: {:?}", a.as_slice::<f32>()?);
    println!("   Subtracting scalar: 3.0");

    let result = a.sub_scalar(3.0f32)?;
    println!("   A - 3.0 = {:?}", result.as_slice::<f32>()?);
    println!();

    // Different data types
    println!("3. Different data types:");

    // f64
    let a_f64 = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b_f64 = Tensor::from_vec(vec![0.1f64, 0.2, 0.3, 0.4], vec![2, 2])?;
    let result_f64 = a_f64.sub(&b_f64)?;
    println!(
        "   f64: {:?} - {:?} = {:?}",
        a_f64.as_slice::<f64>()?,
        b_f64.as_slice::<f64>()?,
        result_f64.as_slice::<f64>()?
    );

    // i32
    let a_i32 = Tensor::from_vec(vec![10i32, 20, 30, 40], vec![2, 2])?;
    let b_i32 = Tensor::from_vec(vec![1i32, 2, 3, 4], vec![2, 2])?;
    let result_i32 = a_i32.sub(&b_i32)?;
    println!(
        "   i32: {:?} - {:?} = {:?}",
        a_i32.as_slice::<i32>()?,
        b_i32.as_slice::<i32>()?,
        result_i32.as_slice::<i32>()?
    );
    println!();

    // Operator overloading
    println!("4. Operator overloading:");
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::from_vec(vec![0.5f32, 1.0, 1.5, 2.0], vec![2, 2])?;

    // Tensor - Tensor
    let result1 = &a - &b;
    println!("   &a - &b = {:?}", result1.as_slice::<f32>()?);

    // Tensor - scalar
    let result2 = &a - 1.0f32;
    println!("   &a - 1.0 = {:?}", result2.as_slice::<f32>()?);

    // scalar - Tensor (not implemented, would need separate trait)
    println!();

    // Edge cases
    println!("5. Edge cases:");

    // Empty tensor
    let empty = Tensor::from_vec(Vec::<f32>::new(), [0])?;
    let result_empty = empty.sub_scalar(5.0f32)?;
    println!(
        "   Empty tensor - 5.0: shape={:?}, numel={}",
        result_empty.dims(),
        result_empty.numel()
    );

    // Single element
    let single = Tensor::from_vec(vec![10.0f32], vec![1])?;
    let result_single = single.sub_scalar(3.0f32)?;
    println!(
        "   Single element: 10.0 - 3.0 = {:?}",
        result_single.as_slice::<f32>()?
    );
    println!();

    // Performance comparison
    println!("6. Performance comparison:");
    let size = 1000;
    let data1: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let data2: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

    let tensor1 = Tensor::from_vec(data1, vec![size])?;
    let tensor2 = Tensor::from_vec(data2, vec![size])?;

    let start = std::time::Instant::now();
    let _result = &tensor1 - &tensor2;
    let elapsed = start.elapsed();

    println!("   Subtracting {size} elements took: {elapsed:?}");
    println!();

    println!("=== Demo completed successfully! ===");
    Ok(())
}
