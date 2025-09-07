use slsl::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Sum Reduction Operations Demo ===\n");

    // Create a 3D tensor for demonstration
    let tensor = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ],
        [2, 3, 4],
    )?;

    println!("Original tensor:");
    println!("Shape: {:?}", tensor.dims());
    println!("Data: {:?}", tensor.as_slice::<f32>()?);
    println!();

    // 1. sum_all - sum all elements
    let sum_all = tensor.sum_all()?;
    println!("1. sum_all(): {sum_all}");
    println!("   Sum all elements");
    println!();

    // 2. Sum along single dimension
    println!("2. Sum along single dimension:");

    // Sum along dimension 0
    let sum_dim0 = tensor.sum(0)?;
    println!("   Sum along dimension 0:");
    println!("   Shape: {:?} -> {:?}", tensor.dims(), sum_dim0.dims());
    println!("   Result: {:?}", sum_dim0.as_slice::<f64>()?);

    // Sum along dimension 1
    let sum_dim1 = tensor.sum(1)?;
    println!("   Sum along dimension 1:");
    println!("   Shape: {:?} -> {:?}", tensor.dims(), sum_dim1.dims());
    println!("   Result: {:?}", sum_dim1.as_slice::<f64>()?);

    // Sum along dimension 2
    let sum_dim2 = tensor.sum(2)?;
    println!("   Sum along dimension 2:");
    println!("   Shape: {:?} -> {:?}", tensor.dims(), sum_dim2.dims());
    println!("   Result: {:?}", sum_dim2.as_slice::<f64>()?);
    println!();

    // 3. Sum along multiple dimensions
    println!("3. Sum along multiple dimensions:");

    // Sum along dimensions [0,1]
    let sum_dims_01 = tensor.sum([0, 1])?;
    println!("   Sum along dimensions [0,1]:");
    println!("   Shape: {:?} -> {:?}", tensor.dims(), sum_dims_01.dims());
    println!("   Result: {:?}", sum_dims_01.as_slice::<f64>()?);

    // Sum along dimensions [1,2]
    let sum_dims_12 = tensor.sum([1, 2])?;
    println!("   Sum along dimensions [1,2]:");
    println!("   Shape: {:?} -> {:?}", tensor.dims(), sum_dims_12.dims());
    println!("   Result: {:?}", sum_dims_12.as_slice::<f64>()?);

    // Sum along all dimensions
    let sum_all_dims = tensor.sum([0, 1, 2])?;
    println!("   Sum along dimensions [0,1,2]:");
    println!("   Shape: {:?} -> {:?}", tensor.dims(), sum_all_dims.dims());
    println!("   Result: {:?}", sum_all_dims.as_slice::<f64>()?);
    println!();

    // 4. sum_keepdim - sum with keeping dimensions
    println!("4. Sum with keeping dimensions (sum_keepdim):");

    // Sum along dimension 0, keeping dimensions
    let sum_keepdim_0 = tensor.sum_keepdim(0)?;
    println!("   Sum along dimension 0 (keeping dimensions):");
    println!(
        "   Shape: {:?} -> {:?}",
        tensor.dims(),
        sum_keepdim_0.dims()
    );
    println!("   Result shape remains [1, 3, 4]");

    // Sum along multiple dimensions, keeping dimensions
    let sum_keepdim_01 = tensor.sum_keepdim([0, 1])?;
    println!("   Sum along dimensions [0,1] (keeping dimensions):");
    println!(
        "   Shape: {:?} -> {:?}",
        tensor.dims(),
        sum_keepdim_01.dims()
    );
    println!("   Result: {:?}", sum_keepdim_01.as_slice::<f64>()?);
    println!();

    // 5. Support for different data types
    println!("5. Support for different data types:");

    // f64 type
    let tensor_f64 = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], [2, 2])?;
    let sum_f64 = tensor_f64.sum_all()?;
    println!("   f64 tensor sum: {sum_f64}");

    // i32 type
    let tensor_i32 = Tensor::from_vec(vec![1i32, 2, 3, 4], [2, 2])?;
    let sum_i32 = tensor_i32.sum_all()?;
    println!("   i32 tensor sum: {sum_i32}");

    // u8 type
    let tensor_u8 = Tensor::from_vec(vec![1u8, 2, 3, 4], [2, 2])?;
    let sum_u8 = tensor_u8.sum_all()?;
    println!("   u8 tensor sum: {sum_u8}");
    println!();

    // 6. Contiguous optimization demonstration
    println!("6. Contiguous optimization:");

    // Contiguous tensor
    let continuous_tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2])?;
    println!(
        "   Contiguous tensor: is_contiguous = {}",
        continuous_tensor.is_contiguous()
    );
    let sum_continuous = continuous_tensor.sum_all()?;
    println!("   Contiguous tensor sum (using backend optimization): {sum_continuous}");

    println!("=== Demo completed ===");

    Ok(())
}
