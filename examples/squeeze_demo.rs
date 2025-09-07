use slsl::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Squeeze Function Demo ===\n");

    // Demo 1: Squeeze all dimensions of size 1
    println!("1. Squeeze all dimensions of size 1:");
    let tensor = Tensor::from_vec(vec![1, 2, 3, 4], [1, 4, 1, 1])?;
    println!(
        "   Original tensor: {:?} -> {:?}",
        tensor.dims(),
        tensor.to_flat_vec::<i32>()?
    );

    let squeezed = tensor.squeeze_all()?;
    println!(
        "   After squeeze_all(): {:?} -> {:?}",
        squeezed.dims(),
        squeezed.to_flat_vec::<i32>()?
    );

    println!();

    // Demo 2: Squeeze specific dimensions
    println!("2. Squeeze specific dimensions:");
    let tensor_2d = Tensor::from_vec(vec![1, 2, 3, 4], [2, 1, 2])?;
    println!(
        "   Original tensor: {:?} -> {:?}",
        tensor_2d.dims(),
        tensor_2d.to_flat_vec::<i32>()?
    );

    let squeezed_dim1 = tensor_2d.squeeze(1)?;
    println!(
        "   After squeeze(1): {:?} -> {:?}",
        squeezed_dim1.dims(),
        squeezed_dim1.to_flat_vec::<i32>()?
    );

    println!();

    // Demo 3: Squeeze multiple specific dimensions
    println!("3. Squeeze multiple specific dimensions:");
    let tensor_3d = Tensor::from_vec(vec![1, 2, 3, 4], [1, 2, 1, 2, 1])?;
    println!(
        "   Original tensor: {:?} -> {:?}",
        tensor_3d.dims(),
        tensor_3d.to_flat_vec::<i32>()?
    );

    let squeezed_multi = tensor_3d.squeeze([0, 2, 4])?;
    println!(
        "   After squeeze([0, 2, 4]): {:?} -> {:?}",
        squeezed_multi.dims(),
        squeezed_multi.to_flat_vec::<i32>()?
    );

    println!();

    // Demo 4: Squeeze with negative dimensions
    println!("4. Squeeze with negative dimensions:");
    let tensor_neg = Tensor::from_vec(vec![1, 2, 3, 4], [1, 2, 1, 2])?;
    println!(
        "   Original tensor: {:?} -> {:?}",
        tensor_neg.dims(),
        tensor_neg.to_flat_vec::<i32>()?
    );

    let squeezed_neg = tensor_neg.squeeze([-1, -3])?;
    println!(
        "   After squeeze([-1, -3]): {:?} -> {:?}",
        squeezed_neg.dims(),
        squeezed_neg.to_flat_vec::<i32>()?
    );

    println!();

    // Demo 5: Squeeze dimensions that are not size 1 (no effect)
    println!("5. Squeeze dimensions that are not size 1 (no effect):");
    let tensor_no_squeeze = Tensor::from_vec(vec![1, 2, 3, 4], [2, 2])?;
    println!(
        "   Original tensor: {:?} -> {:?}",
        tensor_no_squeeze.dims(),
        tensor_no_squeeze.to_flat_vec::<i32>()?
    );

    let squeezed_no_effect = tensor_no_squeeze.squeeze(0)?;
    println!(
        "   After squeeze(0): {:?} -> {:?}",
        squeezed_no_effect.dims(),
        squeezed_no_effect.to_flat_vec::<i32>()?
    );

    println!();

    // Demo 6: Squeeze all dimensions to scalar
    println!("6. Squeeze all dimensions to scalar:");
    let scalar_tensor = Tensor::from_vec(vec![42], [1, 1, 1])?;
    println!(
        "   Original tensor: {:?} -> {:?}",
        scalar_tensor.dims(),
        scalar_tensor.to_flat_vec::<i32>()?
    );

    let squeezed_scalar = scalar_tensor.squeeze_all()?;
    println!(
        "   After squeeze_all(): {:?} -> {:?}",
        squeezed_scalar.dims(),
        squeezed_scalar.to_flat_vec::<i32>()?
    );

    println!();

    // Demo 7: Squeeze with different data types
    println!("7. Squeeze with different data types:");
    let bool_tensor = Tensor::from_vec(vec![true, false], [1, 2, 1])?;
    println!(
        "   Original bool tensor: {:?} -> {:?}",
        bool_tensor.dims(),
        bool_tensor.to_flat_vec::<bool>()?
    );

    let squeezed_bool = bool_tensor.squeeze_all()?;
    println!(
        "   After squeeze_all(): {:?} -> {:?}",
        squeezed_bool.dims(),
        squeezed_bool.to_flat_vec::<bool>()?
    );

    let float_tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [1, 1, 2, 2])?;
    println!(
        "   Original float tensor: {:?} -> {:?}",
        float_tensor.dims(),
        float_tensor.to_flat_vec::<f32>()?
    );

    let squeezed_float = float_tensor.squeeze_all()?;
    println!(
        "   After squeeze_all(): {:?} -> {:?}",
        squeezed_float.dims(),
        squeezed_float.to_flat_vec::<f32>()?
    );

    println!();

    // Demo 8: Squeeze with different input types
    println!("8. Squeeze with different input types:");
    let tensor_mixed = Tensor::from_vec(vec![1, 2, 3, 4], [1, 2, 1, 2])?;
    println!(
        "   Original tensor: {:?} -> {:?}",
        tensor_mixed.dims(),
        tensor_mixed.to_flat_vec::<i32>()?
    );

    // Tuple input
    let squeezed_tuple = tensor_mixed.squeeze((0, 2))?;
    println!(
        "   After squeeze((0, 2)): {:?} -> {:?}",
        squeezed_tuple.dims(),
        squeezed_tuple.to_flat_vec::<i32>()?
    );

    // Array input
    let squeezed_array = tensor_mixed.squeeze([0usize, 2])?;
    println!(
        "   After squeeze([0usize, 2]): {:?} -> {:?}",
        squeezed_array.dims(),
        squeezed_array.to_flat_vec::<i32>()?
    );

    // Vector input
    let squeezed_vec = tensor_mixed.squeeze(vec![0, 2])?;
    println!(
        "   After squeeze(vec![0, 2]): {:?} -> {:?}",
        squeezed_vec.dims(),
        squeezed_vec.to_flat_vec::<i32>()?
    );

    println!();

    // Demo 9: Complex squeeze operations
    println!("9. Complex squeeze operations:");
    let complex_tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 1, 2, 1, 2])?;
    println!(
        "   Original tensor: {:?} -> {:?}",
        complex_tensor.dims(),
        complex_tensor.to_flat_vec::<i32>()?
    );

    // Squeeze only some dimensions
    let squeezed_partial = complex_tensor.squeeze([0, 2, 4])?;
    println!(
        "   After squeeze([0, 2, 4]): {:?} -> {:?}",
        squeezed_partial.dims(),
        squeezed_partial.to_flat_vec::<i32>()?
    );

    // Squeeze all dimensions
    let squeezed_all = complex_tensor.squeeze_all()?;
    println!(
        "   After squeeze_all(): {:?} -> {:?}",
        squeezed_all.dims(),
        squeezed_all.to_flat_vec::<i32>()?
    );

    println!();

    // Demo 10: Edge cases
    println!("10. Edge cases:");

    // Single element tensor
    let single_element = Tensor::from_vec(vec![99], [1])?;
    println!(
        "   Single element tensor: {:?} -> {:?}",
        single_element.dims(),
        single_element.to_flat_vec::<i32>()?
    );

    let squeezed_single = single_element.squeeze_all()?;
    println!(
        "   After squeeze_all(): {:?} -> {:?}",
        squeezed_single.dims(),
        squeezed_single.to_flat_vec::<i32>()?
    );

    // All dimensions are 1
    let all_ones = Tensor::from_vec(vec![42], [1, 1, 1, 1])?;
    println!(
        "   All ones tensor: {:?} -> {:?}",
        all_ones.dims(),
        all_ones.to_flat_vec::<i32>()?
    );

    let squeezed_all_ones = all_ones.squeeze_all()?;
    println!(
        "   After squeeze_all(): {:?} -> {:?}",
        squeezed_all_ones.dims(),
        squeezed_all_ones.to_flat_vec::<i32>()?
    );

    println!("\n=== Demo completed successfully! ===");

    Ok(())
}
