use slsl::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Stack and Cat Functions Demo ===\n");

    // Demo 1: Stacking 1D tensors
    println!("1. Stacking 1D tensors:");
    let tensor1 = Tensor::from_vec(vec![1, 2, 3], [3])?;
    let tensor2 = Tensor::from_vec(vec![4, 5, 6], [3])?;
    let tensor3 = Tensor::from_vec(vec![7, 8, 9], [3])?;

    println!("   Original tensors:");
    println!(
        "   Tensor 1: {:?} -> {:?}",
        tensor1.dims(),
        tensor1.to_flat_vec::<i32>()?
    );
    println!(
        "   Tensor 2: {:?} -> {:?}",
        tensor2.dims(),
        tensor2.to_flat_vec::<i32>()?
    );
    println!(
        "   Tensor 3: {:?} -> {:?}",
        tensor3.dims(),
        tensor3.to_flat_vec::<i32>()?
    );

    let tensors = vec![&tensor1, &tensor2, &tensor3];

    // Stack along dimension 0
    let stacked_dim0 = Tensor::stack(&tensors, 0)?;
    println!(
        "   Stacked along dim 0: {:?} -> {:?}",
        stacked_dim0.dims(),
        stacked_dim0.to_flat_vec::<i32>()?
    );

    // Stack along dimension 1 (creates new dimension)
    let stacked_dim1 = Tensor::stack(&tensors, 1)?;
    println!(
        "   Stacked along dim 1: {:?} -> {:?}",
        stacked_dim1.dims(),
        stacked_dim1.to_flat_vec::<i32>()?
    );

    println!();

    // Demo 2: Stacking 2D tensors
    println!("2. Stacking 2D tensors:");
    let tensor2d_1 = Tensor::from_vec(vec![1, 2, 3, 4], [2, 2])?;
    let tensor2d_2 = Tensor::from_vec(vec![5, 6, 7, 8], [2, 2])?;

    println!("   Original 2D tensors:");
    println!(
        "   Tensor 1: {:?} -> {:?}",
        tensor2d_1.dims(),
        tensor2d_1.to_flat_vec::<i32>()?
    );
    println!(
        "   Tensor 2: {:?} -> {:?}",
        tensor2d_2.dims(),
        tensor2d_2.to_flat_vec::<i32>()?
    );

    let tensors_2d = vec![&tensor2d_1, &tensor2d_2];

    // Stack along different dimensions
    let stacked_2d_dim0 = Tensor::stack(&tensors_2d, 0)?;
    println!(
        "   Stacked along dim 0: {:?} -> {:?}",
        stacked_2d_dim0.dims(),
        stacked_2d_dim0.to_flat_vec::<i32>()?
    );

    let stacked_2d_dim1 = Tensor::stack(&tensors_2d, 1)?;
    println!(
        "   Stacked along dim 1: {:?} -> {:?}",
        stacked_2d_dim1.dims(),
        stacked_2d_dim1.to_flat_vec::<i32>()?
    );

    let stacked_2d_dim2 = Tensor::stack(&tensors_2d, 2)?;
    println!(
        "   Stacked along dim 2: {:?} -> {:?}",
        stacked_2d_dim2.dims(),
        stacked_2d_dim2.to_flat_vec::<i32>()?
    );

    println!();

    // Demo 3: Concatenating 1D tensors
    println!("3. Concatenating 1D tensors:");
    let concatenated = Tensor::cat(&tensors, 0)?;
    println!(
        "   Concatenated along dim 0: {:?} -> {:?}",
        concatenated.dims(),
        concatenated.to_flat_vec::<i32>()?
    );

    println!();

    // Demo 4: Concatenating 2D tensors
    println!("4. Concatenating 2D tensors:");

    // Concatenate along dimension 0 (rows)
    let concatenated_2d_rows = Tensor::cat(&tensors_2d, 0)?;
    println!(
        "   Concatenated along dim 0 (rows): {:?} -> {:?}",
        concatenated_2d_rows.dims(),
        concatenated_2d_rows.to_flat_vec::<i32>()?
    );

    // Concatenate along dimension 1 (columns)
    let concatenated_2d_cols = Tensor::cat(&tensors_2d, 1)?;
    println!(
        "   Concatenated along dim 1 (columns): {:?} -> {:?}",
        concatenated_2d_cols.dims(),
        concatenated_2d_cols.to_flat_vec::<i32>()?
    );

    println!();

    // Demo 5: Concatenating tensors with different sizes
    println!("5. Concatenating tensors with different sizes:");
    let tensor_small = Tensor::from_vec(vec![1, 2], [2])?;
    let tensor_large = Tensor::from_vec(vec![3, 4, 5], [3])?;

    println!(
        "   Small tensor: {:?} -> {:?}",
        tensor_small.dims(),
        tensor_small.to_flat_vec::<i32>()?
    );
    println!(
        "   Large tensor: {:?} -> {:?}",
        tensor_large.dims(),
        tensor_large.to_flat_vec::<i32>()?
    );

    let tensors_diff_size = vec![&tensor_small, &tensor_large];
    let concatenated_diff_size = Tensor::cat(&tensors_diff_size, 0)?;
    println!(
        "   Concatenated: {:?} -> {:?}",
        concatenated_diff_size.dims(),
        concatenated_diff_size.to_flat_vec::<i32>()?
    );

    println!();

    // Demo 6: Different data types
    println!("6. Working with different data types:");

    // Bool tensors
    let bool_tensor1 = Tensor::from_vec(vec![true, false], [2])?;
    let bool_tensor2 = Tensor::from_vec(vec![false, true], [2])?;

    let bool_tensors = vec![&bool_tensor1, &bool_tensor2];
    let stacked_bool = Tensor::stack(&bool_tensors, 0)?;
    println!(
        "   Bool tensors stacked: {:?} -> {:?}",
        stacked_bool.dims(),
        stacked_bool.to_flat_vec::<bool>()?
    );

    let concatenated_bool = Tensor::cat(&bool_tensors, 0)?;
    println!(
        "   Bool tensors concatenated: {:?} -> {:?}",
        concatenated_bool.dims(),
        concatenated_bool.to_flat_vec::<bool>()?
    );

    // Float tensors
    let float_tensor1 = Tensor::from_vec(vec![1.0f32, 2.0f32], [2])?;
    let float_tensor2 = Tensor::from_vec(vec![3.0f32, 4.0f32], [2])?;

    let float_tensors = vec![&float_tensor1, &float_tensor2];
    let stacked_float = Tensor::stack(&float_tensors, 0)?;
    println!(
        "   Float tensors stacked: {:?} -> {:?}",
        stacked_float.dims(),
        stacked_float.to_flat_vec::<f32>()?
    );

    let concatenated_float = Tensor::cat(&float_tensors, 0)?;
    println!(
        "   Float tensors concatenated: {:?} -> {:?}",
        concatenated_float.dims(),
        concatenated_float.to_flat_vec::<f32>()?
    );

    println!();

    // Demo 7: Error handling
    println!("7. Error handling:");

    // Empty tensor list
    let empty_tensors: Vec<&Tensor> = vec![];
    let stack_result = Tensor::stack(&empty_tensors, 0);
    println!("   Stack empty list: {stack_result:?}");

    let cat_result = Tensor::cat(&empty_tensors, 0);
    println!("   Cat empty list: {cat_result:?}");

    // Different shapes
    let different_shape_tensor = Tensor::from_vec(vec![1, 2], [2])?;
    let tensors_diff_shape = vec![&tensor1, &different_shape_tensor];
    let stack_diff_shape = Tensor::stack(&tensors_diff_shape, 0);
    println!("   Stack different shapes: {stack_diff_shape:?}");

    // Different dtypes
    let different_dtype_tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3])?;
    let tensors_diff_dtype = vec![&tensor1, &different_dtype_tensor];
    let stack_diff_dtype = Tensor::stack(&tensors_diff_dtype, 0);
    println!("   Stack different dtypes: {stack_diff_dtype:?}");

    println!();

    // Demo 8: Negative dimensions
    println!("8. Negative dimensions:");

    // Stack with negative dimension
    let stacked_neg = Tensor::stack(&tensors, -1)?;
    println!(
        "   Stacked with dim -1: {:?} -> {:?}",
        stacked_neg.dims(),
        stacked_neg.to_flat_vec::<i32>()?
    );

    // Cat with negative dimension
    let concatenated_neg = Tensor::cat(&tensors, -1)?;
    println!(
        "   Concatenated with dim -1: {:?} -> {:?}",
        concatenated_neg.dims(),
        concatenated_neg.to_flat_vec::<i32>()?
    );

    println!("\n=== Demo completed successfully! ===");

    Ok(())
}
