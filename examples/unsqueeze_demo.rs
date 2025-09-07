use slsl::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Unsqueeze Function Demo ===\n");

    // Demo 1: Unsqueeze 1D tensor at dimension 0
    println!("Demo 1: Unsqueeze 1D tensor at dimension 0");
    let tensor = Tensor::from_vec(vec![1, 2, 3, 4], [4])?;
    println!(
        "Original tensor: {:?} with shape {:?}",
        tensor.to_flat_vec::<i32>()?,
        tensor.dims()
    );

    let unsqueezed = tensor.unsqueeze(0)?;
    println!(
        "After unsqueeze(0): {:?} with shape {:?}",
        unsqueezed.to_flat_vec::<i32>()?,
        unsqueezed.dims()
    );
    println!();

    // Demo 2: Unsqueeze 1D tensor at dimension 1
    println!("Demo 2: Unsqueeze 1D tensor at dimension 1");
    let unsqueezed_dim1 = tensor.unsqueeze(1)?;
    println!(
        "After unsqueeze(1): {:?} with shape {:?}",
        unsqueezed_dim1.to_flat_vec::<i32>()?,
        unsqueezed_dim1.dims()
    );
    println!();

    // Demo 3: Unsqueeze 1D tensor at dimension 2 (beyond current rank) - should error
    println!("Demo 3: Unsqueeze 1D tensor at dimension 2 (beyond current rank) - should error");
    let result = tensor.unsqueeze(2);
    match result {
        Ok(_) => println!("Unexpected: unsqueeze(2) succeeded on 1D tensor"),
        Err(e) => println!("Expected error: {e}"),
    }
    println!();

    // Demo 4: Unsqueeze 2D tensor at different dimensions
    println!("Demo 4: Unsqueeze 2D tensor at different dimensions");
    let tensor_2d = Tensor::from_vec(vec![1, 2, 3, 4], [2, 2])?;
    println!(
        "Original 2D tensor: {:?} with shape {:?}",
        tensor_2d.to_flat_vec::<i32>()?,
        tensor_2d.dims()
    );

    let unsqueezed_2d_dim0 = tensor_2d.unsqueeze(0)?;
    println!(
        "After unsqueeze(0): {:?} with shape {:?}",
        unsqueezed_2d_dim0.to_flat_vec::<i32>()?,
        unsqueezed_2d_dim0.dims()
    );

    let unsqueezed_2d_dim1 = tensor_2d.unsqueeze(1)?;
    println!(
        "After unsqueeze(1): {:?} with shape {:?}",
        unsqueezed_2d_dim1.to_flat_vec::<i32>()?,
        unsqueezed_2d_dim1.dims()
    );

    let unsqueezed_2d_dim2 = tensor_2d.unsqueeze(2)?;
    println!(
        "After unsqueeze(2): {:?} with shape {:?}",
        unsqueezed_2d_dim2.to_flat_vec::<i32>()?,
        unsqueezed_2d_dim2.dims()
    );
    println!();

    // Demo 5: Unsqueeze with negative dimensions
    println!("Demo 5: Unsqueeze with negative dimensions");
    let unsqueezed_neg = tensor.unsqueeze(-1)?;
    println!(
        "After unsqueeze(-1): {:?} with shape {:?}",
        unsqueezed_neg.to_flat_vec::<i32>()?,
        unsqueezed_neg.dims()
    );

    let unsqueezed_neg2 = tensor.unsqueeze(-2)?;
    println!(
        "After unsqueeze(-2): {:?} with shape {:?}",
        unsqueezed_neg2.to_flat_vec::<i32>()?,
        unsqueezed_neg2.dims()
    );
    println!();

    // Demo 6: Unsqueeze with different data types
    println!("Demo 6: Unsqueeze with different data types");
    let bool_tensor = Tensor::from_vec(vec![true, false], [2])?;
    let unsqueezed_bool = bool_tensor.unsqueeze(0)?;
    println!(
        "Bool tensor after unsqueeze(0): {:?} with shape {:?}",
        unsqueezed_bool.to_flat_vec::<bool>()?,
        unsqueezed_bool.dims()
    );

    let float_tensor = Tensor::from_vec(vec![1.0f32, 2.0], [2])?;
    let unsqueezed_float = float_tensor.unsqueeze(1)?;
    println!(
        "Float tensor after unsqueeze(1): {:?} with shape {:?}",
        unsqueezed_float.to_flat_vec::<f32>()?,
        unsqueezed_float.dims()
    );
    println!();

    // Demo 7: Multiple unsqueeze operations
    println!("Demo 7: Multiple unsqueeze operations");
    let tensor_single = Tensor::from_vec(vec![42], [1])?;
    println!(
        "Original single tensor: {:?} with shape {:?}",
        tensor_single.to_flat_vec::<i32>()?,
        tensor_single.dims()
    );

    let unsqueezed_multi1 = tensor_single.unsqueeze(0)?;
    println!(
        "After first unsqueeze(0): {:?} with shape {:?}",
        unsqueezed_multi1.to_flat_vec::<i32>()?,
        unsqueezed_multi1.dims()
    );

    let unsqueezed_multi2 = unsqueezed_multi1.unsqueeze(2)?;
    println!(
        "After second unsqueeze(2): {:?} with shape {:?}",
        unsqueezed_multi2.to_flat_vec::<i32>()?,
        unsqueezed_multi2.dims()
    );
    println!();

    // Demo 8: Edge cases
    println!("Demo 8: Edge cases");
    let empty_tensor = Tensor::from_vec::<i32, _>(vec![], [0])?;
    println!(
        "Original empty tensor: {:?} with shape {:?}",
        empty_tensor.to_flat_vec::<i32>()?,
        empty_tensor.dims()
    );

    let unsqueezed_empty = empty_tensor.unsqueeze(0)?;
    println!(
        "After unsqueeze(0): {:?} with shape {:?}",
        unsqueezed_empty.to_flat_vec::<i32>()?,
        unsqueezed_empty.dims()
    );
    println!();

    // Demo 9: Visual representation of unsqueeze
    println!("Demo 9: Visual representation of unsqueeze");
    let visual_tensor = Tensor::from_vec(vec![1, 2], [2])?;
    println!("Original 1D tensor: [1, 2] with shape [2]");

    let _visual_unsqueeze0 = visual_tensor.unsqueeze(0)?;
    println!(
        "After unsqueeze(0): shape = {:?}",
        _visual_unsqueeze0.shape()
    );

    let _visual_unsqueeze1 = visual_tensor.unsqueeze(1)?;
    println!(
        "After unsqueeze(1): shape = {:?}",
        _visual_unsqueeze1.shape()
    );
    println!();

    // Demo 10: Practical use case - batch dimension
    println!("Demo 10: Practical use case - batch dimension");
    let single_sample = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3])?;
    println!(
        "Single sample: {:?} with shape {:?}",
        single_sample.to_flat_vec::<f32>()?,
        single_sample.dims()
    );

    let batch_sample = single_sample.unsqueeze(0)?;
    println!(
        "Batch sample (unsqueeze(0)): {:?} with shape {:?}",
        batch_sample.to_flat_vec::<f32>()?,
        batch_sample.dims()
    );
    println!("This is useful for adding a batch dimension to a single sample!");
    println!();

    println!("=== Unsqueeze Demo Complete ===");
    Ok(())
}
