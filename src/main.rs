use slsl::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, [2, 3])?;

    // Create non-contiguous tensor by permuting dimensions
    let permuted = tensor.clone().permute([1, 0])?; // [3, 2]
    println!("Permuted tensor: {permuted:?}");

    // Test min_max_argmin_argmax along different dimensions
    let (min_result, max_result, argmin_result, argmax_result) =
        permuted.min_max_argmin_argmax(0)?;
    assert_eq!(min_result.dims(), &[2]);
    assert_eq!(max_result.dims(), &[2]);
    assert_eq!(argmin_result.dims(), &[2]);
    assert_eq!(argmax_result.dims(), &[2]);

    let min_vals = min_result.to_vec::<f32>()?;
    let max_vals = max_result.to_vec::<f32>()?;
    let argmin_vals = argmin_result.to_vec::<u64>()?;
    let argmax_vals = argmax_result.to_vec::<u64>()?;

    println!("Min values: {min_vals:?}");
    println!("Max values: {max_vals:?}");
    println!("Argmin indices: {argmin_vals:?}");
    println!("Argmax indices: {argmax_vals:?}");

    // // Test broadcasting subtraction
    // let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3])?;
    // let scalar = Tensor::from_vec(vec![1.0f64], [1])?;

    // println!("A: {:?}", a.to_vec::<f64>()?);
    // println!(
    //     "A shape: {:?}, strides: {:?}, is_contiguous: {}",
    //     a.shape(),
    //     a.strides(),
    //     a.is_contiguous()
    // );
    // println!("Scalar: {:?}", scalar.to_vec::<f64>()?);

    // // Test broadcast subtraction
    // let broadcasted_scalar = scalar.broadcast_to([3])?;
    // println!(
    //     "Broadcasted scalar shape: {:?}, strides: {:?}",
    //     broadcasted_scalar.shape(),
    //     broadcasted_scalar.strides()
    // );

    // // Test direct iteration of broadcasted scalar
    // println!("Broadcasted scalar elements:");
    // for (i, elem) in broadcasted_scalar.iter().enumerate() {
    //     let ptr = unsafe { elem.as_ptr(broadcasted_scalar.as_ptr()) };
    //     let val = unsafe { *(ptr as *const f64) };
    //     println!("  [{i}]: {val}");
    // }

    // // Test direct iteration of a
    // println!("A elements:");
    // for (i, elem) in a.iter().enumerate() {
    //     let ptr = unsafe { elem.as_ptr(a.as_ptr()) };
    //     let val = unsafe { *(ptr as *const f64) };
    //     println!("  [{i}]: {val}");
    // }

    // let result = &a - &broadcasted_scalar;
    // println!("A - scalar: {:?}", result.to_vec::<f64>()?);

    // // Manual calculation for verification
    // println!("Expected: [0.0, 1.0, 2.0]");
    // println!(
    //     "Actual calculation: [{} - {} = {}, {} - {} = {}, {} - {} = {}]",
    //     1.0,
    //     1.0,
    //     0.0,
    //     2.0,
    //     1.0,
    //     2.0 - 1.0,
    //     3.0,
    //     1.0,
    //     3.0 - 1.0
    // );

    // // Test softmax calculation
    // println!("\n=== Softmax Calculation ===");
    // let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3])?;
    // println!("Input: {:?}", input.to_vec::<f64>()?);

    // // Step 1: Find max value
    // let max_val = input.max_keepdim(0)?;
    // println!("Max value: {:?}", max_val.to_vec::<f64>()?);

    // // Step 2: Subtract max from input (broadcasting)
    // let centered = &input - &max_val;
    // println!("Centered: {:?}", centered.to_vec::<f64>()?);

    // // Step 3: Apply exponential
    // let exp_vals = centered.exp()?;
    // println!("Exp values: {:?}", exp_vals.to_vec::<f64>()?);

    // // Step 4: Sum the exponentials
    // let sum_exp = exp_vals.sum_keepdim(0)?;
    // if sum_exp.dims().is_empty() {
    //     println!("Sum exp (scalar): {:?}", sum_exp.to_scalar::<f64>()?);
    // } else {
    //     println!("Sum exp: {:?}", sum_exp.to_vec::<f64>()?);
    // }

    // // Step 5: Divide by sum to get probabilities (broadcasting)
    // let softmax_result = &exp_vals / &sum_exp;
    // // Check if result is scalar or vector
    // if softmax_result.dims().is_empty() {
    //     println!(
    //         "Softmax result (scalar): {:?}",
    //         softmax_result.to_scalar::<f64>()?
    //     );
    // } else {
    //     println!("Softmax result: {:?}", softmax_result.to_vec::<f64>()?);
    // }

    // // Verify sum equals 1
    // let sum_check = softmax_result.sum([0])?;
    // if sum_check.dims().is_empty() {
    //     println!(
    //         "Sum verification (scalar): {:?}",
    //         sum_check.to_scalar::<f64>()?
    //     );
    // } else {
    //     println!("Sum verification: {:?}", sum_check.to_vec::<f64>()?);
    // }

    Ok(())
}
