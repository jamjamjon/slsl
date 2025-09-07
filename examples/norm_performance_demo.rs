use slsl::Tensor;
use std::time::Instant;

fn main() {
    println!("=== Norm Performance Demo ===\n");

    // Create a large tensor for performance testing
    let size = 1000;
    let data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.01).collect();
    let tensor = Tensor::from_vec(data, [size, size]).unwrap();

    println!("Testing tensor of size: {size}x{size}");
    println!("Total elements: {}", size * size);

    // Test L1 norm performance
    println!("\n--- L1 Norm Performance ---");
    let start = Instant::now();
    let _norm1 = tensor.norm1(0).unwrap();
    let l1_duration = start.elapsed();
    println!("L1 norm (dim 0): {l1_duration:?}");

    let start = Instant::now();
    let _norm1 = tensor.norm1(1).unwrap();
    let l1_duration2 = start.elapsed();
    println!("L1 norm (dim 1): {l1_duration2:?}");

    // Test L2 norm performance
    println!("\n--- L2 Norm Performance ---");
    let start = Instant::now();
    let _norm2 = tensor.norm2(0).unwrap();
    let l2_duration = start.elapsed();
    println!("L2 norm (dim 0): {l2_duration:?}");

    let start = Instant::now();
    let _norm2 = tensor.norm2(1).unwrap();
    let l2_duration2 = start.elapsed();
    println!("L2 norm (dim 1): {l2_duration2:?}");

    // Test Lp norm performance
    println!("\n--- Lp Norm Performance ---");
    let start = Instant::now();
    let _normp = tensor.normp(0, 3.0).unwrap();
    let lp_duration = start.elapsed();
    println!("L3 norm (dim 0): {lp_duration:?}");

    let start = Instant::now();
    let _normp = tensor.normp(1, 3.0).unwrap();
    let lp_duration2 = start.elapsed();
    println!("L3 norm (dim 1): {lp_duration2:?}");

    // Test general norm performance
    println!("\n--- General Norm Performance ---");
    let start = Instant::now();
    let _norm = tensor.norm(0, 1.0).unwrap();
    let general_duration = start.elapsed();
    println!("General norm L1 (dim 0): {general_duration:?}");

    let start = Instant::now();
    let _norm = tensor.norm(0, 2.0).unwrap();
    let general_duration2 = start.elapsed();
    println!("General norm L2 (dim 0): {general_duration2:?}");

    // Test different data types
    println!("\n--- Different Data Types Performance ---");

    // F64
    let data_f64: Vec<f64> = (0..size * size).map(|i| (i as f64) * 0.01).collect();
    let tensor_f64 = Tensor::from_vec(data_f64, [size, size]).unwrap();

    let start = Instant::now();
    let _norm_f64 = tensor_f64.norm2(0).unwrap();
    let f64_duration = start.elapsed();
    println!("F64 L2 norm (dim 0): {f64_duration:?}");

    // F16
    let data_f16: Vec<half::f16> = (0..size * size)
        .map(|i| half::f16::from_f32((i as f32) * 0.01))
        .collect();
    let tensor_f16 = Tensor::from_vec(data_f16, [size, size]).unwrap();

    let start = Instant::now();
    let _norm_f16 = tensor_f16.norm2(0).unwrap();
    let f16_duration = start.elapsed();
    println!("F16 L2 norm (dim 0): {f16_duration:?}");

    println!("\n=== Performance Demo Complete ===");
    println!("Note: The new implementation uses efficient backend acceleration");
    println!("for L1 and L2 norms, while Lp norms use optimized manual calculation.");
}
