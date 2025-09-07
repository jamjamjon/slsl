use slsl::{global_backend, OpsTrait};

fn main() {
    let backend = global_backend();

    println!("=== Mean Operations Demo ===\n");

    // Test floating point types
    let f32_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32];
    let f64_data = vec![1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64];

    println!("Floating point mean operations:");
    println!(
        "  f32: {:?} -> mean = {}",
        f32_data,
        backend.mean_f32(&f32_data)
    );
    println!(
        "  f64: {:?} -> mean = {}",
        f64_data,
        backend.mean_f64(&f64_data)
    );

    // Test half precision types
    let f16_data = vec![
        half::f16::from_f32(1.0),
        half::f16::from_f32(2.0),
        half::f16::from_f32(3.0),
        half::f16::from_f32(4.0),
        half::f16::from_f32(5.0),
    ];
    let bf16_data = vec![
        half::bf16::from_f32(1.0),
        half::bf16::from_f32(2.0),
        half::bf16::from_f32(3.0),
        half::bf16::from_f32(4.0),
        half::bf16::from_f32(5.0),
    ];

    println!("\nHalf precision mean operations:");
    println!("  f16:  mean = {}", backend.mean_f16(&f16_data));
    println!("  bf16: mean = {}", backend.mean_bf16(&bf16_data));

    // Test edge cases
    println!("\nEdge cases:");
    let empty_f32: Vec<f32> = vec![];
    let single_f32 = vec![42.0f32];
    let negative_f32 = vec![-1.0f32, -2.0f32, -3.0f32];

    println!(
        "  Empty vector: {:?} -> mean = {}",
        empty_f32,
        backend.mean_f32(&empty_f32)
    );
    println!(
        "  Single element: {:?} -> mean = {}",
        single_f32,
        backend.mean_f32(&single_f32)
    );
    println!(
        "  Negative values: {:?} -> mean = {}",
        negative_f32,
        backend.mean_f32(&negative_f32)
    );

    // Test odd and even lengths
    println!("\nLength variations:");
    let odd_f32 = vec![1.0f32, 2.0f32, 3.0f32];
    let even_f32 = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];

    println!(
        "  Odd length: {:?} -> mean = {}",
        odd_f32,
        backend.mean_f32(&odd_f32)
    );
    println!(
        "  Even length: {:?} -> mean = {}",
        even_f32,
        backend.mean_f32(&even_f32)
    );

    // Test precision
    println!("\nPrecision testing:");
    let small_f32 = vec![0.1f32, 0.2f32, 0.3f32];
    let small_f64 = vec![0.1f64, 0.2f64, 0.3f64];

    println!(
        "  Small f32: {:?} -> mean = {}",
        small_f32,
        backend.mean_f32(&small_f32)
    );
    println!(
        "  Small f64: {:?} -> mean = {}",
        small_f64,
        backend.mean_f64(&small_f64)
    );

    // Test large numbers
    println!("\nLarge numbers:");
    let large_f32 = vec![1e6f32, 2e6f32, 3e6f32];
    let large_f64 = vec![1e10f64, 2e10f64, 3e10f64];

    println!("  Large f32: mean = {}", backend.mean_f32(&large_f32));
    println!("  Large f64: mean = {}", backend.mean_f64(&large_f64));

    // Consistency check with sum
    println!("\nConsistency check:");
    let test_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32];
    let sum = backend.sum_f32(&test_data);
    let mean = backend.mean_f32(&test_data);
    let expected_mean = sum / (test_data.len() as f32);

    println!("  Data: {test_data:?}");
    println!("  Sum: {sum}");
    println!("  Mean: {mean}");
    println!("  Expected mean (sum/n): {expected_mean}");
    println!("  Difference: {}", (mean - expected_mean).abs());

    // Performance comparison
    println!("\nPerformance comparison:");
    let large_f32: Vec<f32> = (0..1000000).map(|i| (i as f32) * 0.001).collect();

    let start = std::time::Instant::now();
    let mean = backend.mean_f32(&large_f32);
    let elapsed = start.elapsed();

    println!("  Mean of 1M f32 elements: {mean} (took {elapsed:?})");

    // Test with different backend types
    println!("\nBackend information:");
    println!("  Backend type: {:?}", std::any::type_name_of_val(backend));

    // Show the relationship between sum and mean
    println!("\nMathematical relationship:");
    println!("  For data: {test_data:?}");
    println!("  Sum: {}", backend.sum_f32(&test_data));
    println!("  Mean: {}", backend.mean_f32(&test_data));
    println!(
        "  Verification: sum / n = {} / {} = {}",
        backend.sum_f32(&test_data),
        test_data.len(),
        backend.sum_f32(&test_data) / (test_data.len() as f32)
    );
}
