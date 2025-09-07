use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2, Array3, Array4, Array6};
use slsl::Tensor;

fn generate_test_data_f32(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) * 0.1 + 1.0).collect()
}

// ==================== 1D Benchmarks ====================

fn bench_iter_dim_1d_f32(c: &mut Criterion) {
    let size = 1000;
    let data = generate_test_data_f32(size);

    // SLSL
    let slsl_tensor = Tensor::from_vec(data.clone(), [size]).unwrap();
    let slsl_tensor_view = slsl_tensor.view();

    // ndarray
    let ndarray_tensor = Array1::from_vec(data);

    // Test 1: Pure iterator overhead with minimal computation
    let mut group = c.benchmark_group("iter_dim_1d_f32_axis0_pure_overhead");

    group.bench_function("slsl", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (idx, _scalar_view) in slsl_tensor.iter_dim(0).enumerate() {
                // Pure math computation - avoid any view access
                sum += (idx as f32).sin() * 2.0;
            }
            black_box(sum)
        })
    });

    group.bench_function("slsl_view", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (idx, _scalar_view) in slsl_tensor_view.iter_dim(0).enumerate() {
                sum += (idx as f32).sin() * 2.0;
            }
            black_box(sum)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (idx, _scalar_view) in ndarray_tensor.axis_iter(ndarray::Axis(0)).enumerate() {
                sum += (idx as f32).sin() * 2.0;
            }
            black_box(sum)
        })
    });
    group.finish();

    // Test 2: Count baseline
    let mut group = c.benchmark_group("iter_dim_1d_f32_axis0_count");

    group.bench_function("slsl", |b| {
        b.iter(|| black_box(slsl_tensor.iter_dim(0).count()))
    });

    group.bench_function("slsl_view", |b| {
        b.iter(|| black_box(slsl_tensor_view.iter_dim(0).count()))
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| black_box(ndarray_tensor.axis_iter(ndarray::Axis(0)).count()))
    });
    group.finish();
}

// ==================== 2D Benchmarks ====================

fn bench_iter_dim_2d_f32(c: &mut Criterion) {
    let rows = 100;
    let cols = 100;
    let data = generate_test_data_f32(rows * cols);

    // SLSL
    let slsl_tensor = Tensor::from_vec(data.clone(), [rows, cols]).unwrap();
    let slsl_tensor_view = slsl_tensor.view();

    // ndarray
    let ndarray_tensor = Array2::from_shape_vec((rows, cols), data).unwrap();

    // Test 1: Pure iterator overhead - axis 0
    let mut group = c.benchmark_group("iter_dim_2d_f32_axis0_pure_overhead");

    group.bench_function("slsl", |b| {
        b.iter(|| {
            let mut result = 0.0f32;
            for (idx, _row) in slsl_tensor.iter_dim(0).enumerate() {
                // Mathematical computation independent of tensor data
                let x = idx as f32;
                result += x.exp() - x.ln().abs() + (x * 3.5).cos();
            }
            black_box(result)
        })
    });

    group.bench_function("slsl_view", |b| {
        b.iter(|| {
            let mut result = 0.0f32;
            for (idx, _row) in slsl_tensor_view.iter_dim(0).enumerate() {
                let x = idx as f32;
                result += x.exp() - x.ln().abs() + (x * 3.5).cos();
            }
            black_box(result)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let mut result = 0.0f32;
            for (idx, _row) in ndarray_tensor.axis_iter(ndarray::Axis(0)).enumerate() {
                let x = idx as f32;
                result += x.exp() - x.ln().abs() + (x * 3.5).cos();
            }
            black_box(result)
        })
    });
    group.finish();

    // Test 2: Pure iterator overhead - axis 1
    let mut group = c.benchmark_group("iter_dim_2d_f32_axis1_pure_overhead");

    group.bench_function("slsl", |b| {
        b.iter(|| {
            let mut result = 0.0f32;
            for (idx, _col) in slsl_tensor.iter_dim(1).enumerate() {
                let x = idx as f32;
                result += x.sqrt() * (x + 1.0).log10();
            }
            black_box(result)
        })
    });

    group.bench_function("slsl_view", |b| {
        b.iter(|| {
            let mut result = 0.0f32;
            for (idx, _col) in slsl_tensor_view.iter_dim(1).enumerate() {
                let x = idx as f32;
                result += x.sqrt() * (x + 1.0).log10();
            }
            black_box(result)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let mut result = 0.0f32;
            for (idx, _col) in ndarray_tensor.axis_iter(ndarray::Axis(1)).enumerate() {
                let x = idx as f32;
                result += x.sqrt() * (x + 1.0).log10();
            }
            black_box(result)
        })
    });
    group.finish();

    // Test 3: Minimal work (counter increment)
    let mut group = c.benchmark_group("iter_dim_2d_f32_axis0_minimal_work");

    group.bench_function("slsl", |b| {
        b.iter(|| {
            let mut counter = 0usize;
            for _row in slsl_tensor.iter_dim(0) {
                counter += 1;
            }
            black_box(counter)
        })
    });

    group.bench_function("slsl_view", |b| {
        b.iter(|| {
            let mut counter = 0usize;
            for _row in slsl_tensor_view.iter_dim(0) {
                counter += 1;
            }
            black_box(counter)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let mut counter = 0usize;
            for _row in ndarray_tensor.axis_iter(ndarray::Axis(0)) {
                counter += 1;
            }
            black_box(counter)
        })
    });
    group.finish();
}

// ==================== 3D Benchmarks ====================

fn bench_iter_dim_3d_f32(c: &mut Criterion) {
    let dim0 = 20;
    let dim1 = 20;
    let dim2 = 20;
    let data = generate_test_data_f32(dim0 * dim1 * dim2);

    // SLSL
    let slsl_tensor = Tensor::from_vec(data.clone(), [dim0, dim1, dim2]).unwrap();
    let slsl_tensor_view = slsl_tensor.view();

    // ndarray
    let ndarray_tensor = Array3::from_shape_vec((dim0, dim1, dim2), data).unwrap();

    // Test: Complex mathematical computation per iteration
    let mut group = c.benchmark_group("iter_dim_3d_f32_axis0_complex_math");

    group.bench_function("slsl", |b| {
        b.iter(|| {
            let mut result = 0.0f32;
            for (idx, _slice_2d) in slsl_tensor.iter_dim(0).enumerate() {
                let x = idx as f32 + 1.0;
                // Complex computation to simulate real workload
                result += (x.powf(1.5) * (x * 0.5).sin() + (x * 0.3).cos().abs()) / (x + 0.1);
            }
            black_box(result)
        })
    });

    group.bench_function("slsl_view", |b| {
        b.iter(|| {
            let mut result = 0.0f32;
            for (idx, _slice_2d) in slsl_tensor_view.iter_dim(0).enumerate() {
                let x = idx as f32 + 1.0;
                result += (x.powf(1.5) * (x * 0.5).sin() + (x * 0.3).cos().abs()) / (x + 0.1);
            }
            black_box(result)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let mut result = 0.0f32;
            for (idx, _slice_2d) in ndarray_tensor.axis_iter(ndarray::Axis(0)).enumerate() {
                let x = idx as f32 + 1.0;
                result += (x.powf(1.5) * (x * 0.5).sin() + (x * 0.3).cos().abs()) / (x + 0.1);
            }
            black_box(result)
        })
    });
    group.finish();
}

// ==================== 4D Benchmarks ====================

fn bench_iter_dim_4d_f32(c: &mut Criterion) {
    let batch_size = 8;
    let channels = 16;
    let height = 32;
    let width = 32;
    let data = generate_test_data_f32(batch_size * channels * height * width);

    // SLSL
    let slsl_tensor =
        Tensor::from_vec(data.clone(), [batch_size, channels, height, width]).unwrap();
    let slsl_tensor_view = slsl_tensor.view();

    // ndarray
    let ndarray_tensor =
        Array4::from_shape_vec((batch_size, channels, height, width), data).unwrap();

    // Test: Simulate batch processing computation
    let mut group = c.benchmark_group("iter_dim_4d_f32_axis0_batch_computation");

    group.bench_function("slsl", |b| {
        b.iter(|| {
            let mut batch_scores = Vec::new();
            for (batch_idx, _batch_item) in slsl_tensor.iter_dim(0).enumerate() {
                // Simulate feature extraction computation per batch
                let base = batch_idx as f32;
                let score = (base * 2.5).tanh() * (base + 1.0).sqrt() - (base * 0.8).exp() * 0.01;
                batch_scores.push(score);
            }
            black_box(batch_scores)
        })
    });

    group.bench_function("slsl_view", |b| {
        b.iter(|| {
            let mut batch_scores = Vec::new();
            for (batch_idx, _batch_item) in slsl_tensor_view.iter_dim(0).enumerate() {
                let base = batch_idx as f32;
                let score = (base * 2.5).tanh() * (base + 1.0).sqrt() - (base * 0.8).exp() * 0.01;
                batch_scores.push(score);
            }
            black_box(batch_scores)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let mut batch_scores = Vec::new();
            for (batch_idx, _batch_item) in ndarray_tensor.axis_iter(ndarray::Axis(0)).enumerate() {
                let base = batch_idx as f32;
                let score = (base * 2.5).tanh() * (base + 1.0).sqrt() - (base * 0.8).exp() * 0.01;
                batch_scores.push(score);
            }
            black_box(batch_scores)
        })
    });
    group.finish();
}

// ==================== High-dimensional stress test ====================

fn bench_iter_dim_6d_f32(c: &mut Criterion) {
    let dims = [2, 3, 4, 2, 3, 4];
    let total_size = dims.iter().product();
    let data = generate_test_data_f32(total_size);

    // SLSL
    let slsl_tensor = Tensor::from_vec(data.clone(), dims).unwrap();
    let slsl_tensor_view = slsl_tensor.view();

    // ndarray
    let ndarray_tensor =
        Array6::from_shape_vec((dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]), data)
            .unwrap();

    // Test: High-dimensional mathematical computation
    let mut group = c.benchmark_group("iter_dim_6d_f32_axis0_high_dim_math");

    group.bench_function("slsl", |b| {
        b.iter(|| {
            let mut result = 0.0f32;
            for (idx, _slice_5d) in slsl_tensor.iter_dim(0).enumerate() {
                let x = idx as f32;
                // Multi-step computation
                let step1 = (x + 1.0).ln();
                let step2 = step1.sin() + step1.cos();
                let step3 = step2.abs().sqrt();
                let step4 = step3 * (x * 0.1).exp();
                result += step4 / (step3 + 1.0);
            }
            black_box(result)
        })
    });

    group.bench_function("slsl_view", |b| {
        b.iter(|| {
            let mut result = 0.0f32;
            for (idx, _slice_5d) in slsl_tensor_view.iter_dim(0).enumerate() {
                let x = idx as f32;
                let step1 = (x + 1.0).ln();
                let step2 = step1.sin() + step1.cos();
                let step3 = step2.abs().sqrt();
                let step4 = step3 * (x * 0.1).exp();
                result += step4 / (step3 + 1.0);
            }
            black_box(result)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let mut result = 0.0f32;
            for (idx, _slice_5d) in ndarray_tensor.axis_iter(ndarray::Axis(0)).enumerate() {
                let x = idx as f32;
                let step1 = (x + 1.0).ln();
                let step2 = step1.sin() + step1.cos();
                let step3 = step2.abs().sqrt();
                let step4 = step3 * (x * 0.1).exp();
                result += step4 / (step3 + 1.0);
            }
            black_box(result)
        })
    });
    group.finish();
}

// ==================== Large scale stress test ====================

fn bench_iter_dim_large_scale(c: &mut Criterion) {
    let large_dim = 10000;
    let small_dim = 10;
    let data = generate_test_data_f32(large_dim * small_dim);

    // SLSL
    let slsl_tensor = Tensor::from_vec(data.clone(), [large_dim, small_dim]).unwrap();
    let slsl_tensor_view = slsl_tensor.view();

    // ndarray
    let ndarray_tensor = Array2::from_shape_vec((large_dim, small_dim), data).unwrap();

    // Test: Large-scale iteration with simple computation
    let mut group = c.benchmark_group("iter_dim_large_scale_simple_computation");

    group.bench_function("slsl", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (idx, _row) in slsl_tensor.iter_dim(0).enumerate() {
                // Simple but meaningful computation
                let x = (idx % 1000) as f32;
                sum += x * 0.001 + (x * 0.01).sin();
            }
            black_box(sum)
        })
    });

    group.bench_function("slsl_view", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (idx, _row) in slsl_tensor_view.iter_dim(0).enumerate() {
                let x = (idx % 1000) as f32;
                sum += x * 0.001 + (x * 0.01).sin();
            }
            black_box(sum)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (idx, _row) in ndarray_tensor.axis_iter(ndarray::Axis(0)).enumerate() {
                let x = (idx % 1000) as f32;
                sum += x * 0.001 + (x * 0.01).sin();
            }
            black_box(sum)
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_iter_dim_1d_f32,
    bench_iter_dim_2d_f32,
    bench_iter_dim_3d_f32,
    bench_iter_dim_4d_f32,
    bench_iter_dim_6d_f32,
    bench_iter_dim_large_scale
);

criterion_main!(benches);
