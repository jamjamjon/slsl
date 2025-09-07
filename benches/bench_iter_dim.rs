use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{ArcArray, Array1, Array2, Array3, Array4, Array6};
use rayon::prelude::*;
use slsl::Tensor;
use std::hint::black_box;

fn generate_test_data_f32(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) * 0.1 + 1.0).collect()
}

// ==================== 1D Benchmarks ====================

fn bench_iter_dim_1d_f32(c: &mut Criterion) {
    // Small, Medium, Large sizes
    let sizes = [100, 500, 800, 1000, 4000, 6000, 8000, 10000];

    for size in sizes {
        let data = generate_test_data_f32(size);

        // SLSL
        let slsl_tensor = Tensor::from_vec(data.clone(), [size]).unwrap();
        let slsl_tensor_view = slsl_tensor.view();

        // ndarray
        let ndarray_tensor = Array1::from_vec(data.clone());
        let ndarray_arc_tensor = ArcArray::from_shape_vec((size,), data).unwrap();

        // Test: Pure iterator overhead with minimal computation
        let mut group = c.benchmark_group(format!("iter_dim_1d_f32_size_{}_pure_overhead", size));

        group.bench_function("slsl_seq", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for (idx, _scalar_view) in slsl_tensor.iter_dim(0).enumerate() {
                    // Pure math computation - avoid any view access
                    sum += (idx as f32).sin() * 2.0;
                }
                black_box(sum)
            })
        });

        group.bench_function("slsl_view_seq", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for (idx, _scalar_view) in slsl_tensor_view.iter_dim(0).enumerate() {
                    sum += (idx as f32).sin() * 2.0;
                }
                black_box(sum)
            })
        });

        #[cfg(feature = "rayon")]
        group.bench_function("slsl_par", |b| {
            b.iter(|| {
                let sum: f32 = black_box(&slsl_tensor)
                    .iter_dim(0)
                    .par_iter()
                    .enumerate()
                    .map(|(idx, _scalar_view)| (idx as f32).sin() * 2.0)
                    .sum();
                black_box(sum)
            })
        });

        #[cfg(feature = "rayon")]
        group.bench_function("slsl_view_par", |b| {
            b.iter(|| {
                let sum: f32 = black_box(&slsl_tensor_view)
                    .iter_dim(0)
                    .par_iter()
                    .enumerate()
                    .map(|(idx, _scalar_view)| (idx as f32).sin() * 2.0)
                    .sum();
                black_box(sum)
            })
        });

        group.bench_function("ndarray_array_seq", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for (idx, _scalar_view) in ndarray_tensor.axis_iter(ndarray::Axis(0)).enumerate() {
                    sum += (idx as f32).sin() * 2.0;
                }
                black_box(sum)
            })
        });

        group.bench_function("ndarray_arc_array_seq", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for (idx, _scalar_view) in
                    ndarray_arc_tensor.axis_iter(ndarray::Axis(0)).enumerate()
                {
                    sum += (idx as f32).sin() * 2.0;
                }
                black_box(sum)
            })
        });

        group.bench_function("ndarray_array_par", |b| {
            b.iter(|| {
                let sum: f32 = black_box(&ndarray_tensor)
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, _scalar_view)| (idx as f32).sin() * 2.0)
                    .sum();
                black_box(sum)
            })
        });

        group.bench_function("ndarray_arc_array_par", |b| {
            b.iter(|| {
                let sum: f32 = black_box(&ndarray_arc_tensor)
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, _scalar_view)| (idx as f32).sin() * 2.0)
                    .sum();
                black_box(sum)
            })
        });

        group.finish();
    }
}

// ==================== 2D Benchmarks ====================

fn bench_iter_dim_2d_f32(c: &mut Criterion) {
    // Small, Medium, Large sizes
    let sizes = [
        (50, 50),
        (200, 200),
        (400, 400),
        (600, 600),
        (800, 800),
        (1000, 1000),
    ];

    for (rows, cols) in sizes {
        let data = generate_test_data_f32(rows * cols);

        // SLSL
        let slsl_tensor = Tensor::from_vec(data.clone(), [rows, cols]).unwrap();
        let slsl_tensor_view = slsl_tensor.view();

        // ndarray
        let ndarray_tensor = Array2::from_shape_vec((rows, cols), data.clone()).unwrap();
        let ndarray_arc_tensor = ArcArray::from_shape_vec((rows, cols), data).unwrap();

        // Test: Pure iterator overhead - axis 0
        let mut group = c.benchmark_group(format!(
            "iter_dim_2d_f32_size_{}x{}_axis0_overhead",
            rows, cols
        ));

        group.bench_function("slsl_seq", |b| {
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

        group.bench_function("slsl_view_seq", |b| {
            b.iter(|| {
                let mut result = 0.0f32;
                for (idx, _row) in slsl_tensor_view.iter_dim(0).enumerate() {
                    let x = idx as f32;
                    result += x.exp() - x.ln().abs() + (x * 3.5).cos();
                }
                black_box(result)
            })
        });

        #[cfg(feature = "rayon")]
        group.bench_function("slsl_par", |b| {
            b.iter(|| {
                let result: f32 = black_box(&slsl_tensor)
                    .iter_dim(0)
                    .par_iter()
                    .enumerate()
                    .map(|(idx, _row)| {
                        let x = idx as f32;
                        x.exp() - x.ln().abs() + (x * 3.5).cos()
                    })
                    .sum();
                black_box(result)
            })
        });

        #[cfg(feature = "rayon")]
        group.bench_function("slsl_view_par", |b| {
            b.iter(|| {
                let result: f32 = black_box(&slsl_tensor_view)
                    .iter_dim(0)
                    .par_iter()
                    .enumerate()
                    .map(|(idx, _row)| {
                        let x = idx as f32;
                        x.exp() - x.ln().abs() + (x * 3.5).cos()
                    })
                    .sum();
                black_box(result)
            })
        });

        group.bench_function("ndarray_array_seq", |b| {
            b.iter(|| {
                let mut result = 0.0f32;
                for (idx, _row) in ndarray_tensor.axis_iter(ndarray::Axis(0)).enumerate() {
                    let x = idx as f32;
                    result += x.exp() - x.ln().abs() + (x * 3.5).cos();
                }
                black_box(result)
            })
        });

        group.bench_function("ndarray_arc_array_seq", |b| {
            b.iter(|| {
                let mut result = 0.0f32;
                for (idx, _row) in ndarray_arc_tensor.axis_iter(ndarray::Axis(0)).enumerate() {
                    let x = idx as f32;
                    result += x.exp() - x.ln().abs() + (x * 3.5).cos();
                }
                black_box(result)
            })
        });

        group.bench_function("ndarray_array_par", |b| {
            b.iter(|| {
                let result: f32 = black_box(&ndarray_tensor)
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, _row)| {
                        let x = idx as f32;
                        x.exp() - x.ln().abs() + (x * 3.5).cos()
                    })
                    .sum();
                black_box(result)
            })
        });

        group.bench_function("ndarray_arc_array_par", |b| {
            b.iter(|| {
                let result: f32 = black_box(&ndarray_arc_tensor)
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, _row)| {
                        let x = idx as f32;
                        x.exp() - x.ln().abs() + (x * 3.5).cos()
                    })
                    .sum();
                black_box(result)
            })
        });

        group.finish();
    }
}

// ==================== 3D Benchmarks ====================

fn bench_iter_dim_3d_f32(c: &mut Criterion) {
    // Small, Medium, Large sizes
    let sizes = [
        (20, 20, 20),
        (60, 60, 60),
        (100, 100, 100),
        (200, 200, 200),
        (400, 400, 400),
        (600, 600, 600),
        (800, 800, 800),
    ];

    for (dim0, dim1, dim2) in sizes {
        let data = generate_test_data_f32(dim0 * dim1 * dim2);

        // SLSL
        let slsl_tensor = Tensor::from_vec(data.clone(), [dim0, dim1, dim2]).unwrap();
        let slsl_tensor_view = slsl_tensor.view();

        // ndarray
        let ndarray_tensor = Array3::from_shape_vec((dim0, dim1, dim2), data.clone()).unwrap();
        let ndarray_arc_tensor = ArcArray::from_shape_vec((dim0, dim1, dim2), data).unwrap();

        // Test: Complex mathematical computation per iteration
        let mut group = c.benchmark_group(format!(
            "iter_dim_3d_f32_size_{}x{}x{}_axis0_complex_math",
            dim0, dim1, dim2
        ));

        group.bench_function("slsl_seq", |b| {
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

        group.bench_function("slsl_view_seq", |b| {
            b.iter(|| {
                let mut result = 0.0f32;
                for (idx, _slice_2d) in slsl_tensor_view.iter_dim(0).enumerate() {
                    let x = idx as f32 + 1.0;
                    result += (x.powf(1.5) * (x * 0.5).sin() + (x * 0.3).cos().abs()) / (x + 0.1);
                }
                black_box(result)
            })
        });

        #[cfg(feature = "rayon")]
        group.bench_function("slsl_par", |b| {
            b.iter(|| {
                let result: f32 = black_box(&slsl_tensor)
                    .iter_dim(0)
                    .par_iter()
                    .enumerate()
                    .map(|(idx, _slice_2d)| {
                        let x = idx as f32 + 1.0;
                        (x.powf(1.5) * (x * 0.5).sin() + (x * 0.3).cos().abs()) / (x + 0.1)
                    })
                    .sum();
                black_box(result)
            })
        });

        #[cfg(feature = "rayon")]
        group.bench_function("slsl_view_par", |b| {
            b.iter(|| {
                let result: f32 = black_box(&slsl_tensor_view)
                    .iter_dim(0)
                    .par_iter()
                    .enumerate()
                    .map(|(idx, _slice_2d)| {
                        let x = idx as f32 + 1.0;
                        (x.powf(1.5) * (x * 0.5).sin() + (x * 0.3).cos().abs()) / (x + 0.1)
                    })
                    .sum();
                black_box(result)
            })
        });

        group.bench_function("ndarray_array_seq", |b| {
            b.iter(|| {
                let mut result = 0.0f32;
                for (idx, _slice_2d) in ndarray_tensor.axis_iter(ndarray::Axis(0)).enumerate() {
                    let x = idx as f32 + 1.0;
                    result += (x.powf(1.5) * (x * 0.5).sin() + (x * 0.3).cos().abs()) / (x + 0.1);
                }
                black_box(result)
            })
        });

        group.bench_function("ndarray_arc_array_seq", |b| {
            b.iter(|| {
                let mut result = 0.0f32;
                for (idx, _slice_2d) in ndarray_arc_tensor.axis_iter(ndarray::Axis(0)).enumerate() {
                    let x = idx as f32 + 1.0;
                    result += (x.powf(1.5) * (x * 0.5).sin() + (x * 0.3).cos().abs()) / (x + 0.1);
                }
                black_box(result)
            })
        });

        group.bench_function("ndarray_array_par", |b| {
            b.iter(|| {
                let result: f32 = black_box(&ndarray_tensor)
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, _slice_2d)| {
                        let x = idx as f32 + 1.0;
                        (x.powf(1.5) * (x * 0.5).sin() + (x * 0.3).cos().abs()) / (x + 0.1)
                    })
                    .sum();
                black_box(result)
            })
        });

        group.bench_function("ndarray_arc_array_par", |b| {
            b.iter(|| {
                let result: f32 = black_box(&ndarray_arc_tensor)
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, _slice_2d)| {
                        let x = idx as f32 + 1.0;
                        (x.powf(1.5) * (x * 0.5).sin() + (x * 0.3).cos().abs()) / (x + 0.1)
                    })
                    .sum();
                black_box(result)
            })
        });

        group.finish();
    }
}

// ==================== 4D Benchmarks ====================

fn bench_iter_dim_4d_f32(c: &mut Criterion) {
    // Small, Medium, Large sizes
    let sizes = [
        (4, 8, 16, 16),
        (8, 16, 32, 32),
        (16, 32, 64, 64),
        (32, 64, 128, 128),
        (64, 128, 256, 256),
    ];

    for (batch_size, channels, height, width) in sizes {
        let data = generate_test_data_f32(batch_size * channels * height * width);

        // SLSL
        let slsl_tensor =
            Tensor::from_vec(data.clone(), [batch_size, channels, height, width]).unwrap();
        let slsl_tensor_view = slsl_tensor.view();

        // ndarray
        let ndarray_tensor =
            Array4::from_shape_vec((batch_size, channels, height, width), data.clone()).unwrap();
        let ndarray_arc_tensor =
            ArcArray::from_shape_vec((batch_size, channels, height, width), data).unwrap();

        // Test: Simulate batch processing computation
        let mut group = c.benchmark_group(format!(
            "iter_dim_4d_f32_size_{}x{}x{}x{}_axis0_batch_computation",
            batch_size, channels, height, width
        ));

        group.bench_function("slsl_seq", |b| {
            b.iter(|| {
                let mut batch_scores = Vec::new();
                for (batch_idx, _batch_item) in slsl_tensor.iter_dim(0).enumerate() {
                    // Simulate feature extraction computation per batch
                    let base = batch_idx as f32;
                    let score =
                        (base * 2.5).tanh() * (base + 1.0).sqrt() - (base * 0.8).exp() * 0.01;
                    batch_scores.push(score);
                }
                black_box(batch_scores)
            })
        });

        group.bench_function("slsl_view_seq", |b| {
            b.iter(|| {
                let mut batch_scores = Vec::new();
                for (batch_idx, _batch_item) in slsl_tensor_view.iter_dim(0).enumerate() {
                    let base = batch_idx as f32;
                    let score =
                        (base * 2.5).tanh() * (base + 1.0).sqrt() - (base * 0.8).exp() * 0.01;
                    batch_scores.push(score);
                }
                black_box(batch_scores)
            })
        });

        #[cfg(feature = "rayon")]
        group.bench_function("slsl_par", |b| {
            b.iter(|| {
                let batch_scores: Vec<f32> = black_box(&slsl_tensor)
                    .iter_dim(0)
                    .par_iter()
                    .enumerate()
                    .map(|(batch_idx, _batch_item)| {
                        let base = batch_idx as f32;
                        (base * 2.5).tanh() * (base + 1.0).sqrt() - (base * 0.8).exp() * 0.01
                    })
                    .collect();
                black_box(batch_scores)
            })
        });

        #[cfg(feature = "rayon")]
        group.bench_function("slsl_view_par", |b| {
            b.iter(|| {
                let batch_scores: Vec<f32> = black_box(&slsl_tensor_view)
                    .iter_dim(0)
                    .par_iter()
                    .enumerate()
                    .map(|(batch_idx, _batch_item)| {
                        let base = batch_idx as f32;
                        (base * 2.5).tanh() * (base + 1.0).sqrt() - (base * 0.8).exp() * 0.01
                    })
                    .collect();
                black_box(batch_scores)
            })
        });

        group.bench_function("ndarray_array_seq", |b| {
            b.iter(|| {
                let mut batch_scores = Vec::new();
                for (batch_idx, _batch_item) in
                    ndarray_tensor.axis_iter(ndarray::Axis(0)).enumerate()
                {
                    let base = batch_idx as f32;
                    let score =
                        (base * 2.5).tanh() * (base + 1.0).sqrt() - (base * 0.8).exp() * 0.01;
                    batch_scores.push(score);
                }
                black_box(batch_scores)
            })
        });

        group.bench_function("ndarray_arc_array_seq", |b| {
            b.iter(|| {
                let mut batch_scores = Vec::new();
                for (batch_idx, _batch_item) in
                    ndarray_arc_tensor.axis_iter(ndarray::Axis(0)).enumerate()
                {
                    let base = batch_idx as f32;
                    let score =
                        (base * 2.5).tanh() * (base + 1.0).sqrt() - (base * 0.8).exp() * 0.01;
                    batch_scores.push(score);
                }
                black_box(batch_scores)
            })
        });

        group.bench_function("ndarray_array_par", |b| {
            b.iter(|| {
                let batch_scores: Vec<f32> = black_box(&ndarray_tensor)
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .map(|(batch_idx, _batch_item)| {
                        let base = batch_idx as f32;
                        (base * 2.5).tanh() * (base + 1.0).sqrt() - (base * 0.8).exp() * 0.01
                    })
                    .collect();
                black_box(batch_scores)
            })
        });

        group.bench_function("ndarray_arc_array_par", |b| {
            b.iter(|| {
                let batch_scores: Vec<f32> = black_box(&ndarray_arc_tensor)
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .map(|(batch_idx, _batch_item)| {
                        let base = batch_idx as f32;
                        (base * 2.5).tanh() * (base + 1.0).sqrt() - (base * 0.8).exp() * 0.01
                    })
                    .collect();
                black_box(batch_scores)
            })
        });

        group.finish();
    }
}

// ==================== High-dimensional stress test ====================

fn bench_iter_dim_6d_f32(c: &mut Criterion) {
    // Small, Medium, Large sizes
    let sizes = [
        (20, 20, 20, 20, 20, 20),
        (40, 40, 40, 40, 40, 40),
        (60, 60, 60, 60, 60, 60),
        (80, 80, 80, 80, 80, 80),
        (100, 100, 100, 100, 100, 100),
        (200, 200, 200, 200, 200, 200),
    ];

    for dims in sizes {
        let total_size = dims.0 * dims.1 * dims.2 * dims.3 * dims.4 * dims.5;
        let data = generate_test_data_f32(total_size);

        // SLSL
        let slsl_tensor = Tensor::from_vec(
            data.clone(),
            [dims.0, dims.1, dims.2, dims.3, dims.4, dims.5],
        )
        .unwrap();
        let slsl_tensor_view = slsl_tensor.view();

        // ndarray
        let ndarray_tensor = Array6::from_shape_vec(
            (dims.0, dims.1, dims.2, dims.3, dims.4, dims.5),
            data.clone(),
        )
        .unwrap();
        let ndarray_arc_tensor =
            ArcArray::from_shape_vec((dims.0, dims.1, dims.2, dims.3, dims.4, dims.5), data)
                .unwrap();

        // Test: High-dimensional mathematical computation
        let mut group = c.benchmark_group(format!(
            "iter_dim_6d_f32_size_{}x{}x{}x{}x{}x{}_axis0_high_dim_math",
            dims.0, dims.1, dims.2, dims.3, dims.4, dims.5
        ));

        group.bench_function("slsl_seq", |b| {
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

        group.bench_function("slsl_view_seq", |b| {
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

        #[cfg(feature = "rayon")]
        group.bench_function("slsl_par", |b| {
            b.iter(|| {
                let result: f32 = black_box(&slsl_tensor)
                    .iter_dim(0)
                    .par_iter()
                    .enumerate()
                    .map(|(idx, _slice_5d)| {
                        let x = idx as f32;
                        let step1 = (x + 1.0).ln();
                        let step2 = step1.sin() + step1.cos();
                        let step3 = step2.abs().sqrt();
                        let step4 = step3 * (x * 0.1).exp();
                        step4 / (step3 + 1.0)
                    })
                    .sum();
                black_box(result)
            })
        });

        #[cfg(feature = "rayon")]
        group.bench_function("slsl_view_par", |b| {
            b.iter(|| {
                let result: f32 = black_box(&slsl_tensor_view)
                    .iter_dim(0)
                    .par_iter()
                    .enumerate()
                    .map(|(idx, _slice_5d)| {
                        let x = idx as f32;
                        let step1 = (x + 1.0).ln();
                        let step2 = step1.sin() + step1.cos();
                        let step3 = step2.abs().sqrt();
                        let step4 = step3 * (x * 0.1).exp();
                        step4 / (step3 + 1.0)
                    })
                    .sum();
                black_box(result)
            })
        });

        group.bench_function("ndarray_array_seq", |b| {
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

        group.bench_function("ndarray_arc_array_seq", |b| {
            b.iter(|| {
                let mut result = 0.0f32;
                for (idx, _slice_5d) in ndarray_arc_tensor.axis_iter(ndarray::Axis(0)).enumerate() {
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

        group.bench_function("ndarray_array_par", |b| {
            b.iter(|| {
                let result: f32 = black_box(&ndarray_tensor)
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, _slice_5d)| {
                        let x = idx as f32;
                        let step1 = (x + 1.0).ln();
                        let step2 = step1.sin() + step1.cos();
                        let step3 = step2.abs().sqrt();
                        let step4 = step3 * (x * 0.1).exp();
                        step4 / (step3 + 1.0)
                    })
                    .sum();
                black_box(result)
            })
        });

        group.bench_function("ndarray_arc_array_par", |b| {
            b.iter(|| {
                let result: f32 = black_box(&ndarray_arc_tensor)
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, _slice_5d)| {
                        let x = idx as f32;
                        let step1 = (x + 1.0).ln();
                        let step2 = step1.sin() + step1.cos();
                        let step3 = step2.abs().sqrt();
                        let step4 = step3 * (x * 0.1).exp();
                        step4 / (step3 + 1.0)
                    })
                    .sum();
                black_box(result)
            })
        });

        group.finish();
    }
}

// ==================== Large scale stress test ====================

fn bench_iter_dim_large_scale(c: &mut Criterion) {
    // Small, Medium, Large sizes
    let sizes = [(1000, 5000), (10000, 5000), (100000, 5000)];

    for (large_dim, small_dim) in sizes {
        let data = generate_test_data_f32(large_dim * small_dim);

        // SLSL
        let slsl_tensor = Tensor::from_vec(data.clone(), [large_dim, small_dim]).unwrap();
        let slsl_tensor_view = slsl_tensor.view();

        // ndarray
        let ndarray_tensor = Array2::from_shape_vec((large_dim, small_dim), data.clone()).unwrap();
        let ndarray_arc_tensor = ArcArray::from_shape_vec((large_dim, small_dim), data).unwrap();

        // Test: Large-scale iteration with simple computation
        let mut group = c.benchmark_group(format!(
            "iter_dim_large_scale_size_{}x{}_simple_computation",
            large_dim, small_dim
        ));

        group.bench_function("slsl_seq", |b| {
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

        group.bench_function("slsl_view_seq", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for (idx, _row) in slsl_tensor_view.iter_dim(0).enumerate() {
                    let x = (idx % 1000) as f32;
                    sum += x * 0.001 + (x * 0.01).sin();
                }
                black_box(sum)
            })
        });

        #[cfg(feature = "rayon")]
        group.bench_function("slsl_par", |b| {
            b.iter(|| {
                let sum: f32 = black_box(&slsl_tensor)
                    .iter_dim(0)
                    .par_iter()
                    .enumerate()
                    .map(|(idx, _row)| {
                        let x = (idx % 1000) as f32;
                        x * 0.001 + (x * 0.01).sin()
                    })
                    .sum();
                black_box(sum)
            })
        });

        #[cfg(feature = "rayon")]
        group.bench_function("slsl_view_par", |b| {
            b.iter(|| {
                let sum: f32 = black_box(&slsl_tensor_view)
                    .iter_dim(0)
                    .par_iter()
                    .enumerate()
                    .map(|(idx, _row)| {
                        let x = (idx % 1000) as f32;
                        x * 0.001 + (x * 0.01).sin()
                    })
                    .sum();
                black_box(sum)
            })
        });

        group.bench_function("ndarray_array_seq", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for (idx, _row) in ndarray_tensor.axis_iter(ndarray::Axis(0)).enumerate() {
                    let x = (idx % 1000) as f32;
                    sum += x * 0.001 + (x * 0.01).sin();
                }
                black_box(sum)
            })
        });

        group.bench_function("ndarray_arc_array_seq", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for (idx, _row) in ndarray_arc_tensor.axis_iter(ndarray::Axis(0)).enumerate() {
                    let x = (idx % 1000) as f32;
                    sum += x * 0.001 + (x * 0.01).sin();
                }
                black_box(sum)
            })
        });

        group.bench_function("ndarray_array_par", |b| {
            b.iter(|| {
                let sum: f32 = black_box(&ndarray_tensor)
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, _row)| {
                        let x = (idx % 1000) as f32;
                        x * 0.001 + (x * 0.01).sin()
                    })
                    .sum();
                black_box(sum)
            })
        });

        group.bench_function("ndarray_arc_array_par", |b| {
            b.iter(|| {
                let sum: f32 = black_box(&ndarray_arc_tensor)
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, _row)| {
                        let x = (idx % 1000) as f32;
                        x * 0.001 + (x * 0.01).sin()
                    })
                    .sum();
                black_box(sum)
            })
        });

        group.finish();
    }
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
