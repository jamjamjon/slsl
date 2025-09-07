use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use ndarray::ArcArray;
use slsl::Tensor;

fn bench_par_iter_dim_slsl(c: &mut Criterion) {
    let mut group = c.benchmark_group("slsl_par_iter_dim");

    // Test different tensor sizes - focus on larger sizes where parallelization helps
    for size in [100, 1000, 10000, 50000, 100000] {
        let data: Vec<f32> = (0..size * 10).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![size, 10]).unwrap();

        // Sequential baseline
        group.bench_function(format!("size_{size}_seq"), |b| {
            b.iter(|| {
                let result: Vec<f32> = black_box(&tensor)
                    .iter_dim(0)
                    .map(|view| {
                        // Compute math functions unrelated to slsl
                        let mut acc = 0.0f32;
                        for j in 0..view.shape()[0] {
                            let val = view.at::<f32>([j]);
                            acc += val.sin() * val.cos() + val.sqrt().abs() + (val * 2.0).tanh();
                        }
                        acc
                    })
                    .collect();
                black_box(result);
            });
        });

        // Parallel iteration
        #[cfg(feature = "rayon")]
        group.bench_function(format!("size_{size}_par"), |b| {
            b.iter(|| {
                let result: Vec<f32> = black_box(&tensor)
                    .iter_dim(0)
                    .par_iter()
                    .map(|view| {
                        // Same math functions for fair comparison
                        let mut acc = 0.0f32;
                        for j in 0..view.shape()[0] {
                            let val = view.at::<f32>([j]);
                            acc += val.sin() * val.cos() + val.sqrt().abs() + (val * 2.0).tanh();
                        }
                        acc
                    })
                    .collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_par_iter_dim_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_par_iter_dim");

    // Test different array sizes - same as slsl for fair comparison
    for size in [100, 1000, 10000, 50000, 100000] {
        let data: Vec<f32> = (0..size * 10).map(|i| i as f32).collect();
        let array = ArcArray::from_shape_vec((size, 10), data).unwrap();

        // Sequential baseline
        group.bench_function(format!("size_{size}_seq"), |b| {
            b.iter(|| {
                let result: Vec<f32> = black_box(&array)
                    .axis_iter(ndarray::Axis(0))
                    .map(|row| {
                        // Compute same math functions for fair comparison
                        let mut acc = 0.0f32;
                        for &val in row.iter() {
                            acc += val.sin() * val.cos() + val.sqrt().abs() + (val * 2.0).tanh();
                        }
                        acc
                    })
                    .collect();
                black_box(result);
            });
        });

        // Parallel iteration with rayon
        group.bench_function(format!("size_{size}_par"), |b| {
            b.iter(|| {
                let result: Vec<f32> = black_box(&array)
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .map(|row| {
                        // Same math functions for fair comparison
                        let mut acc = 0.0f32;
                        for &val in row.iter() {
                            acc += val.sin() * val.cos() + val.sqrt().abs() + (val * 2.0).tanh();
                        }
                        acc
                    })
                    .collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_par_iter_complexity_slsl(c: &mut Criterion) {
    let mut group = c.benchmark_group("slsl_par_iter_complexity");

    let size = 10000; // Use larger size to see parallel benefits
    let data: Vec<f32> = (0..size * 10).map(|i| i as f32).collect();
    let tensor = Tensor::from_vec(data, vec![size, 10]).unwrap();

    // Simple computation - basic math operations
    group.bench_function("seq_simple", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&tensor)
                .iter_dim(0)
                .map(|view| {
                    let mut acc = 0.0f32;
                    for j in 0..view.shape()[0] {
                        let val = view.at::<f32>([j]);
                        acc += val * val + val.sqrt();
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    #[cfg(feature = "rayon")]
    group.bench_function("par_simple", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&tensor)
                .iter_dim(0)
                .par_iter()
                .map(|view| {
                    let mut acc = 0.0f32;
                    for j in 0..view.shape()[0] {
                        let val = view.at::<f32>([j]);
                        acc += val * val + val.sqrt();
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    // Medium complexity - trigonometric operations
    group.bench_function("seq_medium", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&tensor)
                .iter_dim(0)
                .map(|view| {
                    let mut acc = 0.0f32;
                    for j in 0..view.shape()[0] {
                        let val = view.at::<f32>([j]);
                        acc += val.sin() * val.cos() + val.tan().abs();
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    #[cfg(feature = "rayon")]
    group.bench_function("par_medium", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&tensor)
                .iter_dim(0)
                .par_iter()
                .map(|view| {
                    let mut acc = 0.0f32;
                    for j in 0..view.shape()[0] {
                        let val = view.at::<f32>([j]);
                        acc += val.sin() * val.cos() + val.tan().abs();
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    // High complexity - exponential and logarithmic operations
    group.bench_function("seq_complex", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&tensor)
                .iter_dim(0)
                .map(|view| {
                    let mut acc = 0.0f32;
                    for j in 0..view.shape()[0] {
                        let val = view.at::<f32>([j]);
                        acc += val.exp() * val.ln().abs() + val.powf(1.5) + (val + 1.0).recip();
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    #[cfg(feature = "rayon")]
    group.bench_function("par_complex", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&tensor)
                .iter_dim(0)
                .par_iter()
                .map(|view| {
                    let mut acc = 0.0f32;
                    for j in 0..view.shape()[0] {
                        let val = view.at::<f32>([j]);
                        acc += val.exp() * val.ln().abs() + val.powf(1.5) + (val + 1.0).recip();
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    group.finish();
}

fn bench_par_iter_complexity_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_par_iter_complexity");

    let size = 10000; // Same size as slsl for fair comparison
    let data: Vec<f32> = (0..size * 10).map(|i| i as f32).collect();
    let array = ArcArray::from_shape_vec((size, 10), data).unwrap();

    // Simple computation - basic math operations
    group.bench_function("seq_simple", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&array)
                .axis_iter(ndarray::Axis(0))
                .map(|row| {
                    let mut acc = 0.0f32;
                    for &val in row.iter() {
                        acc += val * val + val.sqrt();
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    group.bench_function("par_simple", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&array)
                .axis_iter(ndarray::Axis(0))
                .into_par_iter()
                .map(|row| {
                    let mut acc = 0.0f32;
                    for &val in row.iter() {
                        acc += val * val + val.sqrt();
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    // Medium complexity - trigonometric operations
    group.bench_function("seq_medium", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&array)
                .axis_iter(ndarray::Axis(0))
                .map(|row| {
                    let mut acc = 0.0f32;
                    for &val in row.iter() {
                        acc += val.sin() * val.cos() + val.tan().abs();
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    group.bench_function("par_medium", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&array)
                .axis_iter(ndarray::Axis(0))
                .into_par_iter()
                .map(|row| {
                    let mut acc = 0.0f32;
                    for &val in row.iter() {
                        acc += val.sin() * val.cos() + val.tan().abs();
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    // High complexity - exponential and logarithmic operations
    group.bench_function("seq_complex", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&array)
                .axis_iter(ndarray::Axis(0))
                .map(|row| {
                    let mut acc = 0.0f32;
                    for &val in row.iter() {
                        acc += val.exp() * val.ln().abs() + val.powf(1.5) + (val + 1.0).recip();
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    group.bench_function("par_complex", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&array)
                .axis_iter(ndarray::Axis(0))
                .into_par_iter()
                .map(|row| {
                    let mut acc = 0.0f32;
                    for &val in row.iter() {
                        acc += val.exp() * val.ln().abs() + val.powf(1.5) + (val + 1.0).recip();
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    group.finish();
}

fn bench_par_iter_chunk_sizes_slsl(c: &mut Criterion) {
    let mut group = c.benchmark_group("slsl_par_iter_chunk_sizes");

    let size = 50000; // Large size to see chunking effects
    let data: Vec<f32> = (0..size * 10).map(|i| i as f32).collect();
    let tensor = Tensor::from_vec(data, vec![size, 10]).unwrap();

    // Sequential baseline
    group.bench_function("seq_baseline", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&tensor)
                .iter_dim(0)
                .map(|view| {
                    // Complex computation unrelated to slsl
                    let mut acc = 0.0f32;
                    for j in 0..view.shape()[0] {
                        let val = view.at::<f32>([j]);
                        acc += val.sin() * val.cos() + val.sqrt().abs() + (val * 2.0).tanh();
                        acc += val.exp() * val.ln().abs() + val.powf(1.5);
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    // Parallel with different strategies
    #[cfg(feature = "rayon")]
    group.bench_function("par_default", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&tensor)
                .iter_dim(0)
                .par_iter()
                .map(|view| {
                    let mut acc = 0.0f32;
                    for j in 0..view.shape()[0] {
                        let val = view.at::<f32>([j]);
                        acc += val.sin() * val.cos() + val.sqrt().abs() + (val * 2.0).tanh();
                        acc += val.exp() * val.ln().abs() + val.powf(1.5);
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    group.finish();
}

fn bench_par_iter_chunk_sizes_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_par_iter_chunk_sizes");

    let size = 50000; // Same size as slsl for fair comparison
    let data: Vec<f32> = (0..size * 10).map(|i| i as f32).collect();
    let array = ArcArray::from_shape_vec((size, 10), data).unwrap();

    // Sequential baseline
    group.bench_function("seq_baseline", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&array)
                .axis_iter(ndarray::Axis(0))
                .map(|row| {
                    // Same complex computation for fair comparison
                    let mut acc = 0.0f32;
                    for &val in row.iter() {
                        acc += val.sin() * val.cos() + val.sqrt().abs() + (val * 2.0).tanh();
                        acc += val.exp() * val.ln().abs() + val.powf(1.5);
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    // Parallel with rayon
    group.bench_function("par_default", |b| {
        b.iter(|| {
            let result: Vec<f32> = black_box(&array)
                .axis_iter(ndarray::Axis(0))
                .into_par_iter()
                .map(|row| {
                    let mut acc = 0.0f32;
                    for &val in row.iter() {
                        acc += val.sin() * val.cos() + val.sqrt().abs() + (val * 2.0).tanh();
                        acc += val.exp() * val.ln().abs() + val.powf(1.5);
                    }
                    acc
                })
                .collect();
            black_box(result);
        });
    });

    group.finish();
}

fn bench_par_iter_memory_intensive_slsl(c: &mut Criterion) {
    let mut group = c.benchmark_group("slsl_par_iter_memory_intensive");

    let size = 20000; // Medium-large size for memory tests
    let data: Vec<f32> = (0..size * 20).map(|i| i as f32).collect();
    let tensor = Tensor::from_vec(data, vec![size, 20]).unwrap();

    // Sequential memory-intensive operation
    group.bench_function("seq_memory", |b| {
        b.iter(|| {
            let result: Vec<Vec<f32>> = black_box(&tensor)
                .iter_dim(0)
                .map(|view| {
                    // Create intermediate vectors with complex computations
                    let mut row_data = Vec::with_capacity(view.shape()[0]);
                    for j in 0..view.shape()[0] {
                        let val = view.at::<f32>([j]);
                        row_data.push(val.sin() * val.cos());
                        row_data.push(val.sqrt().abs());
                        row_data.push((val * 2.0).tanh());
                        row_data.push(val.exp() * val.ln().abs());
                    }
                    row_data
                })
                .collect();
            black_box(result);
        });
    });

    // Parallel memory-intensive operation
    #[cfg(feature = "rayon")]
    group.bench_function("par_memory", |b| {
        b.iter(|| {
            let result: Vec<Vec<f32>> = black_box(&tensor)
                .iter_dim(0)
                .par_iter()
                .map(|view| {
                    let mut row_data = Vec::with_capacity(view.shape()[0]);
                    for j in 0..view.shape()[0] {
                        let val = view.at::<f32>([j]);
                        row_data.push(val.sin() * val.cos());
                        row_data.push(val.sqrt().abs());
                        row_data.push((val * 2.0).tanh());
                        row_data.push(val.exp() * val.ln().abs());
                    }
                    row_data
                })
                .collect();
            black_box(result);
        });
    });

    group.finish();
}

fn bench_par_iter_memory_intensive_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_par_iter_memory_intensive");

    let size = 20000; // Same size as slsl for fair comparison
    let data: Vec<f32> = (0..size * 20).map(|i| i as f32).collect();
    let array = ArcArray::from_shape_vec((size, 20), data).unwrap();

    // Sequential memory-intensive operation
    group.bench_function("seq_memory", |b| {
        b.iter(|| {
            let result: Vec<Vec<f32>> = black_box(&array)
                .axis_iter(ndarray::Axis(0))
                .map(|row| {
                    // Same complex computations for fair comparison
                    let mut row_data = Vec::with_capacity(row.len());
                    for &val in row.iter() {
                        row_data.push(val.sin() * val.cos());
                        row_data.push(val.sqrt().abs());
                        row_data.push((val * 2.0).tanh());
                        row_data.push(val.exp() * val.ln().abs());
                    }
                    row_data
                })
                .collect();
            black_box(result);
        });
    });

    // Parallel memory-intensive operation
    group.bench_function("par_memory", |b| {
        b.iter(|| {
            let result: Vec<Vec<f32>> = black_box(&array)
                .axis_iter(ndarray::Axis(0))
                .into_par_iter()
                .map(|row| {
                    let mut row_data = Vec::with_capacity(row.len());
                    for &val in row.iter() {
                        row_data.push(val.sin() * val.cos());
                        row_data.push(val.sqrt().abs());
                        row_data.push((val * 2.0).tanh());
                        row_data.push(val.exp() * val.ln().abs());
                    }
                    row_data
                })
                .collect();
            black_box(result);
        });
    });

    group.finish();
}

fn bench_par_iter_scaling_slsl(c: &mut Criterion) {
    let mut group = c.benchmark_group("slsl_par_iter_scaling");

    // Test scaling with different tensor dimensions
    let dimensions = [(1000, 100), (5000, 50), (10000, 25), (20000, 10)];

    for (rows, cols) in dimensions {
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![rows, cols]).unwrap();

        // Sequential
        group.bench_function(format!("seq_{rows}x{cols}"), |b| {
            b.iter(|| {
                let result: Vec<f32> = black_box(&tensor)
                    .iter_dim(0)
                    .map(|view| {
                        // Complex computation unrelated to slsl
                        let mut acc = 0.0f32;
                        for j in 0..view.shape()[0] {
                            let val = view.at::<f32>([j]);
                            acc += val.sin() * val.cos() + val.sqrt().abs();
                            acc += val.exp() * val.ln().abs() + val.powf(1.5);
                            acc += (val + 1.0).recip() + val.tan().abs();
                        }
                        acc
                    })
                    .collect();
                black_box(result);
            });
        });

        // Parallel
        #[cfg(feature = "rayon")]
        group.bench_function(format!("par_{rows}x{cols}"), |b| {
            b.iter(|| {
                let result: Vec<f32> = black_box(&tensor)
                    .iter_dim(0)
                    .par_iter()
                    .map(|view| {
                        let mut acc = 0.0f32;
                        for j in 0..view.shape()[0] {
                            let val = view.at::<f32>([j]);
                            acc += val.sin() * val.cos() + val.sqrt().abs();
                            acc += val.exp() * val.ln().abs() + val.powf(1.5);
                            acc += (val + 1.0).recip() + val.tan().abs();
                        }
                        acc
                    })
                    .collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_par_iter_scaling_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_par_iter_scaling");

    // Test scaling with different array dimensions - same as slsl for fair comparison
    let dimensions = [(1000, 100), (5000, 50), (10000, 25), (20000, 10)];

    for (rows, cols) in dimensions {
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let array = ArcArray::from_shape_vec((rows, cols), data).unwrap();

        // Sequential
        group.bench_function(format!("seq_{rows}x{cols}"), |b| {
            b.iter(|| {
                let result: Vec<f32> = black_box(&array)
                    .axis_iter(ndarray::Axis(0))
                    .map(|row| {
                        // Same complex computation for fair comparison
                        let mut acc = 0.0f32;
                        for &val in row.iter() {
                            acc += val.sin() * val.cos() + val.sqrt().abs();
                            acc += val.exp() * val.ln().abs() + val.powf(1.5);
                            acc += (val + 1.0).recip() + val.tan().abs();
                        }
                        acc
                    })
                    .collect();
                black_box(result);
            });
        });

        // Parallel
        group.bench_function(format!("par_{rows}x{cols}"), |b| {
            b.iter(|| {
                let result: Vec<f32> = black_box(&array)
                    .axis_iter(ndarray::Axis(0))
                    .into_par_iter()
                    .map(|row| {
                        let mut acc = 0.0f32;
                        for &val in row.iter() {
                            acc += val.sin() * val.cos() + val.sqrt().abs();
                            acc += val.exp() * val.ln().abs() + val.powf(1.5);
                            acc += (val + 1.0).recip() + val.tan().abs();
                        }
                        acc
                    })
                    .collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_par_iter_dim_slsl,
    bench_par_iter_dim_ndarray,
    bench_par_iter_complexity_slsl,
    bench_par_iter_complexity_ndarray,
    bench_par_iter_chunk_sizes_slsl,
    bench_par_iter_chunk_sizes_ndarray,
    bench_par_iter_memory_intensive_slsl,
    bench_par_iter_memory_intensive_ndarray,
    bench_par_iter_scaling_slsl,
    bench_par_iter_scaling_ndarray
);
criterion_main!(benches);
