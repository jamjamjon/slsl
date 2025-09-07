#![allow(unused)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use slsl::*;

// Define data sizes of different scales - remove excessively large data
const SMALL_SIZES: &[usize] = &[100, 500, 1000];
const MEDIUM_SIZES: &[usize] = &[2000, 5000]; // Removed 10000
const LARGE_SIZES: &[usize] = &[20000, 50000]; // Removed 100000

// Define matrix sizes of different dimensions - remove excessively large matrices
const SMALL_MATRICES: &[(usize, usize)] = &[(32, 32), (64, 64), (128, 128)];
const MEDIUM_MATRICES: &[(usize, usize)] = &[(256, 256), (512, 512)]; // Removed 1024x1024
const LARGE_MATRICES: &[(usize, usize)] = &[(2048, 2048)]; // Removed 4096x4096

// Define 3D tensor sizes - remove excessively large tensors
const SMALL_3D: &[(usize, usize, usize)] = &[(16, 16, 16), (32, 32, 32), (64, 64, 64)];
const MEDIUM_3D: &[(usize, usize, usize)] = &[(128, 128, 128), (256, 256, 256)]; // Removed 512x512x512
const LARGE_3D: &[(usize, usize, usize)] = &[]; // Removed 1024x1024x1024

fn benchmark_1d_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("1D Norm Benchmarks");

    // Small scale 1D tensors
    for &size in SMALL_SIZES {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

        let slsl_tensor = Tensor::from_vec(data, [size]).unwrap();

        // L1 norm
        group.bench_function(format!("slsl_norm1_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor.norm1(0).unwrap();
                black_box(result);
            })
        });

        // L2 norm
        group.bench_function(format!("slsl_norm2_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor.norm2(0).unwrap();
                black_box(result);
            })
        });

        // L3 norm
        group.bench_function(format!("slsl_norm3_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor.normp(0, 3.0).unwrap();
                black_box(result);
            })
        });
    }

    // Medium scale 1D tensors
    for &size in MEDIUM_SIZES {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

        let slsl_tensor = Tensor::from_vec(data, [size]).unwrap();

        // L2 norm (main test)
        group.bench_function(format!("slsl_norm2_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor.norm2(0).unwrap();
                black_box(result);
            })
        });
    }

    // Large scale 1D tensors
    for &size in LARGE_SIZES {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

        let slsl_tensor = Tensor::from_vec(data, [size]).unwrap();

        // L2 norm (main test)
        group.bench_function(format!("slsl_norm2_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor.norm2(0).unwrap();
                black_box(result);
            })
        });
    }

    group.finish();
}

fn benchmark_2d_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("2D Norm Benchmarks");

    // Small scale 2D tensors
    for &(rows, cols) in SMALL_MATRICES {
        let size = rows * cols;
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        let slsl_tensor = Tensor::from_vec(data, [rows, cols]).unwrap();

        // Calculate L2 norm along rows
        group.bench_function(
            format!("slsl_norm2_2d_small_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor.norm2(0).unwrap();
                    black_box(result);
                })
            },
        );

        // Calculate L2 norm along columns
        group.bench_function(
            format!("slsl_norm2_2d_small_{rows}x{cols}_dim1"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor.norm2(1).unwrap();
                    black_box(result);
                })
            },
        );
    }

    // Medium scale 2D tensors
    for &(rows, cols) in MEDIUM_MATRICES {
        let size = rows * cols;
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        let slsl_tensor = Tensor::from_vec(data, [rows, cols]).unwrap();

        // Calculate L2 norm along rows
        group.bench_function(
            format!("slsl_norm2_2d_medium_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor.norm2(0).unwrap();
                    black_box(result);
                })
            },
        );
    }

    // Large scale 2D tensors
    for &(rows, cols) in LARGE_MATRICES {
        let size = rows * cols;
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        let slsl_tensor = Tensor::from_vec(data, [rows, cols]).unwrap();

        // Calculate L2 norm along rows
        group.bench_function(
            format!("slsl_norm2_2d_large_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor.norm2(0).unwrap();
                    black_box(result);
                })
            },
        );
    }

    group.finish();
}

fn benchmark_3d_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("3D Norm Benchmarks");

    // Small scale 3D tensors
    for &(d1, d2, d3) in SMALL_3D {
        let size = d1 * d2 * d3;
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();

        let slsl_tensor = Tensor::from_vec(data, [d1, d2, d3]).unwrap();

        // Calculate L2 norm along different dimensions
        for dim in 0..3 {
            group.bench_function(
                format!("slsl_norm2_3d_small_{d1}x{d2}x{d3}_dim{dim}"),
                |bencher| {
                    bencher.iter(|| {
                        let result = slsl_tensor.norm2(dim).unwrap();
                        black_box(result);
                    })
                },
            );
        }
    }

    // Medium scale 3D tensors
    for &(d1, d2, d3) in MEDIUM_3D {
        let size = d1 * d2 * d3;
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();

        let slsl_tensor = Tensor::from_vec(data, [d1, d2, d3]).unwrap();

        // Calculate L2 norm along the first dimension
        group.bench_function(
            format!("slsl_norm2_3d_medium_{d1}x{d2}x{d3}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor.norm2(0).unwrap();
                    black_box(result);
                })
            },
        );
    }

    group.finish();
}

fn benchmark_norm_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("Norm Types Comparison");

    // Test different types of norms with medium-scale tensors
    let size = 10000;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

    let slsl_tensor = Tensor::from_vec(data, [size]).unwrap();

    // L1 norm
    group.bench_function("slsl_norm1_medium", |bencher| {
        bencher.iter(|| {
            let result = slsl_tensor.norm1(0).unwrap();
            black_box(result);
        })
    });

    // L2 norm
    group.bench_function("slsl_norm2_medium", |bencher| {
        bencher.iter(|| {
            let result = slsl_tensor.norm2(0).unwrap();
            black_box(result);
        })
    });

    // L3 norm
    group.bench_function("slsl_norm3_medium", |bencher| {
        bencher.iter(|| {
            let result = slsl_tensor.normp(0, 3.0).unwrap();
            black_box(result);
        })
    });

    // L-infinity norm
    group.bench_function("slsl_norm_inf_medium", |bencher| {
        bencher.iter(|| {
            let result = slsl_tensor.norm(0, f32::INFINITY).unwrap();
            black_box(result);
        })
    });

    group.finish();
}

fn benchmark_keepdim(c: &mut Criterion) {
    let mut group = c.benchmark_group("Keepdim Norm Benchmarks");

    // Test keepdim functionality for 2D tensors
    let (rows, cols) = (1000, 1000);
    let size = rows * cols;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

    let slsl_tensor = Tensor::from_vec(data, [rows, cols]).unwrap();

    // Without keepdim
    group.bench_function("slsl_norm2_2d_no_keepdim", |bencher| {
        bencher.iter(|| {
            let result = slsl_tensor.norm2(0).unwrap();
            black_box(result);
        })
    });

    // With keepdim
    group.bench_function("slsl_norm2_2d_keepdim", |bencher| {
        bencher.iter(|| {
            let result = slsl_tensor.norm_keepdim(0, 2.0).unwrap();
            black_box(result);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_1d_norm,
    benchmark_2d_norm,
    benchmark_3d_norm,
    benchmark_norm_types,
    benchmark_keepdim
);
criterion_main!(benches);
