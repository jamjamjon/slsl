#![allow(dead_code)]

use candle_core::{Device, Tensor as CandleTensor};
use candle_nn::ops::softmax;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use slsl::Tensor;
use std::hint::black_box;
use std::time::Duration;

const SIZES_1D: &[usize] = &[
    10, 32, 64, 100, 256, 512, 783, 1000, 2000, 4000, 10000, 20000, 50000, 100000, 200000,
];
const SIZES_2D: &[(usize, usize)] = &[
    (10, 10),
    (32, 32),
    (100, 100),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
];

const SIZES_3D: &[(usize, usize, usize)] = &[
    (8, 8, 8),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (384, 384, 384),
];

fn create_slsl_tensor_f32(size: usize) -> Tensor {
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
    Tensor::from_vec(data, [size]).unwrap()
}

fn create_slsl_tensor_f64(size: usize) -> Tensor {
    let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01).collect();
    Tensor::from_vec(data, [size]).unwrap()
}

fn create_slsl_tensor_f16(size: usize) -> Tensor {
    let data: Vec<half::f16> = (0..size)
        .map(|i| half::f16::from_f32((i as f32) * 0.01))
        .collect();
    Tensor::from_vec(data, [size]).unwrap()
}

fn create_slsl_tensor_2d_f16(rows: usize, cols: usize) -> Tensor {
    let size = rows * cols;
    let data: Vec<half::f16> = (0..size)
        .map(|i| half::f16::from_f32((i as f32) * 0.01))
        .collect();
    Tensor::from_vec(data, [rows, cols]).unwrap()
}

fn create_slsl_tensor_3d_f16(d1: usize, d2: usize, d3: usize) -> Tensor {
    let size = d1 * d2 * d3;
    let data: Vec<half::f16> = (0..size)
        .map(|i| half::f16::from_f32((i as f32) * 0.01))
        .collect();
    Tensor::from_vec(data, [d1, d2, d3]).unwrap()
}

fn create_slsl_tensor_3d_f32(d1: usize, d2: usize, d3: usize) -> Tensor {
    let size = d1 * d2 * d3;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
    Tensor::from_vec(data, [d1, d2, d3]).unwrap()
}

fn create_candle_tensor_f32(size: usize) -> CandleTensor {
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
    CandleTensor::from_vec(data, size, &Device::Cpu).unwrap()
}

fn create_candle_tensor_f64(size: usize) -> CandleTensor {
    let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01).collect();
    CandleTensor::from_vec(data, size, &Device::Cpu).unwrap()
}

fn create_candle_tensor_f16(size: usize) -> CandleTensor {
    let data: Vec<half::f16> = (0..size)
        .map(|i| half::f16::from_f32((i as f32) * 0.01))
        .collect();
    CandleTensor::from_vec(data, size, &Device::Cpu).unwrap()
}

fn create_candle_tensor_2d_f16(rows: usize, cols: usize) -> CandleTensor {
    let size = rows * cols;
    let data: Vec<half::f16> = (0..size)
        .map(|i| half::f16::from_f32((i as f32) * 0.01))
        .collect();
    CandleTensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap()
}

fn create_candle_tensor_3d_f16(d1: usize, d2: usize, d3: usize) -> CandleTensor {
    let size = d1 * d2 * d3;
    let data: Vec<half::f16> = (0..size)
        .map(|i| half::f16::from_f32((i as f32) * 0.01))
        .collect();
    CandleTensor::from_vec(data, (d1, d2, d3), &Device::Cpu).unwrap()
}

fn create_candle_tensor_3d_f32(d1: usize, d2: usize, d3: usize) -> CandleTensor {
    let size = d1 * d2 * d3;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
    CandleTensor::from_vec(data, (d1, d2, d3), &Device::Cpu).unwrap()
}

fn bench_1d_softmax_f32_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("1d_softmax_f32_comparison");
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES_1D {
        let slsl_tensor = create_slsl_tensor_f32(size);
        let candle_tensor = create_candle_tensor_f32(size);

        group.bench_with_input(BenchmarkId::new("slsl", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(&slsl_tensor).softmax(0).unwrap();
                black_box(result);
            })
        });

        group.bench_with_input(BenchmarkId::new("candle", size), &size, |b, _| {
            b.iter(|| {
                let result = softmax(black_box(&candle_tensor), 0).unwrap();
                black_box(result);
            })
        });
    }
    group.finish();
}

fn bench_1d_softmax_f64_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("1d_softmax_f64_comparison");
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES_1D {
        let slsl_tensor = create_slsl_tensor_f64(size);
        let candle_tensor = create_candle_tensor_f64(size);

        group.bench_with_input(BenchmarkId::new("slsl", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(&slsl_tensor).softmax(0).unwrap();
                black_box(result);
            })
        });

        group.bench_with_input(BenchmarkId::new("candle", size), &size, |b, _| {
            b.iter(|| {
                let result = softmax(black_box(&candle_tensor), 0).unwrap();
                black_box(result);
            })
        });
    }
    group.finish();
}

fn bench_1d_softmax_f16_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("1d_softmax_f16_comparison");
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES_1D {
        let slsl_tensor = create_slsl_tensor_f16(size);
        let candle_tensor = create_candle_tensor_f16(size);

        group.bench_with_input(BenchmarkId::new("slsl", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(&slsl_tensor).softmax(0).unwrap();
                black_box(result);
            })
        });

        group.bench_with_input(BenchmarkId::new("candle", size), &size, |b, _| {
            b.iter(|| {
                let result = softmax(black_box(&candle_tensor), 0).unwrap();
                black_box(result);
            })
        });
    }
    group.finish();
}

fn bench_2d_softmax_f32_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("2d_softmax_f32_comparison");
    group.measurement_time(Duration::from_secs(10));

    for &(rows, cols) in SIZES_2D {
        let size = rows * cols;
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        let slsl_tensor = Tensor::from_vec(data.clone(), [rows, cols]).unwrap();
        let candle_tensor = CandleTensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap();

        group.bench_with_input(
            BenchmarkId::new("slsl_dim0", format!("{rows}x{cols}")),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    let result = black_box(&slsl_tensor).softmax(0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("candle_dim0", format!("{rows}x{cols}")),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    let result = softmax(black_box(&candle_tensor), 0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("slsl_dim1", format!("{rows}x{cols}")),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    let result = black_box(&slsl_tensor).softmax(1).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("candle_dim1", format!("{rows}x{cols}")),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    let result = softmax(black_box(&candle_tensor), 1).unwrap();
                    black_box(result);
                })
            },
        );
    }
    group.finish();
}

fn bench_2d_softmax_f16_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("2d_softmax_f16_comparison");
    group.measurement_time(Duration::from_secs(10));

    for &(rows, cols) in SIZES_2D {
        let slsl_tensor = create_slsl_tensor_2d_f16(rows, cols);
        let candle_tensor = create_candle_tensor_2d_f16(rows, cols);

        group.bench_with_input(
            BenchmarkId::new("slsl_dim0", format!("{rows}x{cols}")),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    let result = black_box(&slsl_tensor).softmax(0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("candle_dim0", format!("{rows}x{cols}")),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    let result = softmax(black_box(&candle_tensor), 0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("slsl_dim1", format!("{rows}x{cols}")),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    let result = black_box(&slsl_tensor).softmax(1).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("candle_dim1", format!("{rows}x{cols}")),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    let result = softmax(black_box(&candle_tensor), 1).unwrap();
                    black_box(result);
                })
            },
        );
    }
    group.finish();
}

fn bench_3d_softmax_f16_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("3d_softmax_f16_comparison");
    group.measurement_time(Duration::from_secs(10));

    for &(d1, d2, d3) in SIZES_3D {
        let slsl_tensor = create_slsl_tensor_3d_f16(d1, d2, d3);
        let candle_tensor = create_candle_tensor_3d_f16(d1, d2, d3);

        for dim in 0..3 {
            group.bench_with_input(
                BenchmarkId::new("slsl", format!("{d1}x{d2}x{d3}_dim{dim}")),
                &(d1, d2, d3, dim),
                |b, &(_, _, _, dim)| {
                    b.iter(|| {
                        let result = black_box(&slsl_tensor).softmax(dim).unwrap();
                        black_box(result);
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("candle", format!("{d1}x{d2}x{d3}_dim{dim}")),
                &(d1, d2, d3, dim),
                |b, &(_, _, _, dim)| {
                    b.iter(|| {
                        let result = softmax(black_box(&candle_tensor), dim).unwrap();
                        black_box(result);
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_3d_softmax_f32_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("3d_softmax_f32_comparison");
    group.measurement_time(Duration::from_secs(10));

    for &(d1, d2, d3) in SIZES_3D {
        let slsl_tensor = create_slsl_tensor_3d_f32(d1, d2, d3);
        let candle_tensor = create_candle_tensor_3d_f32(d1, d2, d3);

        for dim in 0..3 {
            group.bench_with_input(
                BenchmarkId::new("slsl", format!("{d1}x{d2}x{d3}_dim{dim}")),
                &(d1, d2, d3, dim),
                |b, &(_, _, _, dim)| {
                    b.iter(|| {
                        let result = black_box(&slsl_tensor).softmax(dim).unwrap();
                        black_box(result);
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("candle", format!("{d1}x{d2}x{d3}_dim{dim}")),
                &(d1, d2, d3, dim),
                |b, &(_, _, _, dim)| {
                    b.iter(|| {
                        let result = softmax(black_box(&candle_tensor), dim).unwrap();
                        black_box(result);
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_1d_softmax_f32_comparison,
    bench_1d_softmax_f64_comparison,
    bench_1d_softmax_f16_comparison,
    bench_2d_softmax_f32_comparison,
    bench_2d_softmax_f16_comparison,
    bench_3d_softmax_f16_comparison,
    bench_3d_softmax_f32_comparison
);
criterion_main!(benches);
