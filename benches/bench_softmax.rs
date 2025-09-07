use candle_core::{Device, Tensor as CandleTensor};
use candle_nn::ops::softmax;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use slsl::Tensor;
use std::hint::black_box;
use std::time::Duration;

// Extended test sizes for comprehensive benchmarking
const TEST_SIZES_1D: &[usize] = &[
    100, 128, 378, 512, 640, 783, 800, 900, 1000, 2000, 4000, 10000,
];

// Helper function to create SLSL tensors
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

// Helper function to create Candle tensors
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

// Comprehensive 1D softmax comparison: SLSL vs Candle for f32
fn bench_1d_softmax_f32_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("1d_softmax_f32_comparison");
    group.measurement_time(Duration::from_secs(10));

    for &size in TEST_SIZES_1D {
        // Create tensors with identical data for fair comparison
        let slsl_tensor = create_slsl_tensor_f32(size);
        let candle_tensor = create_candle_tensor_f32(size);

        // SLSL benchmark
        group.bench_with_input(BenchmarkId::new("slsl", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(&slsl_tensor).softmax(0).unwrap();
                black_box(result);
            })
        });

        // Candle benchmark
        group.bench_with_input(BenchmarkId::new("candle", size), &size, |b, _| {
            b.iter(|| {
                let result = softmax(black_box(&candle_tensor), 0).unwrap();
                black_box(result);
            })
        });
    }
    group.finish();
}

// Comprehensive 1D softmax comparison: SLSL vs Candle for f64
fn bench_1d_softmax_f64_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("1d_softmax_f64_comparison");
    group.measurement_time(Duration::from_secs(10));

    for &size in TEST_SIZES_1D {
        // Create tensors with identical data for fair comparison
        let slsl_tensor = create_slsl_tensor_f64(size);
        let candle_tensor = create_candle_tensor_f64(size);

        // SLSL benchmark
        group.bench_with_input(BenchmarkId::new("slsl", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(&slsl_tensor).softmax(0).unwrap();
                black_box(result);
            })
        });

        // Candle benchmark
        group.bench_with_input(BenchmarkId::new("candle", size), &size, |b, _| {
            b.iter(|| {
                let result = softmax(black_box(&candle_tensor), 0).unwrap();
                black_box(result);
            })
        });
    }
    group.finish();
}

// Comprehensive 1D softmax comparison: SLSL vs Candle for f16
fn bench_1d_softmax_f16_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("1d_softmax_f16_comparison");
    group.measurement_time(Duration::from_secs(10));

    for &size in TEST_SIZES_1D {
        // Create tensors with identical data for fair comparison
        let slsl_tensor = create_slsl_tensor_f16(size);
        let candle_tensor = create_candle_tensor_f16(size);

        // SLSL benchmark
        group.bench_with_input(BenchmarkId::new("slsl", size), &size, |b, _| {
            b.iter(|| {
                let result = black_box(&slsl_tensor).softmax(0).unwrap();
                black_box(result);
            })
        });

        // Candle benchmark
        group.bench_with_input(BenchmarkId::new("candle", size), &size, |b, _| {
            b.iter(|| {
                let result = softmax(black_box(&candle_tensor), 0).unwrap();
                black_box(result);
            })
        });
    }
    group.finish();
}

// 2D tensor benchmarks for more realistic scenarios - SLSL vs Candle comparison
fn bench_2d_softmax_f32_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("2d_softmax_f32_comparison");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [(10, 10), (32, 32), (100, 100), (256, 256)];

    for &(rows, cols) in &sizes {
        let size = rows * cols;
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        // Create tensors with identical data for fair comparison
        let slsl_tensor = Tensor::from_vec(data.clone(), [rows, cols]).unwrap();
        let candle_tensor = CandleTensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap();

        // Benchmark softmax along dimension 0
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

        // Benchmark softmax along dimension 1
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

criterion_group!(
    benches,
    bench_1d_softmax_f32_comparison,
    bench_1d_softmax_f64_comparison,
    bench_1d_softmax_f16_comparison,
    bench_2d_softmax_f32_comparison
);
criterion_main!(benches);
