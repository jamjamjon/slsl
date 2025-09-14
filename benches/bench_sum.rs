#![allow(unused)]
use candle_core::{Device, Tensor as CandleTensor};
use criterion::{criterion_group, criterion_main, Criterion};
use slsl::*;
use std::hint::black_box;

// 1D
const SIZES_1D: &[usize] = &[10, 32, 64, 128, 256, 376, 512, 768, 1024, 1344, 2048, 4096];

// 2D
const SIZES_2D: &[(usize, usize)] = &[
    (10, 10),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
];

// 3D
const SIZES_3D: &[(usize, usize, usize)] = &[
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
];

fn benchmark_1d_sum_all(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("1D Sum All");

    // Small scale 1D tensors
    for &size in SIZES_1D {
        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        // u8 data
        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [size]).unwrap();
        let candle_tensor_u8 = CandleTensor::from_vec(data_u8.clone(), size, &device).unwrap();

        // f32 sum_all
        group.bench_function(format!("slsl/f32/{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum_all().unwrap();
                black_box(result);
            })
        });

        // u8 sum_all
        group.bench_function(format!("slsl/u8/{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_u8.sum_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/u8/{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_u8.sum_all().unwrap();
                black_box(result);
            })
        });
    }

    group.finish();
}

fn benchmark_1d_sum(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("1D Sum");

    // Small scale 1D tensors
    for &size in SIZES_1D {
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        // f32 sum along dimension 0
        group.bench_function(format!("slsl/f32/{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum(0).unwrap();
                black_box(result);
            })
        });
    }
    group.finish();
}

fn benchmark_1d_sum_keepdim(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("1D Sum Keepdim");

    for &size in SIZES_1D {
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        // f32 sum_keepdim along dimension 0
        group.bench_function(format!("slsl/f32/{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum_keepdim(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum_keepdim(0).unwrap();
                black_box(result);
            })
        });
    }
    group.finish();
}

// ========== 2D Tensor Benchmarks ==========

fn benchmark_2d_sum_all(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("2D Sum All");

    // Small scale 2D tensors
    for &(rows, cols) in SIZES_2D {
        let size = rows * cols;

        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [rows, cols]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (rows, cols), &device).unwrap();

        // u8 data
        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [rows, cols]).unwrap();
        let candle_tensor_u8 =
            CandleTensor::from_vec(data_u8.clone(), (rows, cols), &device).unwrap();

        // f32 sum_all
        group.bench_function(format!("slsl/f32/{rows}x{cols}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{rows}x{cols}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum_all().unwrap();
                black_box(result);
            })
        });

        // u8 sum_all
        group.bench_function(format!("slsl/u8/{rows}x{cols}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_u8.sum_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/u8/{rows}x{cols}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_u8.sum_all().unwrap();
                black_box(result);
            })
        });
    }

    group.finish();
}

fn benchmark_2d_sum(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("2D Sum");

    // Small scale 2D tensors
    for &(rows, cols) in SIZES_2D {
        let size = rows * cols;
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [rows, cols]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (rows, cols), &device).unwrap();

        // f32 sum along dimension 0 (rows)
        group.bench_function(format!("slsl/f32/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum(0).unwrap();
                black_box(result);
            })
        });

        // f32 sum along dimension 1 (cols)
        group.bench_function(format!("slsl/f32/{rows}x{cols}/dim1"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum(1).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{rows}x{cols}/dim1"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum(1).unwrap();
                black_box(result);
            })
        });
    }
    group.finish();
}

fn benchmark_2d_sum_keepdim(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("2D Sum Keepdim");

    for &(rows, cols) in SIZES_2D {
        let size = rows * cols;

        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [rows, cols]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (rows, cols), &device).unwrap();

        // f32 sum_keepdim along dimension 0
        group.bench_function(format!("slsl/f32/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum_keepdim(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum_keepdim(0).unwrap();
                black_box(result);
            })
        });

        // f32 sum_keepdim along dimension 1
        group.bench_function(format!("slsl/f32/{rows}x{cols}/dim1"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum_keepdim(1).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{rows}x{cols}/dim1"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum_keepdim(1).unwrap();
                black_box(result);
            })
        });
    }

    group.finish();
}

fn benchmark_3d_sum_all(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("3D Sum All");

    // Small scale 3D tensors
    for &(d1, d2, d3) in SIZES_3D {
        let size = d1 * d2 * d3;

        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (d1, d2, d3), &device).unwrap();

        // u8 data
        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_u8 =
            CandleTensor::from_vec(data_u8.clone(), (d1, d2, d3), &device).unwrap();

        // f32 sum_all
        group.bench_function(format!("slsl/f32/{d1}x{d2}x{d3}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{d1}x{d2}x{d3}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum_all().unwrap();
                black_box(result);
            })
        });

        // u8 sum_all
        group.bench_function(format!("slsl/u8/{d1}x{d2}x{d3}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_u8.sum_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/u8/{d1}x{d2}x{d3}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_u8.sum_all().unwrap();
                black_box(result);
            })
        });
    }

    group.finish();
}

fn benchmark_3d_sum(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("3D Sum");

    // Small scale 3D tensors - test multiple dimensions
    for &(d1, d2, d3) in SIZES_3D {
        let size = d1 * d2 * d3;

        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (d1, d2, d3), &device).unwrap();

        // f32 sum along dimension 0
        group.bench_function(format!("slsl/f32/{d1}x{d2}x{d3}/dim0"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{d1}x{d2}x{d3}/dim0"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum(0).unwrap();
                black_box(result);
            })
        });

        // f32 sum along dimension 1
        group.bench_function(format!("slsl/f32/{d1}x{d2}x{d3}/dim1"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum(1).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{d1}x{d2}x{d3}/dim1"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum(1).unwrap();
                black_box(result);
            })
        });

        // f32 sum along dimension 2
        group.bench_function(format!("slsl/f32/{d1}x{d2}x{d3}/dim2"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum(2).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{d1}x{d2}x{d3}/dim2"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum(2).unwrap();
                black_box(result);
            })
        });

        // f32 sum along dimensions 0 and 1
        group.bench_function(format!("slsl/f32/{d1}x{d2}x{d3}/dim01"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum([0, 1]).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{d1}x{d2}x{d3}/dim01"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum([0, 1]).unwrap();
                black_box(result);
            })
        });

        // f32 sum along dimensions 0 and 2
        group.bench_function(format!("slsl/f32/{d1}x{d2}x{d3}/dim02"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum([0, 2]).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{d1}x{d2}x{d3}/dim02"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum([0, 2]).unwrap();
                black_box(result);
            })
        });

        // f32 sum along dimensions 1 and 2
        group.bench_function(format!("slsl/f32/{d1}x{d2}x{d3}/dim12"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum([1, 2]).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{d1}x{d2}x{d3}/dim12"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum([1, 2]).unwrap();
                black_box(result);
            })
        });
    }

    group.finish();
}

fn benchmark_3d_sum_keepdim(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("3D Sum Keepdim");

    for &(d1, d2, d3) in SIZES_3D {
        let size = d1 * d2 * d3;

        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (d1, d2, d3), &device).unwrap();

        // f32 sum_keepdim along dimension 0
        group.bench_function(format!("slsl/f32/{d1}x{d2}x{d3}/dim0"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum_keepdim(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{d1}x{d2}x{d3}/dim0"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum_keepdim(0).unwrap();
                black_box(result);
            })
        });

        // f32 sum_keepdim along dimension 1
        group.bench_function(format!("slsl/f32/{d1}x{d2}x{d3}/dim1"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum_keepdim(1).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{d1}x{d2}x{d3}/dim1"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum_keepdim(1).unwrap();
                black_box(result);
            })
        });

        // f32 sum_keepdim along dimension 2
        group.bench_function(format!("slsl/f32/{d1}x{d2}x{d3}/dim2"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum_keepdim(2).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{d1}x{d2}x{d3}/dim2"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum_keepdim(2).unwrap();
                black_box(result);
            })
        });

        // f32 sum_keepdim along dimensions 0 and 1
        group.bench_function(format!("slsl/f32/{d1}x{d2}x{d3}/dim01"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum_keepdim([0, 1]).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle/f32/{d1}x{d2}x{d3}/dim01"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum_keepdim([0, 1]).unwrap();
                black_box(result);
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_1d_sum_all,
    benchmark_2d_sum_all,
    benchmark_3d_sum_all,
    benchmark_1d_sum,
    benchmark_1d_sum_keepdim,
    benchmark_2d_sum,
    benchmark_2d_sum_keepdim,
    benchmark_3d_sum,
    benchmark_3d_sum_keepdim,
);

criterion_main!(benches);
