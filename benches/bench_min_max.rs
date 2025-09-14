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

fn benchmark_1d_min_max(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("1D Min/Max ");

    for &size in SIZES_1D {
        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        // u8 data
        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [size]).unwrap();
        let candle_tensor_u8 = CandleTensor::from_vec(data_u8.clone(), size, &device).unwrap();

        // ===== Min operations =====
        // f32 min
        group.bench_function(format!("slsl/f32/min/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.min(0).unwrap());
            })
        });
        group.bench_function(format!("candle/f32/min/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.min(0).unwrap());
            })
        });
        group.bench_function(format!("vec/f32/min/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(data_f32.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
            })
        });

        // u8 min
        group.bench_function(format!("slsl/u8/min/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_u8.min(0).unwrap());
            })
        });
        group.bench_function(format!("candle/u8/min/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_u8.min(0).unwrap());
            })
        });
        group.bench_function(format!("vec/u8/min/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(data_u8.iter().fold(u8::MAX, |a, &b| a.min(b)));
            })
        });

        // ===== Max operations =====
        // f32 max
        group.bench_function(format!("slsl/f32/max/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.max(0).unwrap());
            })
        });
        group.bench_function(format!("candle/f32/max/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.max(0).unwrap());
            })
        });
        group.bench_function(format!("vec/f32/max/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(data_f32.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
            })
        });

        // u8 max
        group.bench_function(format!("slsl/u8/max/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_u8.max(0).unwrap());
            })
        });
        group.bench_function(format!("candle/u8/max/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_u8.max(0).unwrap());
            })
        });
        group.bench_function(format!("vec/u8/max/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(data_u8.iter().fold(u8::MIN, |a, &b| a.max(b)));
            })
        });

        // ===== Argmin operations =====
        // f32 argmin
        group.bench_function(format!("slsl/f32/argmin/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.argmin(0).unwrap());
            })
        });
        group.bench_function(format!("candle/f32/argmin/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.argmin(0).unwrap());
            })
        });
        group.bench_function(format!("vec/f32/argmin/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(
                    data_f32
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i)
                        .unwrap(),
                );
            })
        });

        // u8 argmin
        group.bench_function(format!("slsl/u8/argmin/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_u8.argmin(0).unwrap());
            })
        });
        group.bench_function(format!("candle/u8/argmin/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_u8.argmin(0).unwrap());
            })
        });
        group.bench_function(format!("vec/u8/argmin/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(
                    data_u8
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.cmp(b))
                        .map(|(i, _)| i)
                        .unwrap(),
                );
            })
        });

        // ===== Argmax operations =====
        // f32 argmax
        group.bench_function(format!("slsl/f32/argmax/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.argmax(0).unwrap());
            })
        });
        group.bench_function(format!("candle/f32/argmax/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.argmax(0).unwrap());
            })
        });
        group.bench_function(format!("vec/f32/argmax/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(
                    data_f32
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i)
                        .unwrap(),
                );
            })
        });

        // u8 argmax
        group.bench_function(format!("slsl/u8/argmax/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_u8.argmax(0).unwrap());
            })
        });
        group.bench_function(format!("candle/u8/argmax/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_u8.argmax(0).unwrap());
            })
        });
        group.bench_function(format!("vec/u8/argmax/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(
                    data_u8
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.cmp(b))
                        .map(|(i, _)| i)
                        .unwrap(),
                );
            })
        });
    }

    group.finish();
}

fn benchmark_2d_min_max(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("2D Min/Max ");

    // Small scale 2D tensors
    for &(rows, cols) in SIZES_2D {
        let size = rows * cols;

        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [rows, cols]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (rows, cols), &device).unwrap();

        // u8 data
        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [rows, cols]).unwrap();
        let candle_tensor_u8 =
            CandleTensor::from_vec(data_u8.clone(), (rows, cols), &device).unwrap();

        // f32 min dim0
        group.bench_function(format!("slsl/f32/min/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.min(0).unwrap());
            })
        });
        group.bench_function(format!("candle/f32/min/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.min(0).unwrap());
            })
        });
        group.bench_function(format!("vec/f32/min/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                black_box({
                    let mut result = vec![f32::INFINITY; cols];
                    for col in 0..cols {
                        for row in 0..rows {
                            result[col] = result[col].min(data_f32[row * cols + col]);
                        }
                    }
                    result
                });
            })
        });

        // u8 min dim0
        group.bench_function(format!("slsl/u8/min/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_u8.min(0).unwrap());
            })
        });
        group.bench_function(format!("candle/u8/min/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_u8.min(0).unwrap());
            })
        });
        group.bench_function(format!("vec/u8/min/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                black_box({
                    let mut result = vec![u8::MAX; cols];
                    for col in 0..cols {
                        for row in 0..rows {
                            result[col] = result[col].min(data_u8[row * cols + col]);
                        }
                    }
                    result
                });
            })
        });

        // f32 min dim1
        group.bench_function(format!("slsl/f32/min/{rows}x{cols}/dim1"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.min(1).unwrap());
            })
        });
        group.bench_function(format!("candle/f32/min/{rows}x{cols}/dim1"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.min(1).unwrap());
            })
        });
        group.bench_function(format!("vec/f32/min/{rows}x{cols}/dim1"), |bencher| {
            bencher.iter(|| {
                black_box({
                    let mut result = vec![f32::INFINITY; rows];
                    for row in 0..rows {
                        for col in 0..cols {
                            result[row] = result[row].min(data_f32[row * cols + col]);
                        }
                    }
                    result
                });
            })
        });

        // u8 min dim1
        group.bench_function(format!("slsl/u8/min/{rows}x{cols}/dim1"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_u8.min(1).unwrap());
            })
        });
        group.bench_function(format!("candle/u8/min/{rows}x{cols}/dim1"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_u8.min(1).unwrap());
            })
        });
        group.bench_function(format!("vec/u8/min/{rows}x{cols}/dim1"), |bencher| {
            bencher.iter(|| {
                black_box({
                    let mut result = vec![u8::MAX; rows];
                    for row in 0..rows {
                        for col in 0..cols {
                            result[row] = result[row].min(data_u8[row * cols + col]);
                        }
                    }
                    result
                });
            })
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_1d_min_max, benchmark_2d_min_max,);
criterion_main!(benches);
