// #![allow(unused)]
use candle_core::{DType, Device, Tensor as CandleTensor};
use criterion::{criterion_group, criterion_main, Criterion};
use half::f16;
use slsl::*;
use std::hint::black_box;

// 1D sizes as specified
const SIZES_1D: &[usize] = &[10, 32, 64, 128, 256, 512, 1024, 4096];

// 2D sizes as specified
const SIZES_2D: &[(usize, usize)] = &[
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (768, 768),
    (1024, 1024),
];

// 3D sizes as specified
const SIZES_3D: &[(usize, usize, usize)] = &[
    (32, 32, 32),
    (48, 48, 48),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
];

/// Native Vec implementation for L1 norm
fn native_l1_norm_f32(data: &[f32]) -> f32 {
    data.iter().map(|x| x.abs()).sum()
}

fn native_l1_norm_f16(data: &[f16]) -> f16 {
    data.iter().map(|x| f16::from_f32(x.to_f32().abs())).sum()
}

fn native_l1_norm_u8(data: &[u8]) -> u32 {
    data.iter().map(|&x| x as u32).sum()
}

/// Native Vec implementation for L2 norm
fn native_l2_norm_f32(data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn native_l2_norm_f16(data: &[f16]) -> f16 {
    f16::from_f32(
        data.iter()
            .map(|x| x.to_f32() * x.to_f32())
            .sum::<f32>()
            .sqrt(),
    )
}

/// Helper function to create Candle L1 norm
fn candle_l1_norm(tensor: &CandleTensor) -> anyhow::Result<CandleTensor> {
    Ok(tensor.abs()?.sum_all()?)
}

/// Helper function to create Candle L2 norm
fn candle_l2_norm(tensor: &CandleTensor) -> anyhow::Result<CandleTensor> {
    Ok(tensor.sqr()?.sum_all()?.sqrt()?)
}

/// Benchmark L1 norm across different data types and dimensions
fn benchmark_norm_l1(c: &mut Criterion) {
    let mut group = c.benchmark_group("L1 Norm Benchmarks");

    // 1D L1 norm benchmarks
    for &size in SIZES_1D {
        // f32 benchmarks
        {
            let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 - 5.0).collect();
            let slsl_tensor = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
            let candle_tensor =
                CandleTensor::from_vec(data_f32.clone(), size, &Device::Cpu).unwrap();

            group.bench_function(format!("slsl_l1_1d_f32_{size}"), |b| {
                b.iter(|| {
                    let result = slsl_tensor.norm_l1(0).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("candle_l1_1d_f32_{size}"), |b| {
                b.iter(|| {
                    let result = candle_l1_norm(&candle_tensor).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("native_l1_1d_f32_{size}"), |b| {
                b.iter(|| {
                    let result = native_l1_norm_f32(&data_f32);
                    black_box(result);
                })
            });
        }

        // f16 benchmarks
        {
            let data_f16: Vec<f16> = (0..size)
                .map(|i| f16::from_f32((i as f32) * 0.1 - 5.0))
                .collect();
            let slsl_tensor = Tensor::from_vec(data_f16.clone(), [size]).unwrap();
            let candle_tensor = CandleTensor::from_vec(
                data_f16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>(),
                size,
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

            group.bench_function(format!("slsl_l1_1d_f16_{size}"), |b| {
                b.iter(|| {
                    let result = slsl_tensor.norm_l1(0).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("candle_l1_1d_f16_{size}"), |b| {
                b.iter(|| {
                    let result = candle_l1_norm(&candle_tensor).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("native_l1_1d_f16_{size}"), |b| {
                b.iter(|| {
                    let result = native_l1_norm_f16(&data_f16);
                    black_box(result);
                })
            });
        }

        // u8 benchmarks
        {
            let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let slsl_tensor = Tensor::from_vec(data_u8.clone(), [size]).unwrap();
            let candle_tensor = CandleTensor::from_vec(
                data_u8.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
                size,
                &Device::Cpu,
            )
            .unwrap();

            group.bench_function(format!("slsl_l1_1d_u8_{size}"), |b| {
                b.iter(|| {
                    let result = slsl_tensor.norm_l1(0).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("candle_l1_1d_u8_{size}"), |b| {
                b.iter(|| {
                    let result = candle_l1_norm(&candle_tensor).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("native_l1_1d_u8_{size}"), |b| {
                b.iter(|| {
                    let result = native_l1_norm_u8(&data_u8);
                    black_box(result);
                })
            });
        }
    }

    // 2D L1 norm benchmarks
    for &(rows, cols) in SIZES_2D {
        let size = rows * cols;

        // f32 benchmarks
        {
            let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 5.0).collect();
            let slsl_tensor = Tensor::from_vec(data_f32.clone(), [rows, cols]).unwrap();
            let candle_tensor =
                CandleTensor::from_vec(data_f32.clone(), (rows, cols), &Device::Cpu).unwrap();

            // Test along different dimensions
            for dim in 0..2 {
                group.bench_function(format!("slsl_l1_2d_f32_{rows}x{cols}_dim{dim}"), |b| {
                    b.iter(|| {
                        let result = slsl_tensor.norm_l1(dim).unwrap();
                        black_box(result);
                    })
                });

                group.bench_function(format!("candle_l1_2d_f32_{rows}x{cols}_dim{dim}"), |b| {
                    b.iter(|| {
                        let result = candle_tensor.abs().unwrap().sum(dim).unwrap();
                        black_box(result);
                    })
                });
            }
        }

        // f16 benchmarks (only dim 0 for performance)
        {
            let data_f16: Vec<f16> = (0..size)
                .map(|i| f16::from_f32((i as f32) * 0.01 - 5.0))
                .collect();
            let slsl_tensor = Tensor::from_vec(data_f16.clone(), [rows, cols]).unwrap();
            let candle_tensor = CandleTensor::from_vec(
                data_f16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>(),
                (rows, cols),
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

            let dim = 0;
            group.bench_function(format!("slsl_l1_2d_f16_{rows}x{cols}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = slsl_tensor.norm_l1(dim).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("candle_l1_2d_f16_{rows}x{cols}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = candle_tensor.abs().unwrap().sum(dim).unwrap();
                    black_box(result);
                })
            });
        }

        // u8 benchmarks (only dim 0 for performance)
        {
            let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let slsl_tensor = Tensor::from_vec(data_u8.clone(), [rows, cols]).unwrap();
            let candle_tensor = CandleTensor::from_vec(
                data_u8.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
                (rows, cols),
                &Device::Cpu,
            )
            .unwrap();

            let dim = 0;
            group.bench_function(format!("slsl_l1_2d_u8_{rows}x{cols}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = slsl_tensor.norm_l1(dim).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("candle_l1_2d_u8_{rows}x{cols}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = candle_tensor.abs().unwrap().sum(dim).unwrap();
                    black_box(result);
                })
            });
        }
    }

    // 3D L1 norm benchmarks
    for &(d1, d2, d3) in SIZES_3D {
        let size = d1 * d2 * d3;

        // f32 benchmarks (only dim 0 for performance)
        {
            let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001 - 5.0).collect();
            let slsl_tensor = Tensor::from_vec(data_f32.clone(), [d1, d2, d3]).unwrap();
            let candle_tensor =
                CandleTensor::from_vec(data_f32.clone(), (d1, d2, d3), &Device::Cpu).unwrap();

            let dim = 0;
            group.bench_function(format!("slsl_l1_3d_f32_{d1}x{d2}x{d3}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = slsl_tensor.norm_l1(dim).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("candle_l1_3d_f32_{d1}x{d2}x{d3}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = candle_tensor.abs().unwrap().sum(dim).unwrap();
                    black_box(result);
                })
            });
        }

        // f16 benchmarks (only dim 0 for performance)
        {
            let data_f16: Vec<f16> = (0..size)
                .map(|i| f16::from_f32((i as f32) * 0.001 - 5.0))
                .collect();
            let slsl_tensor = Tensor::from_vec(data_f16.clone(), [d1, d2, d3]).unwrap();
            let candle_tensor = CandleTensor::from_vec(
                data_f16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>(),
                (d1, d2, d3),
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

            let dim = 0;
            group.bench_function(format!("slsl_l1_3d_f16_{d1}x{d2}x{d3}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = slsl_tensor.norm_l1(dim).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("candle_l1_3d_f16_{d1}x{d2}x{d3}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = candle_tensor.abs().unwrap().sum(dim).unwrap();
                    black_box(result);
                })
            });
        }

        // u8 benchmarks (only dim 0 for performance)
        {
            let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let slsl_tensor = Tensor::from_vec(data_u8.clone(), [d1, d2, d3]).unwrap();
            let candle_tensor = CandleTensor::from_vec(
                data_u8.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
                (d1, d2, d3),
                &Device::Cpu,
            )
            .unwrap();

            let dim = 0;
            group.bench_function(format!("slsl_l1_3d_u8_{d1}x{d2}x{d3}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = slsl_tensor.norm_l1(dim).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("candle_l1_3d_u8_{d1}x{d2}x{d3}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = candle_tensor.abs().unwrap().sum(dim).unwrap();
                    black_box(result);
                })
            });
        }
    }

    group.finish();
}

/// Benchmark L2 norm across different data types and dimensions
fn benchmark_norm_l2(c: &mut Criterion) {
    let mut group = c.benchmark_group("L2 Norm Benchmarks");

    // 1D L2 norm benchmarks
    for &size in SIZES_1D {
        // f32 benchmarks
        {
            let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 - 5.0).collect();
            let slsl_tensor = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
            let candle_tensor =
                CandleTensor::from_vec(data_f32.clone(), size, &Device::Cpu).unwrap();

            group.bench_function(format!("slsl_l2_1d_f32_{size}"), |b| {
                b.iter(|| {
                    let result = slsl_tensor.norm_l2(0).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("candle_l2_1d_f32_{size}"), |b| {
                b.iter(|| {
                    let result = candle_l2_norm(&candle_tensor).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("native_l2_1d_f32_{size}"), |b| {
                b.iter(|| {
                    let result = native_l2_norm_f32(&data_f32);
                    black_box(result);
                })
            });
        }

        // f16 benchmarks
        {
            let data_f16: Vec<f16> = (0..size)
                .map(|i| f16::from_f32((i as f32) * 0.1 - 5.0))
                .collect();
            let slsl_tensor = Tensor::from_vec(data_f16.clone(), [size]).unwrap();
            let candle_tensor = CandleTensor::from_vec(
                data_f16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>(),
                size,
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

            group.bench_function(format!("slsl_l2_1d_f16_{size}"), |b| {
                b.iter(|| {
                    let result = slsl_tensor.norm_l2(0).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("candle_l2_1d_f16_{size}"), |b| {
                b.iter(|| {
                    let result = candle_l2_norm(&candle_tensor).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("native_l2_1d_f16_{size}"), |b| {
                b.iter(|| {
                    let result = native_l2_norm_f16(&data_f16);
                    black_box(result);
                })
            });
        }
    }

    // 2D L2 norm benchmarks
    for &(rows, cols) in SIZES_2D {
        let size = rows * cols;

        // f32 benchmarks
        {
            let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 5.0).collect();
            let slsl_tensor = Tensor::from_vec(data_f32.clone(), [rows, cols]).unwrap();
            let candle_tensor =
                CandleTensor::from_vec(data_f32.clone(), (rows, cols), &Device::Cpu).unwrap();

            // Test along different dimensions
            for dim in 0..2 {
                group.bench_function(format!("slsl_l2_2d_f32_{rows}x{cols}_dim{dim}"), |b| {
                    b.iter(|| {
                        let result = slsl_tensor.norm_l2(dim).unwrap();
                        black_box(result);
                    })
                });

                group.bench_function(format!("candle_l2_2d_f32_{rows}x{cols}_dim{dim}"), |b| {
                    b.iter(|| {
                        let result = candle_tensor
                            .sqr()
                            .unwrap()
                            .sum(dim)
                            .unwrap()
                            .sqrt()
                            .unwrap();
                        black_box(result);
                    })
                });
            }
        }

        // f16 benchmarks (only dim 0 for performance)
        {
            let data_f16: Vec<f16> = (0..size)
                .map(|i| f16::from_f32((i as f32) * 0.01 - 5.0))
                .collect();
            let slsl_tensor = Tensor::from_vec(data_f16.clone(), [rows, cols]).unwrap();
            let candle_tensor = CandleTensor::from_vec(
                data_f16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>(),
                (rows, cols),
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

            let dim = 0;
            group.bench_function(format!("slsl_l2_2d_f16_{rows}x{cols}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = slsl_tensor.norm_l2(dim).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("candle_l2_2d_f16_{rows}x{cols}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = candle_tensor
                        .sqr()
                        .unwrap()
                        .sum(dim)
                        .unwrap()
                        .sqrt()
                        .unwrap();
                    black_box(result);
                })
            });
        }
    }

    // 3D L2 norm benchmarks
    for &(d1, d2, d3) in SIZES_3D {
        let size = d1 * d2 * d3;

        // f32 benchmarks (only dim 0 for performance)
        {
            let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001 - 5.0).collect();
            let slsl_tensor = Tensor::from_vec(data_f32.clone(), [d1, d2, d3]).unwrap();
            let candle_tensor =
                CandleTensor::from_vec(data_f32.clone(), (d1, d2, d3), &Device::Cpu).unwrap();

            let dim = 0;
            group.bench_function(format!("slsl_l2_3d_f32_{d1}x{d2}x{d3}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = slsl_tensor.norm_l2(dim).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("candle_l2_3d_f32_{d1}x{d2}x{d3}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = candle_tensor
                        .sqr()
                        .unwrap()
                        .sum(dim)
                        .unwrap()
                        .sqrt()
                        .unwrap();
                    black_box(result);
                })
            });
        }

        // f16 benchmarks (only dim 0 for performance)
        {
            let data_f16: Vec<f16> = (0..size)
                .map(|i| f16::from_f32((i as f32) * 0.001 - 5.0))
                .collect();
            let slsl_tensor = Tensor::from_vec(data_f16.clone(), [d1, d2, d3]).unwrap();
            let candle_tensor = CandleTensor::from_vec(
                data_f16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>(),
                (d1, d2, d3),
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

            let dim = 0;
            group.bench_function(format!("slsl_l2_3d_f16_{d1}x{d2}x{d3}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = slsl_tensor.norm_l2(dim).unwrap();
                    black_box(result);
                })
            });

            group.bench_function(format!("candle_l2_3d_f16_{d1}x{d2}x{d3}_dim{dim}"), |b| {
                b.iter(|| {
                    let result = candle_tensor
                        .sqr()
                        .unwrap()
                        .sum(dim)
                        .unwrap()
                        .sqrt()
                        .unwrap();
                    black_box(result);
                })
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_norm_l1,
    benchmark_norm_l2
);
criterion_main!(benches);
