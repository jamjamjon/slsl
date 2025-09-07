#![allow(unused)]

use candle_core::{Device, Tensor as CandleTensor};
use criterion::{criterion_group, criterion_main, Criterion};
use slsl::*;
use std::hint::black_box;

// Define data sizes of different scales
const SMALL_SIZES: &[usize] = &[100, 500];
const MEDIUM_SIZES: &[usize] = &[1000, 2000];
const LARGE_SIZES: &[usize] = &[5000];

// Define matrix sizes of different dimensions
const SMALL_MATRICES: &[(usize, usize)] = &[(32, 32), (64, 64)];
const MEDIUM_MATRICES: &[(usize, usize)] = &[(128, 128)];
const LARGE_MATRICES: &[(usize, usize)] = &[(256, 256)];

// Define 3D tensor sizes
const SMALL_3D: &[(usize, usize, usize)] = &[(16, 16, 16), (32, 32, 32)];
const MEDIUM_3D: &[(usize, usize, usize)] = &[(64, 64, 64)];
const LARGE_3D: &[(usize, usize, usize)] = &[(128, 128, 128)];

// ========== 1D Tensor Benchmarks ==========

fn benchmark_1d_min_max(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("1D Min/Max Benchmarks");

    // Small scale 1D tensors
    for &size in SMALL_SIZES {
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
        group.bench_function(format!("slsl_f32_min_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.min(0).unwrap());
            })
        });
        group.bench_function(format!("candle_f32_min_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.min(0).unwrap());
            })
        });
        group.bench_function(format!("vec_f32_min_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(data_f32.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
            })
        });

        // u8 min
        group.bench_function(format!("slsl_u8_min_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_u8.min(0).unwrap());
            })
        });
        group.bench_function(format!("candle_u8_min_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_u8.min(0).unwrap());
            })
        });
        group.bench_function(format!("vec_u8_min_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(data_u8.iter().fold(u8::MAX, |a, &b| a.min(b)));
            })
        });

        // ===== Max operations =====
        // f32 max
        group.bench_function(format!("slsl_f32_max_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.max(0).unwrap());
            })
        });
        group.bench_function(format!("candle_f32_max_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.max(0).unwrap());
            })
        });
        group.bench_function(format!("vec_f32_max_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(data_f32.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
            })
        });

        // u8 max
        group.bench_function(format!("slsl_u8_max_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_u8.max(0).unwrap());
            })
        });
        group.bench_function(format!("candle_u8_max_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_u8.max(0).unwrap());
            })
        });
        group.bench_function(format!("vec_u8_max_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(data_u8.iter().fold(u8::MIN, |a, &b| a.max(b)));
            })
        });

        // ===== Argmin operations =====
        // f32 argmin
        group.bench_function(format!("slsl_f32_argmin_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.argmin(0).unwrap());
            })
        });
        group.bench_function(format!("candle_f32_argmin_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.argmin(0).unwrap());
            })
        });
        group.bench_function(format!("vec_f32_argmin_1d_small_{size}"), |bencher| {
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
        group.bench_function(format!("slsl_u8_argmin_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_u8.argmin(0).unwrap());
            })
        });
        group.bench_function(format!("candle_u8_argmin_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_u8.argmin(0).unwrap());
            })
        });
        group.bench_function(format!("vec_u8_argmin_1d_small_{size}"), |bencher| {
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
        group.bench_function(format!("slsl_f32_argmax_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.argmax(0).unwrap());
            })
        });
        group.bench_function(format!("candle_f32_argmax_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.argmax(0).unwrap());
            })
        });
        group.bench_function(format!("vec_f32_argmax_1d_small_{size}"), |bencher| {
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
        group.bench_function(format!("slsl_u8_argmax_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_u8.argmax(0).unwrap());
            })
        });
        group.bench_function(format!("candle_u8_argmax_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_u8.argmax(0).unwrap());
            })
        });
        group.bench_function(format!("vec_u8_argmax_1d_small_{size}"), |bencher| {
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

    // Medium scale 1D tensors
    for &size in MEDIUM_SIZES {
        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        // Min operations
        group.bench_function(format!("slsl_f32_min_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.min(0).unwrap());
            })
        });
        group.bench_function(format!("candle_f32_min_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.min(0).unwrap());
            })
        });
        group.bench_function(format!("vec_f32_min_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(data_f32.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
            })
        });

        // Max operations
        group.bench_function(format!("slsl_f32_max_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.max(0).unwrap());
            })
        });
        group.bench_function(format!("candle_f32_max_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.max(0).unwrap());
            })
        });
        group.bench_function(format!("vec_f32_max_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(data_f32.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
            })
        });
    }

    // Large scale 1D tensors
    for &size in LARGE_SIZES {
        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        // Min operations
        group.bench_function(format!("slsl_f32_min_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.min(0).unwrap());
            })
        });
        group.bench_function(format!("candle_f32_min_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.min(0).unwrap());
            })
        });
        group.bench_function(format!("vec_f32_min_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(data_f32.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
            })
        });

        // Max operations
        group.bench_function(format!("slsl_f32_max_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.max(0).unwrap());
            })
        });
        group.bench_function(format!("candle_f32_max_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.max(0).unwrap());
            })
        });
        group.bench_function(format!("vec_f32_max_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                black_box(data_f32.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
            })
        });
    }

    group.finish();
}

// ========== 2D Tensor Benchmarks ==========

fn benchmark_2d_min_max(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("2D Min/Max Benchmarks");

    // Small scale 2D tensors
    for &(rows, cols) in SMALL_MATRICES {
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

        // i64 data
        let data_i64: Vec<i64> = (0..size).map(|i| i as i64).collect();
        let slsl_tensor_i64 = Tensor::from_vec(data_i64.clone(), [rows, cols]).unwrap();
        let candle_tensor_i64 =
            CandleTensor::from_vec(data_i64.clone(), (rows, cols), &device).unwrap();

        // ===== Compute along rows (dim0) =====
        // f32 min dim0
        group.bench_function(
            format!("slsl_f32_min_2d_small_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(slsl_tensor_f32.min(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("candle_f32_min_2d_small_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(candle_tensor_f32.min(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("vec_f32_min_2d_small_{rows}x{cols}_dim0"),
            |bencher| {
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
            },
        );

        // u8 min dim0
        group.bench_function(
            format!("slsl_u8_min_2d_small_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(slsl_tensor_u8.min(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("candle_u8_min_2d_small_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(candle_tensor_u8.min(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("vec_u8_min_2d_small_{rows}x{cols}_dim0"),
            |bencher| {
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
            },
        );

        // i64 min dim0
        group.bench_function(
            format!("slsl_i64_min_2d_small_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(slsl_tensor_i64.min(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("candle_i64_min_2d_small_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(candle_tensor_i64.min(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("vec_i64_min_2d_small_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box({
                        let mut result = vec![i64::MAX; cols];
                        for col in 0..cols {
                            for row in 0..rows {
                                result[col] = result[col].min(data_i64[row * cols + col]);
                            }
                        }
                        result
                    });
                })
            },
        );

        // ===== Compute along columns (dim1) =====
        // f32 min dim1
        group.bench_function(
            format!("slsl_f32_min_2d_small_{rows}x{cols}_dim1"),
            |bencher| {
                bencher.iter(|| {
                    black_box(slsl_tensor_f32.min(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("candle_f32_min_2d_small_{rows}x{cols}_dim1"),
            |bencher| {
                bencher.iter(|| {
                    black_box(candle_tensor_f32.min(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("vec_f32_min_2d_small_{rows}x{cols}_dim1"),
            |bencher| {
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
            },
        );
    }

    // Medium scale 2D tensors
    for &(rows, cols) in MEDIUM_MATRICES {
        let size = rows * cols;

        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [rows, cols]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (rows, cols), &device).unwrap();

        // Min dim0
        group.bench_function(
            format!("slsl_f32_min_2d_medium_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(slsl_tensor_f32.min(1).unwrap());
                })
            },
        );
        group.bench_function(
            format!("candle_f32_min_2d_medium_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(candle_tensor_f32.min(1).unwrap());
                })
            },
        );
        group.bench_function(
            format!("vec_f32_min_2d_medium_{rows}x{cols}_dim0"),
            |bencher| {
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
            },
        );

        // Max dim0
        group.bench_function(
            format!("slsl_f32_max_2d_medium_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(slsl_tensor_f32.max(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("candle_f32_max_2d_medium_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(candle_tensor_f32.max(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("vec_f32_max_2d_medium_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box({
                        let mut result = vec![f32::NEG_INFINITY; cols];
                        for col in 0..cols {
                            for row in 0..rows {
                                result[col] = result[col].max(data_f32[row * cols + col]);
                            }
                        }
                        result
                    });
                })
            },
        );
    }

    // Large scale 2D tensors
    for &(rows, cols) in LARGE_MATRICES {
        let size = rows * cols;

        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [rows, cols]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (rows, cols), &device).unwrap();

        // Min dim0
        group.bench_function(
            format!("slsl_f32_min_2d_large_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(slsl_tensor_f32.min(1).unwrap());
                })
            },
        );
        group.bench_function(
            format!("candle_f32_min_2d_large_{rows}x{cols}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(candle_tensor_f32.min(1).unwrap());
                })
            },
        );
        group.bench_function(
            format!("vec_f32_min_2d_large_{rows}x{cols}_dim0"),
            |bencher| {
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
            },
        );
    }

    group.finish();
}

// ========== 3D Tensor Benchmarks ==========

fn benchmark_3d_min_max(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("3D Min/Max Benchmarks");

    // Small scale 3D tensors
    for &(d1, d2, d3) in SMALL_3D {
        let size = d1 * d2 * d3;

        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (d1, d2, d3), &device).unwrap();

        // ===== Compute along first dimension (dim0) =====
        // f32 min dim0
        group.bench_function(
            format!("slsl_f32_min_3d_small_{d1}x{d2}x{d3}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(slsl_tensor_f32.min(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("candle_f32_min_3d_small_{d1}x{d2}x{d3}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(candle_tensor_f32.min(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("vec_f32_min_3d_small_{d1}x{d2}x{d3}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box({
                        let mut result = vec![f32::INFINITY; d2 * d3];
                        for i2 in 0..d2 {
                            for i3 in 0..d3 {
                                for i1 in 0..d1 {
                                    result[i2 * d3 + i3] = result[i2 * d3 + i3]
                                        .min(data_f32[i1 * d2 * d3 + i2 * d3 + i3]);
                                }
                            }
                        }
                        result
                    });
                })
            },
        );
    }

    // Medium scale 3D tensors
    for &(d1, d2, d3) in MEDIUM_3D {
        let size = d1 * d2 * d3;

        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (d1, d2, d3), &device).unwrap();

        // Min dim0
        group.bench_function(
            format!("slsl_f32_min_3d_medium_{d1}x{d2}x{d3}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(slsl_tensor_f32.min(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("candle_f32_min_3d_medium_{d1}x{d2}x{d3}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(candle_tensor_f32.min(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("vec_f32_min_3d_medium_{d1}x{d2}x{d3}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box({
                        let mut result = vec![f32::INFINITY; d2 * d3];
                        for i2 in 0..d2 {
                            for i3 in 0..d3 {
                                for i1 in 0..d1 {
                                    result[i2 * d3 + i3] = result[i2 * d3 + i3]
                                        .min(data_f32[i1 * d2 * d3 + i2 * d3 + i3]);
                                }
                            }
                        }
                        result
                    });
                })
            },
        );
    }

    // Large scale 3D tensors
    for &(d1, d2, d3) in LARGE_3D {
        let size = d1 * d2 * d3;

        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (d1, d2, d3), &device).unwrap();

        // Min dim0
        group.bench_function(
            format!("slsl_f32_min_3d_large_{d1}x{d2}x{d3}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(slsl_tensor_f32.min(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("candle_f32_min_3d_large_{d1}x{d2}x{d3}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(candle_tensor_f32.min(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("vec_f32_min_3d_large_{d1}x{d2}x{d3}_dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box({
                        let mut result = vec![f32::INFINITY; d2 * d3];
                        for i2 in 0..d2 {
                            for i3 in 0..d3 {
                                for i1 in 0..d1 {
                                    result[i2 * d3 + i3] = result[i2 * d3 + i3]
                                        .min(data_f32[i1 * d2 * d3 + i2 * d3 + i3]);
                                }
                            }
                        }
                        result
                    });
                })
            },
        );
    }

    group.finish();
}

// ========== Main function ==========

criterion_group!(
    benches,
    benchmark_1d_min_max,
    benchmark_2d_min_max,
    benchmark_3d_min_max
);
criterion_main!(benches);
