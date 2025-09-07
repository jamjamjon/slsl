#![allow(unused)]

use candle_core::{Device, Tensor as CandleTensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use slsl::*;

// Define different data sizes
const SMALL_SIZES: &[usize] = &[100, 500];
const MEDIUM_SIZES: &[usize] = &[1000, 2000];
const LARGE_SIZES: &[usize] = &[5000, 10000];

// Define different matrix sizes
const SMALL_MATRICES: &[(usize, usize)] = &[(32, 32), (64, 64)];
const MEDIUM_MATRICES: &[(usize, usize)] = &[(128, 128), (256, 256)];
const LARGE_MATRICES: &[(usize, usize)] = &[(512, 512)];

// Define 3D tensor sizes
const SMALL_3D: &[(usize, usize, usize)] = &[(16, 16, 16), (32, 32, 32)];
const MEDIUM_3D: &[(usize, usize, usize)] = &[(64, 64, 64), (128, 128, 128)];
const LARGE_3D: &[(usize, usize, usize)] = &[(256, 256, 256)];

// ========== 1D Tensor Benchmarks ==========

fn benchmark_1d_mean_all(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("1D Mean All Benchmarks");

    // Small 1D tensors
    for &size in SMALL_SIZES {
        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        // u8 data
        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [size]).unwrap();
        let candle_tensor_u8 = CandleTensor::from_vec(data_u8.clone(), size, &device).unwrap();

        // f32 mean_all
        group.bench_function(format!("slsl_f32_mean_all_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.mean_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_f32_mean_all_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.mean_all().unwrap();
                black_box(result);
            })
        });

        // f32 sum_all / n approach
        group.bench_function(format!("slsl_f32_sum_div_n_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = slsl_tensor_f32.sum_all().unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });
        group.bench_function(format!("candle_f32_sum_div_n_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = candle_tensor_f32.sum_all().unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });

        // u8 mean_all
        group.bench_function(format!("slsl_u8_mean_all_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_u8.mean_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_u8_mean_all_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_u8.mean_all().unwrap();
                black_box(result);
            })
        });

        // u8 sum_all / n approach
        group.bench_function(format!("slsl_u8_sum_div_n_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = slsl_tensor_u8.sum_all().unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });
        group.bench_function(format!("candle_u8_sum_div_n_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = candle_tensor_u8.sum_all().unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });
    }

    // Medium 1D tensors
    for &size in MEDIUM_SIZES {
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [size]).unwrap();
        let candle_tensor_u8 = CandleTensor::from_vec(data_u8.clone(), size, &device).unwrap();

        group.bench_function(format!("slsl_f32_mean_all_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.mean_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_f32_mean_all_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.mean_all().unwrap();
                black_box(result);
            })
        });

        group.bench_function(format!("slsl_f32_sum_div_n_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = slsl_tensor_f32.sum_all().unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });
        group.bench_function(
            format!("candle_f32_sum_div_n_1d_medium_{size}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = candle_tensor_f32.sum_all().unwrap();
                    let result = sum / (size as f64);
                    black_box(result);
                })
            },
        );

        group.bench_function(format!("slsl_u8_mean_all_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_u8.mean_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_u8_mean_all_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_u8.mean_all().unwrap();
                black_box(result);
            })
        });

        group.bench_function(format!("slsl_u8_sum_div_n_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = slsl_tensor_u8.sum_all().unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });
        group.bench_function(format!("candle_u8_sum_div_n_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = candle_tensor_u8.sum_all().unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });
    }

    // Large 1D tensors
    for &size in LARGE_SIZES {
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [size]).unwrap();
        let candle_tensor_u8 = CandleTensor::from_vec(data_u8.clone(), size, &device).unwrap();

        group.bench_function(format!("slsl_f32_mean_all_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.mean_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_f32_mean_all_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.mean_all().unwrap();
                black_box(result);
            })
        });

        group.bench_function(format!("slsl_f32_sum_div_n_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = slsl_tensor_f32.sum_all().unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });
        group.bench_function(format!("candle_f32_sum_div_n_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = candle_tensor_f32.sum_all().unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });

        group.bench_function(format!("slsl_u8_mean_all_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_u8.mean_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_u8_mean_all_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_u8.mean_all().unwrap();
                black_box(result);
            })
        });

        group.bench_function(format!("slsl_u8_sum_div_n_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = slsl_tensor_u8.sum_all().unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });
        group.bench_function(format!("candle_u8_sum_div_n_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = candle_tensor_u8.sum_all().unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });
    }

    group.finish();
}

fn benchmark_1d_mean(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("1D Mean Benchmarks");

    // Small 1D tensors
    for &size in SMALL_SIZES {
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [size]).unwrap();
        let candle_tensor_u8 = CandleTensor::from_vec(data_u8.clone(), size, &device).unwrap();

        // f32 mean along dimension 0
        group.bench_function(format!("slsl_f32_mean_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.mean(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_f32_mean_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.mean(0).unwrap();
                black_box(result);
            })
        });

        // f32 sum / n approach
        group.bench_function(format!("slsl_f32_sum_div_n_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = slsl_tensor_f32.sum(0).unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });
        group.bench_function(format!("candle_f32_sum_div_n_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = candle_tensor_f32.sum(0).unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });

        // u8 mean along dimension 0
        group.bench_function(format!("slsl_u8_mean_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_u8.mean(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_u8_mean_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_u8.mean(0).unwrap();
                black_box(result);
            })
        });

        // u8 sum / n approach
        group.bench_function(format!("slsl_u8_sum_div_n_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = slsl_tensor_u8.sum(0).unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });
        group.bench_function(format!("candle_u8_sum_div_n_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = candle_tensor_u8.sum(0).unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });
    }

    // Medium and large scale tests similar...
    for &size in MEDIUM_SIZES {
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        group.bench_function(format!("slsl_f32_mean_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.mean(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_f32_mean_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.mean(0).unwrap();
                black_box(result);
            })
        });

        group.bench_function(format!("slsl_f32_sum_div_n_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let sum = slsl_tensor_f32.sum(0).unwrap();
                let result = sum / (size as f64);
                black_box(result);
            })
        });
        group.bench_function(
            format!("candle_f32_sum_div_n_1d_medium_{size}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = candle_tensor_f32.sum(0).unwrap();
                    let result = sum / (size as f64);
                    black_box(result);
                })
            },
        );
    }

    group.finish();
}

fn benchmark_1d_mean_keepdim(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("1D Mean Keepdim Benchmarks");

    for &size in SMALL_SIZES {
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [size]).unwrap();
        let candle_tensor_u8 = CandleTensor::from_vec(data_u8.clone(), size, &device).unwrap();

        // f32 mean_keepdim along dimension 0
        group.bench_function(
            format!("slsl_f32_mean_keepdim_1d_small_{size}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.mean_keepdim(0).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_mean_keepdim_1d_small_{size}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.mean_keepdim(0).unwrap();
                    black_box(result);
                })
            },
        );

        // u8 mean_keepdim along dimension 0
        group.bench_function(format!("slsl_u8_mean_keepdim_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_u8.mean_keepdim(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(
            format!("candle_u8_mean_keepdim_1d_small_{size}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.mean_keepdim(0).unwrap();
                    black_box(result);
                })
            },
        );
    }

    group.finish();
}

// ========== 2D Tensor Benchmarks ==========

fn benchmark_2d_mean_all(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("2D Mean All Benchmarks");

    // Small 2D tensors
    for &(rows, cols) in SMALL_MATRICES {
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

        // f32 mean_all
        group.bench_function(
            format!("slsl_f32_mean_all_2d_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.mean_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_mean_all_2d_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.mean_all().unwrap();
                    black_box(result);
                })
            },
        );

        // f32 sum_all / n approach
        group.bench_function(
            format!("slsl_f32_sum_div_n_2d_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = slsl_tensor_f32.sum_all().unwrap();
                    let result = sum / (size as f64);
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_div_n_2d_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = candle_tensor_f32.sum_all().unwrap();
                    let result = sum / (size as f64);
                    black_box(result);
                })
            },
        );

        // u8 mean_all
        group.bench_function(
            format!("slsl_u8_mean_all_2d_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.mean_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_mean_all_2d_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.mean_all().unwrap();
                    black_box(result);
                })
            },
        );

        // u8 sum_all / n approach
        group.bench_function(
            format!("slsl_u8_sum_div_n_2d_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = slsl_tensor_u8.sum_all().unwrap();
                    let result = sum / (size as f64);
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_div_n_2d_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = candle_tensor_u8.sum_all().unwrap();
                    let result = sum / (size as f64);
                    black_box(result);
                })
            },
        );
    }

    // Medium 2D tensors
    for &(rows, cols) in MEDIUM_MATRICES {
        let size = rows * cols;

        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [rows, cols]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (rows, cols), &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [rows, cols]).unwrap();
        let candle_tensor_u8 =
            CandleTensor::from_vec(data_u8.clone(), (rows, cols), &device).unwrap();

        group.bench_function(
            format!("slsl_f32_mean_all_2d_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.mean_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_mean_all_2d_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.mean_all().unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_f32_sum_div_n_2d_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = slsl_tensor_f32.sum_all().unwrap();
                    let result = sum / (size as f64);
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_div_n_2d_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = candle_tensor_f32.sum_all().unwrap();
                    let result = sum / (size as f64);
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_mean_all_2d_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.mean_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_mean_all_2d_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.mean_all().unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_sum_div_n_2d_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = slsl_tensor_u8.sum_all().unwrap();
                    let result = sum / (size as f64);
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_div_n_2d_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = candle_tensor_u8.sum_all().unwrap();
                    let result = sum / (size as f64);
                    black_box(result);
                })
            },
        );
    }

    // Large 2D tensors
    for &(rows, cols) in LARGE_MATRICES {
        let size = rows * cols;

        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [rows, cols]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (rows, cols), &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [rows, cols]).unwrap();
        let candle_tensor_u8 =
            CandleTensor::from_vec(data_u8.clone(), (rows, cols), &device).unwrap();

        group.bench_function(
            format!("slsl_f32_mean_all_2d_large_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.mean_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_mean_all_2d_large_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.mean_all().unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_f32_sum_div_n_2d_large_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = slsl_tensor_f32.sum_all().unwrap();
                    let result = sum / (size as f64);
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_div_n_2d_large_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = candle_tensor_f32.sum_all().unwrap();
                    let result = sum / (size as f64);
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_mean_all_2d_large_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.mean_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_mean_all_2d_large_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.mean_all().unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_sum_div_n_2d_large_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = slsl_tensor_u8.sum_all().unwrap();
                    let result = sum / (size as f64);
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_div_n_2d_large_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = candle_tensor_u8.sum_all().unwrap();
                    let result = sum / (size as f64);
                    black_box(result);
                })
            },
        );
    }

    group.finish();
}

fn benchmark_2d_mean(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("2D Mean Benchmarks");

    // Small 2D tensors
    for &(rows, cols) in SMALL_MATRICES {
        let size = rows * cols;

        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [rows, cols]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (rows, cols), &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [rows, cols]).unwrap();
        let candle_tensor_u8 =
            CandleTensor::from_vec(data_u8.clone(), (rows, cols), &device).unwrap();

        // f32 mean along dimension 0 (rows)
        group.bench_function(
            format!("slsl_f32_mean_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.mean(0).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_mean_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.mean(0).unwrap();
                    black_box(result);
                })
            },
        );

        // f32 sum / n approach along dimension 0
        group.bench_function(
            format!("slsl_f32_sum_div_n_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = slsl_tensor_f32.sum(0).unwrap();
                    let result = sum / (rows as f64);
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_div_n_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = candle_tensor_f32.sum(0).unwrap();
                    let result = sum / (rows as f64);
                    black_box(result);
                })
            },
        );

        // f32 mean along dimension 1 (cols)
        group.bench_function(
            format!("slsl_f32_mean_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.mean(1).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_mean_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.mean(1).unwrap();
                    black_box(result);
                })
            },
        );

        // f32 sum / n approach along dimension 1
        group.bench_function(
            format!("slsl_f32_sum_div_n_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = slsl_tensor_f32.sum(1).unwrap();
                    let result = sum / (cols as f64);
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_div_n_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = candle_tensor_f32.sum(1).unwrap();
                    let result = sum / (cols as f64);
                    black_box(result);
                })
            },
        );

        // u8 mean along both dimensions
        group.bench_function(
            format!("slsl_u8_mean_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.mean(0).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_mean_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.mean(0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_sum_div_n_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = slsl_tensor_u8.sum(0).unwrap();
                    let result = sum / (rows as f64);
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_div_n_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = candle_tensor_u8.sum(0).unwrap();
                    let result = sum / (rows as f64);
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_mean_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.mean(1).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_mean_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.mean(1).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_sum_div_n_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = slsl_tensor_u8.sum(1).unwrap();
                    let result = sum / (cols as f64);
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_div_n_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = candle_tensor_u8.sum(1).unwrap();
                    let result = sum / (cols as f64);
                    black_box(result);
                })
            },
        );
    }

    // Medium and large scale tests similar...
    for &(rows, cols) in MEDIUM_MATRICES {
        let size = rows * cols;

        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [rows, cols]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (rows, cols), &device).unwrap();

        group.bench_function(
            format!("slsl_f32_mean_2d_dim0_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.mean(0).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_mean_2d_dim0_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.mean(0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_f32_sum_div_n_2d_dim0_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = slsl_tensor_f32.sum(0).unwrap();
                    let result = sum / (rows as f64);
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_div_n_2d_dim0_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = candle_tensor_f32.sum(0).unwrap();
                    let result = sum / (rows as f64);
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_f32_mean_2d_dim1_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.mean(1).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_mean_2d_dim1_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.mean(1).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_f32_sum_div_n_2d_dim1_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = slsl_tensor_f32.sum(0).unwrap();
                    let result = sum / (cols as f64);
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_div_n_2d_dim1_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let sum = candle_tensor_f32.sum(0).unwrap();
                    let result = sum / (cols as f64);
                    black_box(result);
                })
            },
        );
    }

    group.finish();
}

// ========== Main Benchmark Registration ==========

criterion_group!(
    benches,
    // 1D benchmarks
    benchmark_1d_mean_all,
    benchmark_1d_mean,
    benchmark_1d_mean_keepdim,
    // 2D benchmarks
    benchmark_2d_mean_all,
    benchmark_2d_mean,
);

criterion_main!(benches);
