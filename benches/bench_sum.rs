#![allow(unused)]

use candle_core::{Device, Tensor as CandleTensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use slsl::*;

// Define data sizes of different scales
const SMALL_SIZES: &[usize] = &[100, 500];
const MEDIUM_SIZES: &[usize] = &[1000, 2000];
const LARGE_SIZES: &[usize] = &[5000, 10000];

// Define matrix sizes of different dimensions
const SMALL_MATRICES: &[(usize, usize)] = &[(32, 32), (64, 64)];
const MEDIUM_MATRICES: &[(usize, usize)] = &[(128, 128), (256, 256)];
const LARGE_MATRICES: &[(usize, usize)] = &[(512, 512)];

// Define 3D tensor sizes
const SMALL_3D: &[(usize, usize, usize)] = &[(16, 16, 16), (32, 32, 32)];
const MEDIUM_3D: &[(usize, usize, usize)] = &[(64, 64, 64), (128, 128, 128)];
const LARGE_3D: &[(usize, usize, usize)] = &[(256, 256, 256)];

// ========== 1D Tensor Benchmarks ==========

fn benchmark_1d_sum_all(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("1D Sum All Benchmarks");

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

        // f32 sum_all
        group.bench_function(format!("slsl_f32_sum_all_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_f32_sum_all_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum_all().unwrap();
                black_box(result);
            })
        });

        // u8 sum_all
        group.bench_function(format!("slsl_u8_sum_all_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_u8.sum_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_u8_sum_all_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_u8.sum_all().unwrap();
                black_box(result);
            })
        });
    }

    // Medium scale 1D tensors
    for &size in MEDIUM_SIZES {
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [size]).unwrap();
        let candle_tensor_u8 = CandleTensor::from_vec(data_u8.clone(), size, &device).unwrap();

        group.bench_function(format!("slsl_f32_sum_all_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_f32_sum_all_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum_all().unwrap();
                black_box(result);
            })
        });

        group.bench_function(format!("slsl_u8_sum_all_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_u8.sum_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_u8_sum_all_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_u8.sum_all().unwrap();
                black_box(result);
            })
        });
    }

    // Large scale 1D tensors
    for &size in LARGE_SIZES {
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [size]).unwrap();
        let candle_tensor_u8 = CandleTensor::from_vec(data_u8.clone(), size, &device).unwrap();

        group.bench_function(format!("slsl_f32_sum_all_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_f32_sum_all_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum_all().unwrap();
                black_box(result);
            })
        });

        group.bench_function(format!("slsl_u8_sum_all_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_u8.sum_all().unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_u8_sum_all_1d_large_{size}"), |bencher| {
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
    let mut group = c.benchmark_group("1D Sum Benchmarks");

    // Small scale 1D tensors
    for &size in SMALL_SIZES {
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [size]).unwrap();
        let candle_tensor_u8 = CandleTensor::from_vec(data_u8.clone(), size, &device).unwrap();

        // f32 sum along dimension 0
        group.bench_function(format!("slsl_f32_sum_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_f32_sum_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_f32.sum(0).unwrap();
                black_box(result);
            })
        });

        // u8 sum along dimension 0
        group.bench_function(format!("slsl_u8_sum_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_u8.sum(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_u8_sum_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor_u8.sum(0).unwrap();
                black_box(result);
            })
        });
    }

    // Medium and large scale tests are similar...
    for &size in MEDIUM_SIZES {
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        group.bench_function(format!("slsl_f32_sum_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(format!("candle_f32_sum_1d_medium_{size}"), |bencher| {
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
    let mut group = c.benchmark_group("1D Sum Keepdim Benchmarks");

    for &size in SMALL_SIZES {
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [size]).unwrap();
        let candle_tensor_u8 = CandleTensor::from_vec(data_u8.clone(), size, &device).unwrap();

        // f32 sum_keepdim along dimension 0
        group.bench_function(format!("slsl_f32_sum_keepdim_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_f32.sum_keepdim(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(
            format!("candle_f32_sum_keepdim_1d_small_{size}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum_keepdim(0).unwrap();
                    black_box(result);
                })
            },
        );

        // u8 sum_keepdim along dimension 0
        group.bench_function(format!("slsl_u8_sum_keepdim_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor_u8.sum_keepdim(0).unwrap();
                black_box(result);
            })
        });
        group.bench_function(
            format!("candle_u8_sum_keepdim_1d_small_{size}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum_keepdim(0).unwrap();
                    black_box(result);
                })
            },
        );
    }

    group.finish();
}

// ========== 2D Tensor Benchmarks ==========

fn benchmark_2d_sum_all(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("2D Sum All Benchmarks");

    // Small scale 2D tensors
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

        // f32 sum_all
        group.bench_function(
            format!("slsl_f32_sum_all_2d_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_all_2d_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum_all().unwrap();
                    black_box(result);
                })
            },
        );

        // u8 sum_all
        group.bench_function(
            format!("slsl_u8_sum_all_2d_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_all_2d_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
    }

    // Medium scale 2D tensors
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
            format!("slsl_f32_sum_all_2d_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_all_2d_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum_all().unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_sum_all_2d_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_all_2d_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
    }

    // Large scale 2D tensors
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
            format!("slsl_f32_sum_all_2d_large_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_all_2d_large_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum_all().unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_sum_all_2d_large_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_all_2d_large_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
    }

    group.finish();
}

fn benchmark_2d_sum(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("2D Sum Benchmarks");

    // Small scale 2D tensors
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

        // f32 sum along dimension 0 (rows)
        group.bench_function(
            format!("slsl_f32_sum_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum(0).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum(0).unwrap();
                    black_box(result);
                })
            },
        );

        // f32 sum along dimension 1 (cols)
        group.bench_function(
            format!("slsl_f32_sum_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum(1).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum(1).unwrap();
                    black_box(result);
                })
            },
        );

        // u8 sum along both dimensions
        group.bench_function(
            format!("slsl_u8_sum_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.sum(0).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum(0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_sum_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.sum(1).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum(1).unwrap();
                    black_box(result);
                })
            },
        );
    }

    // Medium and large scale tests are similar...
    for &(rows, cols) in MEDIUM_MATRICES {
        let size = rows * cols;

        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [rows, cols]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (rows, cols), &device).unwrap();

        group.bench_function(
            format!("slsl_f32_sum_2d_dim0_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum(0).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_2d_dim0_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum(0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_f32_sum_2d_dim1_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum(1).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_2d_dim1_medium_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum(1).unwrap();
                    black_box(result);
                })
            },
        );
    }

    group.finish();
}

fn benchmark_2d_sum_keepdim(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("2D Sum Keepdim Benchmarks");

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

        // f32 sum_keepdim along dimension 0
        group.bench_function(
            format!("slsl_f32_sum_keepdim_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum_keepdim(0).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_keepdim_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum_keepdim(0).unwrap();
                    black_box(result);
                })
            },
        );

        // f32 sum_keepdim along dimension 1
        group.bench_function(
            format!("slsl_f32_sum_keepdim_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum_keepdim(1).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_keepdim_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum_keepdim(1).unwrap();
                    black_box(result);
                })
            },
        );

        // u8 sum_keepdim along both dimensions
        group.bench_function(
            format!("slsl_u8_sum_keepdim_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.sum_keepdim(0).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_keepdim_2d_dim0_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum_keepdim(0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_sum_keepdim_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.sum_keepdim(1).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_keepdim_2d_dim1_small_{rows}x{cols}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum_keepdim(1).unwrap();
                    black_box(result);
                })
            },
        );
    }

    group.finish();
}

// ========== 3D Tensor Benchmarks ==========

fn benchmark_3d_sum_all(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("3D Sum All Benchmarks");

    // Small scale 3D tensors
    for &(d1, d2, d3) in SMALL_3D {
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
        group.bench_function(
            format!("slsl_f32_sum_all_3d_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_all_3d_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum_all().unwrap();
                    black_box(result);
                })
            },
        );

        // u8 sum_all
        group.bench_function(
            format!("slsl_u8_sum_all_3d_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_all_3d_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
    }

    // Medium scale 3D tensors
    for &(d1, d2, d3) in MEDIUM_3D {
        let size = d1 * d2 * d3;

        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (d1, d2, d3), &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_u8 =
            CandleTensor::from_vec(data_u8.clone(), (d1, d2, d3), &device).unwrap();

        group.bench_function(
            format!("slsl_f32_sum_all_3d_medium_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_all_3d_medium_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum_all().unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_sum_all_3d_medium_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_all_3d_medium_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
    }

    // Large scale 3D tensors
    for &(d1, d2, d3) in LARGE_3D {
        let size = d1 * d2 * d3;

        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (d1, d2, d3), &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_u8 =
            CandleTensor::from_vec(data_u8.clone(), (d1, d2, d3), &device).unwrap();

        group.bench_function(
            format!("slsl_f32_sum_all_3d_large_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_all_3d_large_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum_all().unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_sum_all_3d_large_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_all_3d_large_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum_all().unwrap();
                    black_box(result);
                })
            },
        );
    }

    group.finish();
}

fn benchmark_3d_sum(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("3D Sum Benchmarks");

    // Small scale 3D tensors - test multiple dimensions
    for &(d1, d2, d3) in SMALL_3D {
        let size = d1 * d2 * d3;

        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (d1, d2, d3), &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_u8 =
            CandleTensor::from_vec(data_u8.clone(), (d1, d2, d3), &device).unwrap();

        // f32 sum along dimension 0
        group.bench_function(
            format!("slsl_f32_sum_3d_dim0_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum(0).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_3d_dim0_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum(0).unwrap();
                    black_box(result);
                })
            },
        );

        // f32 sum along dimension 1
        group.bench_function(
            format!("slsl_f32_sum_3d_dim1_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum(1).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_3d_dim1_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum(1).unwrap();
                    black_box(result);
                })
            },
        );

        // f32 sum along dimension 2
        group.bench_function(
            format!("slsl_f32_sum_3d_dim2_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum(2).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_3d_dim2_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum(2).unwrap();
                    black_box(result);
                })
            },
        );

        // f32 sum along dimensions 0 and 1
        group.bench_function(
            format!("slsl_f32_sum_3d_dim01_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum([0, 1]).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_3d_dim01_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum([0, 1]).unwrap();
                    black_box(result);
                })
            },
        );

        // f32 sum along dimensions 0 and 2
        group.bench_function(
            format!("slsl_f32_sum_3d_dim02_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum([0, 2]).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_3d_dim02_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum([0, 2]).unwrap();
                    black_box(result);
                })
            },
        );

        // f32 sum along dimensions 1 and 2
        group.bench_function(
            format!("slsl_f32_sum_3d_dim12_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum([1, 2]).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_3d_dim12_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum([1, 2]).unwrap();
                    black_box(result);
                })
            },
        );

        // u8 sum along multiple dimensions
        group.bench_function(
            format!("slsl_u8_sum_3d_dim0_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.sum(0).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_3d_dim0_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum(0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_sum_3d_dim01_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.sum([0, 1]).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_3d_dim01_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum([0, 1]).unwrap();
                    black_box(result);
                })
            },
        );
    }

    // Medium scale 3D tensors
    for &(d1, d2, d3) in MEDIUM_3D {
        let size = d1 * d2 * d3;

        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (d1, d2, d3), &device).unwrap();

        // Test single dimensions
        group.bench_function(
            format!("slsl_f32_sum_3d_dim0_medium_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum(0).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_3d_dim0_medium_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum(0).unwrap();
                    black_box(result);
                })
            },
        );

        // Test multiple dimensions
        group.bench_function(
            format!("slsl_f32_sum_3d_dim01_medium_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum([0, 1]).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_3d_dim01_medium_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum([0, 1]).unwrap();
                    black_box(result);
                })
            },
        );
    }

    group.finish();
}

fn benchmark_3d_sum_keepdim(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("3D Sum Keepdim Benchmarks");

    for &(d1, d2, d3) in SMALL_3D {
        let size = d1 * d2 * d3;

        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_f32 =
            CandleTensor::from_vec(data_f32.clone(), (d1, d2, d3), &device).unwrap();

        let data_u8: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let slsl_tensor_u8 = Tensor::from_vec(data_u8.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor_u8 =
            CandleTensor::from_vec(data_u8.clone(), (d1, d2, d3), &device).unwrap();

        // f32 sum_keepdim along dimension 0
        group.bench_function(
            format!("slsl_f32_sum_keepdim_3d_dim0_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum_keepdim(0).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_keepdim_3d_dim0_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum_keepdim(0).unwrap();
                    black_box(result);
                })
            },
        );

        // f32 sum_keepdim along dimension 1
        group.bench_function(
            format!("slsl_f32_sum_keepdim_3d_dim1_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum_keepdim(1).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_keepdim_3d_dim1_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum_keepdim(1).unwrap();
                    black_box(result);
                })
            },
        );

        // f32 sum_keepdim along dimension 2
        group.bench_function(
            format!("slsl_f32_sum_keepdim_3d_dim2_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum_keepdim(2).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_keepdim_3d_dim2_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum_keepdim(2).unwrap();
                    black_box(result);
                })
            },
        );

        // f32 sum_keepdim along dimensions 0 and 1
        group.bench_function(
            format!("slsl_f32_sum_keepdim_3d_dim01_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_f32.sum_keepdim([0, 1]).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_f32_sum_keepdim_3d_dim01_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_f32.sum_keepdim([0, 1]).unwrap();
                    black_box(result);
                })
            },
        );

        // u8 sum_keepdim along multiple dimensions
        group.bench_function(
            format!("slsl_u8_sum_keepdim_3d_dim0_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.sum_keepdim(0).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_keepdim_3d_dim0_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum_keepdim(0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("slsl_u8_sum_keepdim_3d_dim01_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor_u8.sum_keepdim([0, 1]).unwrap();
                    black_box(result);
                })
            },
        );
        group.bench_function(
            format!("candle_u8_sum_keepdim_3d_dim01_small_{d1}x{d2}x{d3}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor_u8.sum_keepdim([0, 1]).unwrap();
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
    benchmark_1d_sum_all,
    benchmark_1d_sum,
    benchmark_1d_sum_keepdim,
    // 2D benchmarks
    benchmark_2d_sum_all,
    benchmark_2d_sum,
    benchmark_2d_sum_keepdim,
    // 3D benchmarks
    benchmark_3d_sum_all,
    benchmark_3d_sum,
    benchmark_3d_sum_keepdim,
);

criterion_main!(benches);
