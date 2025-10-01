#![allow(unused)]

use candle_core::{Device, Tensor as CandleTensor};
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2};
use slsl::s;
use slsl::*;
use std::hint::black_box;

// 1D
const SIZES_1D: &[usize] = &[
    10, 64, 256, 376, 512, 1024, 1344, 2048, 4096, 8192, 10240, 25600, 51200, 102400,
];

// 2D
const SIZES_2D: &[(usize, usize)] = &[
    (64, 64),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
];

fn benchmark_1d_min_max(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("1D Min/Max ");

    for &size in SIZES_1D {
        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let candle_tensor_f32 = CandleTensor::from_vec(data_f32.clone(), size, &device).unwrap();
        let ndarray_tensor_f32 = Array1::from_shape_vec([size], data_f32.clone()).unwrap();

        // ===== Min operations =====
        group.bench_function(format!("slsl/min_tensor/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.min(0).unwrap());
            })
        });
        group.bench_function(format!("slsl/iter_min_value/{size}"), |bencher| {
            bencher.iter(|| {
                let m = slsl_tensor_f32
                    .iter::<f32>()
                    .fold(f32::INFINITY, |a, &b| a.min(b));
                black_box(m)
            })
        });
        group.bench_function(format!("candle/min_tensor/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.min(0).unwrap());
            })
        });
        group.bench_function(format!("vec/min_value/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(data_f32.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
            })
        });
        group.bench_function(format!("ndarray/min_value/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(
                    ndarray_tensor_f32
                        .iter()
                        .fold(f32::INFINITY, |a, &b| a.min(b)),
                );
            })
        });

        // ===== Max operations =====
        group.bench_function(format!("slsl/max_tensor/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.max(0).unwrap());
            })
        });
        group.bench_function(format!("slsl/iter_max_value/{size}"), |bencher| {
            bencher.iter(|| {
                let m = slsl_tensor_f32
                    .iter::<f32>()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                black_box(m)
            })
        });
        group.bench_function(format!("candle/max_tensor/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.max(0).unwrap());
            })
        });
        group.bench_function(format!("vec/max_value/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(data_f32.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
            })
        });
        group.bench_function(format!("ndarray/max_value/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(
                    ndarray_tensor_f32
                        .iter()
                        .fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                );
            })
        });

        // ===== Max_argmax operations =====
        group.bench_function(format!("slsl/max_argmax_tensor/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.max_argmax(0).unwrap());
            })
        });
        group.bench_function(format!("slsl/iter_max_argmax_value/{size}"), |bencher| {
            bencher.iter(|| {
                let (idx, val) = slsl_tensor_f32.iter::<f32>().enumerate().fold(
                    (0, f32::NEG_INFINITY),
                    |(i, a), (j, b)| if a >= *b { (i, a) } else { (j, *b) },
                );
                black_box((idx, val));
            })
        });
        group.bench_function(format!("candle/max_argmax_tensor/{size}"), |bencher| {
            bencher.iter(|| {
                black_box((
                    candle_tensor_f32.max(0).unwrap(),
                    candle_tensor_f32.argmax(0).unwrap(),
                ));
            })
        });
        group.bench_function(format!("vec/max_argmax_value/{size}"), |bencher| {
            bencher.iter(|| {
                let (idx, val) =
                    data_f32
                        .iter()
                        .enumerate()
                        .fold((0, f32::NEG_INFINITY), |(i, a), (j, b)| {
                            if a >= *b {
                                (i, a)
                            } else {
                                (j, *b)
                            }
                        });
                black_box((idx, val));
            })
        });
        group.bench_function(format!("ndarray/max_argmax_value/{size}"), |bencher| {
            bencher.iter(|| {
                let (idx, val) = ndarray_tensor_f32.iter().enumerate().fold(
                    (0, f32::NEG_INFINITY),
                    |(i, a), (j, b)| if a >= *b { (i, a) } else { (j, *b) },
                );
                black_box((idx, val));
            })
        });

        // ===== Argmin_argmax operations =====
        group.bench_function(format!("slsl/argmin_argmax_tensor/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.argmin_argmax(0).unwrap());
            })
        });
        group.bench_function(format!("slsl/iter_argmin_argmax_value/{size}"), |bencher| {
            bencher.iter(|| {
                let (min_idx, min_val) = slsl_tensor_f32.iter::<f32>().enumerate().fold(
                    (0, f32::INFINITY),
                    |(i, a), (j, b)| if a <= *b { (i, a) } else { (j, *b) },
                );
                let (max_idx, max_val) = slsl_tensor_f32.iter::<f32>().enumerate().fold(
                    (0, f32::NEG_INFINITY),
                    |(i, a), (j, b)| if a >= *b { (i, a) } else { (j, *b) },
                );
                black_box((min_idx, min_val, max_idx, max_val));
            })
        });
        group.bench_function(format!("candle/argmin_argmax_tensor/{size}"), |bencher| {
            bencher.iter(|| {
                black_box((
                    candle_tensor_f32.argmin(0).unwrap(),
                    candle_tensor_f32.argmax(0).unwrap(),
                ));
            })
        });
        group.bench_function(format!("vec/argmin_argmax_value/{size}"), |bencher| {
            bencher.iter(|| {
                let (min_idx, min_val) =
                    data_f32
                        .iter()
                        .enumerate()
                        .fold(
                            (0, f32::INFINITY),
                            |(i, a), (j, b)| if a <= *b { (i, a) } else { (j, *b) },
                        );
                let (max_idx, max_val) =
                    data_f32
                        .iter()
                        .enumerate()
                        .fold((0, f32::NEG_INFINITY), |(i, a), (j, b)| {
                            if a >= *b {
                                (i, a)
                            } else {
                                (j, *b)
                            }
                        });
                black_box((min_idx, min_val, max_idx, max_val));
            })
        });
        group.bench_function(format!("ndarray/argmin_argmax_value/{size}"), |bencher| {
            bencher.iter(|| {
                let (min_idx, min_val) = ndarray_tensor_f32.iter().enumerate().fold(
                    (0, f32::INFINITY),
                    |(i, a), (j, b)| if a <= *b { (i, a) } else { (j, *b) },
                );
                let (max_idx, max_val) = ndarray_tensor_f32.iter().enumerate().fold(
                    (0, f32::NEG_INFINITY),
                    |(i, a), (j, b)| if a >= *b { (i, a) } else { (j, *b) },
                );
                black_box((min_idx, min_val, max_idx, max_val));
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
        let ndarray_tensor_f32 = Array2::from_shape_vec((rows, cols), data_f32.clone()).unwrap();

        // ===== Min operations =====
        group.bench_function(format!("slsl/min_tensor/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.min(0).unwrap());
            })
        });
        group.bench_function(
            format!("slsl/iter_min_value/{rows}x{cols}/dim0"),
            |bencher| {
                bencher.iter(|| {
                    let mut result = vec![f32::INFINITY; cols];
                    for (c, result_item) in result.iter_mut().enumerate().take(cols) {
                        let slice = slsl_tensor_f32.slice(s![.., c]);
                        let m = slice.iter::<f32>().fold(f32::INFINITY, |a, &b| a.min(b));
                        *result_item = m;
                    }
                    black_box(result)
                })
            },
        );
        group.bench_function(format!("candle/min_tensor/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.min(0).unwrap());
            })
        });
        group.bench_function(format!("vec/min_value/{rows}x{cols}/dim0"), |bencher| {
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
        group.bench_function(format!("ndarray/min_value/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                black_box({
                    let mut result = vec![f32::INFINITY; cols];
                    for col in 0..cols {
                        for row in 0..rows {
                            result[col] = result[col].min(ndarray_tensor_f32[[row, col]]);
                        }
                    }
                    result
                });
            })
        });

        // ===== Max operations =====
        group.bench_function(format!("slsl/max_tensor/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                black_box(slsl_tensor_f32.max(0).unwrap());
            })
        });
        group.bench_function(
            format!("slsl/iter_max_value/{rows}x{cols}/dim0"),
            |bencher| {
                bencher.iter(|| {
                    let mut result = vec![f32::NEG_INFINITY; cols];
                    for (c, result_item) in result.iter_mut().enumerate().take(cols) {
                        let slice = slsl_tensor_f32.slice(s![.., c]);
                        let m = slice
                            .iter::<f32>()
                            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        *result_item = m;
                    }
                    black_box(result)
                })
            },
        );
        group.bench_function(format!("candle/max_tensor/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                black_box(candle_tensor_f32.max(0).unwrap());
            })
        });
        group.bench_function(format!("vec/max_value/{rows}x{cols}/dim0"), |bencher| {
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
        });
        group.bench_function(format!("ndarray/max_value/{rows}x{cols}/dim0"), |bencher| {
            bencher.iter(|| {
                black_box({
                    let mut result = vec![f32::NEG_INFINITY; cols];
                    for col in 0..cols {
                        for row in 0..rows {
                            result[col] = result[col].max(ndarray_tensor_f32[[row, col]]);
                        }
                    }
                    result
                });
            })
        });

        // ===== Max_argmax operations =====
        group.bench_function(
            format!("slsl/max_argmax_tensor/{rows}x{cols}/dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(slsl_tensor_f32.max_argmax(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("slsl/iter_max_argmax_value/{rows}x{cols}/dim0"),
            |bencher| {
                bencher.iter(|| {
                    let mut result = vec![(0usize, f32::NEG_INFINITY); cols];
                    for (c, result_item) in result.iter_mut().enumerate().take(cols) {
                        let slice = slsl_tensor_f32.slice(s![.., c]);
                        let (idx, val) = slice.iter::<f32>().enumerate().fold(
                            (0, f32::NEG_INFINITY),
                            |(i, a), (j, b)| if a >= *b { (i, a) } else { (j, *b) },
                        );
                        *result_item = (idx, val);
                    }
                    black_box(result)
                })
            },
        );
        group.bench_function(
            format!("candle/max_argmax_tensor/{rows}x{cols}/dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box((
                        candle_tensor_f32.max(0).unwrap(),
                        candle_tensor_f32.argmax(0).unwrap(),
                    ));
                })
            },
        );
        group.bench_function(
            format!("vec/max_argmax_value/{rows}x{cols}/dim0"),
            |bencher| {
                bencher.iter(|| {
                    let mut result = vec![(0usize, f32::NEG_INFINITY); cols];
                    for col in 0..cols {
                        for row in 0..rows {
                            let val = data_f32[row * cols + col];
                            if val > result[col].1 {
                                result[col] = (row, val);
                            }
                        }
                    }
                    black_box(result)
                })
            },
        );
        group.bench_function(
            format!("ndarray/max_argmax_value/{rows}x{cols}/dim0"),
            |bencher| {
                bencher.iter(|| {
                    let mut result = vec![(0usize, f32::NEG_INFINITY); cols];
                    for col in 0..cols {
                        for row in 0..rows {
                            let val = ndarray_tensor_f32[[row, col]];
                            if val > result[col].1 {
                                result[col] = (row, val);
                            }
                        }
                    }
                    black_box(result)
                })
            },
        );

        // ===== Argmin_argmax operations =====
        group.bench_function(
            format!("slsl/argmin_argmax_tensor/{rows}x{cols}/dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box(slsl_tensor_f32.argmin_argmax(0).unwrap());
                })
            },
        );
        group.bench_function(
            format!("slsl/iter_argmin_argmax_value/{rows}x{cols}/dim0"),
            |bencher| {
                bencher.iter(|| {
                    let mut result = vec![(0usize, f32::INFINITY, 0usize, f32::NEG_INFINITY); cols];
                    for (c, result_item) in result.iter_mut().enumerate().take(cols) {
                        let slice = slsl_tensor_f32.slice(s![.., c]);
                        let (min_idx, min_val) = slice.iter::<f32>().enumerate().fold(
                            (0, f32::INFINITY),
                            |(i, a), (j, b)| if a <= *b { (i, a) } else { (j, *b) },
                        );
                        let (max_idx, max_val) = slice.iter::<f32>().enumerate().fold(
                            (0, f32::NEG_INFINITY),
                            |(i, a), (j, b)| if a >= *b { (i, a) } else { (j, *b) },
                        );
                        *result_item = (min_idx, min_val, max_idx, max_val);
                    }
                    black_box(result)
                })
            },
        );
        group.bench_function(
            format!("candle/argmin_argmax_tensor/{rows}x{cols}/dim0"),
            |bencher| {
                bencher.iter(|| {
                    black_box((
                        candle_tensor_f32.argmin(0).unwrap(),
                        candle_tensor_f32.argmax(0).unwrap(),
                    ));
                })
            },
        );
        group.bench_function(
            format!("vec/argmin_argmax_value/{rows}x{cols}/dim0"),
            |bencher| {
                bencher.iter(|| {
                    let mut result = vec![(0usize, f32::INFINITY, 0usize, f32::NEG_INFINITY); cols];
                    for col in 0..cols {
                        for row in 0..rows {
                            let val = data_f32[row * cols + col];
                            if val < result[col].1 {
                                result[col] = (row, val, result[col].2, result[col].3);
                            }
                            if val > result[col].3 {
                                result[col] = (result[col].0, result[col].1, row, val);
                            }
                        }
                    }
                    black_box(result)
                })
            },
        );
        group.bench_function(
            format!("ndarray/argmin_argmax_value/{rows}x{cols}/dim0"),
            |bencher| {
                bencher.iter(|| {
                    let mut result = vec![(0usize, f32::INFINITY, 0usize, f32::NEG_INFINITY); cols];
                    for col in 0..cols {
                        for row in 0..rows {
                            let val = ndarray_tensor_f32[[row, col]];
                            if val < result[col].1 {
                                result[col] = (row, val, result[col].2, result[col].3);
                            }
                            if val > result[col].3 {
                                result[col] = (result[col].0, result[col].1, row, val);
                            }
                        }
                    }
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_1d_min_max, benchmark_2d_min_max,);
criterion_main!(benches);
