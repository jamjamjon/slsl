use candle_core::{Device, Tensor as CandleTensor};
use criterion::{criterion_group, criterion_main, Criterion};
use slsl::*;
use std::hint::black_box;

// Define uniform data sizes across different scales
const SMALL_SIZES: &[usize] = &[128, 256, 512];
const MEDIUM_SIZES: &[usize] = &[1024, 2048, 4096];
const LARGE_SIZES: &[usize] = &[8192, 16384, 32768];

// Define uniform matrix sizes
const SMALL_MATRICES: &[(usize, usize)] = &[(64, 64), (128, 128), (256, 256)];
const MEDIUM_MATRICES: &[(usize, usize)] = &[(512, 512), (768, 768), (1024, 1024)];
const LARGE_MATRICES: &[(usize, usize)] = &[(1536, 1536), (2048, 2048), (3072, 3072)];

// Define uniform 3D tensor sizes
const SMALL_3D: &[(usize, usize, usize)] = &[(32, 32, 32), (48, 48, 48), (64, 64, 64)];
const MEDIUM_3D: &[(usize, usize, usize)] = &[(96, 96, 96), (128, 128, 128), (192, 192, 192)];
const LARGE_3D: &[(usize, usize, usize)] = &[(256, 256, 256), (384, 384, 384), (512, 512, 512)];

fn benchmark_1d_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("1D Norm Benchmarks");

    // Small scale 1D tensors
    for &size in SMALL_SIZES {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

        let slsl_tensor = Tensor::from_vec(data.clone(), [size]).unwrap();
        let candle_tensor = CandleTensor::from_vec(data, size, &Device::Cpu).unwrap();

        // L1 norm comparison
        group.bench_function(format!("slsl_norm1_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor.norm1(0).unwrap();
                black_box(result);
            })
        });

        group.bench_function(format!("candle_norm1_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor.abs().unwrap().sum_all().unwrap();
                black_box(result);
            })
        });

        // L2 norm comparison
        group.bench_function(format!("slsl_norm2_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor.norm2(0).unwrap();
                black_box(result);
            })
        });

        group.bench_function(format!("candle_norm2_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor
                    .sqr()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .sqrt()
                    .unwrap();
                black_box(result);
            })
        });

        // L2 norm with keepdim comparison
        group.bench_function(format!("slsl_norm2_keepdim_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor.norm_keepdim(0, 2.0).unwrap();
                black_box(result);
            })
        });

        group.bench_function(format!("candle_norm2_keepdim_1d_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor
                    .sqr()
                    .unwrap()
                    .sum_keepdim(0)
                    .unwrap()
                    .sqrt()
                    .unwrap();
                black_box(result);
            })
        });
    }

    // Medium scale 1D tensors
    for &size in MEDIUM_SIZES {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

        let slsl_tensor = Tensor::from_vec(data.clone(), [size]).unwrap();
        let candle_tensor = CandleTensor::from_vec(data, size, &Device::Cpu).unwrap();

        // L2 norm comparison (main test)
        group.bench_function(format!("slsl_norm2_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor.norm2(0).unwrap();
                black_box(result);
            })
        });

        group.bench_function(format!("candle_norm2_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor
                    .sqr()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .sqrt()
                    .unwrap();
                black_box(result);
            })
        });

        // L2 norm with keepdim comparison
        group.bench_function(format!("slsl_norm2_keepdim_1d_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor.norm_keepdim(0, 2.0).unwrap();
                black_box(result);
            })
        });

        group.bench_function(
            format!("candle_norm2_keepdim_1d_medium_{size}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor
                        .sqr()
                        .unwrap()
                        .sum_keepdim(0)
                        .unwrap()
                        .sqrt()
                        .unwrap();
                    black_box(result);
                })
            },
        );
    }

    // Large scale 1D tensors
    for &size in LARGE_SIZES {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

        let slsl_tensor = Tensor::from_vec(data.clone(), [size]).unwrap();
        let candle_tensor = CandleTensor::from_vec(data, size, &Device::Cpu).unwrap();

        // L2 norm comparison (main test)
        group.bench_function(format!("slsl_norm2_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor.norm2(0).unwrap();
                black_box(result);
            })
        });

        group.bench_function(format!("candle_norm2_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor
                    .sqr()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .sqrt()
                    .unwrap();
                black_box(result);
            })
        });

        // L2 norm with keepdim comparison
        group.bench_function(format!("slsl_norm2_keepdim_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let result = slsl_tensor.norm_keepdim(0, 2.0).unwrap();
                black_box(result);
            })
        });

        group.bench_function(format!("candle_norm2_keepdim_1d_large_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_tensor
                    .sqr()
                    .unwrap()
                    .sum_keepdim(0)
                    .unwrap()
                    .sqrt()
                    .unwrap();
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

        let slsl_tensor = Tensor::from_vec(data.clone(), [rows, cols]).unwrap();
        let candle_tensor = CandleTensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap();

        // Test norm along each dimension
        for dim in 0..2 {
            // L2 norm comparison
            group.bench_function(
                format!("slsl_norm2_2d_small_{rows}x{cols}_dim{dim}"),
                |bencher| {
                    bencher.iter(|| {
                        let result = slsl_tensor.norm2(dim).unwrap();
                        black_box(result);
                    })
                },
            );

            group.bench_function(
                format!("candle_norm2_2d_small_{rows}x{cols}_dim{dim}"),
                |bencher| {
                    bencher.iter(|| {
                        let result = candle_tensor
                            .sqr()
                            .unwrap()
                            .sum(dim)
                            .unwrap()
                            .sqrt()
                            .unwrap();
                        black_box(result);
                    })
                },
            );

            // L2 norm with keepdim comparison
            group.bench_function(
                format!("slsl_norm2_keepdim_2d_small_{rows}x{cols}_dim{dim}"),
                |bencher| {
                    bencher.iter(|| {
                        let result = slsl_tensor.norm_keepdim(dim, 2.0).unwrap();
                        black_box(result);
                    })
                },
            );

            group.bench_function(
                format!("candle_norm2_keepdim_2d_small_{rows}x{cols}_dim{dim}"),
                |bencher| {
                    bencher.iter(|| {
                        let result = candle_tensor
                            .sqr()
                            .unwrap()
                            .sum_keepdim(dim)
                            .unwrap()
                            .sqrt()
                            .unwrap();
                        black_box(result);
                    })
                },
            );
        }
    }

    // Medium scale 2D tensors
    for &(rows, cols) in MEDIUM_MATRICES {
        let size = rows * cols;
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        let slsl_tensor = Tensor::from_vec(data.clone(), [rows, cols]).unwrap();
        let candle_tensor = CandleTensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap();

        // Test norm along dimension 0 only for medium scale
        let dim = 0;

        // L2 norm comparison
        group.bench_function(
            format!("slsl_norm2_2d_medium_{rows}x{cols}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor.norm2(dim).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("candle_norm2_2d_medium_{rows}x{cols}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor
                        .sqr()
                        .unwrap()
                        .sum(dim)
                        .unwrap()
                        .sqrt()
                        .unwrap();
                    black_box(result);
                })
            },
        );

        // L2 norm with keepdim comparison
        group.bench_function(
            format!("slsl_norm2_keepdim_2d_medium_{rows}x{cols}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor.norm_keepdim(dim, 2.0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("candle_norm2_keepdim_2d_medium_{rows}x{cols}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor
                        .sqr()
                        .unwrap()
                        .sum_keepdim(dim)
                        .unwrap()
                        .sqrt()
                        .unwrap();
                    black_box(result);
                })
            },
        );
    }

    // Large scale 2D tensors
    for &(rows, cols) in LARGE_MATRICES {
        let size = rows * cols;
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        let slsl_tensor = Tensor::from_vec(data.clone(), [rows, cols]).unwrap();
        let candle_tensor = CandleTensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap();

        // Test norm along dimension 0 only for large scale
        let dim = 0;

        // L2 norm comparison
        group.bench_function(
            format!("slsl_norm2_2d_large_{rows}x{cols}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor.norm2(dim).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("candle_norm2_2d_large_{rows}x{cols}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor
                        .sqr()
                        .unwrap()
                        .sum(dim)
                        .unwrap()
                        .sqrt()
                        .unwrap();
                    black_box(result);
                })
            },
        );

        // L2 norm with keepdim comparison
        group.bench_function(
            format!("slsl_norm2_keepdim_2d_large_{rows}x{cols}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor.norm_keepdim(dim, 2.0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("candle_norm2_keepdim_2d_large_{rows}x{cols}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor
                        .sqr()
                        .unwrap()
                        .sum_keepdim(dim)
                        .unwrap()
                        .sqrt()
                        .unwrap();
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

        let slsl_tensor = Tensor::from_vec(data.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor = CandleTensor::from_vec(data, (d1, d2, d3), &Device::Cpu).unwrap();

        // Test norm along each dimension
        for dim in 0..3 {
            // L2 norm comparison
            group.bench_function(
                format!("slsl_norm2_3d_small_{d1}x{d2}x{d3}_dim{dim}"),
                |bencher| {
                    bencher.iter(|| {
                        let result = slsl_tensor.norm2(dim).unwrap();
                        black_box(result);
                    })
                },
            );

            group.bench_function(
                format!("candle_norm2_3d_small_{d1}x{d2}x{d3}_dim{dim}"),
                |bencher| {
                    bencher.iter(|| {
                        let result = candle_tensor
                            .sqr()
                            .unwrap()
                            .sum(dim)
                            .unwrap()
                            .sqrt()
                            .unwrap();
                        black_box(result);
                    })
                },
            );

            // L2 norm with keepdim comparison
            group.bench_function(
                format!("slsl_norm2_keepdim_3d_small_{d1}x{d2}x{d3}_dim{dim}"),
                |bencher| {
                    bencher.iter(|| {
                        let result = slsl_tensor.norm_keepdim(dim, 2.0).unwrap();
                        black_box(result);
                    })
                },
            );

            group.bench_function(
                format!("candle_norm2_keepdim_3d_small_{d1}x{d2}x{d3}_dim{dim}"),
                |bencher| {
                    bencher.iter(|| {
                        let result = candle_tensor
                            .sqr()
                            .unwrap()
                            .sum_keepdim(dim)
                            .unwrap()
                            .sqrt()
                            .unwrap();
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

        let slsl_tensor = Tensor::from_vec(data.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor = CandleTensor::from_vec(data, (d1, d2, d3), &Device::Cpu).unwrap();

        // Test norm along dimension 0 only for medium scale
        let dim = 0;

        // L2 norm comparison
        group.bench_function(
            format!("slsl_norm2_3d_medium_{d1}x{d2}x{d3}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor.norm2(dim).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("candle_norm2_3d_medium_{d1}x{d2}x{d3}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor
                        .sqr()
                        .unwrap()
                        .sum(dim)
                        .unwrap()
                        .sqrt()
                        .unwrap();
                    black_box(result);
                })
            },
        );

        // L2 norm with keepdim comparison
        group.bench_function(
            format!("slsl_norm2_keepdim_3d_medium_{d1}x{d2}x{d3}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor.norm_keepdim(dim, 2.0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("candle_norm2_keepdim_3d_medium_{d1}x{d2}x{d3}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor
                        .sqr()
                        .unwrap()
                        .sum_keepdim(dim)
                        .unwrap()
                        .sqrt()
                        .unwrap();
                    black_box(result);
                })
            },
        );
    }

    // Large scale 3D tensors
    for &(d1, d2, d3) in LARGE_3D {
        let size = d1 * d2 * d3;
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();

        let slsl_tensor = Tensor::from_vec(data.clone(), [d1, d2, d3]).unwrap();
        let candle_tensor = CandleTensor::from_vec(data, (d1, d2, d3), &Device::Cpu).unwrap();

        // Test norm along dimension 0 only for large scale
        let dim = 0;

        // L2 norm comparison
        group.bench_function(
            format!("slsl_norm2_3d_large_{d1}x{d2}x{d3}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor.norm2(dim).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("candle_norm2_3d_large_{d1}x{d2}x{d3}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor
                        .sqr()
                        .unwrap()
                        .sum(dim)
                        .unwrap()
                        .sqrt()
                        .unwrap();
                    black_box(result);
                })
            },
        );

        // L2 norm with keepdim comparison
        group.bench_function(
            format!("slsl_norm2_keepdim_3d_large_{d1}x{d2}x{d3}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = slsl_tensor.norm_keepdim(dim, 2.0).unwrap();
                    black_box(result);
                })
            },
        );

        group.bench_function(
            format!("candle_norm2_keepdim_3d_large_{d1}x{d2}x{d3}_dim{dim}"),
            |bencher| {
                bencher.iter(|| {
                    let result = candle_tensor
                        .sqr()
                        .unwrap()
                        .sum_keepdim(dim)
                        .unwrap()
                        .sqrt()
                        .unwrap();
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
    let size = 4096;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

    let slsl_tensor = Tensor::from_vec(data.clone(), [size]).unwrap();
    let candle_tensor = CandleTensor::from_vec(data, size, &Device::Cpu).unwrap();

    // L1 norm comparison
    group.bench_function("slsl_norm1_medium", |bencher| {
        bencher.iter(|| {
            let result = slsl_tensor.norm1(0).unwrap();
            black_box(result);
        })
    });

    group.bench_function("candle_norm1_medium", |bencher| {
        bencher.iter(|| {
            let result = candle_tensor.abs().unwrap().sum_all().unwrap();
            black_box(result);
        })
    });

    // L2 norm comparison
    group.bench_function("slsl_norm2_medium", |bencher| {
        bencher.iter(|| {
            let result = slsl_tensor.norm2(0).unwrap();
            black_box(result);
        })
    });

    group.bench_function("candle_norm2_medium", |bencher| {
        bencher.iter(|| {
            let result = candle_tensor
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .sqrt()
                .unwrap();
            black_box(result);
        })
    });

    // L3 norm comparison
    group.bench_function("slsl_norm3_medium", |bencher| {
        bencher.iter(|| {
            let result = slsl_tensor.normp(0, 3.0).unwrap();
            black_box(result);
        })
    });

    group.bench_function("candle_norm3_medium", |bencher| {
        bencher.iter(|| {
            let result = candle_tensor
                .abs()
                .unwrap()
                .powf(3.0)
                .unwrap()
                .sum_all()
                .unwrap()
                .powf(1.0 / 3.0)
                .unwrap();
            black_box(result);
        })
    });

    // L-infinity norm comparison
    group.bench_function("slsl_norm_inf_medium", |bencher| {
        bencher.iter(|| {
            let result = slsl_tensor.norm(0, f32::INFINITY).unwrap();
            black_box(result);
        })
    });

    group.bench_function("candle_norm_inf_medium", |bencher| {
        bencher.iter(|| {
            let result = candle_tensor.abs().unwrap().max_keepdim(0).unwrap();
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
    benchmark_norm_types
);
criterion_main!(benches);
