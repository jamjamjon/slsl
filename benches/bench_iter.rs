#![allow(unused)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use slsl::Tensor;
use std::hint::black_box;

use ndarray::{Array1, Array2, Array3};
use slsl::*;

// 1D
const SIZES_1D: &[usize] = &[
    64, 256, 376, 512, 768, 1024, 1344, 2048, 4096, 8192, 10240, 25600, 51200, 102400,
];

// 2D
const SIZES_2D: &[(usize, usize)] = &[
    (64, 64),
    (512, 512),
    (1024, 1024),
    (500, 8400),
    (500, 12800),
    (500, 25600),
];

fn generate_test_data_2d(rows: usize, cols: usize) -> Vec<f32> {
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    let mut rng = SmallRng::seed_from_u64(42);
    (0..rows * cols).map(|_| rng.random()).collect()
}

fn benchmark_iter_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("1D");

    for &size in SIZES_1D {
        // f32 data
        let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let slsl_tensor_f32 = Tensor::from_vec(data_f32.clone(), [size]).unwrap();
        let ndarray_tensor_f32 = Array1::from_shape_vec([size], data_f32.clone()).unwrap();
        let _ = &black_box(0);

        // SLSL
        group.bench_function(format!("slsl_iter_{size}"), |b| {
            b.iter(|| {
                slsl_tensor_f32
                    .iter::<f32>()
                    .enumerate()
                    .map(|(i, &v)| v + (i as f32) * 0.0001)
                    .collect::<Vec<f32>>()
            })
        });

        // SLSL for loop + at()
        group.bench_function(format!("slsl_for_at_{size}"), |b| {
            b.iter(|| {
                let mut result = Vec::with_capacity(size);
                for i in 0..size {
                    let v = slsl_tensor_f32.at::<f32>([i]);
                    result.push(v + (i as f32) * 0.0001);
                }
                result
            })
        });

        // NDArray
        group.bench_function(format!("ndarray_iter_{size}"), |b| {
            b.iter(|| {
                ndarray_tensor_f32
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v + (i as f32) * 0.0001)
                    .collect::<Vec<f32>>()
            })
        });

        // vec
        group.bench_function(format!("vec_iter_{size}"), |b| {
            b.iter(|| {
                data_f32
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v + (i as f32) * 0.0001)
                    .collect::<Vec<f32>>()
            })
        });
    }
    group.finish();
}

fn benchmark_iter_2d_contig_non_contig(c: &mut Criterion) {
    const SIZES_2D: &[(usize, usize)] = &[(10, 79), (20, 179), (30, 1179), (40, 5179), (50, 11179)];

    let mut group = c.benchmark_group("iter_vs_at_2d");
    for &(d0, d1) in SIZES_2D {
        let data: Vec<f32> = (0..d0 * d1).map(|i| i as f32).collect();
        let t = Tensor::from_vec(data.clone(), [d0, d1]).unwrap();
        let nd = Array2::from_shape_vec((d0, d1), data.clone()).unwrap();

        group.throughput(Throughput::Elements((d0 * d1) as u64));

        let i = d0 / 2;

        group.bench_with_input(
            BenchmarkId::new("slsl/contig/iter", format!("{}x{}", d0, d1)),
            &d0,
            |b, &_d| {
                b.iter(|| {
                    let (class_id, &confidence) = t
                        .slice(s![i, ..])
                        .iter::<f32>()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap();
                    black_box((confidence, class_id))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("slsl/contig/for_at", format!("{}x{}", d0, d1)),
            &d0,
            |b, &_d| {
                b.iter(|| {
                    let cls_dim = t.shape()[1];
                    let mut confidence = f32::NEG_INFINITY;
                    let mut class_id = 0;
                    for j in 0..cls_dim {
                        let conf: f32 = t.at([i, j]);
                        if conf > confidence {
                            confidence = conf;
                            class_id = j;
                        }
                    }
                    black_box((confidence, class_id))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ndarray/contig/iter", format!("{}x{}", d0, d1)),
            &d0,
            |b, &_d| {
                b.iter(|| {
                    let (class_id, &confidence) = nd
                        .slice(ndarray::s![i, ..])
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap();
                    black_box((confidence, class_id))
                })
            },
        );
    }

    const SIZES_2D_NON_CONTIG: &[(usize, usize)] =
        &[(79, 10), (179, 20), (1179, 30), (5179, 40), (11179, 50)];

    for &(d0, d1) in SIZES_2D_NON_CONTIG {
        let data: Vec<f32> = (0..d0 * d1).map(|i| i as f32).collect();
        let t = Tensor::from_vec(data.clone(), [d0, d1]).unwrap();
        let t = t.permute([1, 0]).unwrap();
        let nd = Array2::from_shape_vec((d0, d1), data.clone()).unwrap();
        let nd = nd.view().reversed_axes();

        group.throughput(Throughput::Elements((d0 * d1) as u64));

        let i = d1 / 2;

        group.bench_with_input(
            BenchmarkId::new("slsl/non-contig/iter", format!("{}x{}", d0, d1)),
            &d0,
            |b, &_d| {
                b.iter(|| {
                    let (class_id, &confidence) = t
                        .slice(s![i, ..])
                        .iter::<f32>()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap();
                    black_box((confidence, class_id))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("slsl/non-contig/for_at", format!("{}x{}", d0, d1)),
            &d0,
            |b, &_d| {
                b.iter(|| {
                    let cls_dim = t.shape()[1];
                    let mut confidence = f32::NEG_INFINITY;
                    let mut class_id = 0;
                    for j in 0..cls_dim {
                        let conf: f32 = t.at([i, j]);
                        if conf > confidence {
                            confidence = conf;
                            class_id = j;
                        }
                    }
                    black_box((confidence, class_id))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ndarray/non-contig/iter", format!("{}x{}", d0, d1)),
            &d0,
            |b, &_d| {
                b.iter(|| {
                    let (class_id, &confidence) = nd
                        .slice(ndarray::s![i, ..])
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap();
                    black_box((confidence, class_id))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    full_benches,
    benchmark_iter_1d,
    benchmark_iter_2d_contig_non_contig,
);

criterion_main!(full_benches);
