#![allow(unused)]
use candle_core::{Device, Tensor as CandleTensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use slsl::*;

const SMALL_SIZES: &[usize] = &[100, 500, 1000];
const MEDIUM_SIZES: &[usize] = &[2000, 5000, 10000];
const LARGE_SIZES: &[usize] = &[20000, 50000, 100000];

const SMALL_MATRICES: &[(usize, usize, usize)] = &[(32, 32, 32), (64, 64, 64), (128, 128, 128)];
const MEDIUM_MATRICES: &[(usize, usize, usize)] = &[(256, 256, 256), (512, 512, 512)];
const LARGE_MATRICES: &[(usize, usize, usize)] = &[(768, 768, 768), (1024, 1024, 1024)];

// TODO: candle seems to have a better way to do this
fn benchmark_matrix_multiplication(c: &mut Criterion) {
    let device = Device::Cpu;

    for &(m, k, n) in SMALL_MATRICES {
        let a_data: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i + 1) as f32).collect();

        let a = Tensor::from_vec(a_data.clone(), [m, k]).unwrap();
        let b = Tensor::from_vec(b_data.clone(), [k, n]).unwrap();

        c.bench_function(&format!("slsl_gemm_f32_small_{m}x{k}x{n}"), |bencher| {
            bencher.iter(|| {
                let result = a.matmul(black_box(&b)).unwrap();
                black_box(result);
            })
        });

        let candle_a = CandleTensor::from_vec(a_data, (m, k), &device).unwrap();
        let candle_b = CandleTensor::from_vec(b_data, (k, n), &device).unwrap();

        c.bench_function(&format!("candle_gemm_f32_small_{m}x{k}x{n}"), |bencher| {
            bencher.iter(|| {
                let result = black_box(&candle_a).matmul(black_box(&candle_b)).unwrap();
                black_box(result);
            })
        });

        let a_data_f64: Vec<f64> = (0..m * k).map(|i| i as f64).collect();
        let b_data_f64: Vec<f64> = (0..k * n).map(|i| (i + 1) as f64).collect();

        let a_f64 = Tensor::from_vec(a_data_f64.clone(), [m, k]).unwrap();
        let b_f64 = Tensor::from_vec(b_data_f64.clone(), [k, n]).unwrap();

        c.bench_function(&format!("slsl_gemm_f64_small_{m}x{k}x{n}"), |bencher| {
            bencher.iter(|| {
                let result = a_f64.matmul(black_box(&b_f64)).unwrap();
                black_box(result);
            })
        });

        let candle_a_f64 = CandleTensor::from_vec(a_data_f64, (m, k), &device).unwrap();
        let candle_b_f64 = CandleTensor::from_vec(b_data_f64, (k, n), &device).unwrap();

        c.bench_function(&format!("candle_gemm_f64_small_{m}x{k}x{n}"), |bencher| {
            bencher.iter(|| {
                let result = black_box(&candle_a_f64)
                    .matmul(black_box(&candle_b_f64))
                    .unwrap();
                black_box(result);
            })
        });

        // f16 benchmarks
        let a_data_f16: Vec<half::f16> =
            (0..m * k).map(|i| half::f16::from_f32(i as f32)).collect();
        let b_data_f16: Vec<half::f16> = (0..k * n)
            .map(|i| half::f16::from_f32((i + 1) as f32))
            .collect();

        let a_f16 = Tensor::from_vec(a_data_f16.clone(), [m, k]).unwrap();
        let b_f16 = Tensor::from_vec(b_data_f16.clone(), [k, n]).unwrap();

        c.bench_function(&format!("slsl_gemm_f16_small_{m}x{k}x{n}"), |bencher| {
            bencher.iter(|| {
                let result = a_f16.matmul(black_box(&b_f16)).unwrap();
                black_box(result);
            })
        });
    }

    for &(m, k, n) in MEDIUM_MATRICES {
        let a_data: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i + 1) as f32).collect();

        let a = Tensor::from_vec(a_data.clone(), [m, k]).unwrap();
        let b = Tensor::from_vec(b_data.clone(), [k, n]).unwrap();

        c.bench_function(&format!("slsl_gemm_f32_medium_{m}x{k}x{n}"), |bencher| {
            bencher.iter(|| {
                let result = a.matmul(black_box(&b)).unwrap();
                black_box(result);
            })
        });

        let candle_a = CandleTensor::from_vec(a_data, (m, k), &device).unwrap();
        let candle_b = CandleTensor::from_vec(b_data, (k, n), &device).unwrap();

        c.bench_function(&format!("candle_gemm_f32_medium_{m}x{k}x{n}"), |bencher| {
            bencher.iter(|| {
                let result = black_box(&candle_a).matmul(black_box(&candle_b)).unwrap();
                black_box(result);
            })
        });

        // f16 benchmarks for medium matrices
        let a_data_f16: Vec<half::f16> =
            (0..m * k).map(|i| half::f16::from_f32(i as f32)).collect();
        let b_data_f16: Vec<half::f16> = (0..k * n)
            .map(|i| half::f16::from_f32((i + 1) as f32))
            .collect();

        let a_f16 = Tensor::from_vec(a_data_f16.clone(), [m, k]).unwrap();
        let b_f16 = Tensor::from_vec(b_data_f16.clone(), [k, n]).unwrap();

        c.bench_function(&format!("slsl_gemm_f16_medium_{m}x{k}x{n}"), |bencher| {
            bencher.iter(|| {
                let result = a_f16.matmul(black_box(&b_f16)).unwrap();
                black_box(result);
            })
        });
    }

    for &(m, k, n) in LARGE_MATRICES {
        let a_data: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i + 1) as f32).collect();

        let a = Tensor::from_vec(a_data.clone(), [m, k]).unwrap();
        let b = Tensor::from_vec(b_data.clone(), [k, n]).unwrap();

        c.bench_function(&format!("slsl_gemm_f32_large_{m}x{k}x{n}"), |bencher| {
            bencher.iter(|| {
                let result = a.matmul(black_box(&b)).unwrap();
                black_box(result);
            })
        });

        let candle_a = CandleTensor::from_vec(a_data, (m, k), &device).unwrap();
        let candle_b = CandleTensor::from_vec(b_data, (k, n), &device).unwrap();

        c.bench_function(&format!("candle_gemm_f32_large_{m}x{k}x{n}"), |bencher| {
            bencher.iter(|| {
                let result = black_box(&candle_a).matmul(black_box(&candle_b)).unwrap();
                black_box(result);
            })
        });

        // f16 benchmarks for large matrices
        let a_data_f16: Vec<half::f16> =
            (0..m * k).map(|i| half::f16::from_f32(i as f32)).collect();
        let b_data_f16: Vec<half::f16> = (0..k * n)
            .map(|i| half::f16::from_f32((i + 1) as f32))
            .collect();

        let a_f16 = Tensor::from_vec(a_data_f16.clone(), [m, k]).unwrap();
        let b_f16 = Tensor::from_vec(b_data_f16.clone(), [k, n]).unwrap();

        c.bench_function(&format!("slsl_gemm_f16_large_{m}x{k}x{n}"), |bencher| {
            bencher.iter(|| {
                let result = a_f16.matmul(black_box(&b_f16)).unwrap();
                black_box(result);
            })
        });
    }
}

criterion_group!(benches, benchmark_matrix_multiplication,);
criterion_main!(benches);
