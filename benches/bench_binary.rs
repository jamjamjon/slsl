#![allow(unused)]

use candle_core::{Device, Tensor as CandleTensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use slsl::*;

const SMALL_SIZES: &[usize] = &[100, 500, 1000];
const MEDIUM_SIZES: &[usize] = &[2000, 5000, 10000];
const LARGE_SIZES: &[usize] = &[20000, 50000, 100000];

const SMALL_MATRICES: &[(usize, usize, usize)] = &[(32, 32, 32), (64, 64, 64), (128, 128, 128)];
const MEDIUM_MATRICES: &[(usize, usize, usize)] =
    &[(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)];
const LARGE_MATRICES: &[(usize, usize, usize)] = &[(2048, 2048, 2048), (4096, 4096, 4096)];

fn benchmark_binary_operations(c: &mut Criterion) {
    let device = Device::Cpu;

    for &size in SMALL_SIZES {
        let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        let a = Tensor::from_vec(a_data.clone(), [size]).unwrap();
        let b = Tensor::from_vec(b_data.clone(), [size]).unwrap();

        let candle_a = CandleTensor::from_vec(a_data, size, &device).unwrap();
        let candle_b = CandleTensor::from_vec(b_data, size, &device).unwrap();

        // Add
        c.bench_function(&format!("slsl_add_f32_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = &a + black_box(&b);
                black_box(result);
            })
        });

        c.bench_function(&format!("candle_add_f32_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_a.add(black_box(&candle_b)).unwrap();
                black_box(result);
            })
        });

        // Mul
        c.bench_function(&format!("slsl_mul_f32_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = a.mul(black_box(&b)).unwrap();
                black_box(result);
            })
        });

        c.bench_function(&format!("candle_mul_f32_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_a.mul(black_box(&candle_b)).unwrap();
                black_box(result);
            })
        });

        // Div
        c.bench_function(&format!("slsl_div_f32_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = a.div(black_box(&b)).unwrap();
                black_box(result);
            })
        });

        c.bench_function(&format!("candle_div_f32_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_a.div(black_box(&candle_b)).unwrap();
                black_box(result);
            })
        });

        // Sub
        c.bench_function(&format!("slsl_sub_f32_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = a.sub(black_box(&b)).unwrap();
                black_box(result);
            })
        });

        c.bench_function(&format!("candle_sub_f32_small_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_a.sub(black_box(&candle_b)).unwrap();
                black_box(result);
            })
        });
    }

    for &size in MEDIUM_SIZES {
        let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        let a = Tensor::from_vec(a_data.clone(), [size]).unwrap();
        let b = Tensor::from_vec(b_data.clone(), [size]).unwrap();

        let candle_a = CandleTensor::from_vec(a_data, size, &device).unwrap();
        let candle_b = CandleTensor::from_vec(b_data, size, &device).unwrap();

        c.bench_function(&format!("slsl_add_f32_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = &a + black_box(&b);
                black_box(result);
            })
        });

        c.bench_function(&format!("candle_add_f32_medium_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_a.add(black_box(&candle_b)).unwrap();
                black_box(result);
            })
        });
    }

    for &size in LARGE_SIZES {
        let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        let a = Tensor::from_vec(a_data.clone(), [size]).unwrap();
        let b = Tensor::from_vec(b_data.clone(), [size]).unwrap();

        let candle_a = CandleTensor::from_vec(a_data, size, &device).unwrap();
        let candle_b = CandleTensor::from_vec(b_data, size, &device).unwrap();

        c.bench_function(&format!("slsl_add_f32_large_{size}"), |bencher| {
            bencher.iter(|| {
                let result = &a + black_box(&b);
                black_box(result);
            })
        });

        c.bench_function(&format!("candle_add_f32_large_{size}"), |bencher| {
            bencher.iter(|| {
                let result = candle_a.add(black_box(&candle_b)).unwrap();
                black_box(result);
            })
        });
    }
}

criterion_group!(benches, benchmark_binary_operations,);
criterion_main!(benches);
