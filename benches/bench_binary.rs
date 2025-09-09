use candle_core::{Device, Tensor as CandleTensor};
use criterion::{criterion_group, criterion_main, Criterion};
use slsl::*;
use std::hint::black_box;

const SIZES: &[usize] = &[
    10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000,
];

fn benchmark_add(c: &mut Criterion) {
    let device = Device::Cpu;

    for &size in SIZES {
        let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32 * 0.2).collect();

        // SLSL tensors
        let slsl_a = Tensor::from_vec(a_data.clone(), [size]).unwrap();
        let slsl_b = Tensor::from_vec(b_data.clone(), [size]).unwrap();

        // Candle tensors
        let candle_a = CandleTensor::from_vec(a_data.clone(), size, &device).unwrap();
        let candle_b = CandleTensor::from_vec(b_data.clone(), size, &device).unwrap();

        // Raw Vec data
        let raw_a = a_data.clone();
        let raw_b = b_data.clone();

        // SLSL benchmark
        c.bench_function(&format!("slsl_add_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = &slsl_a + black_box(&slsl_b);
                black_box(result);
            })
        });

        // Candle benchmark
        c.bench_function(&format!("candle_add_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = (&candle_a + black_box(&candle_b)).unwrap();
                black_box(result);
            })
        });

        // Raw Vec benchmark
        c.bench_function(&format!("raw_vec_add_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result: Vec<f32> = raw_a
                    .iter()
                    .zip(black_box(&raw_b).iter())
                    .map(|(a, b)| a + b)
                    .collect();
                black_box(result);
            })
        });
    }
}

fn benchmark_add_scalar(c: &mut Criterion) {
    let device = Device::Cpu;
    let scalar = 2.5f32;

    for &size in SIZES {
        let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();

        // SLSL tensors
        let slsl_a = Tensor::from_vec(a_data.clone(), [size]).unwrap();

        // Candle tensors
        let candle_a = CandleTensor::from_vec(a_data.clone(), size, &device).unwrap();

        // Raw Vec data
        let raw_a = a_data.clone();

        // SLSL benchmark
        c.bench_function(&format!("slsl_add_scalar_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = &slsl_a + black_box(scalar);
                black_box(result);
            })
        });

        // Candle benchmark
        c.bench_function(&format!("candle_add_scalar_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = (&candle_a + black_box(scalar as f64)).unwrap();
                black_box(result);
            })
        });

        // Raw Vec benchmark
        c.bench_function(&format!("raw_vec_add_scalar_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result: Vec<f32> = raw_a.iter().map(|a| a + black_box(scalar)).collect();
                black_box(result);
            })
        });
    }
}

fn benchmark_sub(c: &mut Criterion) {
    let device = Device::Cpu;

    for &size in SIZES {
        let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32 * 0.2).collect();

        // SLSL tensors
        let slsl_a = Tensor::from_vec(a_data.clone(), [size]).unwrap();
        let slsl_b = Tensor::from_vec(b_data.clone(), [size]).unwrap();

        // Candle tensors
        let candle_a = CandleTensor::from_vec(a_data.clone(), size, &device).unwrap();
        let candle_b = CandleTensor::from_vec(b_data.clone(), size, &device).unwrap();

        // Raw Vec data
        let raw_a = a_data.clone();
        let raw_b = b_data.clone();

        // SLSL benchmark
        c.bench_function(&format!("slsl_sub_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = &slsl_a - black_box(&slsl_b);
                black_box(result);
            })
        });

        // Candle benchmark
        c.bench_function(&format!("candle_sub_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = (&candle_a - black_box(&candle_b)).unwrap();
                black_box(result);
            })
        });

        // Raw Vec benchmark
        c.bench_function(&format!("raw_vec_sub_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result: Vec<f32> = raw_a
                    .iter()
                    .zip(black_box(&raw_b).iter())
                    .map(|(a, b)| a - b)
                    .collect();
                black_box(result);
            })
        });
    }
}

fn benchmark_sub_scalar(c: &mut Criterion) {
    let device = Device::Cpu;
    let scalar = 2.5f32;

    for &size in SIZES {
        let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();

        // SLSL tensors
        let slsl_a = Tensor::from_vec(a_data.clone(), [size]).unwrap();

        // Candle tensors
        let candle_a = CandleTensor::from_vec(a_data.clone(), size, &device).unwrap();

        // Raw Vec data
        let raw_a = a_data.clone();

        // SLSL benchmark
        c.bench_function(&format!("slsl_sub_scalar_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = &slsl_a - black_box(scalar);
                black_box(result);
            })
        });

        // Candle benchmark
        c.bench_function(&format!("candle_sub_scalar_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = (&candle_a - black_box(scalar as f64)).unwrap();
                black_box(result);
            })
        });

        // Raw Vec benchmark
        c.bench_function(&format!("raw_vec_sub_scalar_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result: Vec<f32> = raw_a.iter().map(|a| a - black_box(scalar)).collect();
                black_box(result);
            })
        });
    }
}

fn benchmark_mul(c: &mut Criterion) {
    let device = Device::Cpu;

    for &size in SIZES {
        let a_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..size).map(|i| (i + 2) as f32 * 0.2).collect();

        // SLSL tensors
        let slsl_a = Tensor::from_vec(a_data.clone(), [size]).unwrap();
        let slsl_b = Tensor::from_vec(b_data.clone(), [size]).unwrap();

        // Candle tensors
        let candle_a = CandleTensor::from_vec(a_data.clone(), size, &device).unwrap();
        let candle_b = CandleTensor::from_vec(b_data.clone(), size, &device).unwrap();

        // Raw Vec data
        let raw_a = a_data.clone();
        let raw_b = b_data.clone();

        // SLSL benchmark
        c.bench_function(&format!("slsl_mul_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = &slsl_a * black_box(&slsl_b);
                black_box(result);
            })
        });

        // Candle benchmark
        c.bench_function(&format!("candle_mul_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = (&candle_a * black_box(&candle_b)).unwrap();
                black_box(result);
            })
        });

        // Raw Vec benchmark
        c.bench_function(&format!("raw_vec_mul_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result: Vec<f32> = raw_a
                    .iter()
                    .zip(black_box(&raw_b).iter())
                    .map(|(a, b)| a * b)
                    .collect();
                black_box(result);
            })
        });
    }
}

fn benchmark_mul_scalar(c: &mut Criterion) {
    let device = Device::Cpu;
    let scalar = 2.5f32;

    for &size in SIZES {
        let a_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32 * 0.1).collect();

        // SLSL tensors
        let slsl_a = Tensor::from_vec(a_data.clone(), [size]).unwrap();

        // Candle tensors
        let candle_a = CandleTensor::from_vec(a_data.clone(), size, &device).unwrap();

        // Raw Vec data
        let raw_a = a_data.clone();

        // SLSL benchmark
        c.bench_function(&format!("slsl_mul_scalar_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = &slsl_a * black_box(scalar);
                black_box(result);
            })
        });

        // Candle benchmark
        c.bench_function(&format!("candle_mul_scalar_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = (&candle_a * black_box(scalar as f64)).unwrap();
                black_box(result);
            })
        });

        // Raw Vec benchmark
        c.bench_function(&format!("raw_vec_mul_scalar_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result: Vec<f32> = raw_a.iter().map(|a| a * black_box(scalar)).collect();
                black_box(result);
            })
        });
    }
}

fn benchmark_div(c: &mut Criterion) {
    let device = Device::Cpu;

    for &size in SIZES {
        let a_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..size).map(|i| (i + 2) as f32 * 0.2).collect();

        // SLSL tensors
        let slsl_a = Tensor::from_vec(a_data.clone(), [size]).unwrap();
        let slsl_b = Tensor::from_vec(b_data.clone(), [size]).unwrap();

        // Candle tensors
        let candle_a = CandleTensor::from_vec(a_data.clone(), size, &device).unwrap();
        let candle_b = CandleTensor::from_vec(b_data.clone(), size, &device).unwrap();

        // Raw Vec data
        let raw_a = a_data.clone();
        let raw_b = b_data.clone();

        // SLSL benchmark
        c.bench_function(&format!("slsl_div_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = &slsl_a / black_box(&slsl_b);
                black_box(result);
            })
        });

        // Candle benchmark
        c.bench_function(&format!("candle_div_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = (&candle_a / black_box(&candle_b)).unwrap();
                black_box(result);
            })
        });

        // Raw Vec benchmark
        c.bench_function(&format!("raw_vec_div_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result: Vec<f32> = raw_a
                    .iter()
                    .zip(black_box(&raw_b).iter())
                    .map(|(a, b)| a / b)
                    .collect();
                black_box(result);
            })
        });
    }
}

fn benchmark_div_scalar(c: &mut Criterion) {
    let device = Device::Cpu;
    let scalar = 2.5f32;

    for &size in SIZES {
        let a_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32 * 0.1).collect();

        // SLSL tensors
        let slsl_a = Tensor::from_vec(a_data.clone(), [size]).unwrap();

        // Candle tensors
        let candle_a = CandleTensor::from_vec(a_data.clone(), size, &device).unwrap();

        // Raw Vec data
        let raw_a = a_data.clone();

        // SLSL benchmark
        c.bench_function(&format!("slsl_div_scalar_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = &slsl_a / black_box(scalar);
                black_box(result);
            })
        });

        // Candle benchmark
        c.bench_function(&format!("candle_div_scalar_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result = (&candle_a / black_box(scalar as f64)).unwrap();
                black_box(result);
            })
        });

        // Raw Vec benchmark
        c.bench_function(&format!("raw_vec_div_scalar_f32_{size}"), |bencher| {
            bencher.iter(|| {
                let result: Vec<f32> = raw_a.iter().map(|a| a / black_box(scalar)).collect();
                black_box(result);
            })
        });
    }
}

criterion_group!(
    benches,
    benchmark_add,
    benchmark_add_scalar,
    benchmark_sub,
    benchmark_sub_scalar,
    benchmark_mul,
    benchmark_mul_scalar,
    benchmark_div,
    benchmark_div_scalar
);
criterion_main!(benches);
