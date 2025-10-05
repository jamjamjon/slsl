use candle_core::{DType as CandleDType, Device, Tensor as CandleTensor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use slsl::Tensor;
use std::hint::black_box;

// ========== Test Data Generation ==========

fn generate_f32(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i % 1000) as f32 * 0.1).collect()
}

fn generate_f64(size: usize) -> Vec<f64> {
    generate_f32(size).iter().map(|&x| x as f64).collect()
}

fn generate_u8(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

fn generate_i64(size: usize) -> Vec<i64> {
    (0..size).map(|i| (i % 1000) as i64).collect()
}

// ========== Unified Benchmark Macro ==========

macro_rules! bench_conversion {
    ($name:ident, $src_ty:ty, $dst_ty:ty, $gen_fn:ident, $candle_dst:expr) => {
        fn $name(c: &mut Criterion) {
            let sizes = vec![
                ("tiny_16", vec![16, 16]),         // 256
                ("small_64", vec![64, 64]),        // 4K
                ("medium_256", vec![256, 256]),    // 64K
                ("large_512", vec![512, 512]),     // 256K
                ("xlarge_1024", vec![1024, 1024]), // 1M
                ("3d_small", vec![32, 32, 8]),     // 8K
                ("3d_medium", vec![64, 64, 16]),   // 64K
            ];

            let mut group = c.benchmark_group(stringify!($name));

            for (name, shape) in sizes {
                let numel: usize = shape.iter().product();
                let data = $gen_fn(numel);

                // SLSL tensors
                let slsl_cont = Tensor::from_vec(data.clone(), shape.as_slice()).unwrap();
                let slsl_nc = if shape.len() == 2 {
                    slsl_cont.clone().permute([1, 0]).unwrap()
                } else if shape.len() == 3 {
                    slsl_cont.clone().permute([2, 0, 1]).unwrap()
                } else {
                    slsl_cont.clone()
                };

                // Candle tensors
                let device = Device::Cpu;
                let candle_cont = match shape.len() {
                    2 => CandleTensor::from_vec(data.clone(), &[shape[0], shape[1]], &device)
                        .unwrap(),
                    3 => CandleTensor::from_vec(
                        data.clone(),
                        &[shape[0], shape[1], shape[2]],
                        &device,
                    )
                    .unwrap(),
                    _ => panic!("Unsupported shape"),
                };
                let candle_nc = if shape.len() == 2 {
                    candle_cont.permute((1, 0)).unwrap()
                } else if shape.len() == 3 {
                    candle_cont.permute((2, 0, 1)).unwrap()
                } else {
                    candle_cont.clone()
                };

                // Benchmarks
                group.bench_with_input(BenchmarkId::new("slsl_cont", name), &slsl_cont, |b, t| {
                    b.iter(|| black_box(t.to_dtype::<$dst_ty>().unwrap()))
                });

                group.bench_with_input(BenchmarkId::new("slsl_nc", name), &slsl_nc, |b, t| {
                    b.iter(|| black_box(t.to_dtype::<$dst_ty>().unwrap()))
                });

                group.bench_with_input(
                    BenchmarkId::new("candle_cont", name),
                    &candle_cont,
                    |b, t| b.iter(|| black_box(t.to_dtype($candle_dst).unwrap())),
                );

                group.bench_with_input(BenchmarkId::new("candle_nc", name), &candle_nc, |b, t| {
                    b.iter(|| black_box(t.to_dtype($candle_dst).unwrap()))
                });
            }

            group.finish();
        }
    };
}

// ========== Generate Benchmark Functions ==========

bench_conversion!(bench_f32_to_f64, f32, f64, generate_f32, CandleDType::F64);
bench_conversion!(bench_f64_to_f32, f64, f32, generate_f64, CandleDType::F32);
bench_conversion!(bench_f32_to_u8, f32, u8, generate_f32, CandleDType::U8);
bench_conversion!(bench_u8_to_f32, u8, f32, generate_u8, CandleDType::F32);
bench_conversion!(bench_f32_to_i64, f32, i64, generate_f32, CandleDType::I64);
bench_conversion!(bench_i64_to_f32, i64, f32, generate_i64, CandleDType::F32);

// ========== Benchmark Group Registration ==========

criterion_group!(
    benches,
    bench_f32_to_f64,
    bench_f64_to_f32,
    bench_f32_to_u8,
    bench_u8_to_f32,
    bench_f32_to_i64,
    bench_i64_to_f32
);
criterion_main!(benches);
