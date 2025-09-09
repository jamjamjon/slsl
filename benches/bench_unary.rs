use candle_core::{Device, Tensor as CandleTensor};
use criterion::{criterion_group, criterion_main, Criterion};
use slsl::Tensor;
use std::hint::black_box;

// ========== Test Data Generation ==========

fn generate_test_data_f32(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i % 1000) as f32 * 0.1 - 500.0).collect()
}

// ========== Helper Functions ==========

fn create_candle_tensor_f32(data: &[f32]) -> CandleTensor {
    let device = Device::Cpu;
    CandleTensor::from_slice(data, &[data.len()], &device).unwrap()
}

// ========== Generic Benchmark Function ==========

fn bench_unary_operation_f32<F, G, H>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    slsl_op: F,
    candle_op: G,
    vec_op: H,
) where
    F: Fn(&Tensor) -> Result<Tensor, anyhow::Error> + Copy,
    G: Fn(&CandleTensor) -> Result<CandleTensor, candle_core::Error> + Copy,
    H: Fn(&[f32]) -> Vec<f32> + Copy,
{
    let sizes = [
        100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000,
    ];

    for &size in &sizes {
        let data = generate_test_data_f32(size);

        let slsl_tensor = Tensor::from_vec(data.clone(), [size]).unwrap();
        let candle_tensor = create_candle_tensor_f32(&data);
        let vec_data = data.clone();

        group.bench_function(format!("slsl_{}", size), |b| {
            b.iter(|| black_box(slsl_op(&slsl_tensor).unwrap()))
        });

        group.bench_function(format!("candle_{}", size), |b| {
            b.iter(|| black_box(candle_op(&candle_tensor).unwrap()))
        });

        group.bench_function(format!("vec_{}", size), |b| {
            b.iter(|| black_box(vec_op(&vec_data)))
        });
    }
}

// ========== Specific Operation Benchmarks ==========

fn bench_abs(c: &mut Criterion) {
    let mut group = c.benchmark_group("abs");

    bench_unary_operation_f32(
        &mut group,
        |t| t.abs(),
        |t| t.abs(),
        |data| data.iter().map(|&x| x.abs()).collect(),
    );

    group.finish();
}

fn bench_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu");

    bench_unary_operation_f32(
        &mut group,
        |t| t.relu(),
        |t| t.relu(),
        |data| data.iter().map(|&x| x.max(0.0)).collect(),
    );

    group.finish();
}

fn bench_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid");

    bench_unary_operation_f32(
        &mut group,
        |t| t.sigmoid(),
        |t| {
            // Manual sigmoid implementation for candle
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data
                .iter()
                .map(|&x: &f32| 1.0 / (1.0 + (-x).exp()))
                .collect();
            CandleTensor::from_vec(result, &[data.len()], &Device::Cpu)
        },
        |data| data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
    );

    group.finish();
}

fn bench_sin(c: &mut Criterion) {
    let mut group = c.benchmark_group("sin");

    bench_unary_operation_f32(
        &mut group,
        |t| t.sin(),
        |t| t.sin(),
        |data| data.iter().map(|&x| x.sin()).collect(),
    );

    group.finish();
}

fn bench_cos(c: &mut Criterion) {
    let mut group = c.benchmark_group("cos");

    bench_unary_operation_f32(
        &mut group,
        |t| t.cos(),
        |t| t.cos(),
        |data| data.iter().map(|&x| x.cos()).collect(),
    );

    group.finish();
}

fn bench_tan(c: &mut Criterion) {
    let mut group = c.benchmark_group("tan");

    bench_unary_operation_f32(
        &mut group,
        |t| t.tan(),
        |t| {
            // Manual tan implementation for candle
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.tan()).collect();
            CandleTensor::from_vec(result, &[data.len()], &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.tan()).collect(),
    );

    group.finish();
}

fn bench_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp");

    bench_unary_operation_f32(
        &mut group,
        |t| t.exp(),
        |t| t.exp(),
        |data| data.iter().map(|&x| x.exp()).collect(),
    );

    group.finish();
}

fn bench_log(c: &mut Criterion) {
    let mut group = c.benchmark_group("log");

    bench_unary_operation_f32(
        &mut group,
        |t| t.log(),
        |t| {
            // Manual log implementation for candle
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.ln()).collect();
            CandleTensor::from_vec(result, &[data.len()], &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.ln()).collect(),
    );

    group.finish();
}

fn bench_sqrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqrt");

    bench_unary_operation_f32(
        &mut group,
        |t| t.sqrt(),
        |t| {
            // Manual sqrt implementation for candle
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.sqrt()).collect();
            CandleTensor::from_vec(result, &[data.len()], &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.sqrt()).collect(),
    );

    group.finish();
}

fn bench_floor(c: &mut Criterion) {
    let mut group = c.benchmark_group("floor");

    bench_unary_operation_f32(
        &mut group,
        |t| t.floor(),
        |t| {
            // Manual floor implementation for candle
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.floor()).collect();
            CandleTensor::from_vec(result, &[data.len()], &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.floor()).collect(),
    );

    group.finish();
}

fn bench_ceil(c: &mut Criterion) {
    let mut group = c.benchmark_group("ceil");

    bench_unary_operation_f32(
        &mut group,
        |t| t.ceil(),
        |t| {
            // Manual ceil implementation for candle
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.ceil()).collect();
            CandleTensor::from_vec(result, &[data.len()], &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.ceil()).collect(),
    );

    group.finish();
}

fn bench_round(c: &mut Criterion) {
    let mut group = c.benchmark_group("round");

    bench_unary_operation_f32(
        &mut group,
        |t| t.round(),
        |t| {
            // Manual round implementation for candle
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.round()).collect();
            CandleTensor::from_vec(result, &[data.len()], &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.round()).collect(),
    );

    group.finish();
}

fn bench_neg(c: &mut Criterion) {
    let mut group = c.benchmark_group("neg");

    bench_unary_operation_f32(
        &mut group,
        |t| t.neg(),
        |t| t.neg(),
        |data| data.iter().map(|&x| -x).collect(),
    );

    group.finish();
}

fn bench_recip(c: &mut Criterion) {
    let mut group = c.benchmark_group("recip");

    bench_unary_operation_f32(
        &mut group,
        |t| t.recip(),
        |t| {
            // Manual reciprocal implementation for candle
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| 1.0 / x).collect();
            CandleTensor::from_vec(result, &[data.len()], &Device::Cpu)
        },
        |data| data.iter().map(|&x| 1.0 / x).collect(),
    );

    group.finish();
}

fn bench_tanh(c: &mut Criterion) {
    let mut group = c.benchmark_group("tanh");

    bench_unary_operation_f32(
        &mut group,
        |t| t.tanh(),
        |t| {
            // Manual tanh implementation for candle
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.tanh()).collect();
            CandleTensor::from_vec(result, &[data.len()], &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.tanh()).collect(),
    );

    group.finish();
}

fn bench_clamp(c: &mut Criterion) {
    let mut group = c.benchmark_group("clamp");

    bench_unary_operation_f32(
        &mut group,
        |t| t.clamp(Some(-1.0), Some(1.0)),
        |t| {
            // Manual clamp implementation for candle
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.clamp(-1.0, 1.0)).collect();
            CandleTensor::from_vec(result, &[data.len()], &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.clamp(-1.0, 1.0)).collect(),
    );

    group.finish();
}

// ========== Main Benchmark Group ==========

criterion_group!(
    benches,
    bench_abs,
    bench_relu,
    bench_sigmoid,
    bench_sin,
    bench_cos,
    bench_tan,
    bench_exp,
    bench_log,
    bench_sqrt,
    bench_floor,
    bench_ceil,
    bench_round,
    bench_neg,
    bench_recip,
    bench_tanh,
    bench_clamp
);
criterion_main!(benches);
