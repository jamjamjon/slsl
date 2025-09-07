use candle_core::{Device, Tensor as CandleTensor};
use criterion::{criterion_group, criterion_main, Criterion};
use slsl::Tensor;
use std::hint::black_box;

// ========== Test Data Generation ==========

fn generate_test_data_f32(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i % 1000) as f32 * 0.1 - 500.0).collect()
}

// ========== Dimension Shapes ==========

// 1D shapes
const SHAPE_1D_SMALL: [usize; 1] = [1000];
const SHAPE_1D_MEDIUM: [usize; 1] = [10000];
const SHAPE_1D_LARGE: [usize; 1] = [99856];

// 2D shapes
const SHAPE_2D_SMALL: [usize; 2] = [100, 10];
const SHAPE_2D_MEDIUM: [usize; 2] = [100, 100];
const SHAPE_2D_LARGE: [usize; 2] = [316, 316];

// 3D shapes
const SHAPE_3D_SMALL: [usize; 3] = [10, 10, 10];
const SHAPE_3D_MEDIUM: [usize; 3] = [25, 20, 20];
const SHAPE_3D_LARGE: [usize; 3] = [50, 50, 40];

// 4D shapes
const SHAPE_4D_SMALL: [usize; 4] = [5, 10, 10, 20];
const SHAPE_4D_MEDIUM: [usize; 4] = [10, 10, 10, 10];
const SHAPE_4D_LARGE: [usize; 4] = [20, 20, 15, 15];

// ========== Helper Functions ==========

fn create_candle_tensor_f32(data: &[f32], shape: &[usize]) -> CandleTensor {
    let device = Device::Cpu;
    CandleTensor::from_slice(data, shape, &device).unwrap()
}

// ========== Generic Benchmark Functions ==========

fn bench_unary_operation<F, G, H>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    operation_name: &str,
    slsl_op: F,
    candle_op: G,
    vec_op: H,
) where
    F: Fn(&Tensor) -> Result<Tensor, anyhow::Error> + Copy,
    G: Fn(&CandleTensor) -> Result<CandleTensor, candle_core::Error> + Copy,
    H: Fn(&[f32]) -> Vec<f32> + Copy,
{
    // 1D benchmarks
    bench_unary_1d(group, operation_name, slsl_op, candle_op, vec_op);

    // 2D benchmarks
    bench_unary_2d(group, operation_name, slsl_op, candle_op, vec_op);

    // 3D benchmarks
    bench_unary_3d(group, operation_name, slsl_op, candle_op, vec_op);

    // 4D benchmarks
    bench_unary_4d(group, operation_name, slsl_op, candle_op, vec_op);
}

fn bench_unary_1d<F, G, H>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    operation_name: &str,
    slsl_op: F,
    candle_op: G,
    vec_op: H,
) where
    F: Fn(&Tensor) -> Result<Tensor, anyhow::Error> + Copy,
    G: Fn(&CandleTensor) -> Result<CandleTensor, candle_core::Error> + Copy,
    H: Fn(&[f32]) -> Vec<f32> + Copy,
{
    // Small size: 1K elements
    {
        let small_data = generate_test_data_f32(1000);
        let small_shape = SHAPE_1D_SMALL;

        let slsl_contiguous = Tensor::from_vec(small_data.clone(), small_shape).unwrap();
        let candle_tensor = create_candle_tensor_f32(&small_data, &small_shape);
        let vec_data = small_data.clone();

        group.bench_function(format!("slsl_contiguous_1d_small_{operation_name}"), |b| {
            b.iter(|| black_box(slsl_op(&slsl_contiguous).unwrap()))
        });

        group.bench_function(format!("candle_1d_small_{operation_name}"), |b| {
            b.iter(|| black_box(candle_op(&candle_tensor).unwrap()))
        });

        group.bench_function(format!("vec_1d_small_{operation_name}"), |b| {
            b.iter(|| black_box(vec_op(&vec_data)))
        });
    }

    // Medium size: 10K elements
    {
        let medium_data = generate_test_data_f32(10000);
        let medium_shape = SHAPE_1D_MEDIUM;

        let slsl_contiguous = Tensor::from_vec(medium_data.clone(), medium_shape).unwrap();
        let candle_tensor = create_candle_tensor_f32(&medium_data, &medium_shape);
        let vec_data = medium_data.clone();

        group.bench_function(format!("slsl_contiguous_1d_medium_{operation_name}"), |b| {
            b.iter(|| black_box(slsl_op(&slsl_contiguous).unwrap()))
        });

        group.bench_function(format!("candle_1d_medium_{operation_name}"), |b| {
            b.iter(|| black_box(candle_op(&candle_tensor).unwrap()))
        });

        group.bench_function(format!("vec_1d_medium_{operation_name}"), |b| {
            b.iter(|| black_box(vec_op(&vec_data)))
        });
    }

    // Large size: ~100K elements
    {
        let large_data = generate_test_data_f32(99856);
        let large_shape = SHAPE_1D_LARGE;

        let slsl_contiguous = Tensor::from_vec(large_data.clone(), large_shape).unwrap();
        let candle_tensor = create_candle_tensor_f32(&large_data, &large_shape);
        let vec_data = large_data.clone();

        group.bench_function(format!("slsl_contiguous_1d_large_{operation_name}"), |b| {
            b.iter(|| black_box(slsl_op(&slsl_contiguous).unwrap()))
        });

        group.bench_function(format!("candle_1d_large_{operation_name}"), |b| {
            b.iter(|| black_box(candle_op(&candle_tensor).unwrap()))
        });

        group.bench_function(format!("vec_1d_large_{operation_name}"), |b| {
            b.iter(|| black_box(vec_op(&vec_data)))
        });
    }
}

fn bench_unary_2d<F, G, H>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    operation_name: &str,
    slsl_op: F,
    candle_op: G,
    vec_op: H,
) where
    F: Fn(&Tensor) -> Result<Tensor, anyhow::Error> + Copy,
    G: Fn(&CandleTensor) -> Result<CandleTensor, candle_core::Error> + Copy,
    H: Fn(&[f32]) -> Vec<f32> + Copy,
{
    // Small size: 1K elements
    {
        let small_data = generate_test_data_f32(1000);
        let small_shape = SHAPE_2D_SMALL;

        let slsl_contiguous = Tensor::from_vec(small_data.clone(), small_shape).unwrap();
        let candle_tensor = create_candle_tensor_f32(&small_data, &small_shape);
        let vec_data = small_data.clone();

        // Note: Non-contiguous SLSL view omitted in this unified bench

        group.bench_function(format!("slsl_contiguous_2d_small_{operation_name}"), |b| {
            b.iter(|| black_box(slsl_op(&slsl_contiguous).unwrap()))
        });

        // Note: SLSL non-contiguous views return TensorView; to keep the closure type simple (&Tensor),
        // we benchmark only the contiguous owned tensor for SLSL here.

        group.bench_function(format!("candle_2d_small_{operation_name}"), |b| {
            b.iter(|| black_box(candle_op(&candle_tensor).unwrap()))
        });

        group.bench_function(format!("vec_2d_small_{operation_name}"), |b| {
            b.iter(|| black_box(vec_op(&vec_data)))
        });
    }

    // Medium size: 10K elements
    {
        let medium_data = generate_test_data_f32(10000);
        let medium_shape = SHAPE_2D_MEDIUM;

        let slsl_contiguous = Tensor::from_vec(medium_data.clone(), medium_shape).unwrap();
        let candle_tensor = create_candle_tensor_f32(&medium_data, &medium_shape);
        let vec_data = medium_data.clone();

        group.bench_function(format!("slsl_contiguous_2d_medium_{operation_name}"), |b| {
            b.iter(|| black_box(slsl_op(&slsl_contiguous).unwrap()))
        });

        // see note above on SLSL non-contiguous

        group.bench_function(format!("candle_2d_medium_{operation_name}"), |b| {
            b.iter(|| black_box(candle_op(&candle_tensor).unwrap()))
        });

        group.bench_function(format!("vec_2d_medium_{operation_name}"), |b| {
            b.iter(|| black_box(vec_op(&vec_data)))
        });
    }

    // Large size: ~100K elements
    {
        let large_data = generate_test_data_f32(99856);
        let large_shape = SHAPE_2D_LARGE;

        let slsl_contiguous = Tensor::from_vec(large_data.clone(), large_shape).unwrap();
        let candle_tensor = create_candle_tensor_f32(&large_data, &large_shape);
        let vec_data = large_data.clone();

        group.bench_function(format!("slsl_contiguous_2d_large_{operation_name}"), |b| {
            b.iter(|| black_box(slsl_op(&slsl_contiguous).unwrap()))
        });

        // see note above on SLSL non-contiguous

        group.bench_function(format!("candle_2d_large_{operation_name}"), |b| {
            b.iter(|| black_box(candle_op(&candle_tensor).unwrap()))
        });

        group.bench_function(format!("vec_2d_large_{operation_name}"), |b| {
            b.iter(|| black_box(vec_op(&vec_data)))
        });
    }
}

fn bench_unary_3d<F, G, H>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    operation_name: &str,
    slsl_op: F,
    candle_op: G,
    vec_op: H,
) where
    F: Fn(&Tensor) -> Result<Tensor, anyhow::Error> + Copy,
    G: Fn(&CandleTensor) -> Result<CandleTensor, candle_core::Error> + Copy,
    H: Fn(&[f32]) -> Vec<f32> + Copy,
{
    // Small size: 1K elements
    {
        let small_data = generate_test_data_f32(1000);
        let small_shape = SHAPE_3D_SMALL;

        let slsl_contiguous = Tensor::from_vec(small_data.clone(), small_shape).unwrap();
        let candle_tensor = create_candle_tensor_f32(&small_data, &small_shape);
        let vec_data = small_data.clone();

        group.bench_function(format!("slsl_contiguous_3d_small_{operation_name}"), |b| {
            b.iter(|| black_box(slsl_op(&slsl_contiguous).unwrap()))
        });

        // see note above on SLSL non-contiguous

        group.bench_function(format!("candle_3d_small_{operation_name}"), |b| {
            b.iter(|| black_box(candle_op(&candle_tensor).unwrap()))
        });

        group.bench_function(format!("vec_3d_small_{operation_name}"), |b| {
            b.iter(|| black_box(vec_op(&vec_data)))
        });
    }

    // Medium size: 10K elements
    {
        let medium_data = generate_test_data_f32(10000);
        let medium_shape = SHAPE_3D_MEDIUM;

        let slsl_contiguous = Tensor::from_vec(medium_data.clone(), medium_shape).unwrap();
        let candle_tensor = create_candle_tensor_f32(&medium_data, &medium_shape);
        let vec_data = medium_data.clone();

        group.bench_function(format!("slsl_contiguous_3d_medium_{operation_name}"), |b| {
            b.iter(|| black_box(slsl_op(&slsl_contiguous).unwrap()))
        });

        // see note above on SLSL non-contiguous

        group.bench_function(format!("candle_3d_medium_{operation_name}"), |b| {
            b.iter(|| black_box(candle_op(&candle_tensor).unwrap()))
        });

        group.bench_function(format!("vec_3d_medium_{operation_name}"), |b| {
            b.iter(|| black_box(vec_op(&vec_data)))
        });
    }

    // Large size: 100K elements
    {
        let large_data = generate_test_data_f32(100000);
        let large_shape = SHAPE_3D_LARGE;

        let slsl_contiguous = Tensor::from_vec(large_data.clone(), large_shape).unwrap();
        let candle_tensor = create_candle_tensor_f32(&large_data, &large_shape);
        let vec_data = large_data.clone();

        group.bench_function(format!("slsl_contiguous_3d_large_{operation_name}"), |b| {
            b.iter(|| black_box(slsl_op(&slsl_contiguous).unwrap()))
        });

        // see note above on SLSL non-contiguous

        group.bench_function(format!("candle_3d_large_{operation_name}"), |b| {
            b.iter(|| black_box(candle_op(&candle_tensor).unwrap()))
        });

        group.bench_function(format!("vec_3d_large_{operation_name}"), |b| {
            b.iter(|| black_box(vec_op(&vec_data)))
        });
    }
}

fn bench_unary_4d<F, G, H>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    operation_name: &str,
    slsl_op: F,
    candle_op: G,
    vec_op: H,
) where
    F: Fn(&Tensor) -> Result<Tensor, anyhow::Error> + Copy,
    G: Fn(&CandleTensor) -> Result<CandleTensor, candle_core::Error> + Copy,
    H: Fn(&[f32]) -> Vec<f32> + Copy,
{
    // Small size: 10K elements (5*10*10*20 = 10000)
    {
        let small_data = generate_test_data_f32(10000);
        let small_shape = SHAPE_4D_SMALL;

        let slsl_contiguous = Tensor::from_vec(small_data.clone(), small_shape).unwrap();
        let candle_tensor = create_candle_tensor_f32(&small_data, &small_shape);
        let vec_data = small_data.clone();

        group.bench_function(format!("slsl_contiguous_4d_small_{operation_name}"), |b| {
            b.iter(|| black_box(slsl_op(&slsl_contiguous).unwrap()))
        });

        // see note above on SLSL non-contiguous

        group.bench_function(format!("candle_4d_small_{operation_name}"), |b| {
            b.iter(|| black_box(candle_op(&candle_tensor).unwrap()))
        });

        group.bench_function(format!("vec_4d_small_{operation_name}"), |b| {
            b.iter(|| black_box(vec_op(&vec_data)))
        });
    }

    // Medium size: 10K elements
    {
        let medium_data = generate_test_data_f32(10000);
        let medium_shape = SHAPE_4D_MEDIUM;

        let slsl_contiguous = Tensor::from_vec(medium_data.clone(), medium_shape).unwrap();
        let candle_tensor = create_candle_tensor_f32(&medium_data, &medium_shape);
        let vec_data = medium_data.clone();

        group.bench_function(format!("slsl_contiguous_4d_medium_{operation_name}"), |b| {
            b.iter(|| black_box(slsl_op(&slsl_contiguous).unwrap()))
        });

        // see note above on SLSL non-contiguous

        group.bench_function(format!("candle_4d_medium_{operation_name}"), |b| {
            b.iter(|| black_box(candle_op(&candle_tensor).unwrap()))
        });

        group.bench_function(format!("vec_4d_medium_{operation_name}"), |b| {
            b.iter(|| black_box(vec_op(&vec_data)))
        });
    }

    // Large size: 90K elements
    {
        let large_data = generate_test_data_f32(90000);
        let large_shape = SHAPE_4D_LARGE;

        let slsl_contiguous = Tensor::from_vec(large_data.clone(), large_shape).unwrap();
        let candle_tensor = create_candle_tensor_f32(&large_data, &large_shape);
        let vec_data = large_data.clone();

        group.bench_function(format!("slsl_contiguous_4d_large_{operation_name}"), |b| {
            b.iter(|| black_box(slsl_op(&slsl_contiguous).unwrap()))
        });

        // see note above on SLSL non-contiguous

        group.bench_function(format!("candle_4d_large_{operation_name}"), |b| {
            b.iter(|| black_box(candle_op(&candle_tensor).unwrap()))
        });

        group.bench_function(format!("vec_4d_large_{operation_name}"), |b| {
            b.iter(|| black_box(vec_op(&vec_data)))
        });
    }
}

// ========== Specific Operation Benchmarks ==========

fn bench_abs(c: &mut Criterion) {
    let mut group = c.benchmark_group("abs");

    bench_unary_operation(
        &mut group,
        "abs",
        |t| t.abs(),
        |t| t.abs(),
        |data| data.iter().map(|&x| x.abs()).collect(),
    );

    group.finish();
}

fn bench_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu");

    bench_unary_operation(
        &mut group,
        "relu",
        |t| t.relu(),
        |t| t.relu(),
        |data| data.iter().map(|&x| x.max(0.0)).collect(),
    );

    group.finish();
}

fn bench_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid");

    bench_unary_operation(
        &mut group,
        "sigmoid",
        |t| t.sigmoid(),
        |t| {
            // Manual sigmoid implementation for candle
            let shape = t.shape();
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data
                .iter()
                .map(|&x: &f32| 1.0 / (1.0 + (-x).exp()))
                .collect();
            CandleTensor::from_vec(result, shape, &Device::Cpu)
        },
        |data| data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
    );

    group.finish();
}

fn bench_sin(c: &mut Criterion) {
    let mut group = c.benchmark_group("sin");

    bench_unary_operation(
        &mut group,
        "sin",
        |t| t.sin(),
        |t| t.sin(),
        |data| data.iter().map(|&x| x.sin()).collect(),
    );

    group.finish();
}

fn bench_cos(c: &mut Criterion) {
    let mut group = c.benchmark_group("cos");

    bench_unary_operation(
        &mut group,
        "cos",
        |t| t.cos(),
        |t| t.cos(),
        |data| data.iter().map(|&x| x.cos()).collect(),
    );

    group.finish();
}

fn bench_tan(c: &mut Criterion) {
    let mut group = c.benchmark_group("tan");

    bench_unary_operation(
        &mut group,
        "tan",
        |t| t.tan(),
        |t| {
            // Manual tan implementation for candle
            let shape = t.shape();
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.tan()).collect();
            CandleTensor::from_vec(result, shape, &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.tan()).collect(),
    );

    group.finish();
}

fn bench_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp");

    bench_unary_operation(
        &mut group,
        "exp",
        |t| t.exp(),
        |t| t.exp(),
        |data| data.iter().map(|&x| x.exp()).collect(),
    );

    group.finish();
}

fn bench_log(c: &mut Criterion) {
    let mut group = c.benchmark_group("log");

    bench_unary_operation(
        &mut group,
        "log",
        |t| t.log(),
        |t| {
            // Manual log implementation for candle
            let shape = t.shape();
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.ln()).collect();
            CandleTensor::from_vec(result, shape, &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.ln()).collect(),
    );

    group.finish();
}

fn bench_sqrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqrt");

    bench_unary_operation(
        &mut group,
        "sqrt",
        |t| t.sqrt(),
        |t| {
            // Manual sqrt implementation for candle
            let shape = t.shape();
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.sqrt()).collect();
            CandleTensor::from_vec(result, shape, &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.sqrt()).collect(),
    );

    group.finish();
}

fn bench_floor(c: &mut Criterion) {
    let mut group = c.benchmark_group("floor");

    bench_unary_operation(
        &mut group,
        "floor",
        |t| t.floor(),
        |t| {
            // Manual floor implementation for candle
            let shape = t.shape();
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.floor()).collect();
            CandleTensor::from_vec(result, shape, &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.floor()).collect(),
    );

    group.finish();
}

fn bench_ceil(c: &mut Criterion) {
    let mut group = c.benchmark_group("ceil");

    bench_unary_operation(
        &mut group,
        "ceil",
        |t| t.ceil(),
        |t| {
            // Manual ceil implementation for candle
            let shape = t.shape();
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.ceil()).collect();
            CandleTensor::from_vec(result, shape, &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.ceil()).collect(),
    );

    group.finish();
}

fn bench_round(c: &mut Criterion) {
    let mut group = c.benchmark_group("round");

    bench_unary_operation(
        &mut group,
        "round",
        |t| t.round(),
        |t| {
            // Manual round implementation for candle
            let shape = t.shape();
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.round()).collect();
            CandleTensor::from_vec(result, shape, &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.round()).collect(),
    );

    group.finish();
}

fn bench_neg(c: &mut Criterion) {
    let mut group = c.benchmark_group("neg");

    bench_unary_operation(
        &mut group,
        "neg",
        |t| t.neg(),
        |t| t.neg(),
        |data| data.iter().map(|&x| -x).collect(),
    );

    group.finish();
}

fn bench_recip(c: &mut Criterion) {
    let mut group = c.benchmark_group("recip");

    bench_unary_operation(
        &mut group,
        "recip",
        |t| t.recip(),
        |t| {
            // Manual reciprocal implementation for candle
            let shape = t.shape();
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| 1.0 / x).collect();
            CandleTensor::from_vec(result, shape, &Device::Cpu)
        },
        |data| data.iter().map(|&x| 1.0 / x).collect(),
    );

    group.finish();
}

fn bench_tanh(c: &mut Criterion) {
    let mut group = c.benchmark_group("tanh");

    bench_unary_operation(
        &mut group,
        "tanh",
        |t| t.tanh(),
        |t| {
            // Manual tanh implementation for candle
            let shape = t.shape();
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.tanh()).collect();
            CandleTensor::from_vec(result, shape, &Device::Cpu)
        },
        |data| data.iter().map(|&x| x.tanh()).collect(),
    );

    group.finish();
}

fn bench_clamp(c: &mut Criterion) {
    let mut group = c.benchmark_group("clamp");

    bench_unary_operation(
        &mut group,
        "clamp",
        |t| t.clamp(Some(-1.0), Some(1.0)),
        |t| {
            // Manual clamp implementation for candle
            let shape = t.shape();
            let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let result: Vec<f32> = data.iter().map(|&x: &f32| x.clamp(-1.0, 1.0)).collect();
            CandleTensor::from_vec(result, shape, &Device::Cpu)
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
