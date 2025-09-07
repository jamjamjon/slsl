use candle_core::{DType as CandleDType, Device, Tensor as CandleTensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2, Array3};
use slsl::{s, Tensor};

// ========== to_scalar Benchmarks ==========

fn bench_to_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_scalar");

    // Create scalar tensors
    let slsl_tensor = Tensor::from_vec(vec![42.0f32], []).unwrap();
    let slsl_tensor_slice = slsl_tensor.slice(s![]);
    let ndarray_tensor = Array1::from_vec(vec![42.0f32]).into_shape(()).unwrap();
    let device = Device::Cpu;
    let candle_tensor = CandleTensor::from_vec(vec![42.0f32], &[], &device).unwrap();

    group.bench_function("slsl_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor.to_scalar::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_slice_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor_slice.to_scalar::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("candle_core", |b| {
        b.iter(|| {
            let result = candle_tensor.to_scalar::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let result = ndarray_tensor.clone().into_scalar();
            black_box(result)
        })
    });

    group.finish();
}

// ========== to_vec (1D) Benchmarks ==========

fn bench_to_vec_1d_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_vec_1d_contiguous");
    let size = 10000;

    // Create contiguous test data
    let data: Vec<f32> = (0..size).map(|i| i as f32 / (size - 1) as f32).collect();
    let slsl_tensor = Tensor::from_vec(data.clone(), [size]).unwrap();
    let slsl_tensor_slice = slsl_tensor.slice(s![..]);
    let ndarray_tensor = Array1::<f32>::linspace(0.0, 1.0, size);
    let device = Device::Cpu;
    let candle_tensor = CandleTensor::from_vec(data, &[size], &device).unwrap();

    group.bench_function("slsl_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor.to_vec::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_slice_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor_slice.to_vec::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("candle_core", |b| {
        b.iter(|| {
            let result = candle_tensor.to_vec1::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let result = ndarray_tensor.to_vec();
            black_box(result)
        })
    });

    group.finish();
}

fn bench_to_vec_1d_non_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_vec_1d_non_contiguous");
    let size = 5000;

    // Create non-contiguous 1D tensor by slicing a 2D tensor
    let data: Vec<f32> = (0..size * 2).map(|i| i as f32).collect();
    let slsl_2d = Tensor::from_vec(data.clone(), [size, 2]).unwrap();
    let slsl_tensor = slsl_2d.slice(s![.., 0]); // Non-contiguous 1D slice
    let slsl_tensor_slice = slsl_tensor.slice(s![..]);

    let ndarray_2d =
        Array2::<f32>::from_shape_vec((size, 2), (0..size * 2).map(|i| i as f32).collect())
            .unwrap();
    let ndarray_tensor = ndarray_2d.column(0);

    group.bench_function("slsl_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor.to_vec::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_slice_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor_slice.to_vec::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let result = ndarray_tensor.to_vec();
            black_box(result)
        })
    });

    group.finish();
}

// ========== to_vec2 (2D) Benchmarks ==========

fn bench_to_vec2_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_vec2_contiguous");
    let shape = [100, 100];

    // Create contiguous test data
    let slsl_tensor = Tensor::zeros::<f32>(shape).unwrap();
    let slsl_tensor_slice = slsl_tensor.slice(s![.., ..]);
    let ndarray_tensor = Array2::<f32>::zeros((shape[0], shape[1]));
    let device = Device::Cpu;
    let candle_tensor =
        CandleTensor::zeros(&[shape[0], shape[1]], CandleDType::F32, &device).unwrap();

    group.bench_function("slsl_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor.to_vec2::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_slice_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor_slice.to_vec2::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("candle_core", |b| {
        b.iter(|| {
            let result = candle_tensor.to_vec2::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let result = ndarray_tensor
                .outer_iter()
                .map(|row| row.to_vec())
                .collect::<Vec<Vec<f32>>>();
            black_box(result)
        })
    });

    group.finish();
}

fn bench_to_vec2_non_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_vec2_non_contiguous");
    let shape = [100, 100];

    // Create non-contiguous 2D tensor by slicing from larger tensor
    let slsl_base = Tensor::zeros::<f32>([shape[0] * 2, shape[1]]).unwrap();
    let slsl_tensor = slsl_base.slice(s![0..shape[0], ..]); // Non-contiguous with stride
                                                            // let slsl_tensor_slice = slsl_tensor.slice(s![.., ..]);

    let ndarray_base = Array2::<f32>::zeros((shape[1], shape[0]));
    let ndarray_tensor = ndarray_base.t(); // Transposed view

    group.bench_function("slsl_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor.to_vec2::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let result = ndarray_tensor
                .outer_iter()
                .map(|row| row.to_vec())
                .collect::<Vec<Vec<f32>>>();
            black_box(result)
        })
    });

    group.finish();
}

// ========== to_vec3 (3D) Benchmarks ==========

fn bench_to_vec3_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_vec3_contiguous");
    let shape = [20, 25, 20];

    // Create contiguous test data
    let slsl_tensor = Tensor::zeros::<f32>(shape).unwrap();
    let slsl_tensor_slice = slsl_tensor.slice(s![.., .., ..]);
    let ndarray_tensor = Array3::<f32>::zeros((shape[0], shape[1], shape[2]));
    let device = Device::Cpu;
    let candle_tensor =
        CandleTensor::zeros(&[shape[0], shape[1], shape[2]], CandleDType::F32, &device).unwrap();

    group.bench_function("slsl_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor.to_vec3::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_slice_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor_slice.to_vec3::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("candle_core", |b| {
        b.iter(|| {
            let result = candle_tensor.to_vec3::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let result = ndarray_tensor
                .outer_iter()
                .map(|matrix| {
                    matrix
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect::<Vec<Vec<f32>>>()
                })
                .collect::<Vec<Vec<Vec<f32>>>>();
            black_box(result)
        })
    });

    group.finish();
}

fn bench_to_vec3_non_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_vec3_non_contiguous");
    let shape = [20, 25, 20];

    // Create non-contiguous 3D tensor by slicing from larger tensor
    let slsl_base = Tensor::zeros::<f32>([shape[0] * 2, shape[1], shape[2]]).unwrap();
    let slsl_tensor = slsl_base.slice(s![0..shape[0], .., ..]); // Non-contiguous with stride
                                                                // let slsl_tensor_slice = slsl_tensor.slice(s![.., .., ..]);

    let ndarray_base = Array3::<f32>::zeros((shape[2], shape[0], shape[1]));
    let ndarray_tensor = ndarray_base.permuted_axes([1, 2, 0]); // Permuted view

    group.bench_function("slsl_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor.to_vec3::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let result = ndarray_tensor
                .outer_iter()
                .map(|matrix| {
                    matrix
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect::<Vec<Vec<f32>>>()
                })
                .collect::<Vec<Vec<Vec<f32>>>>();
            black_box(result)
        })
    });

    group.finish();
}

// ========== to_flat_vec Benchmarks ==========

fn bench_to_flat_vec_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_flat_vec_contiguous");
    let shape = [50, 200]; // 10000 elements total

    // Create contiguous test data
    let slsl_tensor = Tensor::zeros::<f32>(shape).unwrap();
    let slsl_tensor_slice = slsl_tensor.slice(s![.., ..]);
    let ndarray_tensor = Array2::<f32>::zeros((shape[0], shape[1]));
    let device = Device::Cpu;
    let candle_tensor =
        CandleTensor::zeros(&[shape[0], shape[1]], CandleDType::F32, &device).unwrap();

    group.bench_function("slsl_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor.to_flat_vec::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_slice_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor_slice.to_flat_vec::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("candle_flatten", |b| {
        b.iter(|| {
            let result = candle_tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            black_box(result)
        })
    });

    group.bench_function("ndarray_flatten", |b| {
        b.iter(|| {
            let result = ndarray_tensor.iter().cloned().collect::<Vec<f32>>();
            black_box(result)
        })
    });

    group.finish();
}

fn bench_to_flat_vec_non_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_flat_vec_non_contiguous");
    let shape = [200, 50]; // 10000 elements total

    // Create non-contiguous test data by slicing from larger tensor
    let slsl_base = Tensor::zeros::<f32>([shape[0] * 2, shape[1]]).unwrap();
    let slsl_tensor = slsl_base.slice(s![0..shape[0], ..]); // Non-contiguous with stride
    let _slsl_tensor_slice = slsl_tensor.slice(s![.., ..]);

    let ndarray_base = Array2::<f32>::zeros((shape[1], shape[0]));
    let ndarray_tensor = ndarray_base.t(); // Transposed view

    group.bench_function("slsl_checked", |b| {
        b.iter(|| {
            let result = slsl_tensor.to_flat_vec::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("ndarray_flatten", |b| {
        b.iter(|| {
            let result = ndarray_tensor.iter().cloned().collect::<Vec<f32>>();
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_to_scalar,
    bench_to_vec_1d_contiguous,
    bench_to_vec_1d_non_contiguous,
    bench_to_vec2_contiguous,
    bench_to_vec2_non_contiguous,
    bench_to_vec3_contiguous,
    bench_to_vec3_non_contiguous,
    bench_to_flat_vec_contiguous,
    bench_to_flat_vec_non_contiguous,
);

criterion_main!(benches);
