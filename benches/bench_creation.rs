use candle_core::{Device, Tensor as CandleTensor};
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2, Array3};
use slsl::Tensor;
use std::f64::consts::PI;
use std::hint::black_box;

// ========== Basic Creation Benchmarks ==========

fn bench_zeros_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("zeros_1d");
    let size = 100000;

    group.bench_function("slsl_Tensor", |b| {
        b.iter(|| {
            let _tensor = Tensor::zeros::<f32>([black_box(size)]).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let tensor = Array1::<f32>::zeros(black_box(size));
            black_box(tensor)
        })
    });

    group.bench_function("candle", |b| {
        b.iter(|| {
            let device = Device::Cpu;
            let _tensor =
                CandleTensor::zeros(black_box(size), candle_core::DType::F32, &device).unwrap();
            black_box(_tensor)
        })
    });

    group.finish();
}

fn bench_zeros_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("zeros_2d");
    let shape = [1000, 1000];

    group.bench_function("slsl_Tensor", |b| {
        b.iter(|| {
            let _tensor = Tensor::zeros::<f64>(black_box(shape)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let tensor = Array2::<f64>::zeros(black_box((shape[0], shape[1])));
            black_box(tensor)
        })
    });

    group.bench_function("candle", |b| {
        b.iter(|| {
            let device = Device::Cpu;
            let _tensor = CandleTensor::zeros(
                (black_box(shape[0]), black_box(shape[1])),
                candle_core::DType::F64,
                &device,
            )
            .unwrap();
            black_box(_tensor)
        })
    });

    group.finish();
}

fn bench_zeros_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("zeros_3d");
    let shape = [100, 100, 20];

    group.bench_function("slsl_Tensor", |b| {
        b.iter(|| {
            let _tensor = Tensor::zeros::<f32>(black_box(shape)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let tensor = Array3::<f32>::zeros(black_box((shape[0], shape[1], shape[2])));
            black_box(tensor)
        })
    });

    group.bench_function("candle", |b| {
        b.iter(|| {
            let device = Device::Cpu;
            let tensor = CandleTensor::zeros(
                (
                    black_box(shape[0]),
                    black_box(shape[1]),
                    black_box(shape[2]),
                ),
                candle_core::DType::F32,
                &device,
            )
            .unwrap();
            black_box(tensor)
        })
    });

    group.finish();
}

fn bench_ones_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("ones_2d");
    let shape = [1000, 1000];

    group.bench_function("slsl_Tensor", |b| {
        b.iter(|| {
            let _tensor = Tensor::ones::<f64>(black_box(shape)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let tensor = Array2::<f64>::from_elem(black_box((shape[0], shape[1])), PI);
            black_box(tensor)
        })
    });

    group.bench_function("candle", |b| {
        b.iter(|| {
            let device = Device::Cpu;
            let tensor = CandleTensor::ones(
                (black_box(shape[0]), black_box(shape[1])),
                candle_core::DType::F64,
                &device,
            )
            .unwrap();
            black_box(tensor)
        })
    });

    group.finish();
}

// ========== Linspace Creation Benchmarks ==========

fn bench_linspace_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("linspace_small");
    let size = 1000;

    group.bench_function("slsl_Tensor", |b| {
        b.iter(|| {
            let _tensor =
                Tensor::linspace(black_box(0.0), black_box(1.0), black_box(size)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let _tensor = Array1::<f32>::linspace(black_box(0.0), black_box(1.0), black_box(size));
            black_box(_tensor)
        })
    });

    group.finish();
}

fn bench_linspace_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("linspace_large");
    let size = 1000000;

    group.bench_function("slsl_Tensor", |b| {
        b.iter(|| {
            let _tensor =
                Tensor::linspace(black_box(0.0), black_box(1.0), black_box(size)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let tensor = Array1::<f32>::linspace(black_box(0.0), black_box(1.0), black_box(size));
            black_box(tensor)
        })
    });

    // Note: Candle doesn't have a direct linspace function for large arrays
    // group.bench_function("candle", |b| {
    //     b.iter(|| {
    //         // Candle linspace equivalent would need manual implementation
    //     })
    // });

    group.finish();
}

// ========== from_vec Creation Benchmarks ==========

fn bench_from_vec_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_vec_1d");
    let size = 100000;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

    group.bench_function("slsl_Tensor", |b| {
        b.iter(|| {
            let _tensor = Tensor::from_vec(black_box(data.clone()), [size]).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let _tensor = Array1::from_vec(black_box(data.clone()));
            black_box(_tensor)
        })
    });

    group.finish();
}

fn bench_from_vec_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_vec_2d");
    let shape = [1000, 1000];
    let size = shape[0] * shape[1];
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

    group.bench_function("slsl_Tensor", |b| {
        b.iter(|| {
            let _tensor = Tensor::from_vec(black_box(data.clone()), black_box(shape)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let _tensor = Array1::from_vec(black_box(data.clone()))
                .into_shape_with_order(black_box((shape[0], shape[1])))
                .unwrap();
            black_box(_tensor)
        })
    });

    group.finish();
}

// ========== Memory Layout Creation Tests ==========

fn bench_creation_different_dtypes(c: &mut Criterion) {
    let mut group = c.benchmark_group("creation_different_dtypes");
    let shape = [1000, 1000];

    group.bench_function("slsl_f32", |b| {
        b.iter(|| {
            let _tensor = Tensor::zeros::<f32>(black_box(shape)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("slsl_f64", |b| {
        b.iter(|| {
            let tensor = Tensor::zeros::<f64>(black_box(shape)).unwrap();
            black_box(tensor)
        })
    });

    group.bench_function("slsl_i32", |b| {
        b.iter(|| {
            let tensor = Tensor::zeros::<i32>(black_box(shape)).unwrap();
            black_box(tensor)
        })
    });

    group.bench_function("slsl_i64", |b| {
        b.iter(|| {
            let tensor = Tensor::zeros::<i64>(black_box(shape)).unwrap();
            black_box(tensor)
        })
    });

    group.finish();
}

// ========== Multi-dtype Function Benchmarks ==========

fn bench_arange_dtypes(c: &mut Criterion) {
    let mut group = c.benchmark_group("arange_dtypes");

    group.bench_function("slsl_i32", |b| {
        b.iter(|| {
            let tensor =
                Tensor::arange::<i32>(black_box(0), black_box(10000), black_box(1)).unwrap();
            black_box(tensor)
        })
    });

    group.bench_function("slsl_i64", |b| {
        b.iter(|| {
            let tensor =
                Tensor::arange::<i64>(black_box(0), black_box(10000), black_box(1)).unwrap();
            black_box(tensor)
        })
    });

    group.bench_function("slsl_f32", |b| {
        b.iter(|| {
            let tensor =
                Tensor::arange::<f32>(black_box(0.0), black_box(10000.0), black_box(1.0)).unwrap();
            black_box(tensor)
        })
    });

    group.bench_function("slsl_f64", |b| {
        b.iter(|| {
            let tensor =
                Tensor::arange::<f64>(black_box(0.0), black_box(10000.0), black_box(1.0)).unwrap();
            black_box(tensor)
        })
    });

    group.finish();
}

fn bench_linspace_dtypes(c: &mut Criterion) {
    let mut group = c.benchmark_group("linspace_dtypes");
    let n = 10000;

    group.bench_function("slsl_f32", |b| {
        b.iter(|| {
            let _tensor =
                Tensor::linspace::<f32>(black_box(0.0), black_box(1000.0), black_box(n)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("slsl_f64", |b| {
        b.iter(|| {
            let _tensor =
                Tensor::linspace::<f64>(black_box(0.0), black_box(1000.0), black_box(n)).unwrap();
            black_box(_tensor)
        })
    });

    group.finish();
}

fn bench_rand_dtypes(c: &mut Criterion) {
    let mut group = c.benchmark_group("rand_dtypes");
    let shape = [1000, 1000];

    group.bench_function("slsl_i32", |b| {
        b.iter(|| {
            let _tensor =
                Tensor::rand::<i32>(black_box(0), black_box(100), black_box(shape)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("slsl_i64", |b| {
        b.iter(|| {
            let _tensor =
                Tensor::rand::<i64>(black_box(0), black_box(100), black_box(shape)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("slsl_f32", |b| {
        b.iter(|| {
            let _tensor =
                Tensor::rand::<f32>(black_box(0.0), black_box(1.0), black_box(shape)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("slsl_f64", |b| {
        b.iter(|| {
            let _tensor =
                Tensor::rand::<f64>(black_box(0.0), black_box(1.0), black_box(shape)).unwrap();
            black_box(_tensor)
        })
    });

    group.finish();
}

fn bench_randn_dtypes(c: &mut Criterion) {
    let mut group = c.benchmark_group("randn_dtypes");
    let shape = [1000, 1000];

    group.bench_function("slsl_f32", |b| {
        b.iter(|| {
            let _tensor = Tensor::randn::<f32>(black_box(shape)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("slsl_f64", |b| {
        b.iter(|| {
            let _tensor = Tensor::randn::<f64>(black_box(shape)).unwrap();
            black_box(_tensor)
        })
    });

    group.finish();
}

// ========== Large Tensor Creation Stress Tests ==========

fn bench_large_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_tensor_creation");

    // Test different large sizes
    let sizes = [
        [10000, 1000], // 10M elements
        [5000, 2000],  // 10M elements, different aspect ratio
        [1000, 10000], // 10M elements, wide
    ];

    let sizes_3d = [
        [100, 100, 1000], // 10M elements, 3D
    ];

    for (i, shape) in sizes.iter().enumerate() {
        let bench_name = format!("shape_{i}");

        if shape.len() == 2 {
            group.bench_function(format!("slsl_2d_{bench_name}"), |b| {
                b.iter(|| {
                    let _tensor = Tensor::zeros::<f32>([shape[0], shape[1]]).unwrap();
                    black_box(_tensor)
                })
            });
        }
    }

    // Handle 3D shapes separately
    for (i, shape) in sizes_3d.iter().enumerate() {
        let bench_name = format!("3d_shape_{i}");

        group.bench_function(format!("slsl_3d_{bench_name}").as_str(), |b| {
            b.iter(|| {
                let _tensor = Tensor::zeros::<f32>([shape[0], shape[1], shape[2]]).unwrap();
                black_box(_tensor)
            })
        });
    }

    group.finish();
}

// ========== Scalar Tensor Creation ==========

fn bench_scalar_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_creation");

    group.bench_function("slsl_Tensor", |b| {
        b.iter(|| {
            let _tensor = Tensor::from_vec(vec![black_box(42.0f32)], []).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let _tensor = Array1::from_vec(vec![black_box(42.0f32)])
                .into_shape_with_order(())
                .unwrap();
            black_box(_tensor)
        })
    });

    group.finish();
}

// ========== New Creation Function Benchmarks ==========

fn bench_eye(c: &mut Criterion) {
    let mut group = c.benchmark_group("eye");
    let n = 1000;

    group.bench_function("slsl_Tensor", |b| {
        b.iter(|| {
            let _tensor = Tensor::eye::<f32>(black_box(n)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let _tensor = Array2::<f32>::eye(black_box(n));
            black_box(_tensor)
        })
    });

    group.bench_function("candle", |b| {
        b.iter(|| {
            let device = Device::Cpu;
            let _tensor =
                CandleTensor::eye(black_box(n), candle_core::DType::F32, &device).unwrap();
            black_box(_tensor)
        })
    });

    group.finish();
}

fn bench_arange(c: &mut Criterion) {
    let mut group = c.benchmark_group("arange");
    let start = 0.0f32;
    let end = 100000.0f32;
    let step = 1.0f32;

    group.bench_function("slsl", |b| {
        b.iter(|| {
            let _tensor =
                Tensor::arange(black_box(start), black_box(end), black_box(step)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let _tensor = Array1::<f32>::range(black_box(start), black_box(end), black_box(step));
            black_box(_tensor)
        })
    });

    group.bench_function("candle", |b| {
        let device = candle_core::Device::Cpu;
        b.iter(|| {
            let _tensor =
                candle_core::Tensor::arange(black_box(start), black_box(end), &device).unwrap();
            black_box(_tensor)
        })
    });

    group.finish();
}

fn bench_linspace(c: &mut Criterion) {
    let mut group = c.benchmark_group("linspace");
    let n = 10000;

    group.bench_function("slsl", |b| {
        b.iter(|| {
            let _tensor =
                Tensor::linspace::<f32>(black_box(0.0), black_box(1000.0), black_box(n)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let _tensor = Array1::<f32>::linspace(black_box(0.0), black_box(1000.0), black_box(n));
            black_box(_tensor)
        })
    });

    group.finish();
}

fn bench_ones_like(c: &mut Criterion) {
    let mut group = c.benchmark_group("ones_like");
    let shape = [1000, 1000];
    let template = Tensor::zeros::<f32>(shape).unwrap();
    let ndarray_template = Array2::<f32>::zeros((shape[0], shape[1]));
    let device = Device::Cpu;
    let candle_template =
        CandleTensor::zeros((shape[0], shape[1]), candle_core::DType::F32, &device).unwrap();

    group.bench_function("slsl_Tensor", |b| {
        b.iter(|| {
            let _tensor = Tensor::ones_like::<f32>(black_box(&template)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let _tensor = Array2::<f32>::ones(black_box(ndarray_template.raw_dim()));
            black_box(_tensor)
        })
    });

    group.bench_function("candle", |b| {
        b.iter(|| {
            let _tensor = black_box(&candle_template).ones_like().unwrap();
            black_box(_tensor)
        })
    });

    group.finish();
}

fn bench_zeros_like(c: &mut Criterion) {
    let mut group = c.benchmark_group("zeros_like");
    let shape = [1000, 1000];
    let template = Tensor::ones::<f32>(shape).unwrap();
    let ndarray_template = Array2::<f32>::ones((shape[0], shape[1]));
    let device = Device::Cpu;
    let candle_template =
        CandleTensor::ones((shape[0], shape[1]), candle_core::DType::F32, &device).unwrap();

    group.bench_function("slsl_Tensor", |b| {
        b.iter(|| {
            let _tensor = Tensor::zeros_like::<f32>(black_box(&template)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let _tensor = Array2::<f32>::zeros(black_box(ndarray_template.raw_dim()));
            black_box(_tensor)
        })
    });

    group.bench_function("candle", |b| {
        b.iter(|| {
            let _tensor = black_box(&candle_template).zeros_like().unwrap();
            black_box(_tensor)
        })
    });

    group.finish();
}

fn bench_rand(c: &mut Criterion) {
    let mut group = c.benchmark_group("rand");
    let shape = [1000, 1000];
    let low = 0.0f32;
    let high = 1.0f32;

    group.bench_function("slsl_Tensor", |b| {
        b.iter(|| {
            let _tensor = Tensor::rand(black_box(low), black_box(high), black_box(shape)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("candle", |b| {
        b.iter(|| {
            let device = Device::Cpu;
            let _tensor = CandleTensor::rand(
                black_box(low),
                black_box(high),
                (black_box(shape[0]), black_box(shape[1])),
                &device,
            )
            .unwrap();
            black_box(_tensor)
        })
    });

    group.finish();
}

fn bench_randn(c: &mut Criterion) {
    let mut group = c.benchmark_group("randn");
    let shape = [1000, 1000];

    group.bench_function("slsl_Tensor", |b| {
        b.iter(|| {
            let _tensor = Tensor::randn::<f32>(black_box(shape)).unwrap();
            black_box(_tensor)
        })
    });

    group.bench_function("candle", |b| {
        b.iter(|| {
            let device = Device::Cpu;
            let _tensor = CandleTensor::randn(
                0.0f32,
                1.0f32,
                (black_box(shape[0]), black_box(shape[1])),
                &device,
            )
            .unwrap();
            black_box(_tensor)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_zeros_1d,
    bench_zeros_2d,
    bench_zeros_3d,
    bench_ones_2d,
    bench_linspace_small,
    bench_linspace_large,
    bench_from_vec_1d,
    bench_from_vec_2d,
    bench_creation_different_dtypes,
    bench_arange_dtypes,
    bench_linspace_dtypes,
    bench_rand_dtypes,
    bench_randn_dtypes,
    bench_large_tensor_creation,
    bench_scalar_creation,
    bench_eye,
    bench_arange,
    bench_linspace,
    bench_ones_like,
    bench_zeros_like,
    bench_rand,
    bench_randn
);

criterion_main!(benches);
