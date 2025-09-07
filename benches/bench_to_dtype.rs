use candle_core::{DType as CandleDType, Device, Tensor as CandleTensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use slsl::{s, Tensor};

// ========== Test Data Generation ==========

fn generate_test_data_f32(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i % 1000) as f32 * 0.1).collect()
}

fn generate_test_data_i64(size: usize) -> Vec<i64> {
    (0..size).map(|i| (i % 1000) as i64).collect()
}

// ========== 2D DType Conversion Benchmarks ==========

fn bench_dtype_2d_f32_to_i64(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_2d_f32_to_i64");
    let shape = [200, 200];
    let numel = shape[0] * shape[1];
    let data = generate_test_data_f32(numel);

    // SLSL Tensor - Contiguous
    let slsl_tensor_cont = Tensor::from_vec(data.clone(), shape).unwrap();
    assert!(
        slsl_tensor_cont.is_contiguous(),
        "SLSL tensor should be contiguous"
    );
    assert_eq!(
        slsl_tensor_cont.numel(),
        numel,
        "SLSL tensor numel mismatch"
    );

    // SLSL Tensor - Non-contiguous via permute
    let slsl_tensor_nc = slsl_tensor_cont.clone().permute([1, 0]).unwrap();
    assert!(
        !slsl_tensor_nc.is_contiguous(),
        "SLSL tensor should be non-contiguous"
    );
    assert_eq!(slsl_tensor_nc.numel(), numel, "SLSL tensor numel mismatch");

    // SLSL TensorView - Contiguous (slice of contiguous tensor)
    let slsl_view_cont = slsl_tensor_cont.slice(s![.., ..]);
    assert!(
        slsl_view_cont.is_contiguous(),
        "SLSL view should be contiguous"
    );
    assert_eq!(slsl_view_cont.numel(), numel, "SLSL view numel mismatch");

    // SLSL TensorView - Non-contiguous via slice
    let slsl_view_nc = slsl_tensor_cont.slice(s![10..190, 10..190]);
    assert!(
        !slsl_view_nc.is_contiguous(),
        "SLSL view should be non-contiguous"
    );
    let view_numel = 180 * 180;
    assert_eq!(slsl_view_nc.numel(), view_numel, "SLSL view numel mismatch");

    // Candle Tensor - Contiguous
    let device = Device::Cpu;
    let candle_tensor_cont =
        CandleTensor::from_vec(data.clone(), &[shape[0], shape[1]], &device).unwrap();
    assert!(
        candle_tensor_cont.is_contiguous(),
        "Candle tensor should be contiguous"
    );
    assert_eq!(
        candle_tensor_cont.elem_count(),
        numel,
        "Candle tensor numel mismatch"
    );

    // Candle Tensor - Non-contiguous via permute
    let candle_tensor_nc = candle_tensor_cont.permute((1, 0)).unwrap();
    assert!(
        !candle_tensor_nc.is_contiguous(),
        "Candle tensor should be non-contiguous"
    );
    assert_eq!(
        candle_tensor_nc.elem_count(),
        numel,
        "Candle tensor numel mismatch"
    );

    group.bench_function("slsl_tensor_contiguous", |b| {
        b.iter(|| {
            let result = slsl_tensor_cont.to_dtype::<i64>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_tensor_non_contiguous", |b| {
        b.iter(|| {
            let result = slsl_tensor_nc.to_dtype::<i64>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_view_contiguous", |b| {
        b.iter(|| {
            let result = slsl_view_cont.to_dtype::<i64>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_view_non_contiguous", |b| {
        b.iter(|| {
            let result = slsl_view_nc.to_dtype::<i64>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("candle_tensor_contiguous", |b| {
        b.iter(|| {
            let result = candle_tensor_cont.to_dtype(CandleDType::I64).unwrap();
            black_box(result)
        })
    });

    group.bench_function("candle_tensor_non_contiguous", |b| {
        b.iter(|| {
            let result = candle_tensor_nc.to_dtype(CandleDType::I64).unwrap();
            black_box(result)
        })
    });

    group.finish();
}

fn bench_dtype_2d_i64_to_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_2d_i64_to_f32");
    let shape = [200, 200];
    let numel = shape[0] * shape[1];
    let data = generate_test_data_i64(numel);

    // SLSL Tensor - Contiguous
    let slsl_tensor_cont = Tensor::from_vec(data.clone(), shape).unwrap();
    assert!(
        slsl_tensor_cont.is_contiguous(),
        "SLSL tensor should be contiguous"
    );
    assert_eq!(
        slsl_tensor_cont.numel(),
        numel,
        "SLSL tensor numel mismatch"
    );

    // SLSL Tensor - Non-contiguous via permute
    let slsl_tensor_nc = slsl_tensor_cont.clone().permute([1, 0]).unwrap();
    assert!(
        !slsl_tensor_nc.is_contiguous(),
        "SLSL tensor should be non-contiguous"
    );
    assert_eq!(slsl_tensor_nc.numel(), numel, "SLSL tensor numel mismatch");

    // SLSL TensorView - Contiguous (slice of contiguous tensor)
    let slsl_view_cont = slsl_tensor_cont.slice(s![.., ..]);
    assert!(
        slsl_view_cont.is_contiguous(),
        "SLSL view should be contiguous"
    );
    assert_eq!(slsl_view_cont.numel(), numel, "SLSL view numel mismatch");

    // SLSL TensorView - Non-contiguous via slice
    let slsl_view_nc = slsl_tensor_cont.slice(s![10..190, 10..190]);
    assert!(
        !slsl_view_nc.is_contiguous(),
        "SLSL view should be non-contiguous"
    );
    let view_numel = 180 * 180;
    assert_eq!(slsl_view_nc.numel(), view_numel, "SLSL view numel mismatch");

    // Candle Tensor - Contiguous
    let device = Device::Cpu;
    let candle_tensor_cont =
        CandleTensor::from_vec(data.clone(), &[shape[0], shape[1]], &device).unwrap();
    assert!(
        candle_tensor_cont.is_contiguous(),
        "Candle tensor should be contiguous"
    );
    assert_eq!(
        candle_tensor_cont.elem_count(),
        numel,
        "Candle tensor numel mismatch"
    );

    // Candle Tensor - Non-contiguous via permute
    let candle_tensor_nc = candle_tensor_cont.permute((1, 0)).unwrap();
    assert!(
        !candle_tensor_nc.is_contiguous(),
        "Candle tensor should be non-contiguous"
    );
    assert_eq!(
        candle_tensor_nc.elem_count(),
        numel,
        "Candle tensor numel mismatch"
    );

    group.bench_function("slsl_tensor_contiguous", |b| {
        b.iter(|| {
            let result = slsl_tensor_cont.to_dtype::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_tensor_non_contiguous", |b| {
        b.iter(|| {
            let result = slsl_tensor_nc.to_dtype::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_view_contiguous", |b| {
        b.iter(|| {
            let result = slsl_view_cont.to_dtype::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_view_non_contiguous", |b| {
        b.iter(|| {
            let result = slsl_view_nc.to_dtype::<f32>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("candle_tensor_contiguous", |b| {
        b.iter(|| {
            let result = candle_tensor_cont.to_dtype(CandleDType::F32).unwrap();
            black_box(result)
        })
    });

    group.bench_function("candle_tensor_non_contiguous", |b| {
        b.iter(|| {
            let result = candle_tensor_nc.to_dtype(CandleDType::F32).unwrap();
            black_box(result)
        })
    });

    group.finish();
}

// ========== 3D DType Conversion Benchmarks ==========

fn bench_dtype_3d_f32_to_i64(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_3d_f32_to_i64");
    let shape = [50, 40, 50];
    let numel = shape[0] * shape[1] * shape[2];
    let data = generate_test_data_f32(numel);

    // SLSL Tensor - Contiguous
    let slsl_tensor_cont = Tensor::from_vec(data.clone(), shape).unwrap();
    assert!(
        slsl_tensor_cont.is_contiguous(),
        "SLSL tensor should be contiguous"
    );
    assert_eq!(
        slsl_tensor_cont.numel(),
        numel,
        "SLSL tensor numel mismatch"
    );

    // SLSL Tensor - Non-contiguous via permute
    let slsl_tensor_nc = slsl_tensor_cont.clone().permute([2, 0, 1]).unwrap();
    assert!(
        !slsl_tensor_nc.is_contiguous(),
        "SLSL tensor should be non-contiguous"
    );
    assert_eq!(slsl_tensor_nc.numel(), numel, "SLSL tensor numel mismatch");

    // SLSL TensorView - Contiguous (slice of contiguous tensor)
    let slsl_view_cont = slsl_tensor_cont.slice(s![.., .., ..]);
    assert!(
        slsl_view_cont.is_contiguous(),
        "SLSL view should be contiguous"
    );
    assert_eq!(slsl_view_cont.numel(), numel, "SLSL view numel mismatch");

    // SLSL TensorView - Non-contiguous via slice
    let slsl_view_nc = slsl_tensor_cont.slice(s![5..45, 5..35, 5..45]);
    assert!(
        !slsl_view_nc.is_contiguous(),
        "SLSL view should be non-contiguous"
    );
    let view_numel = 40 * 30 * 40;
    assert_eq!(slsl_view_nc.numel(), view_numel, "SLSL view numel mismatch");

    // Candle Tensor - Contiguous
    let device = Device::Cpu;
    let candle_tensor_cont =
        CandleTensor::from_vec(data.clone(), &[shape[0], shape[1], shape[2]], &device).unwrap();
    assert!(
        candle_tensor_cont.is_contiguous(),
        "Candle tensor should be contiguous"
    );
    assert_eq!(
        candle_tensor_cont.elem_count(),
        numel,
        "Candle tensor numel mismatch"
    );

    // Candle Tensor - Non-contiguous via permute
    let candle_tensor_nc = candle_tensor_cont.permute((2, 0, 1)).unwrap();
    assert!(
        !candle_tensor_nc.is_contiguous(),
        "Candle tensor should be non-contiguous"
    );
    assert_eq!(
        candle_tensor_nc.elem_count(),
        numel,
        "Candle tensor numel mismatch"
    );

    group.bench_function("slsl_tensor_contiguous", |b| {
        b.iter(|| {
            let result = slsl_tensor_cont.to_dtype::<i64>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_tensor_non_contiguous", |b| {
        b.iter(|| {
            let result = slsl_tensor_nc.to_dtype::<i64>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_view_contiguous", |b| {
        b.iter(|| {
            let result = slsl_view_cont.to_dtype::<i64>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_view_non_contiguous", |b| {
        b.iter(|| {
            let result = slsl_view_nc.to_dtype::<i64>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("candle_tensor_contiguous", |b| {
        b.iter(|| {
            let result = candle_tensor_cont.to_dtype(CandleDType::I64).unwrap();
            black_box(result)
        })
    });

    group.bench_function("candle_tensor_non_contiguous", |b| {
        b.iter(|| {
            let result = candle_tensor_nc.to_dtype(CandleDType::I64).unwrap();
            black_box(result)
        })
    });

    group.finish();
}

// ========== 4D DType Conversion Benchmarks ==========

fn bench_dtype_4d_f32_to_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_4d_f32_to_u8");
    let shape = [20, 15, 10, 12];
    let numel = shape[0] * shape[1] * shape[2] * shape[3];
    let data = generate_test_data_f32(numel);

    // SLSL Tensor - Contiguous
    let slsl_tensor_cont = Tensor::from_vec(data.clone(), shape).unwrap();
    assert!(
        slsl_tensor_cont.is_contiguous(),
        "SLSL tensor should be contiguous"
    );
    assert_eq!(
        slsl_tensor_cont.numel(),
        numel,
        "SLSL tensor numel mismatch"
    );

    // SLSL Tensor - Non-contiguous via permute
    let slsl_tensor_nc = slsl_tensor_cont.clone().permute([3, 1, 0, 2]).unwrap();
    assert!(
        !slsl_tensor_nc.is_contiguous(),
        "SLSL tensor should be non-contiguous"
    );
    assert_eq!(slsl_tensor_nc.numel(), numel, "SLSL tensor numel mismatch");

    // SLSL TensorView - Contiguous (slice of contiguous tensor)
    let slsl_view_cont = slsl_tensor_cont.slice(s![.., .., .., ..]);
    assert!(
        slsl_view_cont.is_contiguous(),
        "SLSL view should be contiguous"
    );
    assert_eq!(slsl_view_cont.numel(), numel, "SLSL view numel mismatch");

    // SLSL TensorView - Non-contiguous via slice
    let slsl_view_nc = slsl_tensor_cont.slice(s![2..18, 2..13, 1..9, 1..11]);
    assert!(
        !slsl_view_nc.is_contiguous(),
        "SLSL view should be non-contiguous"
    );
    let view_numel = 16 * 11 * 8 * 10;
    assert_eq!(slsl_view_nc.numel(), view_numel, "SLSL view numel mismatch");

    // Candle Tensor - Contiguous
    let device = Device::Cpu;
    let candle_tensor_cont = CandleTensor::from_vec(
        data.clone(),
        &[shape[0], shape[1], shape[2], shape[3]],
        &device,
    )
    .unwrap();
    assert!(
        candle_tensor_cont.is_contiguous(),
        "Candle tensor should be contiguous"
    );
    assert_eq!(
        candle_tensor_cont.elem_count(),
        numel,
        "Candle tensor numel mismatch"
    );

    // Candle Tensor - Non-contiguous via permute
    let candle_tensor_nc = candle_tensor_cont.permute((3, 1, 0, 2)).unwrap();
    assert!(
        !candle_tensor_nc.is_contiguous(),
        "Candle tensor should be non-contiguous"
    );
    assert_eq!(
        candle_tensor_nc.elem_count(),
        numel,
        "Candle tensor numel mismatch"
    );

    group.bench_function("slsl_tensor_contiguous", |b| {
        b.iter(|| {
            let result = slsl_tensor_cont.to_dtype::<u8>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_tensor_non_contiguous", |b| {
        b.iter(|| {
            let result = slsl_tensor_nc.to_dtype::<u8>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_view_contiguous", |b| {
        b.iter(|| {
            let result = slsl_view_cont.to_dtype::<u8>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("slsl_view_non_contiguous", |b| {
        b.iter(|| {
            let result = slsl_view_nc.to_dtype::<u8>().unwrap();
            black_box(result)
        })
    });

    group.bench_function("candle_tensor_contiguous", |b| {
        b.iter(|| {
            let result = candle_tensor_cont.to_dtype(CandleDType::U8).unwrap();
            black_box(result)
        })
    });

    group.bench_function("candle_tensor_non_contiguous", |b| {
        b.iter(|| {
            let result = candle_tensor_nc.to_dtype(CandleDType::U8).unwrap();
            black_box(result)
        })
    });

    group.finish();
}

// ========== Benchmark Group Registration ==========

criterion_group!(
    benches,
    bench_dtype_2d_f32_to_i64,
    bench_dtype_2d_i64_to_f32,
    bench_dtype_3d_f32_to_i64,
    bench_dtype_4d_f32_to_u8
);
criterion_main!(benches);
