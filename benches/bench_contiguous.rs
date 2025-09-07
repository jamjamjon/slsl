use candle_core::{Device, Tensor as CandleTensor};
use criterion::{criterion_group, criterion_main, Criterion};
use slsl::Tensor;
use std::hint::black_box;

// ========== Test Data Generation ==========

fn generate_test_data_f32(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i % 1000) as f32 * 0.1).collect()
}

// ========== Debug Test ==========

fn debug_permute() {
    println!("=== Debug Permute ===");

    // Create a simple 2x3 tensor
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, [2, 3]).unwrap();

    println!("Original tensor:");
    println!("  Shape: {:?}", tensor.shape());
    println!("  Strides: {:?}", tensor.strides());
    println!("  Is contiguous: {}", tensor.is_contiguous());

    // Try to permute
    let permuted = tensor.permute([1, 0]).unwrap();
    println!("Permuted tensor:");
    println!("  Shape: {:?}", permuted.shape());
    println!("  Strides: {:?}", permuted.strides());
    println!("  Is contiguous: {}", permuted.is_contiguous());

    // Convert to owned tensor
    let owned = permuted.to_owned().unwrap();
    println!("Owned permuted tensor:");
    println!("  Shape: {:?}", owned.shape());
    println!("  Strides: {:?}", owned.strides());
    println!("  Is contiguous: {}", owned.is_contiguous());

    println!("=== End Debug ===");
}

// ========== 1D Contiguous Benchmarks ==========

fn bench_contiguous_1d(c: &mut Criterion) {
    // Debug permute functionality
    debug_permute();

    let mut group = c.benchmark_group("contiguous_1d");

    // Small size
    let small_shape = [1000];
    let small_numel = small_shape[0];
    let small_data = generate_test_data_f32(small_numel);

    // Medium size
    let medium_shape = [10000];
    let medium_numel = medium_shape[0];
    let medium_data = generate_test_data_f32(medium_numel);

    // Large size
    let large_shape = [100000];
    let large_numel = large_shape[0];
    let large_data = generate_test_data_f32(large_numel);

    // Small size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (2D tensor)
        let slsl_tensor = Tensor::from_vec(small_data.clone(), [100, 10]).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([1, 0]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([1, 0]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(small_data.clone(), &[100, 10], &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((1, 0)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_1d_small", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_1d_small", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_1d_small", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    // Medium size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (2D tensor)
        let slsl_tensor = Tensor::from_vec(medium_data.clone(), [100, 100]).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([1, 0]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([1, 0]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(medium_data.clone(), &[100, 100], &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((1, 0)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_1d_medium", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_1d_medium", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_1d_medium", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    // Large size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (2D tensor)
        let slsl_tensor = Tensor::from_vec(large_data.clone(), [1000, 100]).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([1, 0]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([1, 0]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(large_data.clone(), &[1000, 100], &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((1, 0)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_1d_large", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_1d_large", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_1d_large", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

// ========== 2D Contiguous Benchmarks ==========

fn bench_contiguous_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("contiguous_2d");

    // Small size
    let small_shape = [50, 50];
    let small_numel = small_shape[0] * small_shape[1];
    let small_data = generate_test_data_f32(small_numel);

    // Medium size
    let medium_shape = [200, 200];
    let medium_numel = medium_shape[0] * medium_shape[1];
    let medium_data = generate_test_data_f32(medium_numel);

    // Large size
    let large_shape = [500, 500];
    let large_numel = large_shape[0] * large_shape[1];
    let large_data = generate_test_data_f32(large_numel);

    // Small size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (2D tensor)
        let slsl_tensor = Tensor::from_vec(small_data.clone(), [10, 100]).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([1, 0]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([1, 0]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(small_data.clone(), &small_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((1, 0)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_2d_small", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_2d_small", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_2d_small", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    // Medium size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (2D tensor)
        let slsl_tensor = Tensor::from_vec(medium_data.clone(), medium_shape).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([1, 0]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([1, 0]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(medium_data.clone(), &medium_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((1, 0)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_2d_medium", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_2d_medium", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_2d_medium", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    // Large size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (2D tensor)
        let slsl_tensor = Tensor::from_vec(large_data.clone(), large_shape).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([1, 0]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([1, 0]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(large_data.clone(), &large_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((1, 0)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_2d_large", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_2d_large", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_2d_large", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

// ========== 3D Contiguous Benchmarks ==========

fn bench_contiguous_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("contiguous_3d");

    // Small size
    let small_shape = [20, 20, 20];
    let small_numel = small_shape[0] * small_shape[1] * small_shape[2];
    let small_data = generate_test_data_f32(small_numel);

    // Medium size
    let medium_shape = [50, 40, 50];
    let medium_numel = medium_shape[0] * medium_shape[1] * medium_shape[2];
    let medium_data = generate_test_data_f32(medium_numel);

    // Large size
    let large_shape = [100, 80, 100];
    let large_numel = large_shape[0] * large_shape[1] * large_shape[2];
    let large_data = generate_test_data_f32(large_numel);

    // Small size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (3D tensor)
        let slsl_tensor = Tensor::from_vec(small_data.clone(), small_shape).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([2, 0, 1]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([2, 0, 1]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(small_data.clone(), &small_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((2, 0, 1)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_3d_small", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_3d_small", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_3d_small", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    // Medium size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (3D tensor)
        let slsl_tensor = Tensor::from_vec(medium_data.clone(), medium_shape).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([2, 0, 1]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([2, 0, 1]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(medium_data.clone(), &medium_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((2, 0, 1)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_3d_medium", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_3d_medium", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_3d_medium", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    // Large size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (3D tensor)
        let slsl_tensor = Tensor::from_vec(large_data.clone(), large_shape).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([2, 0, 1]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([2, 0, 1]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(large_data.clone(), &large_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((2, 0, 1)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_3d_large", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_3d_large", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_3d_large", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

// ========== 4D Contiguous Benchmarks ==========

fn bench_contiguous_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("contiguous_4d");

    // Small size
    let small_shape = [10, 10, 10, 10];
    let small_numel = small_shape[0] * small_shape[1] * small_shape[2] * small_shape[3];
    let small_data = generate_test_data_f32(small_numel);

    // Medium size
    let medium_shape = [20, 15, 20, 15];
    let medium_numel = medium_shape[0] * medium_shape[1] * medium_shape[2] * medium_shape[3];
    let medium_data = generate_test_data_f32(medium_numel);

    // Large size
    let large_shape = [40, 30, 40, 30];
    let large_numel = large_shape[0] * large_shape[1] * large_shape[2] * large_shape[3];
    let large_data = generate_test_data_f32(large_numel);

    // Small size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (4D tensor)
        let slsl_tensor = Tensor::from_vec(small_data.clone(), small_shape).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([3, 0, 1, 2]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([3, 0, 1, 2]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(small_data.clone(), &small_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((3, 0, 1, 2)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_4d_small", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_4d_small", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_4d_small", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    // Medium size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (4D tensor)
        let slsl_tensor = Tensor::from_vec(medium_data.clone(), medium_shape).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([3, 0, 1, 2]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([3, 0, 1, 2]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(medium_data.clone(), &medium_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((3, 0, 1, 2)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_4d_medium", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_4d_medium", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_4d_medium", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    // Large size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (4D tensor)
        let slsl_tensor = Tensor::from_vec(large_data.clone(), large_shape).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([3, 0, 1, 2]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([3, 0, 1, 2]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(large_data.clone(), &large_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((3, 0, 1, 2)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_4d_large", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_4d_large", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_4d_large", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

// ========== 5D Contiguous Benchmarks ==========

fn bench_contiguous_5d(c: &mut Criterion) {
    let mut group = c.benchmark_group("contiguous_5d");

    // Small size
    let small_shape = [8, 8, 8, 8, 8];
    let small_numel =
        small_shape[0] * small_shape[1] * small_shape[2] * small_shape[3] * small_shape[4];
    let small_data = generate_test_data_f32(small_numel);

    // Medium size
    let medium_shape = [12, 10, 12, 10, 12];
    let medium_numel =
        medium_shape[0] * medium_shape[1] * medium_shape[2] * medium_shape[3] * medium_shape[4];
    let medium_data = generate_test_data_f32(medium_numel);

    // Large size
    let large_shape = [20, 15, 20, 15, 20];
    let large_numel =
        large_shape[0] * large_shape[1] * large_shape[2] * large_shape[3] * large_shape[4];
    let large_data = generate_test_data_f32(large_numel);

    // Small size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (5D tensor)
        let slsl_tensor = Tensor::from_vec(small_data.clone(), small_shape).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([4, 0, 1, 2, 3]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([4, 0, 1, 2, 3]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(small_data.clone(), &small_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((4, 0, 1, 2, 3)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_5d_small", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_5d_small", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_5d_small", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    // Medium size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (5D tensor)
        let slsl_tensor = Tensor::from_vec(medium_data.clone(), medium_shape).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([4, 0, 1, 2, 3]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([4, 0, 1, 2, 3]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(medium_data.clone(), &medium_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((4, 0, 1, 2, 3)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_5d_medium", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_5d_medium", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_5d_medium", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    // Large size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (5D tensor)
        let slsl_tensor = Tensor::from_vec(large_data.clone(), large_shape).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([4, 0, 1, 2, 3]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([4, 0, 1, 2, 3]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(large_data.clone(), &large_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((4, 0, 1, 2, 3)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_5d_large", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_5d_large", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_5d_large", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

// ========== 6D Contiguous Benchmarks ==========

fn bench_contiguous_6d(c: &mut Criterion) {
    let mut group = c.benchmark_group("contiguous_6d");

    // Small size
    let small_shape = [6, 6, 6, 6, 6, 6];
    let small_numel = small_shape[0]
        * small_shape[1]
        * small_shape[2]
        * small_shape[3]
        * small_shape[4]
        * small_shape[5];
    let small_data = generate_test_data_f32(small_numel);

    // Medium size
    let medium_shape = [8, 7, 8, 7, 8, 7];
    let medium_numel = medium_shape[0]
        * medium_shape[1]
        * medium_shape[2]
        * medium_shape[3]
        * medium_shape[4]
        * medium_shape[5];
    let medium_data = generate_test_data_f32(medium_numel);

    // Large size
    let large_shape = [12, 10, 12, 10, 12, 10];
    let large_numel = large_shape[0]
        * large_shape[1]
        * large_shape[2]
        * large_shape[3]
        * large_shape[4]
        * large_shape[5];
    let large_data = generate_test_data_f32(large_numel);

    // Small size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (6D tensor)
        let slsl_tensor = Tensor::from_vec(small_data.clone(), small_shape).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([5, 0, 1, 2, 3, 4]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([5, 0, 1, 2, 3, 4]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(small_data.clone(), &small_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((5, 0, 1, 2, 3, 4)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_6d_small", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_6d_small", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_6d_small", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    // Medium size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (6D tensor)
        let slsl_tensor = Tensor::from_vec(medium_data.clone(), medium_shape).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([5, 0, 1, 2, 3, 4]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([5, 0, 1, 2, 3, 4]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(medium_data.clone(), &medium_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((5, 0, 1, 2, 3, 4)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_6d_medium", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_6d_medium", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_6d_medium", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    // Large size benchmarks
    {
        // SLSL Tensor - Make non-contiguous via permute (6D tensor)
        let slsl_tensor = Tensor::from_vec(large_data.clone(), large_shape).unwrap();
        let slsl_tensor_nc = slsl_tensor.clone().permute([5, 0, 1, 2, 3, 4]).unwrap();
        assert!(
            !slsl_tensor_nc.is_contiguous(),
            "SLSL tensor should be non-contiguous"
        );

        // SLSL TensorView - Make non-contiguous via permute
        let slsl_view = slsl_tensor.permute([5, 0, 1, 2, 3, 4]).unwrap();
        assert!(
            !slsl_view.is_contiguous(),
            "SLSL view should be non-contiguous"
        );

        // Candle Tensor - Make non-contiguous via permute
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(large_data.clone(), &large_shape, &device).unwrap();
        let candle_tensor_nc = candle_tensor.permute((5, 0, 1, 2, 3, 4)).unwrap();
        assert!(
            !candle_tensor_nc.is_contiguous(),
            "Candle tensor should be non-contiguous"
        );

        group.bench_function("slsl_tensor_6d_large", |b| {
            b.iter(|| {
                let result = slsl_tensor_nc.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("slsl_view_6d_large", |b| {
            b.iter(|| {
                let result = slsl_view.to_contiguous().unwrap();
                black_box(result)
            })
        });

        group.bench_function("candle_tensor_6d_large", |b| {
            b.iter(|| {
                let result = candle_tensor_nc.contiguous().unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

// ========== Benchmark Group Registration ==========

criterion_group!(
    benches,
    bench_contiguous_1d,
    bench_contiguous_2d,
    bench_contiguous_3d,
    bench_contiguous_4d,
    bench_contiguous_5d,
    bench_contiguous_6d
);
criterion_main!(benches);
