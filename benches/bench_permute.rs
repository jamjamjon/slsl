use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array;
use slsl::Tensor;
use std::hint::black_box;

fn bench_permute_2d(c: &mut Criterion) {
    let rows = 1000;
    let cols = 1000;
    let size = rows * cols;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

    // slsl tensor
    let tensor = Tensor::from_vec(data.clone(), [rows, cols]).unwrap();

    // ndarray
    let ndarray = Array::from_shape_vec((rows, cols), data).unwrap();

    c.bench_function("permute_slsl_2d", |b| {
        b.iter(|| {
            let result = tensor.clone().permute(black_box([1, 0])).unwrap();
            black_box(result);
        })
    });

    c.bench_function("permute_ndarray_2d", |b| {
        b.iter(|| {
            let result = ndarray.view().permuted_axes(black_box([1, 0]));
            black_box(result);
        })
    });
}

fn bench_permute_3d(c: &mut Criterion) {
    let dim1 = 100;
    let dim2 = 100;
    let dim3 = 100;
    let size = dim1 * dim2 * dim3;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

    // slsl tensor
    let tensor = Tensor::from_vec(data.clone(), [dim1, dim2, dim3]).unwrap();

    // ndarray
    let ndarray = Array::from_shape_vec((dim1, dim2, dim3), data).unwrap();

    c.bench_function("permute_slsl_3d", |b| {
        b.iter(|| {
            let result = tensor.clone().permute(black_box([2, 0, 1])).unwrap();
            black_box(result);
        })
    });

    c.bench_function("permute_ndarray_3d", |b| {
        b.iter(|| {
            let result = ndarray.view().permuted_axes(black_box([2, 0, 1]));
            black_box(result);
        })
    });
}

fn bench_permute_4d(c: &mut Criterion) {
    let dim1 = 32;
    let dim2 = 32;
    let dim3 = 32;
    let dim4 = 32;
    let size = dim1 * dim2 * dim3 * dim4;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

    // slsl tensor
    let tensor = Tensor::from_vec(data.clone(), [dim1, dim2, dim3, dim4]).unwrap();

    // ndarray
    let ndarray = Array::from_shape_vec((dim1, dim2, dim3, dim4), data).unwrap();

    c.bench_function("permute_slsl_4d", |b| {
        b.iter(|| {
            let result = tensor.clone().permute(black_box([3, 1, 0, 2])).unwrap();
            black_box(result);
        })
    });

    c.bench_function("permute_ndarray_4d", |b| {
        b.iter(|| {
            let result = ndarray.view().permuted_axes(black_box([3, 1, 0, 2]));
            black_box(result);
        })
    });
}

fn bench_permute_image_like(c: &mut Criterion) {
    // Common image-like tensor: NCHW -> NHWC
    let batch = 8;
    let channels = 3;
    let height = 224;
    let width = 224;
    let size = batch * channels * height * width;
    let data: Vec<f32> = (0..size).map(|i| (i % 256) as f32).collect();

    // slsl tensor
    let tensor = Tensor::from_vec(data.clone(), [batch, channels, height, width]).unwrap();

    // ndarray
    let ndarray = Array::from_shape_vec((batch, channels, height, width), data).unwrap();

    c.bench_function("permute_slsl_nchw_to_nhwc", |b| {
        b.iter(|| {
            let result = tensor.clone().permute(black_box([0, 2, 3, 1])).unwrap();
            black_box(result);
        })
    });

    c.bench_function("permute_ndarray_nchw_to_nhwc", |b| {
        b.iter(|| {
            let result = ndarray.view().permuted_axes(black_box([0, 2, 3, 1]));
            black_box(result);
        })
    });
}

fn bench_flip_dims(c: &mut Criterion) {
    let dim1 = 100;
    let dim2 = 100;
    let dim3 = 100;
    let size = dim1 * dim2 * dim3;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

    // slsl tensor
    let tensor = Tensor::from_vec(data.clone(), [dim1, dim2, dim3]).unwrap();

    // ndarray - simulate flip_dims with permuted_axes
    let ndarray = Array::from_shape_vec((dim1, dim2, dim3), data).unwrap();

    c.bench_function("flip_dims_slsl", |b| {
        b.iter(|| {
            let result = tensor.clone().flip_dims().unwrap();
            black_box(result);
        })
    });

    c.bench_function("flip_dims_ndarray_equivalent", |b| {
        b.iter(|| {
            let result = ndarray.view().permuted_axes(black_box([2, 1, 0]));
            black_box(result);
        })
    });
}

fn bench_permute_small_tensors(c: &mut Criterion) {
    // Test with small tensors to see overhead
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();

    // slsl tensor
    let tensor = Tensor::from_vec(data.clone(), [2, 3, 4]).unwrap();

    // ndarray
    let ndarray = Array::from_shape_vec((2, 3, 4), data).unwrap();

    c.bench_function("permute_slsl_small", |b| {
        b.iter(|| {
            let result = tensor.clone().permute(black_box([2, 0, 1])).unwrap();
            black_box(result);
        })
    });

    c.bench_function("permute_ndarray_small", |b| {
        b.iter(|| {
            let result = ndarray.view().permuted_axes(black_box([2, 0, 1]));
            black_box(result);
        })
    });
}

fn bench_permute_validation_overhead(c: &mut Criterion) {
    // Test the validation overhead by comparing with a hypothetical no-validation version
    let dim1 = 100;
    let dim2 = 100;
    let dim3 = 100;
    let size = dim1 * dim2 * dim3;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

    let tensor = Tensor::from_vec(data, [dim1, dim2, dim3]).unwrap();

    c.bench_function("permute_slsl_with_validation", |b| {
        b.iter(|| {
            let result = tensor.clone().permute(black_box([2, 0, 1])).unwrap();
            black_box(result);
        })
    });
}

criterion_group!(
    benches,
    bench_permute_2d,
    bench_permute_3d,
    bench_permute_4d,
    bench_permute_image_like,
    bench_flip_dims,
    bench_permute_small_tensors,
    bench_permute_validation_overhead
);
criterion_main!(benches);
