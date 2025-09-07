use candle_core::{Device, IndexOp, Tensor as CandleTensor};
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2, Array3};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use slsl::{s, Tensor};
use std::hint::black_box;

// Test data configuration
const SIZES_1D: [usize; 4] = [10, 100, 1000, 10000];
const SIZES_2D: [(usize, usize); 4] = [(10, 10), (50, 50), (100, 100), (500, 500)];
const SIZES_3D: [(usize, usize, usize); 3] = [(10, 10, 10), (20, 20, 20), (50, 50, 50)];

fn generate_test_data_1d(size: usize) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(42);
    (0..size).map(|_| rng.random()).collect()
}

fn generate_test_data_2d(rows: usize, cols: usize) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(42);
    (0..rows * cols).map(|_| rng.random()).collect()
}

fn generate_test_data_3d(d1: usize, d2: usize, d3: usize) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(42);
    (0..d1 * d2 * d3).map(|_| rng.random()).collect()
}

// Generate test data
// fn generate_test_data() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
//     let mut rng = SmallRng::seed_from_u64(42);

//     let data_1d: Vec<f32> = (0..SMALL_1D).map(|_| rng.random()).collect();
//     let data_2d_small: Vec<f32> = (0..SMALL_2D[0] * SMALL_2D[1])
//         .map(|_| rng.random())
//         .collect();
//     let data_2d_medium: Vec<f32> = (0..MEDIUM_2D[0] * MEDIUM_2D[1])
//         .map(|_| rng.random())
//         .collect();
//     let data_2d_large: Vec<f32> = (0..LARGE_2D[0] * LARGE_2D[1])
//         .map(|_| rng.random())
//         .collect();

//     (data_1d, data_2d_small, data_2d_medium, data_2d_large)
// }

// 1D single index slice performance test
fn bench_slice_1d_single_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("🎯_slice_1d_single_index");
    group.sample_size(10000);

    for &size in &SIZES_1D {
        let data = generate_test_data_1d(size);
        let idx = size / 2; // Middle index

        // SLSL
        let slsl_tensor = Tensor::from_vec(data.clone(), [size]).unwrap();
        group.bench_function(format!("slsl_size_{size}"), |b| {
            b.iter(|| black_box(slsl_tensor.slice(s![black_box(idx)])))
        });

        // Candle - reference performance
        let device = Device::Cpu;
        let candle_tensor = CandleTensor::from_vec(data.clone(), (size,), &device).unwrap();
        group.bench_function(format!("candle_size_{size}"), |b| {
            b.iter(|| black_box(candle_tensor.i(black_box(idx)).unwrap()))
        });

        // NDArray - reference performance
        let ndarray_tensor = Array1::from_vec(data.clone());
        group.bench_function(format!("ndarray_size_{size}"), |b| {
            b.iter(|| black_box(ndarray_tensor.slice(ndarray::s![black_box(idx)])))
        });
    }
    group.finish();
}

// 2D single index slice performance test
fn bench_slice_2d_single_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("🧊_slice_2d_single_index");
    group.sample_size(10000);

    for &(rows, cols) in &SIZES_2D {
        let data = generate_test_data_2d(rows, cols);
        let (idx_r, idx_c) = (rows / 2, cols / 2); // Middle indices

        // SLSL
        let slsl_tensor = Tensor::from_vec(data.clone(), [rows, cols]).unwrap();
        group.bench_function(format!("slsl_{rows}x{cols}"), |b| {
            b.iter(|| black_box(slsl_tensor.slice(s![black_box(idx_r), black_box(idx_c)])))
        });

        // Candle - reference performance
        let device = Device::Cpu;
        let candle_tensor = CandleTensor::from_vec(data.clone(), (rows, cols), &device).unwrap();
        group.bench_function(format!("candle_{rows}x{cols}"), |b| {
            b.iter(|| {
                black_box(
                    candle_tensor
                        .i((black_box(idx_r), black_box(idx_c)))
                        .unwrap(),
                )
            })
        });

        // NDArray - reference performance
        let ndarray_tensor = Array2::from_shape_vec((rows, cols), data.clone()).unwrap();
        group.bench_function(format!("ndarray_{rows}x{cols}"), |b| {
            b.iter(|| {
                black_box(ndarray_tensor.slice(ndarray::s![black_box(idx_r), black_box(idx_c)]))
            })
        });
    }
    group.finish();
}

// 1D range slice performance test
fn bench_slice_1d_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("📏_slice_1d_range");
    group.sample_size(5000);

    for &size in &SIZES_1D {
        let data = generate_test_data_1d(size);
        let range = 0..size / 2; // Slice half the tensor

        // SLSL
        let slsl_tensor = Tensor::from_vec(data.clone(), [size]).unwrap();
        group.bench_function(format!("slsl_size_{size}"), |b| {
            b.iter(|| black_box(slsl_tensor.slice(s![range.clone()])))
        });

        // Candle - reference performance
        let device = Device::Cpu;
        let candle_tensor = CandleTensor::from_vec(data.clone(), (size,), &device).unwrap();
        group.bench_function(format!("candle_size_{size}"), |b| {
            b.iter(|| black_box(candle_tensor.i(range.clone()).unwrap()))
        });

        // NDArray - reference performance
        let ndarray_tensor = Array1::from_vec(data.clone());
        group.bench_function(format!("ndarray_size_{size}"), |b| {
            b.iter(|| black_box(ndarray_tensor.slice(ndarray::s![range.clone()])))
        });
    }
    group.finish();
}

// 2D range slice performance test
fn bench_slice_2d_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("🧊_slice_2d_range");
    group.sample_size(2000);

    for &(rows, cols) in &SIZES_2D {
        let data = generate_test_data_2d(rows, cols);
        let (r_start, r_end) = (rows / 4, rows * 3 / 4); // Middle half
        let (c_start, c_end) = (cols / 4, cols * 3 / 4); // Middle half

        // SLSL
        let slsl_tensor = Tensor::from_vec(data.clone(), [rows, cols]).unwrap();
        group.bench_function(format!("slsl_{rows}x{cols}"), |b| {
            b.iter(|| black_box(slsl_tensor.slice(s![r_start..r_end, c_start..c_end])))
        });

        // Candle - reference performance
        let device = Device::Cpu;
        let candle_tensor = CandleTensor::from_vec(data.clone(), (rows, cols), &device).unwrap();
        group.bench_function(format!("candle_{rows}x{cols}"), |b| {
            b.iter(|| black_box(candle_tensor.i((r_start..r_end, c_start..c_end)).unwrap()))
        });

        // NDArray - reference performance
        let ndarray_tensor = Array2::from_shape_vec((rows, cols), data.clone()).unwrap();
        group.bench_function(format!("ndarray_{rows}x{cols}"), |b| {
            b.iter(|| black_box(ndarray_tensor.slice(ndarray::s![r_start..r_end, c_start..c_end])))
        });
    }
    group.finish();
}

// Mixed slicing performance test (index + range)
fn bench_slice_mixed_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("🧩_slice_mixed_patterns");
    group.sample_size(2000);

    for &(rows, cols) in &SIZES_2D {
        let data = generate_test_data_2d(rows, cols);
        let row_idx = rows / 2; // Middle row index
        let (c_start, c_end) = (cols / 4, cols * 3 / 4); // Middle half of columns

        // SLSL
        let slsl_tensor = Tensor::from_vec(data.clone(), [rows, cols]).unwrap();
        group.bench_function(format!("slsl_{rows}x{cols}"), |b| {
            b.iter(|| black_box(slsl_tensor.slice(s![row_idx, c_start..c_end])))
        });

        // Candle - reference performance
        let device = Device::Cpu;
        let candle_tensor = CandleTensor::from_vec(data.clone(), (rows, cols), &device).unwrap();
        group.bench_function(format!("candle_{rows}x{cols}"), |b| {
            b.iter(|| black_box(candle_tensor.i((row_idx, c_start..c_end)).unwrap()))
        });

        // NDArray - reference performance
        let ndarray_tensor = Array2::from_shape_vec((rows, cols), data.clone()).unwrap();
        group.bench_function(format!("ndarray_{rows}x{cols}"), |b| {
            b.iter(|| black_box(ndarray_tensor.slice(ndarray::s![row_idx, c_start..c_end])))
        });
    }
    group.finish();
}

// Full slice performance test
fn bench_slice_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("🌐_slice_full");
    group.sample_size(10000);

    for &(rows, cols) in &SIZES_2D {
        let data = generate_test_data_2d(rows, cols);

        // SLSL
        let slsl_tensor = Tensor::from_vec(data.clone(), [rows, cols]).unwrap();
        group.bench_function(format!("slsl_{rows}x{cols}"), |b| {
            b.iter(|| black_box(slsl_tensor.slice(s![..])))
        });

        // Candle - reference performance
        let device = Device::Cpu;
        let candle_tensor = CandleTensor::from_vec(data.clone(), (rows, cols), &device).unwrap();
        group.bench_function(format!("candle_{rows}x{cols}"), |b| {
            b.iter(|| black_box(candle_tensor.i((.., ..)).unwrap()))
        });

        // NDArray - reference performance
        let ndarray_tensor = Array2::from_shape_vec((rows, cols), data.clone()).unwrap();
        group.bench_function(format!("ndarray_{rows}x{cols}"), |b| {
            b.iter(|| black_box(ndarray_tensor.slice(ndarray::s![.., ..])))
        });
    }
    group.finish();
}

// 3D single index slice performance test
fn bench_slice_3d_single_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("🧊_slice_3d_single_index");
    group.sample_size(1000);

    for &(dim0, dim1, dim2) in &SIZES_3D {
        let data = generate_test_data_3d(dim0, dim1, dim2);
        let (idx0, idx1, idx2) = (dim0 / 2, dim1 / 2, dim2 / 2); // Middle indices

        // SLSL
        let slsl_tensor = Tensor::from_vec(data.clone(), [dim0, dim1, dim2]).unwrap();
        group.bench_function(format!("slsl_{dim0}x{dim1}x{dim2}"), |b| {
            b.iter(|| {
                black_box(slsl_tensor.slice(s![black_box(idx0), black_box(idx1), black_box(idx2)]))
            })
        });

        // Candle - reference performance
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(data.clone(), (dim0, dim1, dim2), &device).unwrap();
        group.bench_function(format!("candle_{dim0}x{dim1}x{dim2}"), |b| {
            b.iter(|| {
                black_box(
                    candle_tensor
                        .i((black_box(idx0), black_box(idx1), black_box(idx2)))
                        .unwrap(),
                )
            })
        });

        // NDArray - reference performance
        let ndarray_tensor = Array3::from_shape_vec((dim0, dim1, dim2), data.clone()).unwrap();
        group.bench_function(format!("ndarray_{dim0}x{dim1}x{dim2}"), |b| {
            b.iter(|| ndarray_tensor.slice(ndarray::s![idx0, idx1, idx2]))
        });
    }
    group.finish();
}

// 3D range slice performance test
fn bench_slice_3d_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("🎯_slice_3d_range");
    group.sample_size(10000);

    for &(dim0, dim1, dim2) in &SIZES_3D {
        let data = generate_test_data_3d(dim0, dim1, dim2);
        let (d0_start, d0_end) = (dim0 / 4, dim0 * 3 / 4);
        let (d1_start, d1_end) = (dim1 / 4, dim1 * 3 / 4);
        let (d2_start, d2_end) = (dim2 / 4, dim2 * 3 / 4);

        // SLSL
        let slsl_tensor = Tensor::from_vec(data.clone(), [dim0, dim1, dim2]).unwrap();
        group.bench_function(format!("slsl_{dim0}x{dim1}x{dim2}"), |b| {
            b.iter(|| {
                black_box(slsl_tensor.slice(s![
                    d0_start..d0_end,
                    d1_start..d1_end,
                    d2_start..d2_end
                ]))
            })
        });

        // Candle - reference performance
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(data.clone(), (dim0, dim1, dim2), &device).unwrap();
        group.bench_function(format!("candle_{dim0}x{dim1}x{dim2}"), |b| {
            b.iter(|| {
                black_box(
                    candle_tensor
                        .i((d0_start..d0_end, d1_start..d1_end, d2_start..d2_end))
                        .unwrap(),
                )
            })
        });

        // NDArray - reference performance
        let ndarray_tensor = Array3::from_shape_vec((dim0, dim1, dim2), data.clone()).unwrap();
        group.bench_function(format!("ndarray_{dim0}x{dim1}x{dim2}"), |b| {
            b.iter(|| {
                black_box(ndarray_tensor.slice(ndarray::s![
                    d0_start..d0_end,
                    d1_start..d1_end,
                    d2_start..d2_end
                ]))
            })
        });
    }
    group.finish();
}

// Mixed 3D slicing performance test (index + range)
fn bench_slice_3d_mixed_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("🧩_slice_3d_mixed_patterns");
    group.sample_size(2000);

    for &(dim0, dim1, dim2) in &SIZES_3D {
        let data = generate_test_data_3d(dim0, dim1, dim2);
        let idx0 = dim0 / 2; // Middle index for first dimension
        let (d1_start, d1_end) = (dim1 / 4, dim1 * 3 / 4); // Middle half for second dimension
        let idx2 = dim2 / 2; // Middle index for third dimension

        // SLSL
        let slsl_tensor = Tensor::from_vec(data.clone(), [dim0, dim1, dim2]).unwrap();
        group.bench_function(format!("slsl_{dim0}x{dim1}x{dim2}"), |b| {
            b.iter(|| {
                black_box(slsl_tensor.slice(s![black_box(idx0), d1_start..d1_end, black_box(idx2)]))
            })
        });

        // Candle - reference performance
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(data.clone(), (dim0, dim1, dim2), &device).unwrap();
        group.bench_function(format!("candle_{dim0}x{dim1}x{dim2}"), |b| {
            b.iter(|| {
                black_box(
                    candle_tensor
                        .i((black_box(idx0), d1_start..d1_end, black_box(idx2)))
                        .unwrap(),
                )
            })
        });

        // NDArray - reference performance
        let ndarray_tensor = Array3::from_shape_vec((dim0, dim1, dim2), data.clone()).unwrap();
        group.bench_function(format!("ndarray_{dim0}x{dim1}x{dim2}"), |b| {
            b.iter(|| black_box(ndarray_tensor.slice(ndarray::s![idx0, d1_start..d1_end, idx2])))
        });
    }
    group.finish();
}

// 3D full slice performance test
fn bench_slice_3d_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("🌐_slice_3d_full");
    group.sample_size(10000);

    for &(dim0, dim1, dim2) in &SIZES_3D {
        let data = generate_test_data_3d(dim0, dim1, dim2);

        // SLSL
        let slsl_tensor = Tensor::from_vec(data.clone(), [dim0, dim1, dim2]).unwrap();
        group.bench_function(format!("slsl_{dim0}x{dim1}x{dim2}"), |b| {
            b.iter(|| black_box(slsl_tensor.slice(s![..])))
        });

        // Candle - reference performance
        let device = Device::Cpu;
        let candle_tensor =
            CandleTensor::from_vec(data.clone(), (dim0, dim1, dim2), &device).unwrap();
        group.bench_function(format!("candle_{dim0}x{dim1}x{dim2}"), |b| {
            b.iter(|| black_box(candle_tensor.i((.., .., ..)).unwrap()))
        });

        // NDArray - reference performance
        let ndarray_tensor = Array3::from_shape_vec((dim0, dim1, dim2), data.clone()).unwrap();
        group.bench_function(format!("ndarray_{dim0}x{dim1}x{dim2}"), |b| {
            b.iter(|| black_box(ndarray_tensor.slice(ndarray::s![.., .., ..])))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_slice_1d_single_index,
    bench_slice_2d_single_index,
    bench_slice_1d_range,
    bench_slice_2d_range,
    bench_slice_mixed_patterns,
    bench_slice_full,
    bench_slice_3d_single_index,
    bench_slice_3d_range,
    bench_slice_3d_mixed_patterns,
    bench_slice_3d_full
);
criterion_main!(benches);
