use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2, Array3, Array4, Axis};
use slsl::Tensor;
use std::hint::black_box;

const SIZES_1D: &[usize] = &[10, 64, 256, 512, 1024, 2048, 4096];

const SIZES_2D: &[(usize, usize)] = &[(32, 32), (128, 128), (256, 256), (512, 512), (1024, 1024)];

const SIZES_3D: &[(usize, usize, usize)] =
    &[(16, 16, 16), (64, 64, 64), (128, 128, 128), (256, 256, 256)];

const SIZES_4D: &[(usize, usize, usize, usize)] =
    &[(8, 8, 8, 8), (64, 64, 64, 64), (128, 128, 128, 128)];

// 1D benchmarks
fn bench_cat_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("cat_1d");

    for &size in SIZES_1D {
        let data1: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let data2: Vec<f32> = (size..2 * size).map(|i| i as f32).collect();

        let tensor1 = Tensor::from_vec(data1.clone(), [size]).unwrap();
        let tensor2 = Tensor::from_vec(data2.clone(), [size]).unwrap();

        let array1 = Array1::from_vec(data1);
        let array2 = Array1::from_vec(data2);

        // Test concatenation along dimension 0
        group.bench_function(format!("slsl_dim0_{}", size), |b| {
            b.iter(|| {
                let tensors = vec![tensor1.clone(), tensor2.clone()];
                black_box(Tensor::cat(&tensors, 0).unwrap())
            })
        });

        group.bench_function(format!("ndarray_dim0_{}", size), |b| {
            b.iter(|| {
                let arrays = vec![array1.view(), array2.view()];
                black_box(ndarray::concatenate(Axis(0), &arrays).unwrap())
            })
        });
    }

    group.finish();
}

fn bench_stack_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("stack_1d");

    for &size in SIZES_1D {
        let data1: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let data2: Vec<f32> = (size..2 * size).map(|i| i as f32).collect();

        let tensor1 = Tensor::from_vec(data1.clone(), [size]).unwrap();
        let tensor2 = Tensor::from_vec(data2.clone(), [size]).unwrap();

        let array1 = Array1::from_vec(data1);
        let array2 = Array1::from_vec(data2);

        // Test stacking along dimension 0 (new leading dimension)
        group.bench_function(format!("slsl_dim0_{}", size), |b| {
            b.iter(|| {
                let tensors = vec![tensor1.clone(), tensor2.clone()];
                black_box(Tensor::stack(&tensors, 0).unwrap())
            })
        });

        group.bench_function(format!("ndarray_dim0_{}", size), |b| {
            b.iter(|| {
                let arrays = vec![array1.view(), array2.view()];
                black_box(ndarray::stack(Axis(0), &arrays).unwrap())
            })
        });

        // Test stacking along dimension 1 (new trailing dimension)
        group.bench_function(format!("slsl_dim1_{}", size), |b| {
            b.iter(|| {
                let tensors = vec![tensor1.clone(), tensor2.clone()];
                black_box(Tensor::stack(&tensors, 1).unwrap())
            })
        });

        group.bench_function(format!("ndarray_dim1_{}", size), |b| {
            b.iter(|| {
                let arrays = vec![array1.view(), array2.view()];
                black_box(ndarray::stack(Axis(1), &arrays).unwrap())
            })
        });
    }

    group.finish();
}

// 2D benchmarks
fn bench_cat_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("cat_2d");

    for &(rows, cols) in SIZES_2D {
        let data1: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let data2: Vec<f32> = (rows * cols..2 * rows * cols).map(|i| i as f32).collect();

        let tensor1 = Tensor::from_vec(data1.clone(), [rows, cols]).unwrap();
        let tensor2 = Tensor::from_vec(data2.clone(), [rows, cols]).unwrap();

        let array1 = Array2::from_shape_vec((rows, cols), data1).unwrap();
        let array2 = Array2::from_shape_vec((rows, cols), data2).unwrap();

        // Test concatenation along dimension 0 (rows)
        group.bench_function(format!("slsl_dim0_{}x{}", rows, cols), |b| {
            b.iter(|| {
                let tensors = vec![tensor1.clone(), tensor2.clone()];
                black_box(Tensor::cat(&tensors, 0).unwrap())
            })
        });

        group.bench_function(format!("ndarray_dim0_{}x{}", rows, cols), |b| {
            b.iter(|| {
                let arrays = vec![array1.view(), array2.view()];
                black_box(ndarray::concatenate(Axis(0), &arrays).unwrap())
            })
        });

        // Test concatenation along dimension 1 (columns)
        group.bench_function(format!("slsl_dim1_{}x{}", rows, cols), |b| {
            b.iter(|| {
                let tensors = vec![tensor1.clone(), tensor2.clone()];
                black_box(Tensor::cat(&tensors, 1).unwrap())
            })
        });

        group.bench_function(format!("ndarray_dim1_{}x{}", rows, cols), |b| {
            b.iter(|| {
                let arrays = vec![array1.view(), array2.view()];
                black_box(ndarray::concatenate(Axis(1), &arrays).unwrap())
            })
        });
    }

    group.finish();
}

fn bench_stack_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("stack_2d");

    for &(rows, cols) in SIZES_2D {
        let data1: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let data2: Vec<f32> = (rows * cols..2 * rows * cols).map(|i| i as f32).collect();

        let tensor1 = Tensor::from_vec(data1.clone(), [rows, cols]).unwrap();
        let tensor2 = Tensor::from_vec(data2.clone(), [rows, cols]).unwrap();

        let array1 = Array2::from_shape_vec((rows, cols), data1).unwrap();
        let array2 = Array2::from_shape_vec((rows, cols), data2).unwrap();

        // Test stacking along dimension 0 (new leading dimension)
        group.bench_function(format!("slsl_dim0_{}x{}", rows, cols), |b| {
            b.iter(|| {
                let tensors = vec![tensor1.clone(), tensor2.clone()];
                black_box(Tensor::stack(&tensors, 0).unwrap())
            })
        });

        group.bench_function(format!("ndarray_dim0_{}x{}", rows, cols), |b| {
            b.iter(|| {
                let arrays = vec![array1.view(), array2.view()];
                black_box(ndarray::stack(Axis(0), &arrays).unwrap())
            })
        });

        // Test stacking along dimension 1 (intermediate dimension)
        group.bench_function(format!("slsl_dim1_{}x{}", rows, cols), |b| {
            b.iter(|| {
                let tensors = vec![tensor1.clone(), tensor2.clone()];
                black_box(Tensor::stack(&tensors, 1).unwrap())
            })
        });

        group.bench_function(format!("ndarray_dim1_{}x{}", rows, cols), |b| {
            b.iter(|| {
                let arrays = vec![array1.view(), array2.view()];
                black_box(ndarray::stack(Axis(1), &arrays).unwrap())
            })
        });

        // Test stacking along dimension 2 (new trailing dimension)
        group.bench_function(format!("slsl_dim2_{}x{}", rows, cols), |b| {
            b.iter(|| {
                let tensors = vec![tensor1.clone(), tensor2.clone()];
                black_box(Tensor::stack(&tensors, 2).unwrap())
            })
        });

        group.bench_function(format!("ndarray_dim2_{}x{}", rows, cols), |b| {
            b.iter(|| {
                let arrays = vec![array1.view(), array2.view()];
                black_box(ndarray::stack(Axis(2), &arrays).unwrap())
            })
        });
    }

    group.finish();
}

// 3D benchmarks
fn bench_cat_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("cat_3d");

    for &(d1, d2, d3) in SIZES_3D {
        let size = d1 * d2 * d3;
        let data1: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let data2: Vec<f32> = (size..2 * size).map(|i| i as f32).collect();

        let tensor1 = Tensor::from_vec(data1.clone(), [d1, d2, d3]).unwrap();
        let tensor2 = Tensor::from_vec(data2.clone(), [d1, d2, d3]).unwrap();

        let array1 = Array3::from_shape_vec((d1, d2, d3), data1).unwrap();
        let array2 = Array3::from_shape_vec((d1, d2, d3), data2).unwrap();

        // Test concatenation along all dimensions
        for dim in 0..3 {
            group.bench_function(format!("slsl_dim{}_{}x{}x{}", dim, d1, d2, d3), |b| {
                b.iter(|| {
                    let tensors = vec![tensor1.clone(), tensor2.clone()];
                    black_box(Tensor::cat(&tensors, dim).unwrap())
                })
            });

            group.bench_function(format!("ndarray_dim{}_{}x{}x{}", dim, d1, d2, d3), |b| {
                b.iter(|| {
                    let arrays = vec![array1.view(), array2.view()];
                    black_box(ndarray::concatenate(Axis(dim), &arrays).unwrap())
                })
            });
        }
    }

    group.finish();
}

fn bench_stack_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("stack_3d");

    for &(d1, d2, d3) in SIZES_3D {
        let size = d1 * d2 * d3;
        let data1: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let data2: Vec<f32> = (size..2 * size).map(|i| i as f32).collect();

        let tensor1 = Tensor::from_vec(data1.clone(), [d1, d2, d3]).unwrap();
        let tensor2 = Tensor::from_vec(data2.clone(), [d1, d2, d3]).unwrap();

        let array1 = Array3::from_shape_vec((d1, d2, d3), data1).unwrap();
        let array2 = Array3::from_shape_vec((d1, d2, d3), data2).unwrap();

        // Test stacking along all dimensions (0, 1, 2, 3)
        for dim in 0..4 {
            group.bench_function(format!("slsl_dim{}_{}x{}x{}", dim, d1, d2, d3), |b| {
                b.iter(|| {
                    let tensors = vec![tensor1.clone(), tensor2.clone()];
                    black_box(Tensor::stack(&tensors, dim).unwrap())
                })
            });

            group.bench_function(format!("ndarray_dim{}_{}x{}x{}", dim, d1, d2, d3), |b| {
                b.iter(|| {
                    let arrays = vec![array1.view(), array2.view()];
                    black_box(ndarray::stack(Axis(dim), &arrays).unwrap())
                })
            });
        }
    }

    group.finish();
}

// 4D benchmarks
fn bench_cat_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("cat_4d");

    for &(d1, d2, d3, d4) in SIZES_4D {
        let size = d1 * d2 * d3 * d4;
        let data1: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let data2: Vec<f32> = (size..2 * size).map(|i| i as f32).collect();

        let tensor1 = Tensor::from_vec(data1.clone(), [d1, d2, d3, d4]).unwrap();
        let tensor2 = Tensor::from_vec(data2.clone(), [d1, d2, d3, d4]).unwrap();

        let array1 = Array4::from_shape_vec((d1, d2, d3, d4), data1).unwrap();
        let array2 = Array4::from_shape_vec((d1, d2, d3, d4), data2).unwrap();

        // Test concatenation along all dimensions
        for dim in 0..4 {
            group.bench_function(
                format!("slsl_dim{}_{}x{}x{}x{}", dim, d1, d2, d3, d4),
                |b| {
                    b.iter(|| {
                        let tensors = vec![tensor1.clone(), tensor2.clone()];
                        black_box(Tensor::cat(&tensors, dim).unwrap())
                    })
                },
            );

            group.bench_function(
                format!("ndarray_dim{}_{}x{}x{}x{}", dim, d1, d2, d3, d4),
                |b| {
                    b.iter(|| {
                        let arrays = vec![array1.view(), array2.view()];
                        black_box(ndarray::concatenate(Axis(dim), &arrays).unwrap())
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_stack_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("stack_4d");

    for &(d1, d2, d3, d4) in SIZES_4D {
        let size = d1 * d2 * d3 * d4;
        let data1: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let data2: Vec<f32> = (size..2 * size).map(|i| i as f32).collect();

        let tensor1 = Tensor::from_vec(data1.clone(), [d1, d2, d3, d4]).unwrap();
        let tensor2 = Tensor::from_vec(data2.clone(), [d1, d2, d3, d4]).unwrap();

        let array1 = Array4::from_shape_vec((d1, d2, d3, d4), data1).unwrap();
        let array2 = Array4::from_shape_vec((d1, d2, d3, d4), data2).unwrap();

        // Test stacking along all dimensions (0, 1, 2, 3, 4)
        for dim in 0..5 {
            group.bench_function(
                format!("slsl_dim{}_{}x{}x{}x{}", dim, d1, d2, d3, d4),
                |b| {
                    b.iter(|| {
                        let tensors = vec![tensor1.clone(), tensor2.clone()];
                        black_box(Tensor::stack(&tensors, dim).unwrap())
                    })
                },
            );

            group.bench_function(
                format!("ndarray_dim{}_{}x{}x{}x{}", dim, d1, d2, d3, d4),
                |b| {
                    b.iter(|| {
                        let arrays = vec![array1.view(), array2.view()];
                        black_box(ndarray::stack(Axis(dim), &arrays).unwrap())
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cat_1d,
    bench_stack_1d,
    bench_cat_2d,
    bench_stack_2d,
    bench_cat_3d,
    bench_stack_3d,
    bench_cat_4d,
    bench_stack_4d,
);
criterion_main!(benches);
