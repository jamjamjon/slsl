use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::ArcArray;
use slsl::Tensor;
use std::hint::black_box;

// Data scale definitions
#[derive(Clone, Copy)]
enum DataScale {
    Small,
    Medium,
    Large,
}

impl DataScale {
    fn get_1d_size(&self) -> usize {
        match self {
            DataScale::Small => 100,
            DataScale::Medium => 10_000,
            DataScale::Large => 1_000_000,
        }
    }

    fn get_2d_size(&self) -> (usize, usize) {
        match self {
            DataScale::Small => (10, 10),
            DataScale::Medium => (100, 100),
            DataScale::Large => (1000, 1000),
        }
    }

    fn get_3d_size(&self) -> (usize, usize, usize) {
        match self {
            DataScale::Small => (5, 5, 5),
            DataScale::Medium => (50, 50, 20),
            DataScale::Large => (100, 100, 100),
        }
    }

    fn get_4d_size(&self) -> (usize, usize, usize, usize) {
        match self {
            DataScale::Small => (3, 3, 3, 3),
            DataScale::Medium => (10, 10, 10, 10),
            DataScale::Large => (20, 20, 20, 20),
        }
    }

    fn get_5d_size(&self) -> (usize, usize, usize, usize, usize) {
        match self {
            DataScale::Small => (3, 3, 3, 3, 3),
            DataScale::Medium => (8, 8, 8, 8, 8),
            DataScale::Large => (15, 15, 15, 15, 15),
        }
    }

    fn get_6d_size(&self) -> (usize, usize, usize, usize, usize, usize) {
        match self {
            DataScale::Small => (3, 3, 3, 3, 3, 3),
            DataScale::Medium => (6, 6, 6, 6, 6, 6),
            DataScale::Large => (10, 10, 10, 10, 10, 10),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            DataScale::Small => "small",
            DataScale::Medium => "medium",
            DataScale::Large => "large",
        }
    }
}

// Generic data generation functions - only for Back position
fn generate_1d_data<T>(scale: DataScale) -> (Tensor, ndarray::ArcArray1<T>, Vec<[usize; 1]>)
where
    T: slsl::TensorElement + ndarray::NdFloat + From<f32>,
{
    let size = scale.get_1d_size();
    let data: Vec<T> = (0..size).map(|i| T::from_f32(i as f32)).collect();
    let tensor = Tensor::from_vec(data.clone(), [size]).unwrap();
    let ndarray_tensor = ArcArray::from_vec(data);

    // Only Back position: access from the end
    let indices = (0..50).map(|i| [size - 1 - (i % (size / 10))]).collect();

    (tensor, ndarray_tensor, indices)
}

fn generate_2d_data<T>(scale: DataScale) -> (Tensor, ndarray::ArcArray2<T>, Vec<[usize; 2]>)
where
    T: slsl::TensorElement + ndarray::NdFloat + From<f32>,
{
    let (rows, cols) = scale.get_2d_size();
    let data: Vec<T> = (0..rows * cols).map(|i| T::from_f32(i as f32)).collect();
    let tensor = Tensor::from_vec(data.clone(), [rows, cols]).unwrap();
    let ndarray_tensor = ArcArray::from_shape_vec((rows, cols), data).unwrap();

    // Only Back position: access from the end
    let indices = (0..50)
        .map(|i| [rows - 1 - (i % (rows / 4)), cols - 1 - (i % (cols / 4))])
        .collect();

    (tensor, ndarray_tensor, indices)
}

fn generate_3d_data<T>(
    scale: DataScale,
) -> (Tensor, ndarray::ArcArray<T, ndarray::Ix3>, Vec<[usize; 3]>)
where
    T: slsl::TensorElement + ndarray::NdFloat + From<f32>,
{
    let (d1, d2, d3) = scale.get_3d_size();
    let data: Vec<T> = (0..d1 * d2 * d3).map(|i| T::from_f32(i as f32)).collect();
    let tensor = Tensor::from_vec(data.clone(), [d1, d2, d3]).unwrap();
    let ndarray_tensor = ArcArray::from_shape_vec((d1, d2, d3), data).unwrap();

    // Only Back position: access from the end
    let indices = (0..50)
        .map(|i| {
            [
                d1 - 1 - (i % (d1 / 4)),
                d2 - 1 - (i % (d2 / 4)),
                d3 - 1 - (i % (d3 / 4)),
            ]
        })
        .collect();

    (tensor, ndarray_tensor, indices)
}

fn generate_4d_data<T>(
    scale: DataScale,
) -> (Tensor, ndarray::ArcArray<T, ndarray::Ix4>, Vec<[usize; 4]>)
where
    T: slsl::TensorElement + ndarray::NdFloat + From<f32>,
{
    let (d1, d2, d3, d4) = scale.get_4d_size();
    let data: Vec<T> = (0..d1 * d2 * d3 * d4)
        .map(|i| T::from_f32(i as f32))
        .collect();
    let tensor = Tensor::from_vec(data.clone(), [d1, d2, d3, d4]).unwrap();
    let ndarray_tensor = ArcArray::from_shape_vec((d1, d2, d3, d4), data).unwrap();

    // Only Back position: access from the end
    let indices = (0..50)
        .map(|i| {
            [
                d1 - 1 - (i % (d1.max(4) / 4)),
                d2 - 1 - (i % (d2.max(4) / 4)),
                d3 - 1 - (i % (d3.max(4) / 4)),
                d4 - 1 - (i % (d4.max(4) / 4)),
            ]
        })
        .collect();

    (tensor, ndarray_tensor, indices)
}

fn generate_5d_data<T>(
    scale: DataScale,
) -> (Tensor, ndarray::ArcArray<T, ndarray::Ix5>, Vec<[usize; 5]>)
where
    T: slsl::TensorElement + ndarray::NdFloat + From<f32>,
{
    let (d1, d2, d3, d4, d5) = scale.get_5d_size();
    let data: Vec<T> = (0..d1 * d2 * d3 * d4 * d5)
        .map(|i| T::from_f32(i as f32))
        .collect();
    let tensor = Tensor::from_vec(data.clone(), [d1, d2, d3, d4, d5]).unwrap();
    let ndarray_tensor = ArcArray::from_shape_vec((d1, d2, d3, d4, d5), data).unwrap();

    // Only Back position: access from the end
    let indices = (0..50)
        .map(|i| {
            [
                d1 - 1 - (i % (d1.max(4) / 4)),
                d2 - 1 - (i % (d2.max(4) / 4)),
                d3 - 1 - (i % (d3.max(4) / 4)),
                d4 - 1 - (i % (d4.max(4) / 4)),
                d5 - 1 - (i % (d5.max(4) / 4)),
            ]
        })
        .collect();

    (tensor, ndarray_tensor, indices)
}

fn generate_6d_data<T>(
    scale: DataScale,
) -> (Tensor, ndarray::ArcArray<T, ndarray::Ix6>, Vec<[usize; 6]>)
where
    T: slsl::TensorElement + ndarray::NdFloat + From<f32>,
{
    let (d1, d2, d3, d4, d5, d6) = scale.get_6d_size();
    let data: Vec<T> = (0..d1 * d2 * d3 * d4 * d5 * d6)
        .map(|i| T::from_f32(i as f32))
        .collect();
    let tensor = Tensor::from_vec(data.clone(), [d1, d2, d3, d4, d5, d6]).unwrap();
    let ndarray_tensor = ArcArray::from_shape_vec((d1, d2, d3, d4, d5, d6), data).unwrap();

    // Only Back position: access from the end
    let indices = (0..50)
        .map(|i| {
            [
                d1 - 1 - (i % (d1.max(4) / 4)),
                d2 - 1 - (i % (d2.max(4) / 4)),
                d3 - 1 - (i % (d3.max(4) / 4)),
                d4 - 1 - (i % (d4.max(4) / 4)),
                d5 - 1 - (i % (d5.max(4) / 4)),
                d6 - 1 - (i % (d6.max(4) / 4)),
            ]
        })
        .collect();

    (tensor, ndarray_tensor, indices)
}

// Comprehensive benchmark for f32 only with different dimensions
fn bench_comprehensive_scalar_access(c: &mut Criterion) {
    let scales = [DataScale::Small, DataScale::Medium, DataScale::Large];

    // Test 1D f32
    for &scale in &scales {
        let group_name = format!("1d_f32_{}", scale.name());
        let mut group = c.benchmark_group(&*group_name);

        let (tensor, ndarray_tensor, indices) = generate_1d_data::<f32>(scale);
        let tensor_view = tensor.view();

        group.bench_function("slsl_at", |b| {
            b.iter(|| {
                for &idx in &indices {
                    black_box(tensor.at::<f32>(idx));
                }
            })
        });

        group.bench_function("slsl_view_at", |b| {
            b.iter(|| {
                for &idx in &indices {
                    black_box(tensor_view.at::<f32>(idx));
                }
            })
        });

        group.bench_function("ndarray", |b| {
            b.iter(|| {
                for &[idx] in &indices {
                    black_box(ndarray_tensor[idx]);
                }
            })
        });

        group.finish();
    }

    // Test 2D f32
    for &scale in &scales {
        let group_name = format!("2d_f32_{}", scale.name());
        let mut group = c.benchmark_group(&*group_name);

        let (tensor, ndarray_tensor, indices) = generate_2d_data::<f32>(scale);
        let tensor_view = tensor.view();

        group.bench_function("slsl_at", |b| {
            b.iter(|| {
                for &idx in &indices {
                    black_box(tensor.at::<f32>(idx));
                }
            })
        });

        group.bench_function("slsl_view_at", |b| {
            b.iter(|| {
                for &idx in &indices {
                    black_box(tensor_view.at::<f32>(idx));
                }
            })
        });

        group.bench_function("ndarray", |b| {
            b.iter(|| {
                for &[i, j] in &indices {
                    black_box(ndarray_tensor[[i, j]]);
                }
            })
        });

        group.finish();
    }

    // Test 3D f32
    for &scale in &scales {
        let group_name = format!("3d_f32_{}", scale.name());
        let mut group = c.benchmark_group(&*group_name);

        let (tensor, ndarray_tensor, indices) = generate_3d_data::<f32>(scale);
        let tensor_view = tensor.view();

        group.bench_function("slsl_at", |b| {
            b.iter(|| {
                for &idx in &indices {
                    black_box(tensor.at::<f32>(idx));
                }
            })
        });

        group.bench_function("slsl_view_at", |b| {
            b.iter(|| {
                for &idx in &indices {
                    black_box(tensor_view.at::<f32>(idx));
                }
            })
        });

        group.bench_function("ndarray", |b| {
            b.iter(|| {
                for &[i, j, k] in &indices {
                    black_box(ndarray_tensor[[i, j, k]]);
                }
            })
        });

        group.finish();
    }

    // Test 4D f32
    for &scale in &scales {
        let group_name = format!("4d_f32_{}", scale.name());
        let mut group = c.benchmark_group(&*group_name);

        let (tensor, ndarray_tensor, indices) = generate_4d_data::<f32>(scale);
        let tensor_view = tensor.view();

        group.bench_function("slsl_at", |b| {
            b.iter(|| {
                for &idx in &indices {
                    black_box(tensor.at::<f32>(idx));
                }
            })
        });

        group.bench_function("slsl_view_at", |b| {
            b.iter(|| {
                for &idx in &indices {
                    black_box(tensor_view.at::<f32>(idx));
                }
            })
        });

        group.bench_function("ndarray", |b| {
            b.iter(|| {
                for &idx in &indices {
                    black_box(ndarray_tensor[idx]);
                }
            })
        });

        group.finish();
    }

    // Test 5D f32 (only small and medium to save time)
    for &scale in &[DataScale::Small, DataScale::Medium] {
        let group_name = format!("5d_f32_{}", scale.name());
        let mut group = c.benchmark_group(&*group_name);

        let (tensor, ndarray_tensor, indices) = generate_5d_data::<f32>(scale);
        let tensor_view = tensor.view();

        group.bench_function("slsl_at", |b| {
            b.iter(|| {
                for &idx in &indices {
                    black_box(tensor.at::<f32>(idx));
                }
            })
        });

        group.bench_function("slsl_view_at", |b| {
            b.iter(|| {
                for &idx in &indices {
                    black_box(tensor_view.at::<f32>(idx));
                }
            })
        });

        group.bench_function("ndarray", |b| {
            b.iter(|| {
                for &idx in &indices {
                    black_box(ndarray_tensor[idx]);
                }
            })
        });

        group.finish();
    }

    // Test 6D f32 (only small to save time)
    let group_name = "6d_f32_small";
    let mut group = c.benchmark_group(group_name);

    let (tensor, ndarray_tensor, indices) = generate_6d_data::<f32>(DataScale::Small);
    let tensor_view = tensor.view();

    group.bench_function("slsl_at", |b| {
        b.iter(|| {
            for &idx in &indices {
                black_box(tensor.at::<f32>(idx));
            }
        })
    });

    group.bench_function("slsl_view_at", |b| {
        b.iter(|| {
            for &idx in &indices {
                black_box(tensor_view.at::<f32>(idx));
            }
        })
    });

    group.bench_function("ndarray", |b| {
        b.iter(|| {
            for &idx in &indices {
                black_box(ndarray_tensor[idx]);
            }
        })
    });

    group.finish();
}

// Single element access benchmarks for precise timing
fn bench_single_element_access(c: &mut Criterion) {
    let group_name = "single_element_f32";
    let mut group = c.benchmark_group(group_name);

    let tensor_1d = Tensor::from_vec(vec![1.0f32; 1000], [1000]).unwrap();
    let tensor_2d = Tensor::from_vec(vec![1.0f32; 10000], [100, 100]).unwrap();
    let tensor_3d = Tensor::from_vec(vec![1.0f32; 8000], [20, 20, 20]).unwrap();
    let tensor_4d = Tensor::from_vec(vec![1.0f32; 10000], [10, 10, 10, 10]).unwrap();
    let tensor_5d = Tensor::from_vec(vec![1.0f32; 7776], [6, 6, 6, 6, 6]).unwrap();
    let tensor_6d = Tensor::from_vec(vec![1.0f32; 46656], [6, 6, 6, 6, 6, 6]).unwrap();

    let tensor_1d_view = tensor_1d.view();
    let tensor_2d_view = tensor_2d.view();
    let tensor_3d_view = tensor_3d.view();
    let tensor_4d_view = tensor_4d.view();
    let tensor_5d_view = tensor_5d.view();
    let tensor_6d_view = tensor_6d.view();

    let ndarray_1d = ArcArray::from_vec(vec![1.0f32; 1000]);
    let ndarray_2d = ArcArray::from_shape_vec((100, 100), vec![1.0f32; 10000]).unwrap();
    let ndarray_3d = ArcArray::from_shape_vec((20, 20, 20), vec![1.0f32; 8000]).unwrap();
    let ndarray_4d = ArcArray::from_shape_vec((10, 10, 10, 10), vec![1.0f32; 10000]).unwrap();
    let ndarray_5d = ArcArray::from_shape_vec((6, 6, 6, 6, 6), vec![1.0f32; 7776]).unwrap();
    let ndarray_6d = ArcArray::from_shape_vec((6, 6, 6, 6, 6, 6), vec![1.0f32; 46656]).unwrap();

    // 1D access
    group.bench_function("slsl_at_1d", |b| {
        b.iter(|| black_box(tensor_1d.at::<f32>([999])))
    });

    group.bench_function("slsl_view_at_1d", |b| {
        b.iter(|| black_box(tensor_1d_view.at::<f32>([999])))
    });

    group.bench_function("ndarray_1d", |b| b.iter(|| black_box(ndarray_1d[999])));

    // 2D access
    group.bench_function("slsl_at_2d", |b| {
        b.iter(|| black_box(tensor_2d.at::<f32>([99, 99])))
    });

    group.bench_function("slsl_view_at_2d", |b| {
        b.iter(|| black_box(tensor_2d_view.at::<f32>([99, 99])))
    });

    group.bench_function("ndarray_2d", |b| b.iter(|| black_box(ndarray_2d[[99, 99]])));

    // 3D access
    group.bench_function("slsl_at_3d", |b| {
        b.iter(|| black_box(tensor_3d.at::<f32>([19, 19, 19])))
    });

    group.bench_function("slsl_view_at_3d", |b| {
        b.iter(|| black_box(tensor_3d_view.at::<f32>([19, 19, 19])))
    });

    group.bench_function("ndarray_3d", |b| {
        b.iter(|| black_box(ndarray_3d[[19, 19, 19]]))
    });

    // 4D access
    group.bench_function("slsl_at_4d", |b| {
        b.iter(|| black_box(tensor_4d.at::<f32>([9, 9, 9, 9])))
    });

    group.bench_function("slsl_view_at_4d", |b| {
        b.iter(|| black_box(tensor_4d_view.at::<f32>([9, 9, 9, 9])))
    });

    group.bench_function("ndarray_4d", |b| {
        b.iter(|| black_box(ndarray_4d[[9, 9, 9, 9]]))
    });

    // 5D access
    group.bench_function("slsl_at_5d", |b| {
        b.iter(|| black_box(tensor_5d.at::<f32>([5, 5, 5, 5, 5])))
    });

    group.bench_function("slsl_view_at_5d", |b| {
        b.iter(|| black_box(tensor_5d_view.at::<f32>([5, 5, 5, 5, 5])))
    });

    group.bench_function("ndarray_5d", |b| {
        b.iter(|| black_box(ndarray_5d[[5, 5, 5, 5, 5]]))
    });

    // 6D access
    group.bench_function("slsl_at_6d", |b| {
        b.iter(|| black_box(tensor_6d.at::<f32>([5, 5, 5, 5, 5, 5])))
    });

    group.bench_function("slsl_view_at_6d", |b| {
        b.iter(|| black_box(tensor_6d_view.at::<f32>([5, 5, 5, 5, 5, 5])))
    });

    group.bench_function("ndarray_6d", |b| {
        b.iter(|| black_box(ndarray_6d[[5, 5, 5, 5, 5, 5]]))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_comprehensive_scalar_access,
    bench_single_element_access
);
criterion_main!(benches);
