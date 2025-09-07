use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{s, ArcArray, Axis};

fn gen_data_1d(size: usize) -> ArcArray<f32, ndarray::Ix1> {
    ArcArray::from_iter((0..size).map(|x| x as f32))
}

fn gen_data_2d(dim0: usize, dim1: usize) -> ArcArray<f32, ndarray::Ix2> {
    let data = (0..(dim0 * dim1)).map(|x| x as f32).collect::<Vec<_>>();
    ArcArray::from_shape_vec((dim0, dim1), data).unwrap()
}

fn gen_data_3d(dim0: usize, dim1: usize, dim2: usize) -> ArcArray<f32, ndarray::Ix3> {
    let data = (0..(dim0 * dim1 * dim2))
        .map(|x| x as f32)
        .collect::<Vec<_>>();
    ArcArray::from_shape_vec((dim0, dim1, dim2), data).unwrap()
}

fn bench_axis_iter_vs_slice(c: &mut Criterion) {
    let size_1d = 1000;
    let arr1d = gen_data_1d(size_1d);

    let dim0_2d = 100;
    let dim1_2d = 100;
    let arr2d = gen_data_2d(dim0_2d, dim1_2d);

    let dim0_3d = 30;
    let dim1_3d = 30;
    let dim2_3d = 30;
    let arr3d = gen_data_3d(dim0_3d, dim1_3d, dim2_3d);

    // ----------------- 1D -----------------
    let mut group = c.benchmark_group("ArcArray_1D_axis_iter_vs_slice");

    group.bench_function("axis_iter_dim0", |b| {
        b.iter(|| {
            let mut count = 0f32;
            for subview in arr1d.axis_iter(Axis(0)) {
                count += subview[[]];
            }
            black_box(count);
        })
    });

    group.bench_function("slice_loop_dim0", |b| {
        b.iter(|| {
            let mut count = 0f32;
            let len = arr1d.len_of(Axis(0));
            for i in 0..len {
                let s = arr1d.slice(s![i]);
                count += s[[]]; // scalar view
            }
            black_box(count);
        })
    });

    group.finish();

    // ----------------- 2D -----------------
    let mut group = c.benchmark_group("ArcArray_2D_axis_iter_vs_slice");

    group.bench_function("axis_iter_dim0", |b| {
        b.iter(|| {
            let mut count = 0f32;
            for subview in arr2d.axis_iter(Axis(0)) {
                count += subview[0];
            }
            black_box(count);
        })
    });

    group.bench_function("axis_iter_dim1", |b| {
        b.iter(|| {
            let mut count = 0f32;
            for subview in arr2d.axis_iter(Axis(1)) {
                count += subview[0];
            }
            black_box(count);
        })
    });

    group.bench_function("slice_loop_dim0", |b| {
        b.iter(|| {
            let mut count = 0f32;
            let len = arr2d.len_of(Axis(0));
            for i in 0..len {
                let s = arr2d.slice(s![i, ..]);
                count += s[0];
            }
            black_box(count);
        })
    });

    group.bench_function("slice_loop_dim1", |b| {
        b.iter(|| {
            let mut count = 0f32;
            let len = arr2d.len_of(Axis(1));
            for i in 0..len {
                let s = arr2d.slice(s![.., i]);
                count += s[0];
            }
            black_box(count);
        })
    });

    group.finish();

    // ----------------- 3D -----------------
    let mut group = c.benchmark_group("ArcArray_3D_axis_iter_vs_slice");

    group.bench_function("axis_iter_dim0", |b| {
        b.iter(|| {
            let mut count = 0f32;
            for subview in arr3d.axis_iter(Axis(0)) {
                count += subview[[0, 0]];
            }
            black_box(count);
        })
    });

    group.bench_function("axis_iter_dim1", |b| {
        b.iter(|| {
            let mut count = 0f32;
            for subview in arr3d.axis_iter(Axis(1)) {
                count += subview[[0, 0]];
            }
            black_box(count);
        })
    });

    group.bench_function("axis_iter_dim2", |b| {
        b.iter(|| {
            let mut count = 0f32;
            for subview in arr3d.axis_iter(Axis(2)) {
                count += subview[[0, 0]];
            }
            black_box(count);
        })
    });

    group.bench_function("slice_loop_dim0", |b| {
        b.iter(|| {
            let mut count = 0f32;
            let len = arr3d.len_of(Axis(0));
            for i in 0..len {
                let s = arr3d.slice(s![i, .., ..]);
                count += s[[0, 0]];
            }
            black_box(count);
        })
    });

    group.bench_function("slice_loop_dim1", |b| {
        b.iter(|| {
            let mut count = 0f32;
            let len = arr3d.len_of(Axis(1));
            for i in 0..len {
                let s = arr3d.slice(s![.., i, ..]);
                count += s[[0, 0]];
            }
            black_box(count);
        })
    });

    group.bench_function("slice_loop_dim2", |b| {
        b.iter(|| {
            let mut count = 0f32;
            let len = arr3d.len_of(Axis(2));
            for i in 0..len {
                let s = arr3d.slice(s![.., .., i]);
                count += s[[0, 0]];
            }
            black_box(count);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_axis_iter_vs_slice);
criterion_main!(benches);
