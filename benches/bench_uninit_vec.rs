use criterion::{black_box, criterion_group, criterion_main, Criterion};
use slsl::UninitVec;

fn bench_vec_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec vs UninitVec - Full");

    // Test different sizes
    for size in [100, 1000, 10000, 100000] {
        group.bench_function(format!("vec![value; {size}]"), |b| {
            b.iter(|| {
                let _result: Vec<f32> = vec![42.0f32; black_box(size)];
            });
        });

        group.bench_function(format!("UninitVec::new({size}).full(value)"), |b| {
            b.iter(|| {
                let _result = UninitVec::<f32>::new(black_box(size)).full(42.0f32);
            });
        });
    }

    group.finish();
}

fn bench_vec_fill(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec vs UninitVec - Fill");

    // Test different sizes
    for size in [100, 1000, 10000, 100000] {
        group.bench_function(format!("Vec::with_capacity({size}) + fill"), |b| {
            b.iter(|| {
                let mut vec = Vec::<f32>::with_capacity(black_box(size));
                vec.resize(black_box(size), 42.0f32);
                let _result = vec;
            });
        });

        group.bench_function(format!("UninitVec::new({size}).init_with(fill)"), |b| {
            b.iter(|| {
                let _result = UninitVec::<f32>::new(black_box(size)).init_with(|dst| {
                    dst.fill(42.0f32);
                });
            });
        });
    }

    group.finish();
}

fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Allocation Patterns");

    // Test repeated allocations
    for size in [1000, 10000] {
        group.bench_function(format!("repeated vec![value; {size}]"), |b| {
            b.iter(|| {
                for _ in 0..100 {
                    let _result: Vec<f32> = vec![42.0f32; black_box(size)];
                }
            });
        });

        group.bench_function(format!("repeated UninitVec::new({size}).full()"), |b| {
            b.iter(|| {
                for _ in 0..100 {
                    let _result = UninitVec::<f32>::new(black_box(size)).full(42.0f32);
                }
            });
        });
    }

    group.finish();
}

fn bench_different_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("Different Data Types");

    let size = 10000;

    // Test f32
    group.bench_function("vec![f32; size]", |b| {
        b.iter(|| {
            let _result: Vec<f32> = vec![42.0f32; black_box(size)];
        });
    });

    group.bench_function("UninitVec::<f32>::new(size).full()", |b| {
        b.iter(|| {
            let _result = UninitVec::<f32>::new(black_box(size)).full(42.0f32);
        });
    });

    // Test f64
    group.bench_function("vec![f64; size]", |b| {
        b.iter(|| {
            let _result: Vec<f64> = vec![42.0f64; black_box(size)];
        });
    });

    group.bench_function("UninitVec::<f64>::new(size).full()", |b| {
        b.iter(|| {
            let _result = UninitVec::<f64>::new(black_box(size)).full(42.0f64);
        });
    });

    // Test i32
    group.bench_function("vec![i32; size]", |b| {
        b.iter(|| {
            let _result: Vec<i32> = vec![42i32; black_box(size)];
        });
    });

    group.bench_function("UninitVec::<i32>::new(size).full()", |b| {
        b.iter(|| {
            let _result = UninitVec::<i32>::new(black_box(size)).full(42i32);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_vec_full,
    bench_vec_fill,
    bench_memory_allocation,
    bench_different_types
);
criterion_main!(benches);
