#![allow(unused)]

use criterion::{criterion_group, criterion_main, Criterion};
use slsl::ArrayN;
use smallvec::{smallvec, SmallVec};
use std::hint::black_box as bb;

use slsl::arrayn;

fn bench_from_slice_1d_8d(c: &mut Criterion) {
    c.bench_function("creation_1d/arrayn", |b| {
        b.iter(|| {
            let v: ArrayN<usize, 8> = ArrayN::from_slice(bb(&[1]));
            bb(v)
        })
    });

    c.bench_function("creation_1d/smallvec", |b| {
        b.iter(|| {
            let v: SmallVec<[_; 8]> = SmallVec::from_slice(bb(&[1]));
            bb(v)
        })
    });

    c.bench_function("creation_2d/arrayn", |b| {
        b.iter(|| {
            let v: ArrayN<usize, 8> = ArrayN::from_slice(bb(&[1, 3]));
            bb(v)
        })
    });
    c.bench_function("creation_2d/smallvec", |b| {
        b.iter(|| {
            let v: SmallVec<[_; 8]> = SmallVec::from_slice(bb(&[1, 3]));
            bb(v)
        })
    });

    c.bench_function("creation_3d/arrayn", |b| {
        b.iter(|| {
            let v: ArrayN<usize, 8> = ArrayN::from_slice(bb(&[1, 2, 3]));
            bb(v)
        })
    });

    c.bench_function("creation_3d/smallvec", |b| {
        b.iter(|| {
            let v: SmallVec<[_; 8]> = SmallVec::from_slice(bb(&[1, 2, 3]));
            bb(v)
        })
    });
    c.bench_function("creation_8d/arrayn", |b| {
        b.iter(|| {
            let v: ArrayN<usize, 8> = ArrayN::from_slice(bb(&[1, 2, 3, 4, 5, 6, 7, 8]));
            bb(v)
        })
    });

    c.bench_function("creation_8d/smallvec", |b| {
        b.iter(|| {
            let v: SmallVec<[_; 8]> = SmallVec::from_slice(bb(&[1, 2, 3, 4, 5, 6, 7, 8]));
            bb(v)
        })
    });
}

// Creation benchmarks
fn bench_creation_1d_8d(c: &mut Criterion) {
    c.bench_function("creation_1d/arrayn-empty", |b| {
        b.iter(|| {
            let v: ArrayN<usize, 8> = ArrayN::empty().with_len(1);
            bb(v)
        })
    });

    c.bench_function("creation_1d/arrayn", |b| {
        b.iter(|| {
            // let v: ArrayN<usize, 8> = ArrayN::new(bb(&[1]));
            let v: ArrayN<usize, 8> = arrayn![1];
            bb(v)
        })
    });

    c.bench_function("creation_1d/smallvec", |b| {
        b.iter(|| {
            let v: SmallVec<[_; 8]> = smallvec![1,];
            // let v: SmallVec<[_; 8]> = SmallVec::from_slice(bb(&[1]));
            bb(v)
        })
    });

    c.bench_function("creation_1d/smallvec-new", |b| {
        b.iter(|| {
            let v: SmallVec<[usize; 8]> = SmallVec::new();
            bb(v)
        })
    });

    c.bench_function("creation_2d/arrayn-empty", |b| {
        b.iter(|| {
            let v: ArrayN<usize, 8> = ArrayN::empty().with_len(2);
            bb(v)
        })
    });

    c.bench_function("creation_2d/arrayn", |b| {
        b.iter(|| {
            // let v: ArrayN<usize, 8> = ArrayN::new(bb(&[1, 3]));
            let v: ArrayN<usize, 8> = arrayn![1, 3];
            bb(v)
        })
    });

    c.bench_function("creation_2d/smallvec", |b| {
        b.iter(|| {
            let v: SmallVec<[_; 8]> = smallvec![1, 3];
            bb(v)
        })
    });
    c.bench_function("creation_2d/vec", |b| {
        b.iter(|| {
            let v: Vec<u8> = vec![1, 3];
            bb(v)
        })
    });

    c.bench_function("creation_3d/arrayn-empty", |b| {
        b.iter(|| {
            let v: ArrayN<usize, 8> = ArrayN::empty().with_len(3);
            bb(v)
        })
    });

    c.bench_function("creation_3d/arrayn", |b| {
        b.iter(|| {
            // let v: ArrayN<usize, 8> = ArrayN::new(bb(&[1, 2, 3]));
            let v: ArrayN<usize, 8> = arrayn![1, 2, 3];
            bb(v)
        })
    });

    c.bench_function("creation_3d/smallvec", |b| {
        b.iter(|| {
            let v: SmallVec<[_; 8]> = smallvec![1, 2, 3];
            bb(v)
        })
    });
    c.bench_function("creation_3d/vec", |b| {
        b.iter(|| {
            let v: Vec<u8> = vec![1, 2, 3];
            bb(v)
        })
    });

    c.bench_function("creation_4d/arrayn-empty", |b| {
        b.iter(|| {
            let v: ArrayN<usize, 8> = ArrayN::empty().with_len(4);
            bb(v)
        })
    });

    c.bench_function("creation_4d/arrayn", |b| {
        b.iter(|| {
            // let v: ArrayN<usize, 8> = ArrayN::new(bb(&[1, 2, 3, 4]));
            let v: ArrayN<usize, 8> = arrayn![1, 2, 3, 4];
            bb(v)
        })
    });

    c.bench_function("creation_4d/smallvec", |b| {
        b.iter(|| {
            let v: SmallVec<[_; 8]> = smallvec![1, 2, 3, 4];
            bb(v)
        })
    });
    c.bench_function("creation_4d/vec", |b| {
        b.iter(|| {
            let v: Vec<u8> = vec![1, 2, 3, 4];
            bb(v)
        })
    });

    c.bench_function("creation_5d/arrayn-empty", |b| {
        b.iter(|| {
            let v: ArrayN<usize, 8> = ArrayN::empty().with_len(5);
            bb(v)
        })
    });

    c.bench_function("creation_5d/arrayn", |b| {
        b.iter(|| {
            // let v: ArrayN<usize, 8> = ArrayN::new(bb(&[1, 2, 3, 4, 5]));
            let v: ArrayN<usize, 8> = arrayn![1, 2, 3, 4, 5];
            bb(v)
        })
    });

    c.bench_function("creation_5d/smallvec", |b| {
        b.iter(|| {
            let v: SmallVec<[_; 8]> = smallvec![1, 2, 3, 4, 5];
            bb(v)
        })
    });
    c.bench_function("creation_5d/vec", |b| {
        b.iter(|| {
            let v: Vec<u8> = vec![1, 2, 3, 4, 5];
            bb(v)
        })
    });

    c.bench_function("creation_6d/arrayn-empty", |b| {
        b.iter(|| {
            let v: ArrayN<usize, 8> = ArrayN::empty().with_len(6);
            bb(v)
        })
    });

    c.bench_function("creation_6d/arrayn", |b| {
        b.iter(|| {
            // let v: ArrayN<u8, 8> = ArrayN::new(bb(&[1, 2, 3, 4, 5, 6]));
            let v: ArrayN<usize, 8> = arrayn![1, 2, 3, 4, 5, 6];
            bb(v)
        })
    });

    c.bench_function("creation_6d/smallvec", |b| {
        b.iter(|| {
            let v: SmallVec<[_; 8]> = smallvec![1, 2, 3, 4, 5, 6];
            bb(v)
        })
    });

    c.bench_function("creation_6d/vec", |b| {
        b.iter(|| {
            let v: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
            bb(v)
        })
    });

    c.bench_function("creation_7d/arrayn-empty", |b| {
        b.iter(|| {
            let v: ArrayN<usize, 8> = ArrayN::empty().with_len(7);
            bb(v)
        })
    });

    c.bench_function("creation_7d/arrayn", |b| {
        b.iter(|| {
            let v: ArrayN<usize, 8> = arrayn![1, 2, 3, 4, 5, 6, 7];
            bb(v)
        })
    });

    c.bench_function("creation_7d/smallvec", |b| {
        b.iter(|| {
            let v: SmallVec<[_; 8]> = smallvec![1, 2, 3, 4, 5, 6, 7];
            bb(v)
        })
    });

    c.bench_function("creation_7d/vec", |b| {
        b.iter(|| {
            let v: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7];
            bb(v)
        })
    });

    c.bench_function("creation_8d/arrayn-empty", |b| {
        b.iter(|| {
            let v: ArrayN<usize, 8> = ArrayN::empty().with_len(8);
            bb(v)
        })
    });

    c.bench_function("creation_8d/arrayn", |b| {
        b.iter(|| {
            // let v: ArrayN<usize, 8> = ArrayN::new(bb(&[1, 2, 3, 4, 5, 6, 7, 8]));
            let v: ArrayN<usize, 8> = arrayn![1, 2, 3, 4, 5, 6, 7, 8];
            bb(v)
        })
    });

    c.bench_function("creation_8d/smallvec", |b| {
        b.iter(|| {
            let v: SmallVec<[_; 8]> = smallvec![1, 2, 3, 4, 5, 6, 7, 8];
            bb(v)
        })
    });

    c.bench_function("creation_8d/vec", |b| {
        b.iter(|| {
            let v: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
            bb(v)
        })
    });
}

// Access benchmarks
fn bench_iter_8d(c: &mut Criterion) {
    let arrayn: ArrayN<usize, 8> = ArrayN::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let smallv: SmallVec<[_; 8]> = smallvec![1, 2, 3, 4, 5, 6, 7, 8];
    let vecv: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    c.bench_function("access_iter/arrayn", |b| {
        b.iter(|| {
            let mut sum = 0;
            for item in arrayn.as_slice().iter().take(bb(arrayn.len())) {
                sum += bb(*item);
            }
            bb(sum)
        })
    });

    c.bench_function("access_iter/smallvec", |b| {
        b.iter(|| {
            let mut sum = 0;
            for item in smallv.iter().take(bb(smallv.len())) {
                sum += bb(*item);
            }
            bb(sum)
        })
    });
    c.bench_function("access_iter/vec", |b| {
        b.iter(|| {
            let mut sum = 0;
            for item in vecv.iter().take(bb(vecv.len())) {
                sum += bb(*item);
            }
            bb(sum)
        })
    });
}

// Memory usage simulation (copy operations)
fn bench_copy_operations(c: &mut Criterion) {
    let arrayn: ArrayN<usize, 8> = ArrayN::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let smallv: SmallVec<[_; 8]> = smallvec![1, 2, 3, 4, 5, 6, 7, 8];
    let vecv: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    c.bench_function("copy/arrayn", |b| {
        b.iter(|| {
            let copied = bb(arrayn);
            bb(copied)
        })
    });

    c.bench_function("copy/smallvec", |b| {
        b.iter(|| {
            let copied = bb(smallv.clone());
            bb(copied)
        })
    });
    c.bench_function("copy/vec", |b| {
        b.iter(|| {
            let copied = bb(vecv.clone());
            bb(copied)
        })
    });
}

// Modification benchmarks
fn bench_modification(c: &mut Criterion) {
    let mut arrayn: ArrayN<usize, 8> = ArrayN::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut smallv: SmallVec<[_; 8]> = smallvec![1, 2, 3, 4, 5, 6, 7, 8];
    let mut vecv: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    c.bench_function("modification/arrayn", |b| {
        b.iter(|| {
            for (i, item) in arrayn.as_mut_slice().iter_mut().enumerate() {
                *item = bb(i * 2);
            }
            bb(0)
        })
    });

    c.bench_function("modification/smallvec", |b| {
        b.iter(|| {
            for (i, item) in smallv.iter_mut().enumerate() {
                *item = bb(i * 2);
            }
            bb(0)
        })
    });
    c.bench_function("modification/vec", |b| {
        b.iter(|| {
            for (i, item) in vecv.iter_mut().enumerate() {
                *item = bb(i * 2);
            }
            bb(0)
        })
    });
}

fn bench_access_8d(c: &mut Criterion) {
    let arrayn: ArrayN<usize, 8> = ArrayN::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let smallv: SmallVec<[_; 8]> = smallvec![1, 2, 3, 4, 5, 6, 7, 8];
    let vecv: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let ids: Vec<usize> = vec![
        3, 1, 2, 6, 7, 4, 5, 0, 2, 2, 2, 2, 3, 4, 5, 6, 7, 1, 2, 4, 0, 1, 2,
    ];

    c.bench_function("access_random/arrayn", |b| {
        b.iter(|| {
            let mut sum = 0;
            for i in ids.iter() {
                sum += arrayn[*i];
            }
            bb(sum)
        })
    });

    c.bench_function("access_random/smallvec", |b| {
        b.iter(|| {
            let mut sum = 0;
            for i in ids.iter() {
                sum += smallv[*i];
            }
            bb(sum)
        })
    });
    c.bench_function("access_random/vec", |b| {
        b.iter(|| {
            let mut sum = 0;
            for i in ids.iter() {
                sum += vecv[*i];
            }
            bb(sum)
        })
    });
}

criterion_group!(
    benches,
    // bench_from_slice_1d_8d,
    // bench_creation_1d_8d,
    bench_access_8d,
    bench_iter_8d,
    // bench_copy_operations,
    // bench_modification,
);

criterion_main!(benches);
