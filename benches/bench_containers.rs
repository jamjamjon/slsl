use criterion::{
    black_box as bb, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use slsl::{ArrayN, UninitVec};
use smallvec::SmallVec;

// Capacities
const ARRAYN_CAP: usize = 8;

// Size tiers
const SMALL_SIZES: &[usize] = &[10];
const MEDIUM_SIZES: &[usize] = &[64, 128, 200];
const LARGE_SIZES: &[usize] = &[256, 512, 700];
const XLARGE_SIZES: &[usize] = &[1024, 1536, 2048];

type SmallVecF32 = SmallVec<[f32; ARRAYN_CAP]>;
type SmallVecU8 = SmallVec<[u8; ARRAYN_CAP]>;

fn bench_create_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_empty");

    let all = [SMALL_SIZES, MEDIUM_SIZES, LARGE_SIZES, XLARGE_SIZES].concat();
    for &n in &all {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("Vec::with_capacity<f32>", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    let v: Vec<f32> = Vec::with_capacity(bb(n));
                    bb(v.capacity())
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("UninitVec::new<f32>", n), &n, |b, &n| {
            b.iter(|| {
                let u = UninitVec::<f32>::new(bb(n));
                bb(u.capacity())
            })
        });

        group.bench_with_input(BenchmarkId::new("SmallVec::new<f32>", n), &n, |b, &_n| {
            b.iter(|| {
                let v: SmallVecF32 = SmallVec::new();
                bb(v.capacity())
            })
        });

        if n <= ARRAYN_CAP {
            group.bench_with_input(BenchmarkId::new("ArrayN::empty<f32>", n), &n, |b, &n| {
                b.iter(|| {
                    let a: ArrayN<f32, ARRAYN_CAP> = ArrayN::empty().with_len(bb(n));
                    bb(a.len())
                })
            });
        }
    }

    group.finish();
}

fn bench_create_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_full");

    let all = [SMALL_SIZES, MEDIUM_SIZES, LARGE_SIZES, XLARGE_SIZES].concat();
    for &n in &all {
        group.throughput(Throughput::Elements(n as u64));

        // f32
        group.bench_with_input(BenchmarkId::new("vec![f32; n]", n), &n, |b, &n| {
            b.iter(|| {
                let v: Vec<f32> = vec![1.25; bb(n)];
                bb(v)
            })
        });
        group.bench_with_input(BenchmarkId::new("UninitVec::full<f32>", n), &n, |b, &n| {
            b.iter(|| {
                let v = UninitVec::<f32>::new(bb(n)).full(1.25);
                bb(v)
            })
        });
        group.bench_with_input(BenchmarkId::new("SmallVec::resize<f32>", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: SmallVecF32 = SmallVec::new();
                v.resize(bb(n), 1.25);
                bb(v)
            })
        });
        if n <= ARRAYN_CAP {
            group.bench_with_input(
                BenchmarkId::new("ArrayN::full_with_len<f32>", n),
                &n,
                |b, &n| {
                    b.iter(|| {
                        let a: ArrayN<f32, ARRAYN_CAP> = ArrayN::full(1.25, bb(n));
                        bb(a)
                    })
                },
            );
        }

        // u8
        group.bench_with_input(BenchmarkId::new("vec![u8; n]", n), &n, |b, &n| {
            b.iter(|| {
                let v: Vec<u8> = vec![7u8; bb(n)];
                bb(v)
            })
        });
        group.bench_with_input(BenchmarkId::new("UninitVec::full<u8>", n), &n, |b, &n| {
            b.iter(|| {
                let v = UninitVec::<u8>::new(bb(n)).full(7);
                bb(v)
            })
        });
        group.bench_with_input(BenchmarkId::new("SmallVec::resize<u8>", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: SmallVecU8 = SmallVec::new();
                v.resize(bb(n), 7);
                bb(v)
            })
        });
        if n <= ARRAYN_CAP {
            group.bench_with_input(
                BenchmarkId::new("ArrayN::full_with_len<u8>", n),
                &n,
                |b, &n| {
                    b.iter(|| {
                        let a: ArrayN<u8, ARRAYN_CAP> = ArrayN::full(7, bb(n));
                        bb(a)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_index_get_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_get_set");
    let all = [SMALL_SIZES, MEDIUM_SIZES, LARGE_SIZES, XLARGE_SIZES].concat();
    for &n in &all {
        group.throughput(Throughput::Elements(n as u64));

        // Prepare containers (f32)
        let vec_f: Vec<f32> = vec![1.0; n];
        let mut sv_f: SmallVecF32 = SmallVec::new();
        sv_f.resize(n, 1.0);
        let arr_f: Option<ArrayN<f32, ARRAYN_CAP>> =
            (n <= ARRAYN_CAP).then(|| ArrayN::full(1.0, bb(n)));
        let vec_f_uninit = UninitVec::<f32>::new(n).full(1.0);

        group.bench_with_input(BenchmarkId::new("Vec<f32>::get_set", n), &n, |b, &n| {
            let mut v = vec_f.clone();
            b.iter(|| {
                let mut sum = 0.0f32;
                for item in v.iter_mut().take(bb(n)) {
                    sum += bb(*item);
                    *item = bb(sum);
                }
                bb(sum)
            })
        });

        group.bench_with_input(
            BenchmarkId::new("SmallVec<f32>::get_set", n),
            &n,
            |b, &n| {
                let mut v = sv_f.clone();
                b.iter(|| {
                    let mut sum = 0.0f32;
                    for item in v.iter_mut().take(bb(n)) {
                        sum += bb(*item);
                        *item = bb(sum);
                    }
                    bb(sum)
                })
            },
        );

        if let Some(mut a) = arr_f {
            group.bench_with_input(BenchmarkId::new("ArrayN<f32>::get_set", n), &n, |b, &n| {
                let len = n;
                b.iter(|| {
                    let mut sum = 0.0f32;
                    for item in a.as_mut_slice().iter_mut().take(bb(len)) {
                        sum += bb(*item);
                        *item = bb(sum);
                    }
                    bb(sum)
                })
            });
        }

        group.bench_with_input(
            BenchmarkId::new("UninitVec<f32> as Vec::get_set", n),
            &n,
            |b, &n| {
                let mut v = vec_f_uninit.clone();
                b.iter(|| {
                    let mut sum = 0.0f32;
                    for item in v.iter_mut().take(bb(n)) {
                        sum += bb(*item);
                        *item = bb(sum);
                    }
                    bb(sum)
                })
            },
        );
    }
    group.finish();
}

fn bench_clone_copy_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("clone_copy_iter");
    let all = [SMALL_SIZES, MEDIUM_SIZES, LARGE_SIZES, XLARGE_SIZES].concat();
    for &n in &all {
        group.throughput(Throughput::Elements(n as u64));

        // u8
        let base_vec: Vec<u8> = vec![7; n];
        let mut base_sv: SmallVecU8 = SmallVec::new();
        base_sv.resize(n, 7);
        let base_arr: Option<ArrayN<u8, ARRAYN_CAP>> =
            (n <= ARRAYN_CAP).then(|| ArrayN::full(7, n));

        group.bench_with_input(BenchmarkId::new("Vec<u8>::clone", n), &n, |b, _| {
            b.iter(|| {
                let v = bb(&base_vec).clone();
                bb(v.len())
            })
        });
        group.bench_with_input(BenchmarkId::new("SmallVec<u8>::clone", n), &n, |b, _| {
            b.iter(|| {
                let v = bb(&base_sv).clone();
                bb(v.len())
            })
        });
        if let Some(a) = base_arr {
            group.bench_with_input(BenchmarkId::new("ArrayN<u8>::copy", n), &n, |b, _| {
                b.iter(|| {
                    let c = bb(a);
                    bb(c.len())
                })
            });
        }

        group.bench_with_input(BenchmarkId::new("Vec<u8>::iterate_sum", n), &n, |b, _| {
            b.iter(|| {
                let mut s: u64 = 0;
                for &x in bb(&base_vec) {
                    s += x as u64;
                }
                bb(s)
            })
        });
        group.bench_with_input(
            BenchmarkId::new("SmallVec<u8>::iterate_sum", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let mut s: u64 = 0;
                    for &x in bb(&base_sv) {
                        s += x as u64;
                    }
                    bb(s)
                })
            },
        );
        if let Some(a) = base_arr {
            group.bench_with_input(
                BenchmarkId::new("ArrayN<u8>::iterate_sum", n),
                &n,
                |b, _| {
                    b.iter(|| {
                        let mut s: u64 = 0;
                        for &x in bb(a.as_slice()) {
                            s += x as u64;
                        }
                        bb(s)
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_create_empty,
    bench_create_full,
    bench_index_get_set,
    bench_clone_copy_iter
);
criterion_main!(benches);
