#![allow(deprecated)]
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array, ArrayView1, IxDyn, Zip};
use slsl::Tensor;
use std::collections::HashSet;
use std::hint::black_box;

// ndarray implementation from user
pub fn to_shape<D: ndarray::ShapeArg>(
    x: Array<f32, IxDyn>,
    dim: D,
) -> anyhow::Result<Array<f32, IxDyn>> {
    Ok(x.to_shape(dim).map(|x| x.to_owned().into_dyn())?)
}

pub fn standardize_ndarray(
    x: &mut Array<f32, IxDyn>,
    mean: ArrayView1<f32>,
    std: ArrayView1<f32>,
    dim: usize,
) -> anyhow::Result<()> {
    if mean.len() != std.len() {
        anyhow::bail!(
            "`standardize`: `mean` and `std` lengths are not equal. Mean length: {}, Std length: {}.",
            mean.len(),
            std.len()
        );
    }

    let shape = x.shape();
    if dim >= shape.len() || shape[dim] != mean.len() {
        anyhow::bail!(
            "`standardize`: Dimension mismatch. `dim` is {} but shape length is {} or `mean` length is {}.",
            dim,
            shape.len(),
            mean.len()
        );
    }

    // Create broadcast shape for mean and std
    let mut broadcast_shape = vec![1; shape.len()];
    broadcast_shape[dim] = mean.len();

    let mean_reshaped = mean.to_owned().into_shape(IxDyn(&broadcast_shape))?;
    let std_reshaped = std.to_owned().into_shape(IxDyn(&broadcast_shape))?;

    let mean_broadcast = mean_reshaped.broadcast(shape).ok_or_else(|| {
        anyhow::anyhow!("Failed to broadcast `mean` to the shape of the input array.")
    })?;
    let std_broadcast = std_reshaped.broadcast(shape).ok_or_else(|| {
        anyhow::anyhow!("Failed to broadcast `std` to the shape of the input array.")
    })?;

    Zip::from(x)
        .and(mean_broadcast)
        .and(std_broadcast)
        .for_each(|x_val, &mean_val, &std_val| {
            *x_val = (*x_val - mean_val) / std_val;
        });

    Ok(())
}

// Test configurations for 3D tensors
struct TestConfig3D {
    name: String,
    shape: [usize; 3],
    dim: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

// Test configurations for 4D tensors
struct TestConfig4D {
    name: String,
    shape: [usize; 4],
    dim: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

fn gen_3d_configs() -> Vec<TestConfig3D> {
    let means = [
        ("zeros", [0.0f32, 0.0, 0.0], [1.0f32, 1.0, 1.0]),
        ("half", [0.5f32, 0.5, 0.5], [0.5f32, 0.5, 0.5]),
        (
            "imagenet",
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.261_302_6, 0.275_777_1],
        ),
    ];

    let sizes = [(224usize, 224usize), (640, 640), (1024, 1024)];
    let mut v = Vec::new();
    for (h, w) in sizes {
        // Generate three permutations where channel size 3 is placed at each axis
        let shapes = [
            ([3, h, w], 0, format!("3x{}x{}", h, w)),
            ([h, 3, w], 1, format!("{}x3x{}", h, w)),
            ([h, w, 3], 2, format!("{}x{}x3", h, w)),
        ];
        for (shape, dim, tag) in shapes {
            for (case, mean, std) in means.iter() {
                v.push(TestConfig3D {
                    name: format!("{}_{case}", tag),
                    shape,
                    dim,
                    mean: *mean,
                    std: *std,
                });
            }
        }
    }
    v
}

fn gen_4d_configs() -> Vec<TestConfig4D> {
    // Start from 3D base (H,W,3) sizes and insert 10 at all possible positions,
    // then also permute channel (3) to different axes by using the three 3D permutations and inserting 10.
    let sizes = [(224usize, 224usize), (640, 640), (1024, 1024)];
    let mut v = Vec::new();
    let mut seen: HashSet<(usize, usize, usize, usize, usize)> = HashSet::new();
    let mean = [0.48145466, 0.4578275, 0.40821073];
    let std = [0.26862954, 0.2613026, 0.2757771];

    for (h, w) in sizes {
        // Base 3D permutations
        let bases = [
            ([3, h, w], 0, format!("3x{}x{}", h, w)),
            ([h, 3, w], 1, format!("{}x3x{}", h, w)),
            ([h, w, 3], 2, format!("{}x{}x3", h, w)),
        ];

        for (base, ch_dim, tag3d) in bases {
            // Insert 10 at all four positions
            let shapes_4d = [
                (
                    [10, base[0], base[1], base[2]],
                    ch_dim + 1,
                    format!("10x{}", tag3d),
                ),
                (
                    [base[0], 10, base[1], base[2]],
                    ch_dim,
                    format!(
                        "{}x10x{}",
                        if ch_dim == 0 {
                            format!("{}x{}", base[1], base[2])
                        } else {
                            format!("{}", base[0])
                        },
                        if ch_dim == 0 {
                            format!("{}", base[0])
                        } else {
                            format!("{}x{}", base[1], base[2])
                        }
                    ),
                ),
                (
                    [base[0], base[1], 10, base[2]],
                    ch_dim,
                    format!("{}x{}x10x{}", base[0], base[1], base[2]),
                ),
                (
                    [base[0], base[1], base[2], 10],
                    ch_dim,
                    format!("{}x{}x{}x10", base[0], base[1], base[2]),
                ),
            ];

            for (shape4, dim4, _tag4) in shapes_4d {
                // Derive channel dim by locating the index of value 3
                let mut channel_dim = None;
                for (i, &d) in shape4.iter().enumerate() {
                    if d == 3 {
                        channel_dim = Some(i);
                        break;
                    }
                }
                let dim = channel_dim.unwrap_or(dim4);
                let key = (shape4[0], shape4[1], shape4[2], shape4[3], dim);
                if seen.insert(key) {
                    let name = format!(
                        "{}x{}x{}x{}:dim{}",
                        shape4[0], shape4[1], shape4[2], shape4[3], dim
                    );
                    v.push(TestConfig4D {
                        name,
                        shape: shape4,
                        dim,
                        mean,
                        std,
                    });
                }
            }
        }
    }
    v
}

fn bench_compare_standardize_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("standardize_3d_compare");
    let configs = gen_3d_configs();

    for config in configs.iter() {
        // slsl
        group.bench_function(format!("slsl/{}", config.name), |b| {
            b.iter(|| {
                let data: Vec<f32> = (0..config.shape.iter().product::<usize>())
                    .map(|i| (i as f32) * 0.01)
                    .collect();
                let tensor = Tensor::from_vec(data, config.shape).unwrap();
                black_box(
                    tensor
                        .standardize(&config.mean, &config.std, config.dim)
                        .unwrap(),
                )
            });
        });

        // ndarray
        group.bench_function(format!("ndarray/{}", config.name), |b| {
            b.iter(|| {
                let data: Vec<f32> = (0..config.shape.iter().product::<usize>())
                    .map(|i| (i as f32) * 0.01)
                    .collect();
                let mut array = Array::from_shape_vec(IxDyn(&config.shape), data).unwrap();
                let mean = ArrayView1::from(&config.mean);
                let std = ArrayView1::from(&config.std);
                standardize_ndarray(&mut array, mean, std, config.dim).unwrap();
                black_box(&array);
            });
        });
    }

    group.finish();
}

fn bench_compare_standardize_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("standardize_4d_compare");
    let configs = gen_4d_configs();

    for config in configs.iter() {
        // slsl
        group.bench_function(format!("slsl/{}", config.name), |b| {
            b.iter(|| {
                let data: Vec<f32> = (0..config.shape.iter().product::<usize>())
                    .map(|i| (i as f32) * 0.01)
                    .collect();
                let tensor = Tensor::from_vec(data, config.shape).unwrap();
                black_box(
                    tensor
                        .standardize(&config.mean, &config.std, config.dim)
                        .unwrap(),
                )
            });
        });

        // ndarray
        group.bench_function(format!("ndarray/{}", config.name), |b| {
            b.iter(|| {
                let data: Vec<f32> = (0..config.shape.iter().product::<usize>())
                    .map(|i| (i as f32) * 0.01)
                    .collect();
                let mut array = Array::from_shape_vec(IxDyn(&config.shape), data).unwrap();
                let mean = ArrayView1::from(&config.mean);
                let std = ArrayView1::from(&config.std);
                standardize_ndarray(&mut array, mean, std, config.dim).unwrap();
                black_box(&array);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_compare_standardize_3d,
    bench_compare_standardize_4d
);
criterion_main!(benches);
