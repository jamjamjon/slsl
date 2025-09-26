#![allow(deprecated)]
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array, ArrayView1, IxDyn, Zip};
use slsl::Tensor;
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
        .par_for_each(|x_val, &mean_val, &std_val| {
            *x_val = (*x_val - mean_val) / std_val;
        });

    Ok(())
}

// Test configurations
struct TestConfig {
    name: &'static str,
    shape: [usize; 3],
    dim: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

const TEST_CONFIGS: &[TestConfig] = &[
    // 224x224x3 configurations
    TestConfig {
        name: "224x224x3_hwc_zeros",
        shape: [224, 224, 3],
        dim: 2,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    },
    TestConfig {
        name: "224x224x3_hwc_half",
        shape: [224, 224, 3],
        dim: 2,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
    },
    TestConfig {
        name: "224x224x3_hwc_imagenet",
        shape: [224, 224, 3],
        dim: 2,
        mean: [0.48145466, 0.4578275, 0.40821073],
        std: [0.26862954, 0.261_302_6, 0.275_777_1],
    },
    // 3x224x224 configurations
    TestConfig {
        name: "3x224x224_chw_zeros",
        shape: [3, 224, 224],
        dim: 0,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    },
    TestConfig {
        name: "3x224x224_chw_half",
        shape: [3, 224, 224],
        dim: 0,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
    },
    TestConfig {
        name: "3x224x224_chw_imagenet",
        shape: [3, 224, 224],
        dim: 0,
        mean: [0.48145466, 0.4578275, 0.40821073],
        std: [0.26862954, 0.261_302_6, 0.275_777_1],
    },
    // Other sizes
    TestConfig {
        name: "256x256x3_hwc_zeros",
        shape: [256, 256, 3],
        dim: 2,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    },
    TestConfig {
        name: "3x256x256_chw_zeros",
        shape: [3, 256, 256],
        dim: 0,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    },
    TestConfig {
        name: "512x512x3_hwc_zeros",
        shape: [512, 512, 3],
        dim: 2,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    },
    TestConfig {
        name: "3x512x512_chw_zeros",
        shape: [3, 512, 512],
        dim: 0,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    },
    TestConfig {
        name: "1024x1024x3_hwc_zeros",
        shape: [1024, 1024, 3],
        dim: 2,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    },
    TestConfig {
        name: "3x1024x1024_chw_zeros",
        shape: [3, 1024, 1024],
        dim: 0,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    },
];

fn bench_slsl_standardize(c: &mut Criterion) {
    let mut group = c.benchmark_group("standardize_slsl");

    for config in TEST_CONFIGS {
        group.bench_function(config.name, |b| {
            b.iter(|| {
                // Create test data
                let data: Vec<f32> = (0..config.shape.iter().product::<usize>())
                    .map(|i| (i as f32) * 0.01)
                    .collect();
                let tensor = Tensor::from_vec(data, config.shape).unwrap();

                // Perform standardization

                black_box(
                    tensor
                        .standardize(&config.mean, &config.std, config.dim)
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

fn bench_ndarray_standardize(c: &mut Criterion) {
    let mut group = c.benchmark_group("standardize_ndarray");

    for config in TEST_CONFIGS {
        group.bench_function(config.name, |b| {
            b.iter(|| {
                // Create test data
                let data: Vec<f32> = (0..config.shape.iter().product::<usize>())
                    .map(|i| (i as f32) * 0.01)
                    .collect();
                let mut array = Array::from_shape_vec(IxDyn(&config.shape), data).unwrap();
                let mean = ArrayView1::from(&config.mean);
                let std = ArrayView1::from(&config.std);
                standardize_ndarray(&mut array, mean, std, config.dim).unwrap();
                black_box(());
                array
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_slsl_standardize, bench_ndarray_standardize);
criterion_main!(benches);
