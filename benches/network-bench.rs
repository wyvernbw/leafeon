use criterion::{criterion_group, criterion_main, Criterion};
use leafeon_core::{network::Network, parser::load_data};
use leafeon_types::prelude::*;
use ndarray::Array1;
use rand::seq::IteratorRandom;
use std::hint::black_box;

fn benchmark(c: &mut Criterion) {
    let dataset = load_data()
        .data_path("./data/train-images-idx3-ubyte")
        .labels_path("./data/train-labels-idx1-ubyte")
        .call()
        .expect("Failed to load dataset");
    c.bench_function("backprop", |b| {
        b.iter(|| {
            let network = Network::untrained()
                .input_size(28 * 28)
                .layer_spec(&[128, 128, 32, 10])
                .call();
            let (image, label) = dataset
                .images()
                .iter()
                .choose(&mut rand::thread_rng())
                .unwrap();
            black_box({
                let target = Activations(Array1::from_shape_fn(10, |i| match i {
                    i if i == *label as usize => 1.0,
                    _ => 0.0,
                }));
                network
                    .backprop()
                    .target(target)
                    .input(Activations(image.into()))
                    .learning_rate(0.1)
                    .call()
            });
            anyhow::Ok(())
        })
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
