use criterion::{criterion_group, criterion_main, Criterion};
use leafeon_core::{network::Network, parser::load_data};
use ndarray::{Array1, Array2};
use rand::seq::IteratorRandom;
use std::hint::black_box;

pub fn dot_2_cpu(c: &mut Criterion) {
    let array_a = Array2::<f32>::from_shape_fn((1024, 1024), |_| rand::random::<f32>());
    let array_b = Array2::<f32>::from_shape_fn((1024, 1024), |_| rand::random::<f32>());
    let array_a = leafeon_types::prelude::Array2::<_, BaseOps>::from(array_a);
    let array_b = leafeon_types::prelude::Array2::<_, BaseOps>::from(array_b);
}

criterion_group!(benches, dot_2_cpu);
criterion_main!(benches);
