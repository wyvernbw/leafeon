use criterion::{black_box, criterion_group, criterion_main, Criterion};
use leafeon_types::prelude::{BaseOps, GpuOps};
use ndarray::linalg::Dot;

fn dot_2_cpu(c: &mut Criterion) {
    let array_a = ndarray::Array2::<f32>::from_shape_fn((3, 3), |_| rand::random::<f32>());
    let array_b = ndarray::Array2::<f32>::from_shape_fn((3, 3), |_| rand::random::<f32>());
    let array_a =
        leafeon_types::array::Array2::<_, leafeon_types::array::base_ops::BaseOps>::from(array_a);
    let array_b =
        leafeon_types::array::Array2::<_, leafeon_types::array::base_ops::BaseOps>::from(array_b);
    c.bench_function("dot_2_cpu", |b| {
        b.iter(|| {
            let array_a = array_a.dot(&array_b);
            black_box(array_a);
        })
    });
}

fn dot_2_gpu(c: &mut Criterion) {
    let array_a = ndarray::Array2::<f32>::from_shape_fn((3, 3), |_| rand::random::<f32>());
    let array_b = ndarray::Array2::<f32>::from_shape_fn((3, 3), |_| rand::random::<f32>());
    let array_a =
        leafeon_types::array::Array2::<_, leafeon_types::array::gpu_ops::GpuOps>::from(array_a);
    let array_b =
        leafeon_types::array::Array2::<_, leafeon_types::array::gpu_ops::GpuOps>::from(array_b);
    c.bench_function("dot_2_gpu", |b| {
        b.iter(|| {
            let array_a = array_a.dot(&array_b);
            black_box(array_a);
        })
    });
}

criterion_group!(benches, dot_2_cpu, dot_2_gpu);
criterion_main!(benches);
