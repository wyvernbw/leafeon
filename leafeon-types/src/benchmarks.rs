use std::hint::black_box;

use ndarray::linalg::Dot;

extern crate test;

use test::Bencher;

use crate::prelude::{BaseOps, GpuOps};

const N: usize = 4096;

#[test]
fn dot_2_cpu() {
    let array_a = ndarray::Array2::<f32>::from_shape_fn((N, N), |_| rand::random::<f32>());
    let array_b = ndarray::Array2::<f32>::from_shape_fn((N, N), |_| rand::random::<f32>());
    let array_a = crate::array::Array2::<_, BaseOps>::from(array_a);
    let array_b = crate::array::Array2::<_, BaseOps>::from(array_b);
    let array_a = black_box(array_a.dot(&array_b));
}

#[test]
fn dot_2_gpu() {
    tracing_subscriber::fmt::init();
    let array_a = ndarray::Array2::<f32>::from_shape_fn((N, N), |_| rand::random::<f32>());
    let array_b = ndarray::Array2::<f32>::from_shape_fn((N, N), |_| rand::random::<f32>());
    let array_a = crate::array::Array2::<_, GpuOps>::from(array_a);
    let array_b = crate::array::Array2::<_, GpuOps>::from(array_b);
    let array_a = black_box(array_a.dot(&array_b));
}
