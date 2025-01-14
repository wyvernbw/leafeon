use std::{ops::Mul, sync::OnceLock};

use bytemuck::Pod;
use ndarray::{Array2, ArrayView1};

use crate::gpu::State;

struct Executor {
    state: State,
}

impl Executor {
    fn new() -> Self {
        let state = State::try_new_sync().unwrap();
        Executor { state }
    }
}

static EXECUTOR: OnceLock<Executor> = OnceLock::new();

fn executor() -> &'static Executor {
    EXECUTOR.get_or_init(Executor::new)
}

pub fn dot(
    a: ndarray::ArrayView2<'_, f32>,
    b: ndarray::ArrayView2<'_, f32>,
) -> ndarray::Array2<f32> {
    let _ = b;
    let _ = a;
    todo!()
}

pub fn outer_dot_1<T: Mul + Pod>(a: ArrayView1<'_, T>, b: ArrayView1<'_, T>) -> Array2<T> {
    executor().state.try_outer_dot_1(a, b).unwrap()
}
