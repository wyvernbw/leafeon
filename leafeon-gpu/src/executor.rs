use std::{
    cell::OnceCell,
    ops::{Deref, Mul},
    sync::{Arc, Mutex, OnceLock},
};

use bytemuck::Pod;
use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::gpu::State;

#[derive(Debug)]
pub struct Executor {
    pub state: State,
}

impl Executor {
    fn new() -> Self {
        let state = State::try_new_sync().unwrap();
        Executor { state }
    }
}

static EXECUTOR: OnceLock<Mutex<Executor>> = OnceLock::new();

pub fn executor() -> &'static Mutex<Executor> {
    EXECUTOR.get_or_init(|| Mutex::new(Executor::new()))
}

pub fn dot(
    a: ndarray::ArrayView2<'_, f32>,
    b: ndarray::ArrayView2<'_, f32>,
) -> ndarray::Array2<f32> {
    todo!()
}

pub fn outer_dot_1<T: Mul + Pod>(a: ArrayView1<'_, T>, b: ArrayView1<'_, T>) -> Array2<T> {
    executor()
        .lock()
        .unwrap()
        .state
        .try_outer_dot_1(a, b)
        .unwrap()
}
