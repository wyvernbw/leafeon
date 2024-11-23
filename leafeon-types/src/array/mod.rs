use std::sync::{Arc, Mutex};

#[path = "base-ops.rs"]
pub mod base_ops;
#[path = "gpu-ops.rs"]
pub mod gpu_ops;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Array<A, O> {
    data: A,
    operations: O,
}

pub type Array1<T, O> = Array<ndarray::Array1<T>, O>;
pub type Array2<T, O> = Array<ndarray::Array2<T>, O>;

impl<A, O: Default> From<A> for Array<A, O> {
    fn from(value: A) -> Self {
        Array {
            data: value,
            operations: O::default(),
        }
    }
}

impl<A, O> AsRef<A> for Array<A, O> {
    fn as_ref(&self) -> &A {
        &self.data
    }
}
