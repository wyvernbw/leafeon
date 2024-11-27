use std::{
    ops::{Add, Mul, Sub},
    sync::{Arc, Mutex},
};

use base_ops::BaseOps;
use gpu_ops::GpuOps;
use ndarray::{linalg::Dot, ArrayBase, Data, Dimension, RawData};
use serde::{Deserialize, Serialize};

#[path = "base-ops.rs"]
pub mod base_ops;
#[path = "gpu-ops.rs"]
pub mod gpu_ops;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Array<A, O = BaseOps> {
    data: A,
    operations: O,
}

trait Operations<A, Rhs> {}

impl<Rhs, A, T> Operations<A, Rhs> for T
where
    Array<A, T>: Add + Sub + Mul + Dot<Rhs> + Default,
    T: Default,
{
}

pub type Array1<T, O = BaseOps> = Array<ndarray::Array1<T>, O>;
pub type Array2<T, O = BaseOps> = Array<ndarray::Array2<T>, O>;
pub type Array3<T, O = BaseOps> = Array<ndarray::Array3<T>, O>;
pub type Array4<T, O = BaseOps> = Array<ndarray::Array4<T>, O>;
pub type Array5<T, O = BaseOps> = Array<ndarray::Array5<T>, O>;
pub type Array6<T, O = BaseOps> = Array<ndarray::Array6<T>, O>;
pub type ArrayView1<'a, T, O = BaseOps> = Array<ndarray::ArrayView1<'a, T>, O>;
pub type ArrayView2<'a, T, O = BaseOps> = Array<ndarray::ArrayView2<'a, T>, O>;
pub type ArrayView3<'a, T, O = BaseOps> = Array<ndarray::ArrayView3<'a, T>, O>;
pub type ArrayView4<'a, T, O = BaseOps> = Array<ndarray::ArrayView4<'a, T>, O>;
pub type ArrayView5<'a, T, O = BaseOps> = Array<ndarray::ArrayView5<'a, T>, O>;
pub type ArrayView6<'a, T, O = BaseOps> = Array<ndarray::ArrayView6<'a, T>, O>;

impl<A, O> Array<A, O> {
    pub fn operations(&self) -> &O {
        &self.operations
    }
    pub fn operations_mut(&mut self) -> &mut O {
        &mut self.operations
    }
}

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

impl From<BaseOps> for GpuOps {
    fn from(_: BaseOps) -> Self {
        Self::default()
    }
}

impl From<GpuOps> for BaseOps {
    fn from(_: GpuOps) -> Self {
        Self
    }
}
