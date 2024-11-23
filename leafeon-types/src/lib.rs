#![feature(associated_type_defaults)]
use std::{fmt::Display, iter::Sum};

extern crate openblas_src;

use derive_more::derive::{Add, AsRef, Index, IndexMut, Mul as DeriveMoreMul, MulAssign, Sub};
use ndarray::prelude::*;

pub mod array;
pub mod dataset;
pub mod prelude;

#[derive(Debug, Clone, Default, DeriveMoreMul, MulAssign, Add, Sub)]
pub struct WeightGradient(pub Array2<f32>);

impl Sum for WeightGradient {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap_or_default()
    }
}

#[derive(Debug, Clone, Default, MulAssign, DeriveMoreMul, Add, Sub)]
pub struct BiasGradient(pub Array1<f32>);

impl Sum for BiasGradient {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap_or_default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Loss(pub Array1<f32>);

#[derive(Debug, Clone, Default, Index, IndexMut)]
pub struct ZValues(#[index] pub Array1<f32>);

#[derive(Debug, Clone, Default, Index, IndexMut, Add, Sub, DeriveMoreMul, AsRef)]
pub struct Activations(#[index] pub Array1<f32>);

impl Display for Activations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
