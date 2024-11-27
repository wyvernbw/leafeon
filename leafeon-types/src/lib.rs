#![feature(test)]
#![feature(associated_type_defaults)]
use ndarray::LinalgScalar;

use crate::prelude::*;
use std::{
    fmt::Display,
    iter::Sum,
    ops::{Add, Mul, Sub},
};

extern crate blas_src;

pub mod array;
pub mod benchmarks;
pub mod dataset;
pub mod prelude;

#[derive(Debug, Clone, Default)]
pub struct WeightGradient<O = BaseOps>(pub Array2<f32, O>);

impl Add for WeightGradient {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for WeightGradient {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<T: LinalgScalar> Mul<T> for WeightGradient {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        todo!();
        //Self(self.0 * rhs)
    }
}

impl Sum for WeightGradient {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap_or_default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct BiasGradient<O = BaseOps>(pub Array1<f32, O>);

impl Add for BiasGradient {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for BiasGradient {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Sum for BiasGradient {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap_or_default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Loss<O = BaseOps>(pub Array1<f32, O>);

#[derive(Debug, Clone, Default, Index, IndexMut)]
pub struct ZValues<O = BaseOps>(#[index] pub Array1<f32, O>);

#[derive(Debug, Clone, Default, Index, IndexMut, Add, Sub, DeriveMoreMul, AsRef)]
pub struct Activations<O = BaseOps>(#[index] pub Array1<f32, O>);

impl Display for Activations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
