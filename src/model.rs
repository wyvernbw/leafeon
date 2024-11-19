use std::{fmt::Debug, iter::Sum, ops::Deref, os::unix::net, sync::Arc};

use bon::bon;
use derive_more::derive::{Add, AsRef, Index, IndexMut, Mul, MulAssign, Sub};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use ndarray::prelude::*;
use rand::{random, seq::IteratorRandom};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};
use tracing::{instrument, Span};
use tracing_indicatif::span_ext::IndicatifSpanExt;

use crate::parser::Dataset;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// (n_out, n_in)
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
}

pub fn relu(x: ArrayView1<f32>) -> Array1<f32> {
    const EPSILON: f32 = 1e-5;
    x.map(|x| x.max(EPSILON))
}

pub fn relu_derivative(x: ArrayView1<f32>) -> Array1<f32> {
    const EPSILON: f32 = 1e-5;
    x.map(|x| if x > &EPSILON { 1.0 } else { EPSILON })
}

pub enum BackwardKind<'a> {
    Output {
        current_activation: &'a Activations,
        target: &'a Activations,
    },
    Hidden {
        next_layer: &'a Layer,
        next_error: &'a Loss,
    },
}

impl Debug for BackwardKind<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackwardKind::Output { .. } => write!(f, "BackwardKind {{ Output }}"),
            BackwardKind::Hidden { .. } => write!(f, "BackwardKind {{ Hidden }}"),
        }
    }
}

#[derive(Debug, Clone, Default, Mul, MulAssign, Add, Sub)]
pub struct WeightGradient(pub Array2<f32>);

impl Sum for WeightGradient {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap_or_default()
    }
}

#[derive(Debug, Clone, Default, MulAssign, Mul, Add, Sub)]
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

#[derive(Debug, Clone, Default, Index, IndexMut, Add, Sub, Mul, AsRef)]
pub struct Activations(#[index] pub Array1<f32>);

#[bon]
impl Layer {
    pub fn new(weights: Array2<f32>, bias: Array1<f32>) -> Self {
        assert_eq!(weights.dim().1, bias.len());
        Self { weights, bias }
    }
    #[builder]
    pub fn forward(&self, prev_activation: &Activations) -> (ZValues, Activations) {
        let Activations(prev_activation) = prev_activation;
        assert_eq!(self.weights.dim().1, prev_activation.dim());
        let z = self.weights.dot(&prev_activation.view());
        assert_eq!(z.dim(), self.bias.dim());
        let z = z + &self.bias;
        let a = relu(z.view());
        (ZValues(z), Activations(a))
    }
    #[builder]
    pub fn backward(
        &self,
        prev_activations: &Activations,
        z_values: &ZValues,
        pass_data: BackwardKind<'_>,
    ) -> (WeightGradient, BiasGradient, Loss) {
        let Activations(prev_activations) = prev_activations;
        let ZValues(z_values) = z_values;
        //tracing::debug!(?prev_activations, ?z_values, ?pass_data);
        let error = match pass_data {
            BackwardKind::Output {
                current_activation,
                target,
            } => {
                assert_eq!(current_activation.0.dim(), target.0.dim());
                let err = (&current_activation.0 - &target.0) * relu_derivative(z_values.view());
                Loss(err)
            }
            BackwardKind::Hidden {
                next_layer,
                next_error,
            } => {
                let err =
                    next_layer.weights.t().dot(&next_error.0) * relu_derivative(z_values.view());
                Loss(err)
            }
        };
        // FIXME: here error does not increase in size!
        let weight_gradient = error
            .0
            .broadcast((1, error.0.len()))
            .expect("Failed to broadcast error column vector to matrix")
            .dot(
                &prev_activations
                    .broadcast((1, prev_activations.len()))
                    .expect("Failed to broadcast prev_activations column vector to matrix")
                    .t(),
            );
        let bias_gradient = error.0.clone();
        (
            WeightGradient(weight_gradient),
            BiasGradient(bias_gradient),
            error,
        )
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Network {
    layers: Vec<Layer>,
}

#[bon]
impl Network {
    #[builder]
    pub fn untrained(input_size: usize, layer_spec: &[usize]) -> Self {
        let layers = layer_spec
            .iter()
            .scan(None, |prev_size_state, &size| {
                let prev_size = prev_size_state.unwrap_or(input_size);
                let generate = || random::<f32>() * 0.1;
                let weights = Array2::from_shape_fn((size, prev_size), |_| generate());
                let bias = Array1::from_shape_fn(size, |_| generate());
                *prev_size_state = Some(size);
                Some(Layer { weights, bias })
            })
            .collect();
        Network { layers }
    }

    pub fn forward(&self, input: Activations) -> (Vec<ZValues>, Vec<Activations>) {
        let (z_values, activations): (Vec<_>, Vec<_>) = self
            .layers
            .iter()
            .scan(input, |activations, layer| {
                let (z, a) = layer.forward().prev_activation(activations).call();
                *activations = a.clone();
                Some((z, a))
            })
            .unzip();
        (z_values, activations)
    }

    #[builder]
    pub fn backprop(
        &self,
        target: Activations,
        input: Activations,
        learning_rate: f32,
    ) -> (Vec<WeightGradient>, Vec<BiasGradient>) {
        let (z_values, activations) = self.forward(input);
        let (d_weights, d_biases): (Vec<_>, Vec<_>) = self
            .layers
            .iter()
            .enumerate()
            .rev()
            .scan(
                (None, None),
                |(error_state, next_layer): &mut (Option<Loss>, Option<&Layer>),
                 (idx, layer): (usize, &Layer)| {
                    let pass_kind = match error_state {
                        None => BackwardKind::Output {
                            current_activation: &activations[idx],
                            target: &target,
                        },
                        Some(error) => BackwardKind::Hidden {
                            next_layer: next_layer.expect("l + 1 is out of bounds"),
                            next_error: error,
                        },
                    };
                    //tracing::info!("layer: {}, pass_kind: {:?}", idx, pass_kind);
                    let (weights_gradient, bias_gradient, error) = layer
                        .backward()
                        .prev_activations(&activations[idx])
                        .z_values(&z_values[idx])
                        .pass_data(pass_kind)
                        .call();
                    *error_state = Some(error);
                    *next_layer = Some(layer);
                    Some((
                        weights_gradient * learning_rate,
                        bias_gradient * learning_rate,
                    ))
                },
            )
            .unzip();
        (d_weights, d_biases)
    }

    fn compose_gradients(
        d_weights: &[&[WeightGradient]],
        d_biases: &[&[BiasGradient]],
    ) -> (Vec<WeightGradient>, Vec<BiasGradient>) {
        (0..d_weights[0].len())
            .map(|j| {
                d_weights
                    .iter()
                    .map(|row| &row[j])
                    .zip(d_biases.iter().map(|row| &row[j]))
                    .fold(
                        (WeightGradient::default(), BiasGradient::default()),
                        |(weights, biases), (w, b)| {
                            (
                                WeightGradient(weights.0 + &w.0),
                                BiasGradient(biases.0 + &b.0),
                            )
                        },
                    )
            })
            .unzip()
    }

    fn gradient_descent(
        &self,
        (d_weights, d_biases): (Vec<WeightGradient>, Vec<BiasGradient>),
    ) -> Self {
        let new_layers: Vec<_> = self
            .layers
            .par_iter()
            .zip(
                d_weights
                    .into_par_iter()
                    .rev()
                    .zip(d_biases.into_par_iter().rev()),
            )
            .map(|(layer, (weights_gradient, bias_gradient))| Layer {
                weights: &layer.weights - weights_gradient.0,
                bias: &layer.bias - bias_gradient.0,
            })
            .collect();
        Network { layers: new_layers }
    }

    #[builder]
    pub fn train(
        self,
        dataset: Dataset,
        epochs: usize,
        accuracy: Option<f32>,
        learning_rate: Option<f32>,
    ) -> Self {
        let accuracy = accuracy.unwrap_or(1.0);
        let chunk_size = dataset.headers().image_count() as f32 * accuracy;
        let learning_rate = learning_rate.unwrap_or(0.1) / chunk_size;
        let chunk_size = chunk_size as usize;
        tracing::info!(
            target: "model::training",
            "chunk size: {}, learning_rate: {}",
            chunk_size,
            learning_rate
        );
        // epoch loop
        (1..=epochs).fold(self.clone(), |state, epoch| {
            let span = tracing::info_span!("epoch", epoch, epochs);
            span.pb_set_style(&ProgressStyle::default_bar());
            span.pb_set_length(chunk_size as u64);
            let _handle = span.entered();

            tracing::info!(target: "model::training", "epoch: {}/{}", epoch, epochs);
            let shuffled = dataset.images().iter().choose_multiple(
                &mut rand::thread_rng(),
                dataset.headers().image_count() as usize,
            );
            let images = shuffled.chunks(chunk_size);
            // chunk loop

            images.fold(state, |state, chunk| {
                // image loop
                let res = chunk
                    .into_par_iter()
                    .map(|(image, label)| {
                        let target = Activations(Array1::from_shape_fn(10, |i| match i {
                            i if i == *label as usize => 1.0,
                            _ => 0.0,
                        }));
                        let res = self
                            .backprop()
                            .target(target)
                            .input(Activations(image.into()))
                            .learning_rate(learning_rate)
                            .call();
                        res
                        //state.gradient_descent((d_weights, d_bias))
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .fold(state, |state, a| state.gradient_descent(a));
                Span::current().pb_inc(1);
                res
            })
        })
    }
}

impl Debug for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let weights: Vec<_> = self.layers.iter().map(|l| l.weights.dim()).collect();
        let bias: Vec<_> = self.layers.iter().map(|l| l.bias.dim()).collect();
        write!(
            f,
            "Network {{ weights: {:#?}, bias: {:#?} }}",
            weights, bias
        )
    }
}
