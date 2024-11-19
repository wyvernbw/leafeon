use std::{
    cmp::Ordering,
    convert::identity,
    f32::consts::SQRT_2,
    fmt::{Debug, Display},
    io::Write,
    iter::Sum,
    ops::Deref,
    os::unix::net,
    path::PathBuf,
    sync::Arc,
};

use anyhow::Context;
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

use crate::{default_progress_style, default_progress_style_pink, parser::Dataset};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// (n_out, n_in)
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
}

const EPSILON: f32 = 0.1;

pub fn relu(x: ArrayView1<f32>) -> Array1<f32> {
    x.map(|x| if *x <= 0.0 { EPSILON * x } else { *x })
}

pub fn relu_derivative(x: ArrayView1<f32>) -> Array1<f32> {
    x.map(|x| if *x > 0.0 { 1.0 } else { EPSILON })
}

pub fn softmax(logits: ArrayView1<f32>) -> Array1<f32> {
    let exp_logits = logits.map(|x| x.exp()); // Apply exp to each logit
    let sum_exp_logits = exp_logits.sum(); // Sum of exponentials
    exp_logits.map(|x| x / sum_exp_logits) // Normalize by the sum to get probabilities
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

impl Display for Activations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

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
                let err = &current_activation.0 - &target.0; // * relu_derivative(z_values.view());
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
        // DONE: here error does not increase in size!
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
    pub fn from_pretrained(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let weights = std::fs::read(path.into()).context("Failed to read weights file")?;
        bincode::deserialize(&weights).context("Failed to deserialize weights")
    }
    pub fn save_data(&self, path: impl Into<PathBuf>) -> anyhow::Result<()> {
        let data = bincode::serialize(self)?;
        //std::fs::create_dir_all(path.into())?;
        std::fs::File::create(path.into())?.write_all(&data)?;
        Ok(())
    }
    #[builder]
    pub fn untrained(input_size: usize, layer_spec: &[usize]) -> Self {
        let layers = layer_spec
            .iter()
            .scan(None, |prev_size_state, &size| {
                let prev_size = prev_size_state.unwrap_or(input_size);
                let generate = || random::<f32>() * (2.0 / prev_size as f32).sqrt();
                let weights = Array2::from_shape_fn((size, prev_size), |_| generate());
                let bias = Array1::from_shape_fn(size, |_| 0.0);
                *prev_size_state = Some(size);
                Some(Layer { weights, bias })
            })
            .collect();
        Network { layers }
    }

    pub fn forward(&self, input: Activations) -> (Vec<ZValues>, Vec<Activations>) {
        let (z_values, mut activations): (Vec<_>, Vec<_>) = self
            .layers
            .iter()
            .scan(input, |activations, layer| {
                let (z, a) = layer.forward().prev_activation(activations).call();
                *activations = a.clone();
                Some((z, a))
            })
            .unzip();
        let output = activations.last_mut().expect("No output layer!");
        *output = Activations(softmax(output.0.view()));
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
        let chunk_count = dataset.headers().image_count() as usize / chunk_size;
        tracing::info!(
            target: "model::training",
            "chunk size: {}, learning_rate: {}",
            chunk_size,
            learning_rate
        );
        // epoch loop
        let epoch_span = tracing::info_span!("epoch");
        epoch_span.pb_set_length(epochs as u64);
        epoch_span.pb_set_style(&default_progress_style_pink());
        let epoch_span = epoch_span.entered();

        (1..=epochs).fold(self, |state, _epoch| {
            let chunk_span = tracing::info_span!(parent: &epoch_span, "chunk");
            chunk_span.pb_set_length(chunk_count as u64);

            let chunk_span = chunk_span.entered();

            let shuffled = dataset.images().iter().choose_multiple(
                &mut rand::thread_rng(),
                dataset.headers().image_count() as usize,
            );
            let images = shuffled.chunks(chunk_size);
            // chunk loop
            let state = images.fold(state, |chunk_state, chunk| {
                // image loop
                let res = chunk
                    .into_par_iter()
                    .map(|(image, label)| {
                        let target = Activations(Array1::from_shape_fn(10, |i| match i {
                            i if i == *label as usize => 1.0,
                            _ => 0.0,
                        }));
                        let image: Array1<f32> = image.into();
                        let res = chunk_state
                            .backprop()
                            .target(target)
                            .input(Activations(image))
                            .learning_rate(learning_rate)
                            .call();
                        res
                        //state.gradient_descent((d_weights, d_bias))
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .fold(chunk_state.clone(), |state, a| state.gradient_descent(a));
                chunk_span.pb_inc(1);
                res
            });
            epoch_span.pb_inc(1);
            state
        })
    }

    pub fn predict(outputs: Activations) -> (usize, f32) {
        outputs
            .0
            .iter()
            .cloned()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(Ordering::Equal))
            .expect("No output layer!")
    }

    pub fn accuracy(&self, dataset: Dataset) -> f32 {
        let span = tracing::info_span!("accuracy");
        span.pb_set_length(dataset.headers().image_count() as u64);
        span.pb_set_style(&default_progress_style_pink());
        span.pb_set_message("Calculating accuracy...");
        let span = span.entered();
        let correct = dataset
            .images()
            .iter()
            .map(|(image, label)| {
                let output = self
                    .forward(Activations(image.into()))
                    .1
                    .last()
                    .unwrap()
                    .clone();
                let (prediction, _) = Self::predict(output);
                span.pb_inc(1);
                prediction == *label as usize
            })
            .filter(|x| *x)
            .count();
        correct as f32 / dataset.headers().image_count() as f32
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
