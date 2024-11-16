use std::{f32::consts::SQRT_2, random::random};

use nalgebra::{Const, DMatrix, DVector, Dyn, Matrix};
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
};

use crate::parser::Dataset;

#[derive(Debug, Clone)]
pub struct OneNeuronExample {
    bias: DVector<f32>,
    weights: DVector<f32>,
}

#[derive(Debug, Clone)]
pub struct OneNeuronCostDerivatives {
    pub dc_dw: f32,
    pub dc_db: f32,
}

impl OneNeuronExample {
    pub fn random(len: usize) -> Self {
        let init_weight = |_, _| (rand::thread_rng().gen::<f32>() * 2.0 - 1.0) * SQRT_2;
        let init_biases = |_, _| (rand::thread_rng().gen::<f32>() * 2.0 - 1.0) * 0.01;
        let weights = DVector::from_fn(len, init_weight);
        let bias = DVector::from_fn(len, init_biases);
        Self { weights, bias }
    }
    pub fn zero(len: usize) -> Self {
        let weights = DVector::zeros(len);
        let bias = DVector::zeros(len);
        Self { weights, bias }
    }
    fn relu(x: f32) -> f32 {
        x.max(x * 0.01)
    }
    fn calculate_layer(&self, w_l: f32, a_l1: f32, b_l: f32) -> f32 {
        Self::relu(w_l * a_l1 + b_l)
    }
    pub fn predict(&self, input: f32) -> f32 {
        (0..self.weights.len())
            .zip(self.weights.iter().zip(self.bias.iter()))
            .fold(input, |a_l1, (_, (&w_l, &b_l))| {
                self.calculate_layer(w_l, a_l1, b_l)
            })
    }

    fn relu_derivative(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.01
        }
    }

    fn calculate_derivatives(
        &self,
        a_l: f32,
        a_l1: f32,
        y: f32,
        z_l: impl Fn() -> f32,
    ) -> OneNeuronCostDerivatives {
        let z_l = z_l();
        let dc_dw = 2. * (a_l - y) * a_l1 * Self::relu_derivative(z_l);
        let dc_db = 2. * (a_l - y) * Self::relu_derivative(z_l);
        let clip_threshold = 1.0;
        let dc_db = dc_db.clamp(-clip_threshold, clip_threshold);
        OneNeuronCostDerivatives { dc_dw, dc_db }
    }

    fn calculate_all_layers(
        &self,
        input: f32,
        output: f32,
        index: Option<usize>,
    ) -> (DVector<f32>, Vec<OneNeuronCostDerivatives>) {
        let index = index.unwrap_or(self.weights.len() - 1);
        match index {
            0 => {
                let w_l = self.weights[0];
                let b_l = self.bias[0];
                (
                    vec![input].into(),
                    vec![self.calculate_derivatives(input, input, output, || w_l * input + b_l)],
                )
            }
            idx => {
                let (prev_layers, prev_derivatives) =
                    self.calculate_all_layers(input, output, Some(idx - 1));
                let a_l1 = prev_layers[prev_layers.len() - 1];
                let w_l = self.weights[idx];
                let b_l = self.bias[idx];
                let a_l = self.calculate_layer(w_l, a_l1, b_l);
                let z_l = w_l * a_l1 + b_l;
                let d = self.calculate_derivatives(a_l, a_l1, 1.0, || z_l);
                (
                    [&[a_l], prev_layers.as_slice()].concat().into(),
                    [&[d], prev_derivatives.as_slice()].concat(),
                )
            }
        }
    }
    // Function to get the negative gradient for weights and biases
    pub fn get_negative_gradient(&self, target: f32) -> (DVector<f32>, DVector<f32>) {
        let initial_a_l = 0.5; // Assume some starting activation value (e.g., 0.5)

        // Call the recursive function to calculate all layers' activations and gradients
        let (_, gradients) = self.calculate_all_layers(initial_a_l, target, None);

        // Now extract the gradients for weights and biases
        let weights_grad = gradients.iter().map(|d| d.dc_dw).collect::<Vec<_>>();
        let biases_grad = gradients.iter().map(|d| d.dc_db).collect::<Vec<_>>();

        (weights_grad.into(), biases_grad.into())
    }

    pub fn backpropagate(mut self, target: f32, dataset: Dataset, take: usize) -> Self {
        let mut images = dataset
            .images()
            .iter()
            .array_chunks::<100>()
            .collect::<Vec<_>>();
        images.shuffle(&mut rand::thread_rng());
        let learning_rate = 0.1;
        for piece in images.iter().take(take) {
            let grad = piece
                .iter()
                .map(|(image, label)| self.get_negative_gradient(target))
                .fold(
                    Matrix::<f32, Dyn, Const<2>, _>::zeros(self.weights.len()),
                    |acc, (w_grad, b_grad)| {
                        Matrix::<f32, Dyn, Const<2>, _>::from_columns(&[
                            acc.column(0) + w_grad * learning_rate,
                            acc.column(1) + b_grad * learning_rate,
                        ])
                    },
                );
            let grad = grad / (piece.len() as f32);
            self.weights -= grad.column(0);
            self.bias -= grad.column(1);
        }
        self
    }
}
