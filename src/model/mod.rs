use core::panic;
use std::{f32::consts::SQRT_2, random::random};

use nalgebra::{Const, DMatrix, DVector, Dyn, Matrix, SVector};
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

fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.01
    }
}

#[derive(Debug, Clone)]
pub struct DModel {
    layer_weights: Vec<DMatrix<f32>>,
    bias: Vec<DVector<f32>>,
}

pub struct LayerSpec(pub usize, pub usize);

#[derive(Debug, Clone)]
pub struct CostDerivatives {
    pub dc_dw: DMatrix<f32>,
    pub dc_db: DVector<f32>,
}

impl DModel {
    pub fn untrained(format: &[(usize, LayerSpec)]) -> Self {
        let init_weight = |_, _| (rand::thread_rng().gen::<f32>() * 2.0 - 1.0) * SQRT_2;
        let init_biases = |_, _| (rand::thread_rng().gen::<f32>() * 2.0 - 1.0) * 0.01;

        let layer_weights = format
            .iter()
            .flat_map(|(len, space)| match len {
                0..=1 => std::iter::repeat_n(space, *len),
                len => {
                    assert!(
                        space.0 == space.1,
                        "Expected square matrix for repeated layers."
                    );
                    std::iter::repeat_n(space, *len)
                }
            })
            .map(|&LayerSpec(columns, rows)| DMatrix::from_fn(rows, columns, init_weight))
            .collect::<Vec<_>>();
        let bias = layer_weights
            .iter()
            .map(|w| DVector::from_fn(w.nrows(), init_biases))
            .collect();

        Self {
            layer_weights,
            bias,
        }
    }
    fn relu(x: DVector<f32>) -> DVector<f32> {
        x.map(|x| x.max(0.001))
    }
    fn calculate_layer(
        &self,
        w_l: &DMatrix<f32>,
        a_l1: &DVector<f32>,
        b_l: &DVector<f32>,
    ) -> DVector<f32> {
        Self::relu(w_l * a_l1 + b_l)
    }

    fn relu_derivative(x: &DVector<f32>) -> DVector<f32> {
        x.map(relu_derivative)
    }

    fn calculate_output_derivatives(
        &self,
        a_l: &DVector<f32>,
        a_l1: &DVector<f32>,
        y: &DVector<f32>,
        z_l: &DVector<f32>,
    ) -> CostDerivatives {
        let relu_prime = Self::relu_derivative(z_l); // Size: (n_out)

        dbg!(a_l.nrows(), a_l.ncols());
        dbg!(a_l1.nrows(), a_l1.ncols());
        dbg!(z_l.nrows(), z_l.ncols());

        // Compute delta_l = (a_l - y) ⊙ relu_prime
        let delta_l = (a_l - y).component_mul(&relu_prime); // Size: (n_out)

        // Compute dc_dw = delta_l ⊗ a_l1
        let dc_dw = DMatrix::from_fn(delta_l.len(), a_l1.len(), |i, j| delta_l[i] * a_l1[j]); // Shape: (n_out, n_in)

        // Compute dc_db = delta_l
        let dc_db = delta_l.clone(); // Bias gradient is just the error term (size: n_out)

        // Optional: Clip gradients for stability
        let clip_threshold = 1.0;
        let dc_db = dc_db.map(|x| x.clamp(-clip_threshold, clip_threshold));

        CostDerivatives { dc_dw, dc_db }
    }

    fn calculate_hidden_layer_derivatives(
        &self,
        a_l: &DVector<f32>,
        a_l1: &DVector<f32>,
        delta_l: &DVector<f32>,
        z_l: &DVector<f32>,
    ) -> CostDerivatives {
        dbg!(a_l.nrows(), a_l.ncols());
        dbg!(a_l1.nrows(), a_l1.ncols());
        dbg!(z_l.nrows(), z_l.ncols());

        // Compute dc_dw = delta_l ⊗ a_l1
        let dc_dw = DMatrix::from_fn(delta_l.len(), a_l1.len(), |i, j| delta_l[i] * a_l1[j]); // Shape: (n_out, n_in)

        // Compute dc_db = delta_l
        let dc_db = delta_l.clone(); // Bias gradient is just the error term (size: n_out)

        // Optional: Clip gradients for stability
        let clip_threshold = 1.0;
        let dc_db = dc_db.map(|x| x.clamp(-clip_threshold, clip_threshold));

        CostDerivatives { dc_dw, dc_db }
    }

    fn backpropagation(
        &self,
        input: &DVector<f32>,
        output: &DVector<f32>,
        index: Option<usize>,
        activations: Option<Vec<DVector<f32>>>,
        cost_derivatives: Option<Vec<CostDerivatives>>,
    ) -> (Vec<DVector<f32>>, Vec<CostDerivatives>) {
        let index = index.unwrap_or(0);
        let activations = activations.unwrap_or_default();
        let cost_derivatives = cost_derivatives.unwrap_or_default();
        let done_forward = || activations.len() == self.layer_weights.len() + 1;
        match (&activations[..], index) {
            ([], 0) => {
                self.backpropagation(input, output, Some(1), Some(vec![input.clone()]), None)
            }
            (_, index) if !done_forward() => {
                let a_l1 = &activations[index - 1];
                let b_l = &self.bias[index - 1];
                let a_l = self.calculate_layer(&self.layer_weights[index - 1], a_l1, b_l);
                let activations = [&activations[..], &[a_l]].concat();
                let (layers, derivatives) =
                    self.backpropagation(input, output, Some(index + 1), Some(activations), None);
                (layers, derivatives)
            }
            (_, 0) if done_forward() => (activations, cost_derivatives),
            (_, index) if done_forward() && index == activations.len() => {
                // performing the backward pass
                let a_l1 = &activations[index - 2];
                let w_l = &self.layer_weights[index - 1];
                let b_l = &self.bias[index - 1];
                let z_l = w_l * a_l1 + b_l;
                let a_l = self.calculate_layer(&self.layer_weights[index - 1], a_l1, b_l);

                let dc = self.calculate_output_derivatives(&a_l, a_l1, output, &z_l);
                self.backpropagation(
                    input,
                    output,
                    Some(index - 1),
                    Some(activations),
                    Some([cost_derivatives.as_slice(), &[dc]].concat()),
                )
            }

            (_, index) if done_forward() => {
                // performing the backward pass
                let cost_idx = activations.len() - index - 1;
                let a_l = &activations[index - 1];
                let a_l1 = &activations[index - 2];
                let b_l = &self.bias[index - 1];
                let w_l = &self.layer_weights[index - 1];
                let z_l = w_l * a_l1 + b_l;

                let delta_l_next = &cost_derivatives[cost_idx].dc_db; // The error from the next layer

                // Compute the error term for the current layer (hidden layer)
                let delta_l = (self.layer_weights[index].transpose() * delta_l_next)
                    .component_mul(&Self::relu_derivative(&z_l));

                let dc = self.calculate_hidden_layer_derivatives(a_l, a_l1, &delta_l, &z_l);

                self.backpropagation(
                    input,
                    output,
                    Some(index - 1),
                    Some(activations.clone()),
                    Some([cost_derivatives.as_slice(), &[dc]].concat()),
                )
            }
            _ => unreachable!(),
        };
        todo!()
    }
    fn backpropagation2(
        &self,
        input: &DVector<f32>,
        output: &DVector<f32>,
        index: Option<usize>,
        activations: Option<Vec<DVector<f32>>>,
        cost_derivatives: Option<Vec<CostDerivatives>>,
    ) -> (Vec<DVector<f32>>, Vec<CostDerivatives>) {
        let index = index.unwrap_or(0);
        let activations = activations.unwrap_or_default();
        let cost_derivatives = cost_derivatives.unwrap_or_default();
        let done_forward = || activations.len() == self.layer_weights.len() + 1;
        match (&activations[..], index) {
            ([], 0) => {
                self.backpropagation2(input, output, Some(1), Some(vec![input.clone()]), None)
            }
            (_, index) if !done_forward() => {
                let a_l1 = &activations[index - 1];
                let b_l = &self.bias[index - 1];
                let a_l = self.calculate_layer(&self.layer_weights[index - 1], a_l1, b_l);
                let activations = [&activations[..], &[a_l]].concat();
                let (layers, derivatives) =
                    self.backpropagation2(input, output, Some(index + 1), Some(activations), None);
                (layers, derivatives)
            }
            (_, 0) if done_forward() => (activations, cost_derivatives),
            (_, index) if done_forward() && index == activations.len() => {
                // performing the backward pass
                let a_l1 = &activations[index - 2];
                let w_l = &self.layer_weights[index - 1];
                let b_l = &self.bias[index - 1];
                let z_l = w_l * a_l1 + b_l;
                let a_l = self.calculate_layer(&self.layer_weights[index - 1], a_l1, b_l);

                let dc = self.calculate_output_derivatives(&a_l, a_l1, output, &z_l);
                self.backpropagation2(
                    input,
                    output,
                    Some(index - 1),
                    Some(activations),
                    Some([cost_derivatives.as_slice(), &[dc]].concat()),
                )
            }

            (_, index) if done_forward() => {
                // performing the backward pass
                let cost_idx = activations.len() - index - 1;
                let a_l = &activations[index - 1];
                let a_l1 = &activations[index - 2];
                let b_l = &self.bias[index - 1];
                let w_l = &self.layer_weights[index - 1];
                let z_l = w_l * a_l1 + b_l;

                let delta_l_next = &cost_derivatives[cost_idx].dc_db; // The error from the next layer

                // Compute the error term for the current layer (hidden layer)
                let delta_l = (self.layer_weights[index].transpose() * delta_l_next)
                    .component_mul(&Self::relu_derivative(&z_l));

                let dc = self.calculate_hidden_layer_derivatives(a_l, a_l1, &delta_l, &z_l);

                self.backpropagation2(
                    input,
                    output,
                    Some(index - 1),
                    Some(activations.clone()),
                    Some([cost_derivatives.as_slice(), &[dc]].concat()),
                )
            }
            _ => unreachable!(),
        }
    }

    // Function to get the negative gradient for weights and biases
    fn get_negative_gradient(
        &self,
        input: &DVector<f32>,

        target: &DVector<f32>,
    ) -> (Vec<DMatrix<f32>>, Vec<DVector<f32>>) {
        let (_, gradients) = self.backpropagation2(input, target, None, None, None);

        // Now extract the gradients for weights and biases
        let (weights_grad, biases_grad) = gradients.into_iter().map(|d| (d.dc_dw, d.dc_db)).unzip();

        (weights_grad, biases_grad)
    }

    pub fn train(mut self, dataset: Dataset, take: usize) -> Self {
        let mut images = dataset
            .images()
            .iter()
            .array_chunks::<100>()
            .collect::<Vec<_>>();
        images.shuffle(&mut rand::thread_rng());
        tracing::info!("training on {} images...", dataset.images().len());
        let count = images.iter().take(take).count();
        for (idx, piece) in images.iter().take(take).enumerate() {
            let (w_grad, b_grad): (Vec<_>, Vec<_>) = piece
                .iter()
                .map(|(image, label)| {
                    let target =
                        DVector::from_fn(10, |i, _| if i == *label as usize { 1.0 } else { 0.0 });
                    self.get_negative_gradient(&image.into(), &target)
                })
                .unzip();

            let learning_rate = 0.1;
            let coefficient = 1.0 / piece.len() as f32 * learning_rate;

            for (col, grad_col) in w_grad.iter().enumerate() {
                for grad in grad_col.iter() {
                    self.layer_weights[col] += grad * coefficient;
                }
            }

            for (row, grad_row) in b_grad.iter().enumerate() {
                for grad in grad_row.iter() {
                    self.bias[row] += grad * coefficient;
                }
            }
            tracing::info!("{}/{count}", idx + 1);
        }
        self
    }
}
