use nalgebra::{SMatrix, SVector};

#[derive(Debug)]
pub struct Model<const N: usize>
where
    [(); N * N]:,
{
    layer_0: SVector<f32, { N * N }>,
    weights: Vec<SMatrix<f32, { N * N }, { N * N }>>,
}

impl<const N: usize> Default for Model<N>
where
    [(); N * N]:,
{
    fn default() -> Self {
        Self {
            layer_0: SMatrix::repeat(0.0),
            weights: Default::default(),
        }
    }
}
