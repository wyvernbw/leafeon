use std::{
    fmt::Debug,
    marker::PhantomData,
    num::NonZero,
    ops::DerefMut,
    sync::{Arc, Mutex},
};

use bytemuck::Pod;
use leafeon_gpu::{
    executor::{self, executor, Executor},
    gpu::{PipelineSelector, SupportedPipeline},
};
use ndarray::{linalg::Dot, ArrayBase, ArrayView, Data, Dim, Dimension, RawData};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::Array;

pub type GpuOps = Arc<Mutex<GpuOpsInner>>;

#[derive(Debug, Clone)]
pub struct GpuOpsInner {
    pub(crate) storage_buffer: Option<Arc<wgpu::Buffer>>,
    pub(crate) read_buffer: Option<Arc<wgpu::Buffer>>,
    pub(crate) executor: &'static Mutex<Executor>,
}

#[cfg(feature = "serde")]
impl Serialize for GpuOpsInner {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_none()
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for GpuOpsInner {
    fn deserialize<D>(_: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::default())
    }
}

impl Default for GpuOpsInner {
    fn default() -> Self {
        Self {
            storage_buffer: None,
            read_buffer: None,
            executor: executor(),
        }
    }
}

impl GpuOpsInner {
    fn multiply_len<T: RawData + Data, O, D: Dimension>(
        lhs: &Array<ArrayBase<T, D>, O>,
        rhs: &Array<ArrayBase<T, D>, O>,
    ) -> usize {
        let lhs_len = lhs.data.view().len();
        let rhs_len = rhs.data.view().len();
        assert_eq!(lhs.data.shape()[1], rhs.data.shape()[0]);
        let d = lhs.data.shape()[1];
        let len = (lhs_len * rhs_len) / d.pow(2);
        len * std::mem::size_of::<T::Elem>()
    }
    fn multiplication_dims<T: RawData + Data, O, D: Dimension>(
        lhs: &Array<ArrayBase<T, D>, O>,
        rhs: &Array<ArrayBase<T, D>, O>,
    ) -> (usize, usize) {
        let binding = lhs.data.view();
        let lhs = binding.shape();
        let binding = rhs.data.view();
        let rhs = binding.shape();
        assert_eq!(lhs[1], rhs[0]);
        (lhs[0], rhs[1])
    }
    fn size_mismatch(buffer: &wgpu::Buffer, size: usize) -> bool {
        let buffer_size = buffer.as_entire_buffer_binding().size;
        buffer_size != NonZero::<u64>::new(size as u64)
    }

    pub fn update_storage<T: Pod, D: Dimension>(&mut self, array: ArrayView<'_, T, D>) {
        match &self.storage_buffer {
            Some(buffer) => {
                let data = array.as_slice().unwrap();
                let data = bytemuck::cast_slice(data);
                self.executor
                    .lock()
                    .unwrap()
                    .state
                    .queue
                    .write_buffer(buffer, 0, data);
            }
            None => self.prepare_storage_buffer(array),
        }
    }

    pub fn update_read<T: Pod>(&mut self, element_count: usize) {
        let size = element_count * std::mem::size_of::<T>();
        match &self.read_buffer {
            None => self.prepare_read_buffer::<T>(element_count),
            Some(buffer) if Self::size_mismatch(buffer, size) => {
                self.prepare_read_buffer::<T>(element_count);
            }
            _ => {}
        }
    }

    fn prepare_storage_buffer<T: Pod, D: Dimension>(&mut self, array: ArrayView<'_, T, D>) {
        let data = array.as_slice().unwrap();
        let data = bytemuck::cast_slice(data);
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let buffer = self
            .executor
            .lock()
            .unwrap()
            .state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage,
            });
        self.storage_buffer = Some(Arc::new(buffer));
    }

    fn prepare_read_buffer<T: Pod>(&mut self, element_count: usize) {
        let data = vec![0u8; element_count * std::mem::size_of::<T>()];
        let data = data.as_slice();
        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
        let buffer = self
            .executor
            .lock()
            .unwrap()
            .state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage,
            });
        self.read_buffer = Some(Arc::new(buffer));
    }
}

impl<S, D> Dot<Array<ArrayBase<S, D>, GpuOps>> for Array<ArrayBase<S, D>, GpuOps>
where
    S: RawData + Data + ndarray::DataOwned,
    <S as RawData>::Elem: Pod + Debug,
    D: Dimension<Pattern = (usize, usize)>,
    ArrayBase<S, D>: Dot<ArrayBase<S, D>>,
    // womp womp
    PipelineSelector<<S as RawData>::Elem>: SupportedPipeline,
{
    type Output = Array<ArrayBase<S, D>, GpuOps>;

    fn dot(&self, rhs: &Array<ArrayBase<S, D>, GpuOps>) -> Self::Output {
        let ops_clone = self.operations.clone();
        let mut lhs_ops = self.operations.lock().unwrap();
        let mut rhs_ops = rhs.operations.lock().unwrap();
        let result_len = GpuOpsInner::multiply_len::<S, GpuOps, D>(self, rhs);
        let result_dims = GpuOpsInner::multiplication_dims::<S, GpuOps, D>(self, rhs);
        lhs_ops.update_storage::<S::Elem, D>(self.data.view());
        rhs_ops.update_storage(rhs.data.view());
        lhs_ops.update_read::<S::Elem>(result_len / std::mem::size_of::<S::Elem>());
        tracing::debug!(
            "lhs storage: {:?}, lhs size: {:?}",
            lhs_ops.storage_buffer.as_ref().map(|buffer| buffer.size()),
            self.data.view().len()
        );
        tracing::debug!(
            "lhs (result) read: {:?}",
            lhs_ops.read_buffer.as_ref().map(|buffer| buffer.size())
        );
        tracing::debug!(
            "rhs storage: {:?}, rhs size: {:?}",
            rhs_ops.storage_buffer.as_ref().map(|buffer| buffer.size()),
            rhs.data.view().len()
        );
        let state = &mut lhs_ops.executor.lock().unwrap().state;
        assert_eq!(self.data.dim().1, rhs.data.dim().0);
        let dims_slice = &[
            self.data.dim().0 as u32,
            self.data.dim().1 as u32,
            rhs.data.dim().1 as u32,
            0u32,
        ];
        let dimensions_buffer = state.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("mmul_buffers_matrix_dims"),
            contents: bytemuck::cast_slice(dims_slice),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        tracing::info!(lhs = ?self.data.dim(), rhs = ?rhs.data.dim(), results = ?result_dims);
        let (bindings, res_buffer_storage) = state
            .create_binary_op_bind_group()
            .pipeline_selector(&PipelineSelector::<S::Elem>::mmul())
            .lhs_buffer(lhs_ops.storage_buffer.as_ref().unwrap())
            .rhs_buffer(rhs_ops.storage_buffer.as_ref().unwrap())
            .dims_buffer(&dimensions_buffer)
            .result_len(result_len as u64)
            .call();
        state
            .run_compute_pass()
            .bind_group(&bindings)
            .selector(&PipelineSelector::<S::Elem>::mmul())
            .read_buffer(lhs_ops.read_buffer.as_ref().unwrap())
            .result_buffer_storage(&res_buffer_storage)
            .result_len(result_len as u64)
            .dims((result_dims.0 as u32, result_dims.1 as u32, 1))
            .call();
        let data = state.readback::<S, D>(lhs_ops.read_buffer.as_ref().unwrap(), result_dims);
        Self {
            data,
            operations: ops_clone,
        }
    }
}

pub mod tests {
    use ndarray::array;
    use ndarray::linalg::Dot;
    use pretty_assertions::assert_eq;
    use rstest::{fixture, rstest};
    use tracing::instrument::WithSubscriber;

    use crate::array::Array2;
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    use super::GpuOps;

    #[fixture]
    #[once]
    fn init() {
        let filter =
            EnvFilter::new("wgpu_hal=off,wgpu=trace,leafeon_types=debug,leafeon_gpu=debug");

        // Set up the subscriber with filtering and logging to stdout
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().without_time())
            .with(filter)
            .init();
    }

    #[rstest]
    #[should_panic]
    #[case(array![[1, 2, 3], [4, 5, 6]], array![[4, 5, 6], [7, 8, 9]])]
    #[case(array![[1, 2, 3], [4, 5, 6]], array![[4, 5, 6], [7, 8, 9], [10, 11, 12]])]
    #[case(array![[1, 2, 3], [4, 5, 6]], array![[4, 5], [6, 7], [8, 9]])] // Rectangular matrix multiplication
    #[case(array![[0, 0, 0], [0, 0, 0]], array![[0, 0], [0, 0], [0, 0]])] // All zero matrices
    #[case(array![[1]], array![[1]])] // Single-element (1x1) matrices
    #[case(array![[-1, -2, -3], [-4, -5, -6]], array![[1, 2, 3], [4, 5, 6], [7, 8, 9]])] // Negative values in 'a'
    #[case(array![[1, 2, 3], [4, 5, 6]], array![[-4, -5], [-6, -7], [-8, -9]])] // Negative values in 'b'
    #[case(large_array(0), large_array(16))]
    fn test_dot_2_i32(
        #[case] a: impl Into<Array2<i32, GpuOps>>,
        #[case] b: impl Into<Array2<i32, GpuOps>>,
        _init: (),
    ) {
        let a = a.into();
        let b = b.into();
        let expected = a.data.dot(&b.data);
        let result = a.dot(&b);
        result
            .as_ref()
            .indexed_iter()
            .zip(expected.indexed_iter())
            .for_each(|(a, b)| assert_eq!(a, b));
    }

    #[rstest]
    #[case(array![[1, 2, 3], [4, 5, 6], [7, 8, 9]], array![[1, 2, 3], [4, 5, 6], [7, 8, 9]])]
    fn test_dot_2_repeated(
        #[case] a: impl Into<Array2<i32, GpuOps>>,
        #[case] b: impl Into<Array2<i32, GpuOps>>,
        _init: (),
    ) {
        let mut a = a.into();
        let b = b.into();
        let mut expected = a.data.clone();
        for _ in 0..5 {
            expected = expected.dot(&b.data);
        }
        for _ in 0..5 {
            a = a.dot(&b);
        }
        assert_eq!(a.data, expected);
    }

    fn large_array(offset: i32) -> Array2<i32, GpuOps> {
        ndarray::Array2::from_shape_fn((256, 256), |(x, y)| x as i32 + y as i32 + offset).into()
    }

    #[rstest]
    #[case(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], array![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])]
    // square case
    #[case(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], array![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])]
    fn test_dot_2_f32(
        #[case] a: impl Into<Array2<f32, GpuOps>>,
        #[case] b: impl Into<Array2<f32, GpuOps>>,
    ) {
        let a = a.into();
        let b = b.into();
        let expected = a.data.dot(&b.data);
        let result = a.dot(&b);
        assert_eq!(result.data, expected);
    }
}
