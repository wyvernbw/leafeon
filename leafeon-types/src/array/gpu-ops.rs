use std::{
    num::NonZero,
    ops::DerefMut,
    sync::{Arc, Mutex},
};

use bytemuck::Pod;
use leafeon_gpu::executor::{self, executor, Executor};
use ndarray::{linalg::Dot, ArrayBase, ArrayView, Data, Dim, Dimension, RawData};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use super::Array;

pub type GpuOps = Arc<Mutex<GpuOpsInner>>;

#[derive(Debug, Clone)]
pub struct GpuOpsInner {
    pub(crate) storage_buffer: Option<Arc<wgpu::Buffer>>,
    pub(crate) read_buffer: Option<Arc<wgpu::Buffer>>,
    pub(crate) executor: &'static Mutex<Executor>,
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
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
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
    <S as RawData>::Elem: Pod,
    D: Dimension<Pattern = (usize, usize)>,
    ArrayBase<S, D>: Dot<ArrayBase<S, D>>,
{
    type Output = Array<ArrayBase<S, D>, GpuOps>;

    fn dot(&self, rhs: &Array<ArrayBase<S, D>, GpuOps>) -> Self::Output {
        let ops_clone = self.operations.clone();
        let mut lhs_ops = self.operations.lock().unwrap();
        lhs_ops.update_storage::<S::Elem, D>(self.data.view());
        lhs_ops.update_read::<S::Elem>(rhs.data.len());
        let mut rhs_ops = rhs.operations.lock().unwrap();
        rhs_ops.update_storage(rhs.data.view());
        let state = &mut lhs_ops.executor.lock().unwrap().state;
        let dims_slice = &[self.data.dim().1, rhs.data.dim().0];
        let dimensions_buffer = state.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("matrix_dims"),
            contents: bytemuck::cast_slice(dims_slice),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let result_len = GpuOpsInner::multiply_len::<S, GpuOps, D>(self, rhs);
        let dims = rhs.data.dim();
        let (bindings, res_buffer_storage) = state
            .create_binary_op_bind_group()
            .pipeline_selector(&leafeon_gpu::gpu::PipelineSelector::MMul)
            .lhs_buffer(lhs_ops.storage_buffer.as_ref().unwrap())
            .rhs_buffer(rhs_ops.storage_buffer.as_ref().unwrap())
            .dims_buffer(&dimensions_buffer)
            .result_len(result_len as u64)
            .call();
        state
            .run_compute_pass()
            .bind_group(&bindings)
            .selector(&leafeon_gpu::gpu::PipelineSelector::MMul)
            .read_buffer(lhs_ops.read_buffer.as_ref().unwrap())
            .result_buffer_storage(&res_buffer_storage)
            .result_len(result_len as u64)
            .dims((dims.0 as u32, dims.1 as u32, 1))
            .call();
        let data = state.readback::<S, D>(lhs_ops.read_buffer.as_ref().unwrap(), dims);
        Self {
            data,
            operations: ops_clone,
        }
    }
}

pub mod tests {
    use ndarray::array;
    use ndarray::linalg::Dot;
    use rstest::rstest;

    use crate::array::Array2;

    use super::GpuOps;

    #[rstest]
    #[should_panic]
    #[case(array![[1, 2, 3], [4, 5, 6]], array![[4, 5, 6], [7, 8, 9]])]
    #[case(array![[1, 2, 3], [4, 5, 6]], array![[4, 5, 6], [7, 8, 9], [10, 11, 12]])]
    #[case(array![[1, 2, 3], [4, 5, 6]], array![[4, 5], [6, 7], [8, 9]])] // Rectangular matrix multiplication
    #[case(array![[0, 0, 0], [0, 0, 0]], array![[0, 0], [0, 0], [0, 0]])] // All zero matrices
    #[case(array![[1]], array![[1]])] // Single-element (1x1) matrices
    #[case(array![[-1, -2, -3], [-4, -5, -6]], array![[1, 2, 3], [4, 5, 6], [7, 8, 9]])] // Negative values in 'a'
    #[case(array![[1, 2, 3], [4, 5, 6]], array![[-4, -5], [-6, -7], [-8, -9]])] // Negative values in 'b'
    #[case(array![[1, 2]], array![[3], [4]])] // Non-square matrices
    #[case(array![[1, 2, 3]], array![[4], [5], [6]])] // Row-vector times column-vector
    #[case(array![[1], [2], [3]], array![[4, 5, 6]])] // Column-vector times row-vector
    fn test_dot_2(
        #[case] a: impl Into<Array2<i32, GpuOps>>,
        #[case] b: impl Into<Array2<i32, GpuOps>>,
    ) {
        let a = a.into();
        let b = b.into();
        let expected = a.data.dot(&b.data);
        let result = a.dot(&b);
        assert_eq!(result.data, expected);
    }
}
