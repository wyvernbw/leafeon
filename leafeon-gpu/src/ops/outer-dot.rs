use std::{num::NonZeroU32, ops::Mul};

use anyhow::Context;
use bytemuck::Pod;
use ndarray::{Array2, ArrayView1};
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor,
};

use crate::gpu::State;

impl State {
    /// computes the outer product for 2 column vectors, transposing the rhs
    pub fn try_outer_dot_1<T: Mul + Pod>(
        &self,
        lhs: ArrayView1<T>,
        rhs: ArrayView1<T>,
    ) -> anyhow::Result<Array2<T>> {
        let n = lhs.dim() as u32;
        let m = rhs.dim() as u32;
        let res_len = u32::try_from(usize::try_from(n * m)? * std::mem::size_of::<T>())
            .and_then(NonZeroU32::try_from)?;
        let lhs = lhs.as_slice().context("lhs is not contiguous")?;
        let lhs: &[u8] = bytemuck::cast_slice(lhs);
        let rhs = rhs.as_slice().context("rhs is not contiguous")?;
        let rhs: &[u8] = bytemuck::cast_slice(rhs);
        //let len = |arr: &[_]| u32::try_from(arr.len()).and_then(NonZeroU32::try_from);
        //let lhs_len = len(lhs)?;
        //let rhs_len = len(rhs)?;
        let lhs_buffer = self.create_buffer().data(lhs).idx(0).call();
        let rhs_buffer = self.create_buffer().data(rhs).idx(1).call();
        let res_slice = vec![0u8; res_len.get() as usize];
        let res_buffer_storage = self
            .create_buffer()
            .data(res_slice.as_slice())
            .idx(2)
            .usage(BufferUsages::STORAGE | BufferUsages::COPY_SRC)
            .call();
        let res_buffer_copy = self
            .create_buffer()
            .data(res_slice.as_slice())
            .idx(4)
            .usage(BufferUsages::COPY_DST | BufferUsages::MAP_READ)
            .call();
        let dims_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matrix_dims"),
                contents: bytemuck::cast_slice(&[n, m]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Vector outer product"),
                layout: None,
                module: &self.outer_product_shader,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("outerdot bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: lhs_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: rhs_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: res_buffer_storage.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        tracing::debug!("Finished creating bind group");

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("outerdot encoder"),
            });

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("outerdot pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        const WORKGROUP_SIZE: u32 = 32u32;
        let dispatch_x = n.div_ceil(WORKGROUP_SIZE);
        let dispatch_y = m.div_ceil(WORKGROUP_SIZE);

        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        drop(compute_pass);

        encoder.copy_buffer_to_buffer(
            &res_buffer_storage,
            0,
            &res_buffer_copy,
            0,
            res_slice.len() as u64,
        );

        let commands = encoder.finish();
        tracing::debug!(?commands, "Submitted commands");
        self.queue.submit(Some(commands));

        // Read the result from GPU
        let buffer_slice = res_buffer_copy.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |data| {
            tracing::debug!("Read finished");
            tx.send(data).unwrap();
        });
        tracing::debug!("Waiting for read...");
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let data = buffer_slice.get_mapped_range();
        assert_eq!(data.len(), res_len.get() as usize);
        let data = bytemuck::cast_slice::<_, T>(&data);

        Ok(Array2::from_shape_vec(
            (n as usize, m as usize),
            data.to_vec(),
        )?)
    }
}

#[cfg(test)]
pub mod tests {
    use ndarray::{Array1, Array2};
    use rand::Rng;
    use rstest::rstest;
    use std::ops::Mul;

    use crate::{executor::outer_dot_1, ops::mul::tests::equal_approx};

    #[rstest::fixture]
    fn random_col_vec() -> (Array1<f32>, Array1<f32>, Array2<f32>) {
        let n: usize = rand::thread_rng().gen_range(4..1024);
        let m: usize = rand::thread_rng().gen_range(4..1024);
        let gen = |_| rand::random::<f32>() * 100.0;

        let a = Array1::from_shape_fn(n, gen);
        let b = Array1::from_shape_fn(m, gen);

        let a_mat = a.broadcast((b.len(), a.len())).unwrap();
        let b_mat = b.broadcast((a.len(), b.len())).unwrap();
        let b_mat = b_mat.t();

        let expected = a_mat.to_owned().mul(b_mat);
        (a, b, expected)
    }

    #[rstest]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    #[case(random_col_vec())]
    fn fuzz_test(#[case] random_col_vec: (Array1<f32>, Array1<f32>, Array2<f32>)) {
        let (a, b, expected) = random_col_vec;
        let result = outer_dot_1(a.view(), b.view());
        equal_approx(result.view(), expected.view());
    }
}
