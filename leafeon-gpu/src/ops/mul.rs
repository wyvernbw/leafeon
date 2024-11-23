use bytemuck::Pod;
use ndarray::{Array2, ArrayView2};
use pretty_assertions::assert_eq;
use wgpu::{util::DeviceExt, BufferUsages, ShaderModuleDescriptor};

use std::{
    num::{NonZero, NonZeroU32, NonZeroU64},
    ops::{Add, Mul},
};

use anyhow::Context;
use bon::bon;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, Buffer, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, PipelineLayout, PipelineLayoutDescriptor,
    ShaderStages,
};

use crate::gpu::State;

#[bon]
impl State {
    #[builder]
    pub fn create_matrix_bind_group_layout(&self, idx: u32, len: &NonZero<u32>) -> BindGroupLayout {
        let label = format!("matrix_{idx}");
        let label = Some(label.as_str());
        self.device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label,
                entries: &[BindGroupLayoutEntry {
                    binding: idx,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(1).unwrap()),
                    },
                    count: Some(*len),
                }],
            })
    }

    #[builder]
    pub fn create_buffer(&self, data: &[u8], idx: u32, usage: Option<BufferUsages>) -> Buffer {
        let label = format!("buffer_{idx}");
        let label = Some(label.as_str());
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: data,
                usage: usage.unwrap_or(wgpu::BufferUsages::STORAGE),
            })
    }

    #[builder]
    fn create_layout(
        &self,
        lhs_len: &NonZero<u32>,
        rhs_len: &NonZero<u32>,
        res_len: &NonZero<u32>,
    ) -> PipelineLayout {
        self.device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("mmul"),
                bind_group_layouts: &[
                    &self
                        .create_matrix_bind_group_layout()
                        .idx(0)
                        .len(lhs_len)
                        .call(),
                    &self
                        .create_matrix_bind_group_layout()
                        .idx(1)
                        .len(rhs_len)
                        .call(),
                    &self
                        .create_matrix_bind_group_layout()
                        .idx(2)
                        .len(res_len)
                        .call(),
                ],
                push_constant_ranges: &[],
            })
    }

    pub async fn test_read(&self, arr: Array2<f32>) -> anyhow::Result<Array2<f32>> {
        let n = arr.dim().0 as u32;
        let m = arr.dim().1 as u32;
        let old_arr = arr.clone();
        let arr = arr.as_slice().context("lhs is not contiguous")?;
        let arr: &[u8] = bytemuck::cast_slice(arr);
        let arr_buffer = self
            .create_buffer()
            .usage(BufferUsages::COPY_SRC | BufferUsages::STORAGE)
            .data(arr)
            .idx(0)
            .call();
        let arr_copy_buffer = self
            .create_buffer()
            .data(arr)
            .idx(0)
            .usage(BufferUsages::COPY_DST | BufferUsages::MAP_READ)
            .call();
        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Test Read Pipeline"),
                layout: None,
                module: &self.device.create_shader_module(ShaderModuleDescriptor {
                    label: Some("Test Read Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("test.wgsl").into()),
                }),
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("mmul bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[BindGroupEntry {
                binding: 0,
                resource: arr_buffer.as_entire_binding(),
            }],
        });

        tracing::debug!("Finished creating bind group");

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("mmul encoder"),
            });

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("mmul pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        const WORKGROUP_SIZE: u32 = 32u32;

        compute_pass.dispatch_workgroups(WORKGROUP_SIZE, WORKGROUP_SIZE, 1);
        drop(compute_pass);

        encoder.copy_buffer_to_buffer(&arr_buffer, 0, &arr_copy_buffer, 0, arr.len() as u64);

        let commands = encoder.finish();
        tracing::debug!(?commands, "Submitted commands");
        self.queue.submit(Some(commands));

        // Read the result from GPU
        let buffer_slice = arr_copy_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |data| {
            tracing::debug!("Read finished");
            tx.send(data).unwrap();
        });
        tracing::debug!("Waiting for read...");
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let data = buffer_slice.get_mapped_range();
        let data = bytemuck::cast_slice::<_, f32>(&data);

        let new_arr = Array2::from_shape_vec((n as usize, m as usize), data.to_vec())?;
        assert_eq!(new_arr, old_arr);
        Ok(new_arr)
    }

    pub async fn dot<T: Mul + Add + Pod>(
        &self,
        lhs: ArrayView2<'_, T>,
        rhs: ArrayView2<'_, T>,
    ) -> Array2<T> {
        self.try_dot(lhs, rhs).await.unwrap()
    }

    pub async fn try_dot<T: Mul + Add + Pod>(
        &self,
        lhs: ArrayView2<'_, T>,
        rhs: ArrayView2<'_, T>,
    ) -> anyhow::Result<Array2<T>> {
        let n = lhs.dim().0 as u32;
        let m = lhs.dim().1 as u32;
        let p = rhs.dim().1 as u32;
        tracing::info!(?n, ?m, ?p);
        assert_eq!(lhs.dim().1, rhs.dim().0);
        let res_len = u32::try_from(usize::try_from(n * p)? * std::mem::size_of::<T>())
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
                contents: bytemuck::cast_slice(&[n, m, p]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Multiplication Pipeline"),
                layout: None,
                module: &self.multiply_shader,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("mmul bind group"),
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
                label: Some("mmul encoder"),
            });

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("mmul pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        const WORKGROUP_SIZE: u32 = 32u32;
        let dispatch_x = n.div_ceil(WORKGROUP_SIZE);
        let dispatch_y = p.div_ceil(WORKGROUP_SIZE);

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
            (n as usize, p as usize),
            data.to_vec(),
        )?)
    }
}

#[cfg(test)]
pub mod tests {
    use std::{convert::identity, ops::Div};

    use pretty_assertions::{assert_eq, assert_ne};

    use ndarray::{array, Array2, ArrayD, ArrayView2};
    use rand::Rng;
    use rstest::{fixture, rstest};

    use crate::gpu::State;

    #[fixture]
    #[once]
    fn init() -> () {
        tracing_subscriber::fmt::fmt();
        //.with_max_level(tracing::Level::DEBUG)
        //.init();
    }

    #[rstest::rstest]
    fn start_test(_init: &()) -> anyhow::Result<()> {
        smol::block_on(async {
            let state = State::try_new().await?;
            anyhow::Ok(())
        })?;
        Ok(())
    }

    #[rstest::rstest]
    fn test_slice(_init: &()) -> anyhow::Result<()> {
        let a = array![[1.0f32, 2., 3.], [4., 5., 6.]];
        let slice = a.as_slice().unwrap();
        assert_eq!(slice.len(), 6);
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[1], 2.0);
        assert_eq!(slice[2], 3.0);
        assert_eq!(slice[3], 4.0);
        assert_eq!(slice[4], 5.0);
        assert_eq!(slice[5], 6.0);
        let slice = slice.to_vec();
        let b = Array2::from_shape_vec((2, 3), slice)?;
        assert_eq!(a, b);
        Ok(())
    }

    #[rstest::rstest]
    fn test_read(_init: &()) -> anyhow::Result<()> {
        let a = array![[1.0f32, 2., 3.], [4., 5., 6.]];
        smol::block_on(async {
            let state = State::try_new().await?;
            let new_arr = state.test_read(a.clone()).await?;
            assert_eq!(new_arr, a);
            Ok(())
        })
    }

    #[rstest::rstest]
    fn test(_init: &()) -> anyhow::Result<()> {
        let a = array![[1.0f32, 2., 3.], [4., 5., 6.]];
        let b = array![[1., 2.], [3., 4.], [5., 6.]];
        let expected = a.dot(&b);
        smol::block_on(async {
            let state = State::try_new().await?;
            let result = state.dot(a.view(), b.view()).await;
            assert_eq!(result, expected);
            anyhow::Ok(())
        })
    }

    #[rstest::fixture]
    fn random_arrays() -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let n: usize = rand::thread_rng().gen_range(4..1024);
        let m: usize = rand::thread_rng().gen_range(4..1024);
        let p: usize = rand::thread_rng().gen_range(4..1024);
        let gen = |_| rand::random::<f32>() * 100.0;

        let a = Array2::from_shape_fn((n, m), gen);
        let b = Array2::from_shape_fn((m, p), gen);
        let expected = a.dot(&b);
        (a, b, expected)
    }

    pub fn equal_approx(a: ArrayView2<f32>, b: ArrayView2<f32>) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| x.div(y).abs() < 0.001)
    }

    #[rstest::rstest]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    #[case(random_arrays())]
    fn fuzz_test(init: &(), #[case] random_arrays: (Array2<f32>, Array2<f32>, Array2<f32>)) {
        smol::block_on(async {
            let (a, b, expected) = random_arrays;
            let state = State::try_new().await.unwrap();
            let result = state.dot(a.view(), b.view()).await;
            equal_approx(result.view(), expected.view());
        });
    }
}
