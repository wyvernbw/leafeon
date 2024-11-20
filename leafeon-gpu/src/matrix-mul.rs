use bytemuck::Pod;
use ndarray::Array2;
use wgpu::util::DeviceExt;

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

pub struct MatrixMult {
    state: State,
}

#[bon]
impl MatrixMult {
    pub async fn try_new() -> anyhow::Result<Self> {
        let state = State::try_new().await?;
        Ok(Self { state })
    }
    #[builder]
    fn create_matrix_bind_group_layout(&self, idx: u32, len: &NonZero<u32>) -> BindGroupLayout {
        let label = format!("matrix_{idx}");
        let label = Some(label.as_str());
        self.state
            .device
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
    fn create_matrix_buffer(&self, data: &[u8], idx: u32) -> Buffer {
        let label = format!("matrix_{idx}");
        let label = Some(label.as_str());
        self.state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    #[builder]
    fn create_layout(
        &self,
        lhs_len: &NonZero<u32>,
        rhs_len: &NonZero<u32>,
        res_len: &NonZero<u32>,
    ) -> PipelineLayout {
        self.state
            .device
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

    pub async fn dot<T: Mul + Add + Pod>(
        &self,
        lhs: Array2<T>,
        rhs: Array2<T>,
    ) -> anyhow::Result<Array2<T>> {
        let n = lhs.dim().0 as u64;
        let m = lhs.dim().1 as u64;
        let p = rhs.dim().1 as u64;
        let res_len = u32::try_from(usize::try_from(n * p)? * std::mem::size_of::<T>())
            .and_then(NonZeroU32::try_from)?;
        let lhs = lhs.as_slice().context("lhs is not contiguous")?;
        let lhs: &[u8] = bytemuck::cast_slice(lhs);
        let rhs = rhs.as_slice().context("rhs is not contiguous")?;
        let rhs: &[u8] = bytemuck::cast_slice(rhs);
        let len = |arr: &[_]| u32::try_from(arr.len()).and_then(NonZeroU32::try_from);
        let lhs_len = len(lhs)?;
        let rhs_len = len(rhs)?;
        let lhs_buffer = self.create_matrix_buffer().data(lhs).idx(0).call();
        let rhs_buffer = self.create_matrix_buffer().data(rhs).idx(1).call();
        let res_slice = vec![0u8; res_len.get() as usize];
        let res_buffer = self
            .create_matrix_buffer()
            .data(res_slice.as_slice())
            .idx(2)
            .call();
        let dims_buffer = self
            .state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matrix_dims"),
                contents: bytemuck::cast_slice(&[m, n, p]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let pipeline = self
            .state
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Multiplication Pipeline"),
                layout: None,
                module: &self.state.multiply_shader,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });

        let bind_group = self.state.device.create_bind_group(&BindGroupDescriptor {
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
                    resource: res_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .state
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
        compute_pass.dispatch_workgroups(n as u32, m as u32, 1);

        self.state.queue.submit(Some(encoder.finish()));

        // Read the result from GPU
        let buffer_slice = res_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |data| {
            tx.send(data).unwrap();
        });
        rx.recv()??;
        let data = buffer_slice.get_mapped_range();
        let data = bytemuck::cast_slice::<u8, T>(&data);

        Ok(Array2::from_shape_vec(
            (n as usize, p as usize),
            data.to_vec(),
        )?)
    }
}
