use std::{borrow::Cow, fmt::Debug, marker::PhantomData, path::PathBuf};

use anyhow::Context;
use bon::{bon, builder};
use bytemuck::Pod;
use ndarray::{ArrayBase, DataOwned, Dimension, RawData};
use serde::Deserialize;
use wgpu::{
    util::DeviceExt, Adapter, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, DeviceDescriptor, Instance, InstanceDescriptor, Limits, PowerPreference, Queue,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderModuleDescriptorSpirV,
};

use crate::MMUL;

#[derive(Debug)]
pub struct State {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub outer_product_shader: wgpu::ShaderModule,
    pub pipelines: Pipelines,
}

#[derive(Debug)]
pub struct Pipelines {
    pub mmul: MMulPipelines,
}

#[derive(Debug)]
pub struct MMulPipelines {
    pub shader: wgpu::ShaderModule,
    pub mmul_i32: ComputePipeline,
    pub mmul_f32: ComputePipeline,
}

impl MMulPipelines {
    pub fn new(device: &Device) -> Self {
        let mmul = wgpu::util::make_spirv(MMUL);
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mmul_shader"),
            source: mmul,
        });
        let create_pipeline = |name: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let mmul_i32 = create_pipeline("mmul_i32");
        let mmul_f32 = create_pipeline("mmul_f32");
        Self {
            shader,
            mmul_i32,
            mmul_f32,
        }
    }
}

pub trait SupportedPipeline: Debug {
    fn get<'a>(&self, state: &'a State) -> (&'a ComputePipeline, &'a wgpu::ShaderModule);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineSelector<T> {
    MMul(PhantomData<T>),
}

impl<T> PipelineSelector<T> {
    pub fn mmul() -> Self {
        PipelineSelector::MMul(PhantomData)
    }
}

impl SupportedPipeline for PipelineSelector<i32> {
    fn get<'a>(&self, state: &'a State) -> (&'a ComputePipeline, &'a wgpu::ShaderModule) {
        match self {
            PipelineSelector::MMul(_) => {
                (&state.pipelines.mmul.mmul_i32, &state.pipelines.mmul.shader)
            }
        }
    }
}

impl SupportedPipeline for PipelineSelector<f32> {
    fn get<'a>(&self, state: &'a State) -> (&'a ComputePipeline, &'a wgpu::ShaderModule) {
        match self {
            PipelineSelector::MMul(_) => {
                (&state.pipelines.mmul.mmul_f32, &state.pipelines.mmul.shader)
            }
        }
    }
}

#[bon]
impl State {
    pub fn try_new_sync() -> anyhow::Result<Self> {
        smol::block_on(Self::try_new())
    }
    pub async fn try_new() -> anyhow::Result<Self> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("Failed to request GPU adapter")?;
        tracing::info!(adapter_limit = ?adapter.limits());
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    required_limits: Limits {
                        max_compute_invocations_per_workgroup: 1024,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await?;
        let shader = wgpu::ShaderSource::Wgsl(include_str!("./ops/mult.wgsl").into());
        let mmul_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Matrix Multiplication Shader"),
            source: shader,
        });
        let outer_product_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Outer Product Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./ops/outer-product.wgsl").into()),
        });
        let mmul_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("mmul"),
            layout: None,
            module: &mmul_shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });
        let pipelines = Pipelines {
            mmul: MMulPipelines::new(&device),
        };

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            outer_product_shader,
            pipelines,
        })
    }
    #[builder]
    pub fn create_binary_op_bind_group<P: SupportedPipeline>(
        &mut self,
        pipeline_selector: &P,
        lhs_buffer: &wgpu::Buffer,
        rhs_buffer: &wgpu::Buffer,
        dims_buffer: &wgpu::Buffer,
        result_len: u64,
    ) -> (BindGroup, wgpu::Buffer) {
        let (pipeline, _) = pipeline_selector.get(self);
        let res_buffer_storage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("res_buffer_storage"),
            size: result_len,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some(&format!("{pipeline:?}_bind_group")),
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
        (bind_group, res_buffer_storage)
    }

    #[builder]
    pub fn run_compute_pass<P: SupportedPipeline>(
        &mut self,
        selector: &P,
        bind_group: &BindGroup,
        result_buffer_storage: &wgpu::Buffer,
        read_buffer: &wgpu::Buffer,
        dims: (u32, u32, u32),
        result_len: u64,
    ) {
        let label = format!("{selector:?}_pass");
        let label = label.as_str();
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("comptue_pass_encoder"),
            });

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });

        let (pipeline, _) = selector.get(self);
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);

        const WORKGROUP_SIZE: u32 = 32u32;
        let dispatch_x = dims.0.div_ceil(WORKGROUP_SIZE);
        let dispatch_y = dims.1.div_ceil(WORKGROUP_SIZE);
        let dispatch_z = dims.2.div_ceil(WORKGROUP_SIZE);

        tracing::info!("dispatching {:?}", (dispatch_x, dispatch_y, dispatch_z));
        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
        drop(compute_pass);

        encoder.copy_buffer_to_buffer(result_buffer_storage, 0, read_buffer, 0, result_len);

        let commands = encoder.finish();
        tracing::debug!(?commands, "Submitted commands");

        self.queue.submit(Some(commands));
    }

    pub fn readback<S: RawData + DataOwned, D: Dimension>(
        &self,
        read_buffer: &wgpu::Buffer,
        dims: D::Pattern,
    ) -> ArrayBase<S, D>
    where
        S::Elem: Pod + Debug,
    {
        let buffer_slice = read_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |data| {
            tracing::debug!("Read finished");
            tx.send(data).unwrap();
        });
        tracing::debug!("Waiting for read...");
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = buffer_slice.get_mapped_range();
        let byte_count = data.len();
        let data = bytemuck::cast_slice::<u8, S::Elem>(&data);
        assert_eq!(std::mem::size_of_val(data), byte_count);

        ArrayBase::<S, D>::from_shape_vec(dims, data.to_vec()).unwrap()
    }
}
