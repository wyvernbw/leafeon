use anyhow::Context;
use wgpu::{
    Adapter, Backends, Device, DeviceDescriptor, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions, ShaderModuleDescriptor,
};

pub struct State {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub multiply_shader: wgpu::ShaderModule,
    pub outer_product_shader: wgpu::ShaderModule,
}

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
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Matrix Multiplication Shader"),
            source: shader,
        });
        let outer_product_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Outer Product Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./ops/outer-product.wgsl").into()),
        });
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            multiply_shader: shader,
            outer_product_shader,
        })
    }
}
