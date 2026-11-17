//! GPU 纯显存带宽测试：vec4 流式读取，无分支、无 atomic 竞争。
//!
//! 目的：区分「GPU 本身弱」和「负载模式把它拖慢了」。
//! trit_visibility 的 BINARY 只跑到 ~146 GB/s，而 4060 Ti 理论 288 GB/s。
//! 这个基准把分支和 atomicAdd 全去掉，只留 vec4 读取 + 累加。

use std::time::Instant;

const WG: u32 = 256;
const REP: u32 = 100;

const WGSL: &str = r#"
struct Params { words: u32, gx: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> data: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> sink: atomic<u32>;

@compute @workgroup_size(256)
fn bw_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    // 2D dispatch：x 方向最多 65535 组，超出走 y 方向
    let i = gid.x + gid.y * p.gx * 256u;
    if (i >= p.words) { return; }
    let v = data[i];
    let s = v.x + v.y + v.z + v.w;
    if (s > 1e30) {
        atomicAdd(&sink, 1u);
    }
}
"#;

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .expect("adapter");
    let info = adapter.get_info();
    let caps = adapter.limits();
    println!("GPU: {} ({:?})", info.name, info.backend);
    println!(
        "max_storage_buffer_binding_size: {} MB",
        caps.max_storage_buffer_binding_size / 1024 / 1024
    );

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: caps,
        ..Default::default()
    }))
    .expect("device");

    // 4060 Ti：GDDR6 128-bit @ 18 Gbps = 288 GB/s
    const THEORETICAL: f64 = 288.0;
    println!("理论显存带宽: {THEORETICAL:.0} GB/s（4060 Ti, 128-bit GDDR6 @18Gbps）\n");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bw"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bw"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("bw_kernel"),
        compilation_options: Default::default(),
        cache: None,
    });

    let sink = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sink"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    println!(
        "{:>10} | {:>14} | {:>10}",
        "缓冲区", "实测带宽", "占理论"
    );
    println!("{:-<10}-+-{:-<14}-+-{:-<10}", "", "", "");

    for &mb in &[16usize, 64, 128, 256, 512, 1024] {
        let bytes = mb * 1024 * 1024;
        let words = (bytes / 16) as u32;
        let data = vec![1.0f32; bytes / 4];

        use wgpu::util::DeviceExt;
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("data"),
            contents: unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
            },
            usage: wgpu::BufferUsages::STORAGE,
        });

        let groups = words.div_ceil(WG);
        let gx = groups.min(65535);
        let gy = groups.div_ceil(gx);

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut pd = [0u8; 8];
        pd[0..4].copy_from_slice(&words.to_le_bytes());
        pd[4..8].copy_from_slice(&gx.to_le_bytes());
        queue.write_buffer(&params, 0, &pd);

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sink.as_entire_binding(),
                },
            ],
        });

        // 预热
        for _ in 0..3 {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&pipe);
                cp.set_bind_group(0, &bg, &[]);
                cp.dispatch_workgroups(gx, gy, 1);
            }
            queue.submit([enc.finish()]);
        }
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // 测量：REP 次 dispatch，一次 poll
        let start = Instant::now();
        for _ in 0..REP {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&pipe);
                cp.set_bind_group(0, &bg, &[]);
                cp.dispatch_workgroups(gx, gy, 1);
            }
            queue.submit([enc.finish()]);
        }
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let secs = start.elapsed().as_secs_f64();

        let gb = (bytes as f64 * REP as f64) / 1e9;
        let gbps = gb / secs;
        println!(
            "{:>7} MB | {:>10.1} GB/s | {:>9.1}%",
            mb,
            gbps,
            gbps / THEORETICAL * 100.0
        );
    }

    println!();
    println!("读的是纯 streaming（vec4、无分支、无 atomic 竞争）。");
    println!("16 MB 落在 L2（32 MB）内会明显高于显存带宽，这是正常的。");
    println!("大缓冲稳定在多少，就是这张卡的真实显存带宽。");
}
