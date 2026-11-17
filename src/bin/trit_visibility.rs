//! GPU 位并行可见性筛选：同一查询，两种表示。
//!
//! 查询：`a == True && b == True && c != False`（三个三值条件）
//!
//! - BINARY：每对象 3 个 f32（12 MB），一个线程处理一个对象，3 次比较 + 分支
//! - TERNARY：每对象 6 bit（pos/neg 位平面，375 KB），**一个线程处理 64 个对象**，
//!   一条 `pa & pb & ~nc` 判定 64 个，`countOneBits` 一次计数
//!
//! 两个 kernel 做完全相同的逻辑，只有数据表示与线程粒度不同。

use std::time::Instant;

const N: u32 = 10_000_000;
const NW: u32 = N / 32; // 位平面字数（u32 = 32 位）
const REPS: u32 = 200;
const WG: u32 = 256;

const WGSL: &str = r#"
struct Params { n: u32, nw: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read_write> counter: atomic<u32>;

@compute @workgroup_size(256)
fn binary_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    // a == +1 && b == +1 && c != -1
    if (a[i] > 0.5 && b[i] > 0.5 && c[i] > -0.5) {
        atomicAdd(&counter, 1u);
    }
}
"#;

const WGSL_TERNARY: &str = r#"
struct Params { n: u32, nw: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> pa: array<u32>;
@group(0) @binding(2) var<storage, read> pb: array<u32>;
@group(0) @binding(3) var<storage, read> nc: array<u32>;
@group(0) @binding(4) var<storage, read_write> counter: atomic<u32>;

@compute @workgroup_size(256)
fn ternary_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= p.nw) { return; }
    // 一次判定 64 个对象
    let bits = pa[w] & pb[w] & ~nc[w];
    let n = countOneBits(bits);
    if (n > 0u) {
        atomicAdd(&counter, n);
    }
}
"#;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// -1 / 0 / +1
    fn tri(&mut self) -> i32 {
        (self.next() % 3) as i32 - 1
    }
}

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
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("device");
    println!("GPU: {}", info.name);

    // ---- 数据 ----
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut a = vec![0f32; N as usize];
    let mut b = vec![0f32; N as usize];
    let mut c = vec![0f32; N as usize];
    let mut pa = vec![0u32; NW as usize];
    let mut pb = vec![0u32; NW as usize];
    let mut nc = vec![0u32; NW as usize];

    for i in 0..N as usize {
        let va = rng.tri();
        let vb = rng.tri();
        let vc = rng.tri();
        a[i] = va as f32;
        b[i] = vb as f32;
        c[i] = vc as f32;
        let bit = 1u32 << (i % 32);
        let w = i / 32;
        if va == 1 {
            pa[w] |= bit;
        }
        if vb == 1 {
            pb[w] |= bit;
        }
        if vc == -1 {
            nc[w] |= bit;
        }
    }

    // 期望结果（CPU 参考）
    let mut expect = 0u32;
    for i in 0..N as usize {
        if a[i] > 0.5 && b[i] > 0.5 && c[i] > -0.5 {
            expect += 1;
        }
    }
    println!("对象 {N}  期望命中 {expect}\n");

    // ---- 二进制侧缓冲 ----
    let bin_buf = |data: &[f32], label: &str| {
        use wgpu::util::DeviceExt;
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck_cast(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };

    // bytemuck 不在依赖里，手工转字节
    fn bytemuck_cast(data: &[f32]) -> &[u8] {
        // 安全：f32 的任何位模式都是有效 u8 序列
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
    }
    fn u32_cast(data: &[u32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
    }

    let _ = bin_buf; // 占位，下面直接用 create_buffer_init

    use wgpu::util::DeviceExt;
    let mk_f32 = |data: &[f32], label: &str| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck_cast(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let mk_u32 = |data: &[u32], label: &str| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: u32_cast(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };

    let a_buf = mk_f32(&a, "a");
    let b_buf = mk_f32(&b, "b");
    let c_buf = mk_f32(&c, "c");
    let pa_buf = mk_u32(&pa, "pa");
    let pb_buf = mk_u32(&pb, "pb");
    let nc_buf = mk_u32(&nc, "nc");

    let counter = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("counter"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut pd = [0u8; 8];
    pd[0..4].copy_from_slice(&N.to_le_bytes());
    pd[4..8].copy_from_slice(&NW.to_le_bytes());
    queue.write_buffer(&params, 0, &pd);

    let shader_bin = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("binary"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });
    let shader_tri = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ternary"),
        source: wgpu::ShaderSource::Wgsl(WGSL_TERNARY.into()),
    });

    let bgl_bin = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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

    let bg_bin = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_bin,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: c_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: counter.as_entire_binding(),
            },
        ],
    });

    let bg_tri = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_bin,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pa_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pb_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: nc_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: counter.as_entire_binding(),
            },
        ],
    });

    let layout_bin = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl_bin)],
        immediate_size: 0,
    });
    let pipe_bin = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("binary"),
        layout: Some(&layout_bin),
        module: &shader_bin,
        entry_point: Some("binary_kernel"),
        compilation_options: Default::default(),
        cache: None,
    });
    let pipe_tri = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ternary"),
        layout: Some(&layout_bin),
        module: &shader_tri,
        entry_point: Some("ternary_kernel"),
        compilation_options: Default::default(),
        cache: None,
    });

    // ---- 测二进制 ----
    // 批量 dispatch：不在每轮里 clear/copy/poll，避免固定开销淹没真实差异
    let run = |pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, groups: u32| -> f64 {
        for _ in 0..5 {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(pipe);
                cp.set_bind_group(0, bg, &[]);
                cp.dispatch_workgroups(groups, 1, 1);
            }
            queue.submit([enc.finish()]);
        }
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let start = Instant::now();
        for _ in 0..REPS {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(pipe);
                cp.set_bind_group(0, bg, &[]);
                cp.dispatch_workgroups(groups, 1, 1);
            }
            queue.submit([enc.finish()]);
        }
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        start.elapsed().as_secs_f64() / REPS as f64
    };

    let bin_groups = N.div_ceil(WG);
    let tri_groups = NW.div_ceil(WG);

    let t_bin = run(&pipe_bin, &bg_bin, bin_groups);
    let t_tri = run(&pipe_tri, &bg_tri, tri_groups);

    // 正确性：读回计数
    let check = |bg: &wgpu::BindGroup, groups: u32| -> u32 {
        let mut enc = device.create_command_encoder(&Default::default());
        enc.clear_buffer(&counter, 0, None);
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pipe_tri);
            cp.set_bind_group(0, bg, &[]);
            cp.dispatch_workgroups(groups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&counter, 0, &readback, 0, 4);
        queue.submit([enc.finish()]);
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let data = slice.get_mapped_range().expect("map");
        let v = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        readback.unmap();
        v
    };

    let tri_check = check(&bg_tri, tri_groups);
    let bin_check_bg = &bg_bin;
    // 用 ternary pipeline 测会对不上，这里改用 binary 单独跑一次
    let mut enc = device.create_command_encoder(&Default::default());
    enc.clear_buffer(&counter, 0, None);
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&pipe_bin);
        cp.set_bind_group(0, bin_check_bg, &[]);
        cp.dispatch_workgroups(bin_groups, 1, 1);
    }
    enc.copy_buffer_to_buffer(&counter, 0, &readback, 0, 4);
    queue.submit([enc.finish()]);
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let data = slice.get_mapped_range().expect("map");
    let bin_check = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    drop(data);
    readback.unmap();

    println!("{:<14} | {:>10} | {:>14} | {:>12}", "实现", "ms/次", "百万对象/秒", "带宽/次");
    println!("{:-<14}-+-{:-<10}-+-{:-<14}-+-{:-<12}", "", "", "", "");
    let bin_bytes = N as f64 * 12.0 / 1e6; // 3 × f32
    let tri_bytes = N as f64 * 12.0 / 32.0 / 1e6; // 3 平面 × N/32 个 u32
    println!(
        "{:<14} | {:>10.4} | {:>14.1} | {:>11.2} MB",
        "BINARY",
        t_bin * 1000.0,
        N as f64 / t_bin / 1e6,
        bin_bytes
    );
    println!(
        "{:<14} | {:>10.4} | {:>14.1} | {:>11.2} MB",
        "TERNARY",
        t_tri * 1000.0,
        N as f64 / t_tri / 1e6,
        tri_bytes
    );
    println!("\n倍数：{:.2}x（binary / ternary）", t_bin / t_tri);
    println!("正确性：binary 命中 {bin_check} · ternary 命中 {tri_check} · 期望 {expect}");
    assert_eq!(bin_check, expect, "binary 结果错误");
    assert_eq!(tri_check, expect, "ternary 结果错误");
    println!("✓ 两版结果一致");
}
