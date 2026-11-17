//! 三进制 vs 二进制 · GPU 位并行筛选（持续压 GPU 的窗口程序）
//!
//! - 默认 BINARY：每对象 3 个 f32（120 MB），逐对象比较
//! - `--ternary`：位平面（3.75 MB），一个线程判 32 个
//!
//! 每帧真实 dispatch 一次 compute 并等 GPU 完成，窗口内持续显示
//! FPS / 每帧耗时 / 对象数 / 带宽，关窗口才停。

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const N: u32 = 10_000_000;
const NW: u32 = N / 32;
const WG: u32 = 256;
const OVERLAY_W: u32 = 1024;
const OVERLAY_H: u32 = 64;

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
    let bits = pa[w] & pb[w] & ~nc[w];
    let n = countOneBits(bits);
    if (n > 0u) { atomicAdd(&counter, n); }
}
"#;

const OVERLAY_WGSL: &str = r#"
struct G { size: vec2<f32>, _a: f32, _b: f32 };
@group(0) @binding(0) var<uniform> g: G;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs_overlay(@builtin(vertex_index) vi: u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    let px = vec2<f32>(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5) * vec2<f32>(1024.0, 64.0);
    let ndc = vec2<f32>(px.x / g.size.x * 2.0 - 1.0, 1.0 - px.y / g.size.y * 2.0);
    var o: VsOut;
    o.clip = vec4<f32>(ndc, 0.0, 1.0);
    o.uv = c * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return o;
}

@fragment
fn fs_overlay(v: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, v.uv);
}
"#;

fn glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        ' ' => [0; 7],
        _ => [0; 7],
    }
}

fn draw_text(buf: &mut [u8], x0: usize, y0: usize, text: &str, scale: usize, color: [u8; 4]) {
    let mut x = x0;
    for ch in text.chars() {
        let g = glyph(ch);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..5 {
                if bits & (1 << (4 - col)) != 0 {
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let px = x + col * scale + dx;
                            let py = y0 + row * scale + dy;
                            if px < OVERLAY_W as usize && py < OVERLAY_H as usize {
                                let i = (py * OVERLAY_W as usize + px) * 4;
                                buf[i..i + 4].copy_from_slice(&color);
                            }
                        }
                    }
                }
            }
        }
        x += 6 * scale;
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn tri(&mut self) -> i32 {
        (self.next() % 3) as i32 - 1
    }
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipe: wgpu::ComputePipeline,
    bg: wgpu::BindGroup,
    groups: u32,
    overlay_pipeline: wgpu::RenderPipeline,
    overlay_bind: wgpu::BindGroup,
    overlay_globals: wgpu::Buffer,
    overlay_tex: wgpu::Texture,
    overlay_pixels: Vec<u8>,
    ternary: bool,
    fps: f32,
    frame_ms: f32,
    frames: u32,
    timer: Instant,
    start: Instant,
}

impl State {
    fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .expect("adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("device");
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("config");
        config.present_mode = wgpu::PresentMode::AutoNoVsync;
        surface.configure(&device, &config);

        let ternary = std::env::args().any(|a| a == "--ternary");

        // 数据
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

        fn bytes_f32(d: &[f32]) -> &[u8] {
            unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) }
        }
        fn bytes_u32(d: &[u32]) -> &[u8] {
            unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) }
        }
        use wgpu::util::DeviceExt;
        let mk = |data: &[u8], label: &str| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        let (buf1, buf2, buf3) = if ternary {
            (
                mk(bytes_u32(&pa), "pa"),
                mk(bytes_u32(&pb), "pb"),
                mk(bytes_u32(&nc), "nc"),
            )
        } else {
            (
                mk(bytes_f32(&a), "a"),
                mk(bytes_f32(&b), "b"),
                mk(bytes_f32(&c), "c"),
            )
        };

        let counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("k"),
            source: wgpu::ShaderSource::Wgsl(if ternary {
                WGSL_TERNARY.into()
            } else {
                WGSL.into()
            }),
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
                    resource: buf1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: counter.as_entire_binding(),
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("k"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some(if ternary {
                "ternary_kernel"
            } else {
                "binary_kernel"
            }),
            compilation_options: Default::default(),
            cache: None,
        });
        let groups = if ternary {
            NW.div_ceil(WG)
        } else {
            N.div_ceil(WG)
        };

        // overlay
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(OVERLAY_WGSL.into()),
        });
        let overlay_globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("og"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let overlay_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("overlay"),
            size: wgpu::Extent3d {
                width: OVERLAY_W,
                height: OVERLAY_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let overlay_view = overlay_tex.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let overlay_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let overlay_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &overlay_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: overlay_globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&overlay_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let overlay_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&overlay_bgl)],
            immediate_size: 0,
        });
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay"),
            layout: Some(&overlay_layout),
            vertex: wgpu::VertexState {
                module: &overlay_shader,
                entry_point: Some("vs_overlay"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader,
                entry_point: Some("fs_overlay"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let mut state = Self {
            window,
            surface,
            device,
            queue,
            config,
            pipe,
            bg,
            groups,
            overlay_pipeline,
            overlay_bind,
            overlay_globals,
            overlay_tex,
            overlay_pixels: vec![0u8; (OVERLAY_W * OVERLAY_H * 4) as usize],
            ternary,
            fps: 0.0,
            frame_ms: 0.0,
            frames: 0,
            timer: Instant::now(),
            start: Instant::now(),
        };
        state.update_overlay();
        state
    }

    fn update_overlay(&mut self) {
        self.overlay_pixels.fill(0);
        let ok = [120u8, 255, 170, 255];
        let dim = [200u8, 220u8, 255, 255];
        let line1 = format!(
            "{} | FPS {:>8.1} | {:>7.3} MS/FRAME",
            if self.ternary { "TERNARY" } else { "BINARY " },
            self.fps,
            self.frame_ms
        );
        let line2 = format!(
            "OBJECTS {} | BW {} | HIT QUERY: A=TRUE & B=TRUE & C!=FALSE",
            N / 1_000_000,
            if self.ternary { "3.75 MB" } else { "120 MB" }
        );
        draw_text(&mut self.overlay_pixels, 16, 8, &line1, 2, ok);
        draw_text(&mut self.overlay_pixels, 16, 42, &line2, 1, dim);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.overlay_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.overlay_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(OVERLAY_W * 4),
                rows_per_image: Some(OVERLAY_H),
            },
            wgpu::Extent3d {
                width: OVERLAY_W,
                height: OVERLAY_H,
                depth_or_array_layers: 1,
            },
        );
        let mut og = [0u8; 16];
        og[0..4].copy_from_slice(&(self.config.width as f32).to_le_bytes());
        og[4..8].copy_from_slice(&(self.config.height as f32).to_le_bytes());
        self.queue.write_buffer(&self.overlay_globals, 0, &og);
    }

    fn render(&mut self) {
        let frame_start = Instant::now();
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => return,
        };
        let view = output.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.pipe);
            cp.set_bind_group(0, &self.bg, &[]);
            cp.dispatch_workgroups(self.groups, 1, 1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.03,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.overlay_pipeline);
            pass.set_bind_group(0, &self.overlay_bind, &[]);
            pass.draw(0..6, 0..1);
        }
        self.queue.submit([enc.finish()]);
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        self.queue.present(output);

        self.frames += 1;
        let elapsed = self.timer.elapsed().as_secs_f32();
        if elapsed >= 0.5 {
            self.fps = self.frames as f32 / elapsed;
            self.frame_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
            self.frames = 0;
            self.timer = Instant::now();
            self.window.set_title(&format!(
                "{} | FPS {:.1} | {:.3} ms",
                if self.ternary { "TERNARY" } else { "BINARY" },
                self.fps,
                self.frame_ms
            ));
            self.update_overlay();
        }
    }
}

struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("trit vs binary · GPU")
                        .with_inner_size(winit::dpi::LogicalSize::new(1000.0, 300.0)),
                )
                .expect("window"),
        );
        self.state = Some(State::new(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    state.config.width = size.width;
                    state.config.height = size.height;
                    state.surface.configure(&state.device, &state.config);
                }
            }
            WindowEvent::RedrawRequested => {
                state.render();
                state.window.request_redraw();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { state: None };
    event_loop.run_app(&mut app).expect("run");
}
