//! 毒蘑菇分形（volumeshader_bm 内核）原生 wgpu 基准。
//!
//! 与 mushroom.html 使用同一份 WGSL、同一相机与同一分辨率，
//! 离屏渲染、不锁 vsync，用来对照浏览器版的真实 GPU 开销。

use std::time::Instant;

const W: u32 = 879;
const H: u32 = 681;

const WGSL: &str = r#"
struct CamU {
  pos: vec4<f32>,
  right: vec4<f32>,
  upv: vec4<f32>,
  fwd: vec4<f32>,
  m0: vec4<f32>,
  m1: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: CamU;

fn kernel(ver: vec3<f32>, iters: u32, power: f32) -> f32 {
  var a = ver;
  var b = 0.0;
  for (var i = 0u; i < 120u; i++) {
    if (i >= iters) { break; }
    b = length(a);
    let c = atan2(a.y, a.x) * power;
    let e = 1.0 / max(b, 1e-9);
    let d = acos(clamp(a.z * e, -1.0, 1.0)) * power;
    b = pow(b, power);
    a = vec3<f32>(b * sin(d) * cos(c), b * sin(d) * sin(c), b * cos(d)) + ver;
    if (b > 8.0) { break; }
  }
  return 5.0 - dot(a, a);
}

struct VOut { @builtin(position) pos: vec4<f32> };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
  var o: VOut;
  o.pos = vec4<f32>(p[i], 0.0, 1.0);
  return o;
}

@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  var uv = vec2<f32>(fc.x / cam.m1.z, fc.y / cam.m1.w) * 2.0 - vec2<f32>(1.0);
  uv.y = -uv.y;
  let dir = normalize(cam.fwd.xyz
    + uv.x * cam.m0.x * cam.m0.y * cam.right.xyz
    + uv.y * cam.m0.x * cam.upv.xyz);
  let o = cam.pos.xyz;
  let iters = u32(cam.m0.z);
  let power = cam.m0.w;
  let dt = cam.m1.x;
  let tmax = cam.m1.y;

  var t = 0.05;
  var f0 = kernel(o + dir * t, iters, power);
  var hit = false;
  var tHit = t;
  if (f0 > 0.0) {
    hit = true;
  } else {
    for (var k = 0; k < 12000; k++) {
      let t1 = t + dt;
      if (t1 > tmax) { break; }
      let f1 = kernel(o + dir * t1, iters, power);
      if (f1 > 0.0) {
        var a = t;
        var b = t1;
        for (var j = 0; j < 6; j++) {
          let m = (a + b) * 0.5;
          let fm = kernel(o + dir * m, iters, power);
          if (fm > 0.0) { b = m; } else { a = m; }
        }
        tHit = (a + b) * 0.5;
        hit = true;
        break;
      }
      t = t1;
      f0 = f1;
    }
  }

  let bg = mix(vec3<f32>(0.006, 0.007, 0.012), vec3<f32>(0.02, 0.016, 0.03),
               clamp(uv.y * 0.5 + 0.5, 0.0, 1.0));
  var col = bg;
  if (hit) {
    let p = o + dir * tHit;
    let nIters = min(iters, 6u);
    let eps = 0.02;
    let gx = kernel(p + vec3<f32>(eps, 0.0, 0.0), nIters, power) - kernel(p - vec3<f32>(eps, 0.0, 0.0), nIters, power);
    let gy = kernel(p + vec3<f32>(0.0, eps, 0.0), nIters, power) - kernel(p - vec3<f32>(0.0, eps, 0.0), nIters, power);
    let gz = kernel(p + vec3<f32>(0.0, 0.0, eps), nIters, power) - kernel(p - vec3<f32>(0.0, 0.0, eps), nIters, power);
    var n = vec3<f32>(gx, gy, gz);
    let nl = length(n);
    if (nl < 1e-10) {
      n = -dir;
    } else {
      n = n / nl;
      if (dot(n, dir) > 0.0) { n = -n; }
    }

    let L = normalize(vec3<f32>(-0.4, 0.85, -0.45));
    let diff = max(dot(n, L), 0.0);
    let fill = max(dot(n, normalize(cam.pos.xyz - p)), 0.0);
    let ang = atan2(p.x, p.z);
    let phase = 0.5 + 0.55 * p.y + 0.32 * ang + 0.10 * sin(p.x * 2.1 + p.z * 1.7);
    var base = 0.55 + 0.45 * cos(6.28318 * (vec3<f32>(0.0, 0.33, 0.67) + phase));
    base = pow(clamp(base, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(0.7));
    col = base * (0.34 + 0.95 * diff + 0.1 * fill);
  }
  return vec4<f32>(pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(0.85)), 1.0);
}
"#;

struct Preset {
    name: &'static str,
    iters: f32,
    power: f32,
    dt: f32,
    max_t: f32,
    frames: u32,
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
    let caps = adapter.limits();
    println!("GPU: {} ({:?})", info.name, info.backend);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: caps,
        ..Default::default()
    }))
    .expect("device");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mushroom"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mushroom"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

    let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cam"),
        size: 96,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() }],
    });

    // 相机与网页一致：dist 1.9、pitch 0.06、yaw 0、fov 65°
    let yaw = 0.0_f32;
    let pitch = 0.06_f32;
    let dist = 1.9_f32;
    let cp = [
        dist * pitch.cos() * yaw.sin(),
        0.06f32.sin() * dist,
        dist * pitch.cos() * yaw.cos(),
    ];
    let tgt = [0.0f32, 0.0, 0.0];
    let norm = |v: [f32; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    };
    let fwd = norm([tgt[0] - cp[0], tgt[1] - cp[1], tgt[2] - cp[2]]);
    let right = norm(cross(fwd, [0.0, 1.0, 0.0]));
    let up = cross(right, fwd);
    let tan_half = (32.5_f32 * std::f32::consts::PI / 180.0).tan();

    let presets = [
        Preset { name: "低", iters: 3.0, power: 8.0, dt: 0.003, max_t: 4.5, frames: 120 },
        Preset { name: "中", iters: 4.0, power: 8.0, dt: 0.002, max_t: 4.5, frames: 120 },
        Preset { name: "高", iters: 100.0, power: 10.0, dt: 0.0014, max_t: 4.5, frames: 60 },
    ];

    println!("\n分辨率 {}×{}（与网页内部渲染一致），离屏、不锁 vsync\n", W, H);
    println!("{:>4} | {:>6} | {:>5} | {:>10} | {:>8}", "档位", "迭代", "幂", "ms/帧", "FPS");
    println!("{:-<4}-+-{:-<6}-+-{:-<5}-+-{:-<10}-+-{:-<8}", "", "", "", "", "");

    with_pipeline(&device, &queue, &pipe, &bg, &view, &cam_buf, &cp, &right, &up, &fwd, tan_half, &presets);
    println!("\n说明：内核与网页版完全同一份 WGSL；网页版中档 60 FPS 是 vsync 上限，");
    println!("高档 18 FPS 是 GPU 瓶颈，可与上表高档直接对比。");
}

#[allow(clippy::too_many_arguments)]
fn with_pipeline(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipe: &wgpu::RenderPipeline,
    bg: &wgpu::BindGroup,
    view: &wgpu::TextureView,
    cam_buf: &wgpu::Buffer,
    cp: &[f32; 3],
    right: &[f32; 3],
    up: &[f32; 3],
    fwd: &[f32; 3],
    tan_half: f32,
    presets: &[Preset],
) {
    for p in presets {
        let aspect = W as f32 / H as f32;
        let mut u = [0.0f32; 24];
        u[0] = cp[0]; u[1] = cp[1]; u[2] = cp[2]; u[3] = 1.0;
        u[4] = right[0]; u[5] = right[1]; u[6] = right[2]; u[7] = 0.0;
        u[8] = up[0]; u[9] = up[1]; u[10] = up[2]; u[11] = 0.0;
        u[12] = fwd[0]; u[13] = fwd[1]; u[14] = fwd[2]; u[15] = 0.0;
        u[16] = tan_half; u[17] = aspect; u[18] = p.iters; u[19] = p.power;
        u[20] = p.dt; u[21] = p.max_t; u[22] = W as f32; u[23] = H as f32;
        let bytes = unsafe { std::slice::from_raw_parts(u.as_ptr() as *const u8, 96) };
        queue.write_buffer(cam_buf, 0, bytes);

        let draw = || {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Discard,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                    occlusion_query_set: None,
                });
                rp.set_pipeline(pipe);
                rp.set_bind_group(0, bg, &[]);
                rp.draw(0..3, 0..1);
            }
            enc.finish()
        };

        // 预热
        for _ in 0..3 {
            queue.submit([draw()]);
        }
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        let start = Instant::now();
        for _ in 0..p.frames {
            queue.submit([draw()]);
        }
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let secs = start.elapsed().as_secs_f64();
        let ms = secs * 1000.0 / p.frames as f64;
        let fps = p.frames as f64 / secs;

        println!(
            "{:>4} | {:>6.0} | {:>5.0} | {:>10.2} | {:>8.1}",
            p.name, p.iters, p.power, ms, fps
        );
    }
}
