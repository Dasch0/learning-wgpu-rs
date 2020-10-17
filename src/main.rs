use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{EventLoop, ControlFlow},
    window::{Window, WindowBuilder},
};
use std::time::{Instant, Duration};
use bytemuck::{Pod, Zeroable};
use imgui::*;

struct Handle {
    _instance: wgpu::Instance,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[allow(unused)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 4],
    _tex_coord:[f32; 2],
}

fn vertex(pos: [i8; 3], tc:[i8; 2]) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        _tex_coord: [tc[0] as f32, tc[1] as f32],

    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConstant {
    _zoom: f32,
    _offset: [f32; 2],
    _pad: f32,
}

fn push_constant(zoom: f32, offset: [f32; 2]) -> PushConstant {
    PushConstant {
        _zoom: zoom,
        _offset: offset,
        _pad: 0.0
    }
}


struct BufferDimensions {
    width: usize,
    height: usize,
    unpadded_bytes_per_row: usize,
    padded_bytes_per_row: usize,
}

impl BufferDimensions {
    fn new(width: usize, height: usize) -> Self {
        let bytes_per_pixel = std::mem::size_of::<u32>();
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;
        Self {
            width,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }
}

fn create_test_mesh() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1, -1, 1], [0, 0]),
        vertex([1, -1, 1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([-1, 1, 1], [0, 1]),
        // bottom (0, 0, -1)
        vertex([-1, 1, -1], [1, 0]),
        vertex([1, 1, -1], [0, 0]),
        vertex([1, -1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // right (1, 0, 0)
        vertex([1, -1, -1], [0, 0]),
        vertex([1, 1, -1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([1, -1, 1], [0, 1]),
        // left (-1, 0, 0)
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, 1, 1], [0, 0]),
        vertex([-1, 1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // front (0, 1, 0)
        vertex([1, 1, -1], [1, 0]),
        vertex([-1, 1, -1], [0, 0]),
        vertex([-1, 1, 1], [0, 1]),
        vertex([1, 1, 1], [1, 1]),
        // back (0, -1, 0)
        vertex([1, -1, 1], [0, 0]),
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, -1, -1], [1, 1]),
        vertex([1, -1, -1], [0, 1]),
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

fn create_mandelbrot_texture(size: usize) -> Vec<u8>{
    use std::iter;
    (0..size * size)
        .flat_map(|id| {
            let cx = 3.0 * (id % size) as f32 / (size - 1) as f32 - 2.0;
            let cy = 2.0 * (id / size) as f32 / (size - 1) as f32 - 1.0;
            let (mut x, mut y, mut count) = (cx, cy, 0);
            while count < 0xFF && x * x + y * y < 4.0 {
                let old_x = x;
                x = x * x - y * y + cx;
                y = 2.0 * old_x * y + cy;
                count += 1;
            }
            iter::once(0xFF - (count * 5) as u8)
                .chain(iter::once(0xFF - (count * 15) as u8))
                .chain(iter::once(0xFF - (count * 50) as u8))
                .chain(iter::once(1))
        })
        .collect()
}

fn create_test_camera(aspect_ratio: f32, pos: f32) -> cgmath::Matrix4<f32> {
    let proj = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 10.0);
    let view = cgmath::Matrix4::look_at(
        cgmath::Point3::new(pos, -5.0, 3.0),
        cgmath::Point3::new(0f32, 0.0, 0.0),
        cgmath::Vector3::unit_z(),
    );
    let correction = OPENGL_TO_WGPU_MATRIX;
    correction * proj * view
}

async fn setup (window: &Window) -> Handle {
    log::info!("Initializing instance...");
    let _instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    
    log::info!("Obtaining window surface...");
    let (size, surface) = unsafe {
        let size = window.inner_size();
        let surface = _instance.create_surface(window);
        (size, surface)
    };

    log::info!("Initializing adapter...");
    let _adapter = _instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: Some(&surface),
        })
        .await
       .unwrap();

    let optional_features = wgpu::Features::empty();
    
    // TODO: support for setups without unsized_binding_array
    let required_features = 
        wgpu::Features::default() 
        | wgpu::Features::PUSH_CONSTANTS
        //| wgpu::Features::UNSIZED_BINDING_ARRAY
        //| wgpu::Features::SAMPLED_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
        | wgpu::Features::SAMPLED_TEXTURE_BINDING_ARRAY;

    let adapter_features = _adapter.features();

    let required_limits = wgpu::Limits {
        max_push_constant_size: 16,
        ..wgpu::Limits::default()
    };

    let trace_dir = std::env::var("WGPU_TRACE");
    
    log::info!("Initializing device & queue...");
    let(device, queue) = _adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: (adapter_features & optional_features) | required_features,
                limits: required_limits,
                shader_validation: true,
            },
            trace_dir.ok().as_ref().map(std::path::Path::new),
        )
        .await
        .unwrap();
    log::info!("Setup complete!");

    Handle {
        _instance,
        size,
        surface,
        _adapter,
        device,
        queue,
    }
}

fn update_camera(
    swapchain_desc: &wgpu::SwapChainDescriptor,
    queue: &wgpu::Queue,
    camera_buf: &wgpu::Buffer,
    camera_state: f32
) {
    let camera = create_test_camera(
        swapchain_desc.width as f32 / swapchain_desc.height as f32,
        camera_state,
    );
    let camera_ref: &[f32;16] = camera.as_ref();
    queue.write_buffer(&camera_buf, 0, bytemuck::cast_slice(camera_ref));
}

fn create_msaa_target(swapchain_desc: &wgpu::SwapChainDescriptor,
                      device: &wgpu::Device,
                      msaa_samples: u32,
) -> wgpu::TextureView {
    let multisampled_texture_extent = wgpu::Extent3d {
        width: swapchain_desc.width,
        height: swapchain_desc.height,
        depth: 1,
    };
    let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
        size: multisampled_texture_extent,
        mip_level_count: 1,
        sample_count: msaa_samples,
        dimension: wgpu::TextureDimension::D2,
        format: swapchain_desc.format,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        label: None,
    };
    
    device
        .create_texture(multisampled_frame_descriptor)
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn resize(
    swapchain_desc: &wgpu::SwapChainDescriptor,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    camera_buf: &wgpu::Buffer,
    camera_state: f32,
    msaa_view: &mut wgpu::TextureView,
    msaa_samples: u32,
){
    update_camera(swapchain_desc, queue, camera_buf, camera_state);
    *msaa_view = create_msaa_target(swapchain_desc, device, msaa_samples);
}

fn render<T:Pod>(
    msaa_target: &wgpu::TextureView,
    resolve_target: &wgpu::TextureView,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    vertex_buf: &wgpu::Buffer,
    index_buf: &wgpu::Buffer,
    index_cnt: usize,
    push_constants: T,
    _spawner: &impl futures::task::LocalSpawn,
) {
    let mut encoder = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });

    let mut renderpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
            attachment: &msaa_target,
            resolve_target: Some(resolve_target),
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                }),
                store: true,
            },
        }],
        depth_stencil_attachment: None,
    });
    renderpass.push_debug_group("Prepare data for draw.");
    renderpass.set_pipeline(&pipeline);
    renderpass.set_bind_group(0, &bind_group, &[]);
    renderpass.set_index_buffer(index_buf.slice(..));
    renderpass.set_vertex_buffer(0, vertex_buf.slice(..));
    renderpass.set_push_constants(wgpu::ShaderStage::FRAGMENT, 0, bytemuck::cast_slice(&[push_constants]));
    renderpass.pop_debug_group();
    renderpass.insert_debug_marker("Drawing frame");
    renderpass.draw_indexed(0..index_cnt as u32, 0, 0..1);
    drop(renderpass);
    queue.submit(Some(encoder.finish()));
}

fn main() {
    // Init Window
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .unwrap();

    // Init GPU Handle
    let h = futures::executor::block_on(setup(&window));

    // define sample count for MSAA
    let msaa_samples = 8;

    log::info!("Initializing swapchain...");
    let mut swapchain_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: h.size.width,
        height: h.size.height,
        present_mode: wgpu::PresentMode::Immediate,
    };
    let mut swapchain = h
        .device
        .create_swap_chain(&h.surface, &swapchain_desc);

    log::info!("Initializing MSAA target...");
    let mut msaa_view = 
        create_msaa_target(&swapchain_desc, &h.device, msaa_samples);

    log::info!("Initializing screen capture target");
    // It is a webgpu requirement that BufferCopyView.layout.bytes_per_row 
    // % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT == 0
    // So we calculate padded_bytes_per_row by rounding unpadded_bytes_per_row
    // up to the next multiple of wgpu::COPY_BYTES_PER_ROW_ALIGNMENT.
    // https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
    let capture_buffer_dimensions = BufferDimensions::new(
        swapchain_desc.width as usize,
        swapchain_desc.height as usize
    );
    // The output buffer lets us retrieve the data as an array
    let capture_buffer = h.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (capture_buffer_dimensions.padded_bytes_per_row 
               * capture_buffer_dimensions.height) as u64,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    });
    let capture_texture_extent = wgpu::Extent3d {
        width: capture_buffer_dimensions.width as u32,
        height: capture_buffer_dimensions.height as u32,
        depth: 1,
    };
    
    let capture_format= wgpu::TextureFormat::Rgba8UnormSrgb;

    // Create 10 multisampled textures for capture
    let capture_count = 10;
    let capture_msaa_texture :Vec<wgpu::Texture> = (0..capture_count).map(|_| {
        h.device.create_texture(&wgpu::TextureDescriptor {
            size: capture_texture_extent,
            mip_level_count: 1,
            sample_count: msaa_samples,
            dimension: wgpu::TextureDimension::D2,
            format: capture_format,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::COPY_SRC,
            label: None,
        })
    }).collect();

    // Create 10 resolve targets for capture
    let mut capture_resolve_texture : Vec<wgpu::Texture> = (0..capture_count).map(|_| {
        h.device.create_texture(&wgpu::TextureDescriptor {
            size: capture_texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: capture_format,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT 
                | wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::SAMPLED,
            label: None,
        })
    }).collect();

    // Create views for capture msaa and resolve views
    let capture_msaa_view : Vec<wgpu::TextureView> = (0..capture_count).map(|i| {
        capture_msaa_texture[i].create_view(&wgpu::TextureViewDescriptor::default())
    }).collect();

    let capture_resolve_view = capture_resolve_texture[2]
        .create_view(&wgpu::TextureViewDescriptor::default());
    
    log::info!("Initializing Rendering Pipelines...");
    let scene_bind_group_layout = h.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        comparison: false,
                    },
                    count: None,
                },
            ],
        })
    ;

    let capture_bind_group_layout = h.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: wgpu::BufferSize::new(
                            (std::mem::size_of::<cgmath::Matrix4::<f32>>() 
                             * capture_count) as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None, 
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        comparison: false,
                    },
                    count: None,
                },
            ],
        })
    ;

    let scene_pipeline_layout = h.device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Scene pipeline layout"),
            bind_group_layouts: &[&scene_bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStage::FRAGMENT,
                range: (0..16),
            }],
        })
    ;
    
    let capture_pipeline_layout = h.device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Capture pipeline layout"),
            bind_group_layouts: &[&capture_bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStage::FRAGMENT,
                range: (0..16),
            }],
        })
    ;

    let vertex_size = std::mem::size_of::<Vertex>();
    let vertex_state = wgpu::VertexStateDescriptor {
        index_format: wgpu::IndexFormat::Uint16,
        vertex_buffers: &[wgpu::VertexBufferDescriptor {
            stride: vertex_size as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    format: wgpu::VertexFormat::Float4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttributeDescriptor {
                    format: wgpu::VertexFormat::Float2,
                    offset: 4 * 4, // TODO: cleanup
                    shader_location: 1,
                },
            ],
        }],
    };

    let scene_vert_shader = h.device
        .create_shader_module(wgpu::include_spirv!("../shaders/scene.vert.spv"));
    let scene_frag_shader = h.device
        .create_shader_module(wgpu::include_spirv!("../shaders/scene.frag.spv"));
    let capture_vert_shader = h.device
        .create_shader_module(wgpu::include_spirv!("../shaders/capture.vert.spv"));
    let capture_frag_shader = h.device
        .create_shader_module(wgpu::include_spirv!("../shaders/capture.frag.spv"));

    let scene_pipeline = h.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Scene Pipeline"),
            layout: Some(&scene_pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &scene_vert_shader,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &scene_frag_shader,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode:wgpu::CullMode::Back,
                ..Default::default()
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: swapchain_desc.format,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            vertex_state: vertex_state.clone(),
            sample_count: msaa_samples,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        })
    ;

    let capture_pipeline = h.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Capture pipeline"),
            layout: Some(&capture_pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &capture_vert_shader,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &capture_frag_shader,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode:wgpu::CullMode::Back,
                ..Default::default()
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: capture_format,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            vertex_state: vertex_state.clone(),
            sample_count: msaa_samples,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        }
    );

    log::info!("Initializing buffers & textures...");
    let (vertex_data, index_data) = create_test_mesh();

    let vertex_buf = h.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertex_data),
        usage: wgpu::BufferUsage::VERTEX,
    });

    let index_buf = h.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(&index_data),
        usage: wgpu::BufferUsage::INDEX,
    });

    let size = 1048u32;
    let texels = create_mandelbrot_texture(size as usize);
    let texture_extent = wgpu::Extent3d {
        width: size,
        height: size,
        depth: 1,
    };

    let texture = h.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: texture_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    h.queue.write_texture(
        wgpu::TextureCopyView {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        &texels,
        wgpu::TextureDataLayout {
            offset: 0,
            bytes_per_row: 4 * size,
            rows_per_image: 0,
        },
        texture_extent,
    );

    let sampler = h.device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    log::info!("Creating Cameras...");
    let scene_camera = create_test_camera(
        swapchain_desc.width as f32 / swapchain_desc.height as f32,
        0.0,
    );
    let scene_camera_ref: &[f32; 16] = scene_camera.as_ref();
    let scene_camera_buf: wgpu::Buffer = h.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(scene_camera_ref),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        }
    );

    // store as vector of primatives so that bytemuck works
    let capture_camera : Vec<[[f32; 4]; 4]> = (0..capture_count).map(|x| {
        create_test_camera(
            swapchain_desc.width as f32 / swapchain_desc.height as f32,
            x as f32).into()
    }).collect();    
    let capture_camera_buf = h.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&capture_camera),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

    log::info!("Creating bind groups...");
    let scene_bind_group = h.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &scene_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(scene_camera_buf.slice(..)), 
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
        label: Some("Scene Bind Group"),
    });

    let capture_bind_group = h.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &capture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(capture_camera_buf.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
        label: Some("Capture Bind Group"),
    });

    log::info!("Initializing imgui...");
    let hidpi_factor = window.scale_factor();
    let mut imgui = imgui::Context::create();
    let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
    platform.attach_window(
        imgui.io_mut(),
        &window,
        imgui_winit_support::HiDpiMode::Default,
    );
    imgui.set_ini_filename(None);

    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
    imgui
        .fonts()
        .add_font(&[imgui::FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                oversample_h: 1,
                pixel_snap_h: true,
                size_pixels: font_size,
                ..Default::default()
            }),
        }]);

    let imgui_render_config = imgui_wgpu::RendererConfig::new()
        .set_texture_format(swapchain_desc.format);

    let mut imgui_renderer = imgui_wgpu::Renderer::new(
        &mut imgui,
        &h.device,
        &h.queue,
        imgui_render_config,
    );

    // Create imgui texture for drawing rendered textures in UI

    let imgui_texture_bind_group_layout = h.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None, 
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::SampledTexture {
                    multisampled: false,
                    component_type: wgpu::TextureComponentType::Float,
                    dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::Sampler { comparison: false },
                count: None,
            },
        ],
    });
    let imgui_texture_bind_group = h.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &imgui_texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&capture_resolve_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    let imgui_texture_view = capture_resolve_texture[2]
        .create_view(&wgpu::TextureViewDescriptor::default());
    let imgui_texture = imgui_wgpu::Texture::from_raw_parts(
        capture_resolve_texture.remove(2),
        imgui_texture_view,
        imgui_texture_bind_group,
        capture_texture_extent
    );

    let imgui_texture_id = imgui_renderer.textures.insert(imgui_texture);

    // UI States
    let mut last_frame = std::time::Instant::now();
    let mut show_demo = false;
    let mut last_cursor = None;
    let mut camera_state: f32 = 0.0;
    let mut zoom: f32 = 0.1;
    let mut offset: [f32; 2] = [0.0, 0.0];

    log::info!("Starting event loop!");
    let (_pool, spawner) = {
        let local_pool = futures::executor::LocalPool::new();
        let spawner = local_pool.spawner();
        (local_pool, spawner)
    };

    let mut last_update_inst = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        let _ = (
            &h,
            &swapchain_desc,
            &scene_camera_buf,
        );
        
        // Set control flow to wait min(next event, 10ms) 
        *control_flow =
            ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(1));

        // Process events
        match event {
            Event::MainEventsCleared => {
                if last_update_inst.elapsed() > Duration::from_millis(2) {
                    window.request_redraw();
                    last_update_inst = Instant::now();
                }
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                let size = window.inner_size();
                log::info!("Resizing to {:?}", size);
                swapchain_desc.width = if size.width == 0 {1} else { size.width };
                swapchain_desc.height = if size.height == 0 {1} else { size.height };
                resize(
                    &swapchain_desc,
                    &h.device,
                    &h.queue,
                    &scene_camera_buf,
                    camera_state,
                    &mut msaa_view,
                    msaa_samples,
                );
                swapchain = h.device.create_swap_chain(&h.surface, &swapchain_desc);
            }
            Event::WindowEvent {ref event, .. } => match event {
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                }
                | WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            Event::RedrawRequested(_) => {
                let delta_s = last_frame.elapsed();
                let now = Instant::now();
                imgui.io_mut().update_delta_time(now - last_frame);
                last_frame = now;
                let frame = match swapchain.get_current_frame() { Ok(frame) => frame,
                    Err(e) => {
                        log::warn!("dropped frame: {:?}", e);
                        swapchain = h
                            .device
                            .create_swap_chain(&h.surface, &swapchain_desc);
                        swapchain
                            .get_current_frame()
                            .expect("Failed to acquire next swap chain texture!")
                    }
                };

                update_camera(&swapchain_desc, &h.queue, &scene_camera_buf, camera_state);

                // Render Scene
                render(
                    &msaa_view,
                    &frame.output.view,
                    &h.device,
                    &h.queue,
                    &scene_pipeline,
                    &scene_bind_group,
                    &vertex_buf,
                    &index_buf,
                    index_data.len(),
                    push_constant(zoom, offset),
                    &spawner
                );

                render(
                    &capture_msaa_view[2],
                    &capture_resolve_view,
                    &h.device,
                    &h.queue,
                    &capture_pipeline,
                    &capture_bind_group,
                    &vertex_buf,
                    &index_buf,
                    index_data.len(),
                    push_constant(zoom, offset),
                    &spawner
                );

                // Render UI
                platform
                    .prepare_frame(imgui.io_mut(), &window)
                    .expect("Failed to prepare frame");
                let ui = imgui.frame();
                {
                    let imgui_window = imgui::Window::new(im_str!("Hello World"));
                    imgui_window
                        .size([300.0, 300.0], imgui::Condition::FirstUseEver)
                        .build(&ui, || {
                            ui.text(im_str!("Frametime: {:?}", delta_s));
                            ui.separator();
                            let mouse_pos = ui.io().mouse_pos;
                            ui.text(im_str!(
                                "Mouse Position: ({:.1},{:.1})",
                                mouse_pos[0],
                                mouse_pos[1],
                            ));

                            ui.separator();
                            if ui.button(im_str!("Toggle Demo"), [100., 20.]) {
                                show_demo = !show_demo
                            }
                            ui.separator();

                            imgui::Slider::new(im_str!("Camera Rotation"))
                                .range(0.0..=6.0)
                                .build(&ui, &mut camera_state);
                            ui.separator();
                            imgui::Slider::new(im_str!("Texture Zoom"))
                                .range(0.001..=1.0)
                                .build(&ui, &mut zoom);
                            ui.separator();
                            imgui::Slider::new(im_str!("Texture X offset"))
                                .range(0.00..=1.0)
                                .build(&ui, &mut offset[0]);
                            ui.separator();
                            imgui::Slider::new(im_str!("Texture Y offset"))
                                .range(0.00..=1.0)
                                .build(&ui, &mut offset[1]);
                            ui.separator();
                            imgui::Image::new(imgui_texture_id, [500.0, 500.0]).build(&ui);
                            ui.separator();
                        });
                    if show_demo {
                        ui.show_demo_window(&mut false);
                    }
                }
                if last_cursor != Some(ui.mouse_cursor()) {
                    last_cursor = Some(ui.mouse_cursor());
                    platform.prepare_render(&ui, &window);
                }

                let mut encoder: wgpu::CommandEncoder = h
                    .device
                    .create_command_encoder(
                        &wgpu::CommandEncoderDescriptor{label: None }
                    );
                let mut imgui_renderpass = encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments:
                                &[wgpu::RenderPassColorAttachmentDescriptor {
                                attachment: &frame.output.view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: true,
                            },
                        }],
                        depth_stencil_attachment: None,
                });

                imgui_renderer.render(
                    ui.render(),
                    &h.queue,
                    &h.device,
                    &mut imgui_renderpass
                ).expect("Rendering failed");
                
                drop(imgui_renderpass);
                h.queue.submit(Some(encoder.finish()));
           }
            _ => {}
        }
        platform.handle_event(imgui.io_mut(), &window, &event)
    });
}

