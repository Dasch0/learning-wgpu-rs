use winit::{
    event::*,
    event_loop::{EventLoop, ControlFlow},
    window::{Window, WindowBuilder},
};

struct Setup {
    instance: wgpu::Instance,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

async fn setup (window: &Window) -> Setup {
    log::info!("Initializing instance...");
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    
    log::info!("Obtaining window surface...");
    let (size, surface) = unsafe {
        let size = window.inner_size();
        let surface = instance.create_surface(window);
        (size, surface)
    };

    log::info!("Initializing adapter...");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: Some(&surface),
        })
        .await
        .unwrap();

    //TODO: Support features
    let optional_features = wgpu::Features::empty();
    let required_features = wgpu::Features::default();
    let adapter_features = adapter.features();

    //TODO: Support limits
    let required_limits = wgpu::Limits::default();

    let trace_dir = std::env::var("WGPU_TRACE");
    
    log::info!("Initializing device & queue...");
    let(device, queue) = adapter
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

    Setup {
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
    }
}

fn main() {
    // Setup Winit
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .unwrap();

    let setup = futures::executor::block_on(setup(&window));

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput {
                    input,
                    ..
                } => {
                    match input {
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        _ => {}
                    }
                }
                _ => {}
            }
            _ => {}
        }
    }); 
}
