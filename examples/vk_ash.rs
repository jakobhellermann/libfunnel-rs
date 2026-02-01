#![allow(clippy::type_complexity, clippy::too_many_arguments)]
// Translated from https://github.com/hoshinolina/libfunnel/blob/main/demo/test-vk.c
//
// Command-line options:
//   -async         Async mode (default) with FIFO present
//   -single        Single-buffered mode with MAILBOX present
//   -double        Double-buffered mode with MAILBOX present
//   -synchronous   Synchronous mode with MAILBOX present
//   -sync_torture  Run 100 iterations at 1024x1024 for testing

use anyhow::{Context as _, Result};
use ash::ext::debug_utils;
use ash::khr::{surface, swapchain, wayland_surface};
use ash::vk::{self, Handle, PresentModeKHR};
use libfunnel::bindings::{self as funnel, funnel_mode};
use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use wayland_client::protocol::{wl_compositor, wl_registry, wl_surface};
use wayland_client::{Connection, Dispatch, Proxy, QueueHandle};
use wayland_protocols::xdg::decoration::zv1::client::zxdg_decoration_manager_v1;
use wayland_protocols::xdg::decoration::zv1::client::zxdg_toplevel_decoration_v1;
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};

const VK_API_VERSION: u32 = vk::API_VERSION_1_3;
const APP_NAME: &str = "Wayland Vulkan Example";

// Assuming Vulkan 1.3+, most extensions are promoted to core
const INSTANCE_EXTENSIONS: &[&str] = &[
    "VK_EXT_debug_utils",
    "VK_KHR_surface",
    "VK_KHR_wayland_surface",
];

const DEVICE_EXTENSIONS: &[&str] = &[
    "VK_KHR_swapchain",
    "VK_EXT_swapchain_maintenance1",
    "VK_KHR_external_semaphore_fd",
    "VK_KHR_external_memory_fd",
    "VK_EXT_external_memory_dma_buf",
    "VK_EXT_image_drm_format_modifier",
];

const LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];

// Alignment helper for shader bytecode
#[repr(C)]
struct AlignedAs<Align, Bytes: ?Sized> {
    _align: [Align; 0],
    bytes: Bytes,
}

macro_rules! include_aligned_bytes {
    ($align:ty, $source:literal) => {
        &AlignedAs {
            _align: [],
            bytes: *include_bytes!($source),
        }
    };
}

static VERTEX_SHADER: &AlignedAs<u32, [u8]> =
    include_aligned_bytes!(u32, "../shaders/triangle_vert.spv");
static FRAGMENT_SHADER: &AlignedAs<u32, [u8]> =
    include_aligned_bytes!(u32, "../shaders/triangle_frag.spv");

struct SwapchainElement {
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    image_view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    start_semaphore: vk::Semaphore,
    end_semaphore: vk::Semaphore,
    fence: vk::Fence,
    last_fence: vk::Fence,
}

struct SwapchainResources {
    loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    elements: Vec<SwapchainElement>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    #[allow(dead_code)]
    format: vk::Format,
    image_count: u32,
}

#[repr(C)]
struct PushConstants {
    frame: f32,
}

// Funnel buffer callbacks
unsafe extern "C" fn alloc_buffer_cb(
    _opaque: *mut std::os::raw::c_void,
    _stream: *mut funnel::funnel_stream,
    _buf: *mut funnel::funnel_buffer,
) {
    // TODO: Allocate VkImageView for funnel buffer (C lines 231-266)
    // Currently commented out in C code, so we leave empty for now
}

unsafe extern "C" fn free_buffer_cb(
    _opaque: *mut std::os::raw::c_void,
    _stream: *mut funnel::funnel_stream,
    buf: *mut funnel::funnel_buffer,
) {
    // Get user data and destroy the image view if it exists (C lines 268-272)
    // Note: This would need a global device handle to call device.destroy_image_view()
    // like the C code uses. Since alloc_buffer_cb is commented out in C and never
    // creates views, this callback won't have anything to destroy in practice.
    unsafe {
        let view_ptr = funnel::funnel_buffer_get_user_data(buf);
        let view = vk::ImageView::from_raw(view_ptr as u64);
        if !view.is_null() {
            // Would call: device.destroy_image_view(view, None);
            // But we don't have device handle here without using globals
        }
    }
}

// Wayland application state
struct AppState {
    compositor: Option<wl_compositor::WlCompositor>,
    xdg_wm_base: Option<xdg_wm_base::XdgWmBase>,
    decoration_manager: Option<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1>,
    quit: AtomicBool,
    resize: AtomicBool,
    ready_to_resize: AtomicBool,
    new_width: AtomicI32,
    new_height: AtomicI32,
}

impl AppState {
    fn new() -> Self {
        Self {
            compositor: None,
            xdg_wm_base: None,
            decoration_manager: None,
            quit: AtomicBool::new(false),
            resize: AtomicBool::new(false),
            ready_to_resize: AtomicBool::new(false),
            new_width: AtomicI32::new(0),
            new_height: AtomicI32::new(0),
        }
    }
}

// Registry listener - binds globals
impl Dispatch<wl_registry::WlRegistry, ()> for AppState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match interface.as_str() {
                "wl_compositor" => {
                    state.compositor = Some(registry.bind::<wl_compositor::WlCompositor, _, _>(
                        name,
                        version.min(1),
                        qh,
                        (),
                    ));
                }
                "xdg_wm_base" => {
                    state.xdg_wm_base = Some(registry.bind::<xdg_wm_base::XdgWmBase, _, _>(
                        name,
                        version.min(1),
                        qh,
                        (),
                    ));
                }
                "zxdg_decoration_manager_v1" => {
                    state.decoration_manager = Some(
                        registry.bind::<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1, _, _>(
                            name,
                            version.min(1),
                            qh,
                            (),
                        ),
                    );
                }
                _ => {}
            }
        }
    }
}

// XDG WM Base listener - handle ping
impl Dispatch<xdg_wm_base::XdgWmBase, ()> for AppState {
    fn event(
        _state: &mut Self,
        wm_base: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            wm_base.pong(serial);
        }
    }
}

// XDG Surface listener - handle configure
impl Dispatch<xdg_surface::XdgSurface, ()> for AppState {
    fn event(
        state: &mut Self,
        surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            surface.ack_configure(serial);
            if state.resize.load(Ordering::Relaxed) {
                state.ready_to_resize.store(true, Ordering::Relaxed);
            }
        }
    }
}

// XDG Toplevel listener - handle configure and close
impl Dispatch<xdg_toplevel::XdgToplevel, ()> for AppState {
    fn event(
        state: &mut Self,
        _toplevel: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            xdg_toplevel::Event::Configure {
                width,
                height,
                states: _,
            } => {
                if width != 0 && height != 0 {
                    state.resize.store(true, Ordering::Relaxed);
                    state.new_width.store(width, Ordering::Relaxed);
                    state.new_height.store(height, Ordering::Relaxed);
                }
            }
            xdg_toplevel::Event::Close => {
                state.quit.store(true, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}

// Stubs for other protocols we need to implement
impl Dispatch<wl_compositor::WlCompositor, ()> for AppState {
    fn event(
        _: &mut Self,
        _: &wl_compositor::WlCompositor,
        _: wl_compositor::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_surface::WlSurface, ()> for AppState {
    fn event(
        _: &mut Self,
        _: &wl_surface::WlSurface,
        _: wl_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1, ()> for AppState {
    fn event(
        _: &mut Self,
        _: &zxdg_decoration_manager_v1::ZxdgDecorationManagerV1,
        _: zxdg_decoration_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1, ()> for AppState {
    fn event(
        _: &mut Self,
        _: &zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1,
        _: zxdg_toplevel_decoration_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

// Vulkan debug callback (C lines 191-229)
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let data = unsafe { &*p_callback_data };
    let message = unsafe { CStr::from_ptr(data.p_message) }.to_string_lossy();

    let type_str = match type_ {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "general",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "validation",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "performance",
        _ => "unknown",
    };

    let severity_str = match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "(verbose)",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "(info)",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "(warning)",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "(error)",
        _ => "(unknown)",
    };

    println!("Vulkan {} {}: {}", type_str, severity_str, message);

    vk::FALSE
}

// Swapchain creation (C lines 275-551)
#[allow(clippy::too_many_arguments)]
unsafe fn create_swapchain(
    instance: &ash::Instance,
    phys_device: vk::PhysicalDevice,
    device: &ash::Device,
    surface_loader: &surface::Instance,
    vulkan_surface: vk::SurfaceKHR,
    config: &Config,
    vertex_shader_module: vk::ShaderModule,
    fragment_shader_module: vk::ShaderModule,
    command_pool: vk::CommandPool,
) -> Result<SwapchainResources> {
    // Query surface capabilities and formats
    let capabilities = unsafe {
        surface_loader.get_physical_device_surface_capabilities(phys_device, vulkan_surface)?
    };

    let formats =
        unsafe { surface_loader.get_physical_device_surface_formats(phys_device, vulkan_surface)? };

    // Choose format - prefer B8G8R8A8_UNORM
    let chosen_format = formats
        .iter()
        .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
        .unwrap_or(&formats[0]);

    let format = chosen_format.format;

    // Determine image count - prefer min + 1 for better performance, but respect max if set
    let desired_image_count = capabilities.min_image_count + 1;
    let image_count = if capabilities.max_image_count > 0 {
        desired_image_count.min(capabilities.max_image_count)
    } else {
        // max_image_count == 0 means no limit
        desired_image_count
    };

    // Create swapchain
    let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(vulkan_surface)
        .min_image_count(image_count)
        .image_format(chosen_format.format)
        .image_color_space(chosen_format.color_space)
        .image_extent(vk::Extent2D {
            width: config.width,
            height: config.height,
        })
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(config.present_mode)
        .clipped(true);

    let swapchain_loader = swapchain::Device::new(instance, device);
    let swapchain_khr = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

    // Create render pass
    let attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

    let attachment_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&attachment_ref));

    let render_pass_create_info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass));

    let render_pass = unsafe { device.create_render_pass(&render_pass_create_info, None)? };

    // Get swapchain images
    let images = unsafe { swapchain_loader.get_swapchain_images(swapchain_khr)? };
    let actual_image_count = images.len() as u32;

    // Create per-image resources
    let mut elements = Vec::with_capacity(images.len());

    for &image in &images {
        // Command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };
        let command_buffer = command_buffers[0];

        // Image view
        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image(image);

        let image_view = unsafe { device.create_image_view(&image_view_create_info, None)? };

        // Framebuffer
        let framebuffer_create_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(std::slice::from_ref(&image_view))
            .width(config.width)
            .height(config.height)
            .layers(1);

        let framebuffer = unsafe { device.create_framebuffer(&framebuffer_create_info, None)? };

        // Semaphores
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let start_semaphore = unsafe { device.create_semaphore(&semaphore_create_info, None)? };
        let end_semaphore = unsafe { device.create_semaphore(&semaphore_create_info, None)? };

        // Fence
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { device.create_fence(&fence_create_info, None)? };

        elements.push(SwapchainElement {
            command_buffer,
            image,
            image_view,
            framebuffer,
            start_semaphore,
            end_semaphore,
            fence,
            last_fence: vk::Fence::null(),
        });
    }

    // Create graphics pipeline
    let entry_name = CString::new("main")?;

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&entry_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&entry_name),
    ];

    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: config.width as f32,
        height: config.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };

    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: config.width,
            height: config.height,
        },
    };

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(std::slice::from_ref(&viewport))
        .scissors(std::slice::from_ref(&scissor));

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false);

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(false);

    let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(std::slice::from_ref(&color_blend_attachment));

    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(std::mem::size_of::<PushConstants>() as u32);

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));

    let pipeline_layout =
        unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None)? };

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipelines = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_create_info),
            None,
        )
    }
    .map_err(|(_, e)| e)?;

    let pipeline = pipelines[0];

    Ok(SwapchainResources {
        loader: swapchain_loader,
        swapchain: swapchain_khr,
        elements,
        render_pass,
        pipeline_layout,
        pipeline,
        format,
        image_count: actual_image_count,
    })
}

// Load shader module from SPIR-V bytecode (C lines 573-590)
unsafe fn load_shader_module(device: &ash::Device, code: &[u8]) -> Result<vk::ShaderModule> {
    let code_u32: &[u32] = unsafe {
        std::slice::from_raw_parts(
            code.as_ptr() as *const u32,
            code.len() / std::mem::size_of::<u32>(),
        )
    };

    let create_info = vk::ShaderModuleCreateInfo::default().code(code_u32);

    unsafe {
        device
            .create_shader_module(&create_info, None)
            .map_err(Into::into)
    }
}

struct Config {
    funnel_mode: funnel_mode,
    present_mode: PresentModeKHR,
    iterations: u32,
    width: u32,
    height: u32,
}

fn main() -> Result<()> {
    // Parse command line arguments (C lines 592-614)
    let args: Vec<String> = std::env::args().collect();

    let mut config = Config {
        funnel_mode: funnel_mode::FUNNEL_ASYNC,
        present_mode: vk::PresentModeKHR::FIFO,
        iterations: 1,
        width: 512u32,
        height: 512u32,
    };

    for arg in &args[1..] {
        match arg.as_str() {
            "-async" => {
                config.funnel_mode = funnel_mode::FUNNEL_ASYNC;
                config.present_mode = vk::PresentModeKHR::FIFO;
            }
            "-single" => {
                config.funnel_mode = funnel_mode::FUNNEL_SINGLE_BUFFERED;
                config.present_mode = vk::PresentModeKHR::MAILBOX;
            }
            "-double" => {
                config.funnel_mode = funnel_mode::FUNNEL_DOUBLE_BUFFERED;
                config.present_mode = vk::PresentModeKHR::MAILBOX;
            }
            "-synchronous" => {
                config.funnel_mode = funnel_mode::FUNNEL_SYNCHRONOUS;
                config.present_mode = vk::PresentModeKHR::MAILBOX;
            }
            "-sync_torture" => {
                config.iterations = 100;
                config.width = 1024;
                config.height = 1024;
            }
            _ => {}
        }
    }

    println!(
        "Starting Wayland Vulkan Example ({}x{}, iterations={})",
        config.width, config.height, config.iterations
    );

    // Wayland setup (C lines 616-642)
    let conn = Connection::connect_to_env()?;
    let display = conn.display();

    let mut event_queue = conn.new_event_queue();
    let qh = event_queue.handle();

    let mut state = AppState::new();
    let _registry = display.get_registry(&qh, ());

    // Roundtrip to get globals
    event_queue.roundtrip(&mut state)?;

    let compositor = state
        .compositor
        .as_ref()
        .context("Compositor not available")?;
    let xdg_wm_base = state
        .xdg_wm_base
        .as_ref()
        .context("xdg_wm_base not available")?;
    let decoration_manager = state
        .decoration_manager
        .as_ref()
        .context("decoration_manager not available")?;

    let surface = compositor.create_surface(&qh, ());
    let xdg_surface = xdg_wm_base.get_xdg_surface(&surface, &qh, ());
    let toplevel = xdg_surface.get_toplevel(&qh, ());

    let decoration = decoration_manager.get_toplevel_decoration(&toplevel, &qh, ());

    toplevel.set_title(APP_NAME.to_string());
    toplevel.set_app_id(APP_NAME.to_string());
    decoration.set_mode(zxdg_toplevel_decoration_v1::Mode::ServerSide);

    surface.commit();
    event_queue.roundtrip(&mut state)?;
    surface.commit();

    println!("Wayland setup complete");

    // Vulkan instance creation (C lines 645-704)
    let entry = unsafe { ash::Entry::load()? };

    // Check for validation layers
    let available_layers = unsafe { entry.enumerate_instance_layer_properties()? };
    let mut found_layers = 0;
    for layer in &available_layers {
        let layer_name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
        for &requested_layer in LAYERS {
            if layer_name.to_str().unwrap_or("") == requested_layer {
                found_layers += 1;
            }
        }
    }

    let enable_layers = found_layers >= LAYERS.len();

    // Convert extension names to CStrings
    let extension_names_raw: Vec<CString> = INSTANCE_EXTENSIONS
        .iter()
        .map(|&s| CString::new(s).unwrap())
        .collect();
    let extension_names_ptrs: Vec<*const i8> =
        extension_names_raw.iter().map(|s| s.as_ptr()).collect();

    // Convert layer names to CStrings
    let layer_names_raw: Vec<CString> = LAYERS.iter().map(|&s| CString::new(s).unwrap()).collect();
    let layer_names_ptrs: Vec<*const i8> = layer_names_raw.iter().map(|s| s.as_ptr()).collect();

    let app_name = CString::new(APP_NAME)?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&app_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(VK_API_VERSION);

    let mut create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extension_names_ptrs);

    if enable_layers {
        create_info = create_info.enabled_layer_names(&layer_names_ptrs);
    }

    let instance = unsafe { entry.create_instance(&create_info, None)? };

    // Create debug messenger
    let debug_utils = debug_utils::Instance::new(&entry, &instance);

    let debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));

    let debug_messenger =
        unsafe { debug_utils.create_debug_utils_messenger(&debug_create_info, None)? };

    println!("Vulkan instance created");

    // Create Vulkan surface from Wayland surface (C lines 706-714)
    let wayland_surface_loader = wayland_surface::Instance::new(&entry, &instance);

    // Get raw pointers from wayland-client objects for Vulkan
    let wl_display_ptr = display.id().as_ptr().cast();
    let wl_surface_ptr = surface.id().as_ptr().cast();

    let surface_create_info = vk::WaylandSurfaceCreateInfoKHR::default()
        .display(wl_display_ptr)
        .surface(wl_surface_ptr);

    let vulkan_surface =
        unsafe { wayland_surface_loader.create_wayland_surface(&surface_create_info, None)? };

    let surface_loader = surface::Instance::new(&entry, &instance);

    println!("Vulkan surface created");

    // Helper function to prioritize physical devices (higher is better)
    fn device_type_priority(instance: &ash::Instance, device: vk::PhysicalDevice) -> u32 {
        let properties = unsafe { instance.get_physical_device_properties(device) };
        match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 5,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 4,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 3,
            vk::PhysicalDeviceType::CPU => 2,
            vk::PhysicalDeviceType::OTHER => 1,
            _ => u32::MIN,
        }
    }

    // Physical device selection (C lines 716-756)
    let phys_devices = unsafe { instance.enumerate_physical_devices()? };

    let phys_device = phys_devices
        .iter()
        .max_by_key(|&&device| device_type_priority(&instance, device));

    let Some(&phys_device) = phys_device else {
        anyhow::bail!("No suitable physical device found");
    };

    println!("Physical device selected");

    // Find queue family with graphics + present support (C lines 758-831)
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(phys_device) };

    let mut queue_family_index = None;
    for (index, queue_family) in queue_families.iter().enumerate() {
        let supports_present = unsafe {
            surface_loader.get_physical_device_surface_support(
                phys_device,
                index as u32,
                vulkan_surface,
            )?
        };

        if supports_present && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            queue_family_index = Some(index as u32);
            break;
        }
    }

    let queue_family_index = queue_family_index.context("No suitable queue family found")?;

    let queue_priorities = [1.0f32];
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);

    // Enable swapchain maintenance feature
    let mut swapchain_maint_features =
        vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default().swapchain_maintenance1(true);

    // Convert device extension names to CStrings
    let device_extension_names_raw: Vec<CString> = DEVICE_EXTENSIONS
        .iter()
        .map(|&s| CString::new(s).unwrap())
        .collect();
    let device_extension_names_ptrs: Vec<*const i8> = device_extension_names_raw
        .iter()
        .map(|s| s.as_ptr())
        .collect();

    // Device layers are deprecated in Vulkan 1.1+, layers are only at instance level
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_create_info))
        .enabled_extension_names(&device_extension_names_ptrs)
        .push_next(&mut swapchain_maint_features);

    let device = unsafe { instance.create_device(phys_device, &device_create_info, None)? };

    let _queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    println!("Logical device and queue created");

    // Command pool creation (C lines 833-841)
    let command_pool_create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None)? };

    println!("Command pool created");

    // Load shaders (C lines 843-844)
    let vertex_shader_module = unsafe { load_shader_module(&device, &VERTEX_SHADER.bytes)? };
    let fragment_shader_module = unsafe { load_shader_module(&device, &FRAGMENT_SHADER.bytes)? };

    println!("Shaders loaded");

    // Create swapchain
    let mut swapchain = unsafe {
        create_swapchain(
            &instance,
            phys_device,
            &device,
            &surface_loader,
            vulkan_surface,
            &config,
            vertex_shader_module,
            fragment_shader_module,
            command_pool,
        )?
    };

    println!("Swapchain created with {} images", swapchain.image_count);

    let (ctx, stream) = setup_funnel(&config, &instance, phys_device, &device);

    println!("Funnel stream initialized and started");

    render_loop(
        &mut config,
        event_queue,
        state,
        surface,
        &instance,
        vulkan_surface,
        &surface_loader,
        phys_device,
        queue_family_index,
        &device,
        _queue,
        command_pool,
        vertex_shader_module,
        fragment_shader_module,
        &mut swapchain,
        stream,
    )?;

    println!("Exiting main loop");

    // Wait for device idle before cleanup
    unsafe { device.device_wait_idle()? };

    // Funnel cleanup (C lines 1137-1142)
    unsafe {
        let ret = funnel::funnel_stream_stop(stream);
        assert_eq!(ret, 0, "funnel_stream_stop failed");

        funnel::funnel_stream_destroy(stream);
        funnel::funnel_shutdown(ctx);
    }

    println!("Funnel stream stopped and cleaned up");

    // Clean up
    unsafe {
        destroy_swapchain(&device, command_pool, &swapchain);
        device.destroy_command_pool(command_pool, None);
        device.destroy_shader_module(vertex_shader_module, None);
        device.destroy_shader_module(fragment_shader_module, None);
        device.destroy_device(None);
        surface_loader.destroy_surface(vulkan_surface, None);
        debug_utils.destroy_debug_utils_messenger(debug_messenger, None);
        instance.destroy_instance(None);
    }

    Ok(())
}

/// Cleanup swapchain resources
unsafe fn destroy_swapchain(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    swapchain: &SwapchainResources,
) {
    unsafe {
        for element in &swapchain.elements {
            device.destroy_fence(element.fence, None);
            device.destroy_semaphore(element.end_semaphore, None);
            device.destroy_semaphore(element.start_semaphore, None);
            device.destroy_framebuffer(element.framebuffer, None);
            device.destroy_image_view(element.image_view, None);
            device.free_command_buffers(command_pool, &[element.command_buffer]);
        }
        device.destroy_pipeline(swapchain.pipeline, None);
        device.destroy_pipeline_layout(swapchain.pipeline_layout, None);
        device.destroy_render_pass(swapchain.render_pass, None);
        swapchain
            .loader
            .destroy_swapchain(swapchain.swapchain, None);
    }
}

fn render_loop(
    config: &mut Config,
    mut event_queue: wayland_client::EventQueue<AppState>,
    mut state: AppState,
    surface: wl_surface::WlSurface,
    instance: &ash::Instance,
    vulkan_surface: vk::SurfaceKHR,
    surface_loader: &surface::Instance,
    phys_device: vk::PhysicalDevice,
    queue_family_index: u32,
    device: &ash::Device,
    _queue: vk::Queue,
    command_pool: vk::CommandPool,
    vertex_shader_module: vk::ShaderModule,
    fragment_shader_module: vk::ShaderModule,
    swapchain: &mut SwapchainResources,
    stream: *mut funnel::funnel_stream,
) -> Result<(), anyhow::Error> {
    let mut frame = 0u32;
    let mut current_frame = 0u32;

    while !state.quit.load(Ordering::Relaxed) {
        // Handle resize (C lines 901-925)
        if state.ready_to_resize.load(Ordering::Relaxed) && state.resize.load(Ordering::Relaxed) {
            let new_width = state.new_width.load(Ordering::Relaxed) as u32;
            let new_height = state.new_height.load(Ordering::Relaxed) as u32;

            if config.width != new_width || config.height != new_height {
                config.width = new_width;
                config.height = new_height;

                unsafe { device.device_wait_idle()? };

                // Destroy old swapchain and create new one
                unsafe { destroy_swapchain(device, command_pool, swapchain) };
                *swapchain = unsafe {
                    create_swapchain(
                        instance,
                        phys_device,
                        device,
                        surface_loader,
                        vulkan_surface,
                        config,
                        vertex_shader_module,
                        fragment_shader_module,
                        command_pool,
                    )?
                };

                current_frame = 0;

                surface.commit();

                // Update funnel stream size (C lines 920-923)
                unsafe {
                    let ret = funnel::funnel_stream_set_size(stream, config.width, config.height);
                    assert_eq!(ret, 0);
                    let ret = funnel::funnel_stream_configure(stream);
                    assert_eq!(ret, 0);
                };
            }

            state.ready_to_resize.store(false, Ordering::Relaxed);
            state.resize.store(false, Ordering::Relaxed);
        }

        // Dequeue a funnel buffer for this frame (C line 928)
        let mut funnel_buf: *mut funnel::funnel_buffer = std::ptr::null_mut();
        let _ret = unsafe { funnel::funnel_stream_dequeue(stream, &mut funnel_buf) };
        // _ret == 0 means we got a buffer, non-zero means no buffer available (skip frame)

        // Copy values from current_element to avoid long-lived borrow
        let (current_fence, current_start_semaphore, current_end_semaphore) = {
            let current_element = &swapchain.elements[current_frame as usize];
            (
                current_element.fence,
                current_element.start_semaphore,
                current_element.end_semaphore,
            )
        };

        // Wait for fence and acquire next image (C lines 932-945)
        unsafe {
            device.wait_for_fences(&[current_fence], true, u64::MAX)?;
        }

        let image_index = match unsafe {
            swapchain.loader.acquire_next_image(
                swapchain.swapchain,
                u64::MAX,
                current_start_semaphore,
                vk::Fence::null(),
            )
        } {
            Ok((index, _)) => index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                // Swapchain out of date, recreate
                unsafe { device.device_wait_idle()? };

                // Destroy old swapchain and create new one
                unsafe { destroy_swapchain(device, command_pool, swapchain) };
                *swapchain = unsafe {
                    create_swapchain(
                        instance,
                        phys_device,
                        device,
                        surface_loader,
                        vulkan_surface,
                        config,
                        vertex_shader_module,
                        fragment_shader_module,
                        command_pool,
                    )?
                };
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        // Wait for last fence if it exists (C lines 949-954)
        {
            let element = &mut swapchain.elements[image_index as usize];
            if !element.last_fence.is_null() {
                unsafe {
                    device.wait_for_fences(&[element.last_fence], true, u64::MAX)?;
                }
            }
            element.last_fence = current_fence;
        }

        unsafe {
            device.reset_fences(&[current_fence])?;
        }

        let element = &swapchain.elements[image_index as usize];

        // Begin command buffer (C lines 958-962)
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device.begin_command_buffer(element.command_buffer, &begin_info)?;
        }

        // Render pass (C lines 964-988)
        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        };

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(swapchain.render_pass)
            .framebuffer(element.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: config.width,
                    height: config.height,
                },
            })
            .clear_values(std::slice::from_ref(&clear_value));

        unsafe {
            device.cmd_begin_render_pass(
                element.command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );

            device.cmd_bind_pipeline(
                element.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                swapchain.pipeline,
            );

            // Push constants (C lines 983-986)
            let push_constants = PushConstants {
                frame: frame as f32,
            };
            device.cmd_push_constants(
                element.command_buffer,
                swapchain.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_raw_parts(
                    &push_constants as *const _ as *const u8,
                    std::mem::size_of::<PushConstants>(),
                ),
            );

            device.cmd_draw(element.command_buffer, 3, 1, 0, 0);
            device.cmd_end_render_pass(element.command_buffer);
        }

        // Blit to funnel buffer if we have one (C lines 989-1037)
        if !funnel_buf.is_null() {
            unsafe {
                let mut buffer_width: u32 = 0;
                let mut buffer_height: u32 = 0;
                funnel::funnel_buffer_get_size(funnel_buf, &mut buffer_width, &mut buffer_height);

                let mut funnel_image: funnel::VkImage = std::ptr::null_mut();
                let ret = funnel::funnel_buffer_get_vk_image(funnel_buf, &mut funnel_image);
                assert_eq!(ret, 0, "funnel_buffer_get_vk_image failed");
                assert!(!funnel_image.is_null(), "funnel image is null");

                let funnel_vk_image = vk::Image::from_raw(funnel_image as u64);

                let blit_region = vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: config.width as i32,
                            y: config.height as i32,
                            z: 1,
                        },
                    ],
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    dst_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: buffer_width as i32,
                            y: buffer_height as i32,
                            z: 1,
                        },
                    ],
                };

                let subresource_range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };

                let clear_color = vk::ClearColorValue {
                    float32: [1.0, 0.0, 0.0, 1.0],
                };

                for _ in 0..config.iterations {
                    device.cmd_clear_color_image(
                        element.command_buffer,
                        funnel_vk_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &clear_color,
                        &[subresource_range],
                    );

                    device.cmd_blit_image(
                        element.command_buffer,
                        element.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        funnel_vk_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[blit_region],
                        vk::Filter::NEAREST,
                    );
                }
            }
        }

        // Image barrier for present (C lines 1039-1060)
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::empty())
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index)
            .image(element.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            device.cmd_pipeline_barrier(
                element.command_buffer,
                vk::PipelineStageFlags::ALL_GRAPHICS,
                vk::PipelineStageFlags::ALL_GRAPHICS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            device.end_command_buffer(element.command_buffer)?;
        }

        // Queue submit (C lines 1065-1092)
        // Get funnel semaphores and fence if we have a buffer (C lines 1074-1080)
        let (wait_semaphores, signal_semaphores, submit_fence) = if !funnel_buf.is_null() {
            unsafe {
                let mut funnel_wait_sema: funnel::VkSemaphore = std::ptr::null_mut();
                let mut funnel_signal_sema: funnel::VkSemaphore = std::ptr::null_mut();
                let ret = funnel::funnel_buffer_get_vk_semaphores(
                    funnel_buf,
                    &mut funnel_wait_sema,
                    &mut funnel_signal_sema,
                );
                assert_eq!(ret, 0, "funnel_buffer_get_vk_semaphores failed");

                let mut funnel_fence: funnel::VkFence = std::ptr::null_mut();
                let ret = funnel::funnel_buffer_get_vk_fence(funnel_buf, &mut funnel_fence);
                assert_eq!(ret, 0, "funnel_buffer_get_vk_fence failed");

                (
                    vec![
                        current_start_semaphore,
                        vk::Semaphore::from_raw(funnel_wait_sema as u64),
                    ],
                    vec![
                        current_end_semaphore,
                        vk::Semaphore::from_raw(funnel_signal_sema as u64),
                    ],
                    vk::Fence::from_raw(funnel_fence as u64),
                )
            }
        } else {
            (
                vec![current_start_semaphore],
                vec![current_end_semaphore],
                current_fence,
            )
        };

        let wait_stages = [
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ];
        let command_buffers = [element.command_buffer];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages[..wait_semaphores.len()])
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            device.queue_submit(_queue, &[submit_info], submit_fence)?;
        }

        // Enqueue funnel buffer if we have one (C lines 1094-1102)
        if !funnel_buf.is_null() {
            let ret = unsafe { funnel::funnel_stream_enqueue(stream, funnel_buf) };
            if ret < 0 {
                eprintln!("Queue failed: {}", ret);
                assert!(ret >= 0, "funnel_stream_enqueue failed");
            } else if ret != 1 {
                eprintln!("Buffer dropped (stream renegotiated or paused)");
            }
        }

        // Present (C lines 1104-1128)
        let swapchains = [swapchain.swapchain];
        let image_indices = [image_index];
        let fences = [current_fence];
        let present_wait_semaphores = [current_end_semaphore];

        let mut swapchain_present_fence_info =
            vk::SwapchainPresentFenceInfoEXT::default().fences(&fences);

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&present_wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .push_next(&mut swapchain_present_fence_info);

        match unsafe { swapchain.loader.queue_present(_queue, &present_info) } {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                unsafe { device.device_wait_idle()? };

                // Destroy old swapchain and create new one
                unsafe { destroy_swapchain(device, command_pool, swapchain) };
                *swapchain = unsafe {
                    create_swapchain(
                        instance,
                        phys_device,
                        device,
                        surface_loader,
                        vulkan_surface,
                        config,
                        vertex_shader_module,
                        fragment_shader_module,
                        command_pool,
                    )?
                };
                continue;
            }
            Err(e) => return Err(e.into()),
        }

        current_frame = (current_frame + 1) % swapchain.image_count;
        frame += 1;

        // Process Wayland events (C line 1132)
        event_queue.dispatch_pending(&mut state)?;
    }

    Ok(())
}

fn setup_funnel(
    config: &Config,
    instance: &ash::Instance,
    phys_device: vk::PhysicalDevice,
    device: &ash::Device,
) -> (*mut funnel::funnel_ctx, *mut funnel::funnel_stream) {
    // Funnel initialization (C lines 849-898)
    let mut ctx: *mut funnel::funnel_ctx = std::ptr::null_mut();
    let mut stream: *mut funnel::funnel_stream = std::ptr::null_mut();

    unsafe {
        let ret = funnel::funnel_init(&mut ctx);
        assert_eq!(ret, 0, "funnel_init failed");

        let stream_name = CString::new("Funnel Test").unwrap();
        let ret = funnel::funnel_stream_create(ctx, stream_name.as_ptr(), &mut stream);
        assert_eq!(ret, 0, "funnel_stream_create failed");

        funnel::funnel_stream_set_buffer_callbacks(
            stream,
            Some(alloc_buffer_cb),
            Some(free_buffer_cb),
            std::ptr::null_mut(),
        );

        let ret = funnel::funnel_stream_init_vulkan(
            stream,
            instance.handle().as_raw() as funnel::VkInstance,
            phys_device.as_raw() as funnel::VkPhysicalDevice,
            device.handle().as_raw() as funnel::VkDevice,
        );
        assert_eq!(ret, 0, "funnel_stream_init_vulkan failed");

        let ret = funnel::funnel_stream_set_size(stream, config.width, config.height);
        assert_eq!(ret, 0, "funnel_stream_set_size failed");

        let ret = funnel::funnel_stream_set_mode(stream, config.funnel_mode);
        assert_eq!(ret, 0, "funnel_stream_set_mode failed");

        // FUNNEL_RATE_VARIABLE = {0, 1} from funnel.h
        let rate_variable = funnel::funnel_fraction { num: 0, den: 1 };
        let ret = funnel::funnel_stream_set_rate(
            stream,
            rate_variable,
            funnel::funnel_fraction { num: 1, den: 1 },
            funnel::funnel_fraction { num: 1000, den: 1 },
        );
        assert_eq!(ret, 0, "funnel_stream_set_rate failed");

        let ret =
            funnel::funnel_stream_vk_set_usage(stream, vk::ImageUsageFlags::TRANSFER_DST.as_raw());
        assert_eq!(ret, 0, "funnel_stream_vk_set_usage failed");

        // Try multiple formats, at least one should work
        let mut have_format = false;
        let ret = funnel::funnel_stream_vk_add_format(
            stream,
            vk::Format::R8G8B8A8_SRGB.as_raw() as u32,
            true,
            vk::FormatFeatureFlags::BLIT_DST.as_raw(),
        );
        have_format |= ret == 0;

        let ret = funnel::funnel_stream_vk_add_format(
            stream,
            vk::Format::B8G8R8A8_SRGB.as_raw() as u32,
            true,
            vk::FormatFeatureFlags::BLIT_DST.as_raw(),
        );
        have_format |= ret == 0;

        let ret = funnel::funnel_stream_vk_add_format(
            stream,
            vk::Format::R8G8B8A8_SRGB.as_raw() as u32,
            false,
            vk::FormatFeatureFlags::BLIT_DST.as_raw(),
        );
        have_format |= ret == 0;

        let ret = funnel::funnel_stream_vk_add_format(
            stream,
            vk::Format::B8G8R8A8_SRGB.as_raw() as u32,
            false,
            vk::FormatFeatureFlags::BLIT_DST.as_raw(),
        );
        have_format |= ret == 0;

        assert!(have_format, "No suitable format found for funnel stream");

        let ret = funnel::funnel_stream_configure(stream);
        assert_eq!(ret, 0, "funnel_stream_configure failed");

        let ret = funnel::funnel_stream_start(stream);
        assert_eq!(ret, 0, "funnel_stream_start failed");
    }
    (ctx, stream)
}
