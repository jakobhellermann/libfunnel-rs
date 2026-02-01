# libfunnel-rs

Rust bindings for [libfunnel](https://github.com/hoshinolina/libfunnel) - a library to make creating PipeWire video streams easy, using zero-copy DMA-BUF frame sharing. "Spout2 / Syphon, but for Linux".

## Status

- [x] Bindings for the core `libfunnel` API
- [x] Bindings for the vulkan integration
- [ ] Bindings for the egl integration
- [ ] Bindings for the gbm integration
- [x] [Example](./examples/vk_ash.rs) using [ash](https://docs.rs/ash)
- [ ] Example using [wgpu](https://docs.rs/wgpu) (requires explicit wait semaphores)

## Documentation

See [libfunnel Documentation](https://libfunnel.readthedocs.io/en/latest/) for the upstream docs. The documentation for the bindings are available on [docs.rs](https://docs.rs/libfunnel).

Specifically the [`funnel_mode`](https://libfunnel.readthedocs.io/en/latest/funnel_8h.html#a302307074c18579ebd57b9088f76c4c7) and [Buffer synchronization guide](https://libfunnel.readthedocs.io/en/latest/buffersync.html) pages are very helpful to read.

## Installation

Run `cargo add libfunnel` to add it to your `Cargo.toml`.

```toml
[dependencies]
libfunnel = "0.1.0"
```

You'll also need the C libfunnel library installed on your system.

## Usages

Here's an rough outline of how you would integrate this library into the draw loop of a vulkan app. A fully working example can be found in [./examples](./examples).

```rust
use libfunnel::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create funnel context and stream
    let ctx = FunnelContext::new()?;
    let mut stream = ctx.create_stream(c"MyStream")?;

    // Initialize Vulkan integration
    unsafe {
        stream.init_vulkan(vk_instance, vk_physical_device, vk_device)?;
    }
    stream.vk_set_usage(VK_IMAGE_USAGE_TRANSFER_DST_BIT)?;
    stream.vk_add_format(VK_FORMAT_B8G8R8A8_SRGB, true, VK_FORMAT_FEATURE_BLIT_DST_BIT)?;

    // Configure stream
    stream.set_size(1920, 1080)?;
    stream.set_mode(funnel_mode::FUNNEL_ASYNC)?;
    stream.set_rate(funnel_fraction::VARIABLE, 1.into(), 144.into())?;
    stream.configure()?;
    stream.start()?;

    // Render loop
    loop {
        // Try to get a funnel buffer to stream the frame
        let mut funnel_buffer = stream.dequeue()?;

        // Render your application...

        // If we have a funnel buffer, copy to it
        if let Some(buffer) = &mut funnel_buffer {
            let vk_image = buffer.vk_get_image()?;
            let (acquire_sema, release_sema) = unsafe { buffer.vk_get_semaphores()? };
            let fence = unsafe { buffer.vk_get_fence()? };

            // Submit GPU commands with synchronization:
            //   wait_semaphores: [..., acquire_sema]
            //   signal_semaphores: [..., release_sema]
            //   fence: fence
        }

        // Enqueue buffer back to stream for PipeWire to send
        if let Some(buffer) = funnel_buffer {
            unsafe { stream.enqueue(buffer)?; }
        }
    }
}
```
