//! libfunnel is a library to make creating PipeWire video streams easy, using zero-copy DMA-BUF frame sharing. "Spout2 / Syphon, but for Linux".
//!
//! This crate provides bindings for the C library. You can find the upstream documentation here:
//! - Docs: <https://libfunnel.readthedocs.io>
//! - Github: <https://github.com/hoshinolina/libfunnel>
//!
//! # Example
//!
//! ```no_run
//! use libfunnel::*;
//! use std::ffi::CStr;
//!
//! # const FORMAT_FEATURE_BLIT_DST: u32 = 0;
//! # const FORMAT_B8G8R8A8_SRGB: u32 = 0;
//! # const TRANSFER_DST: u32 = 0;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create funnel context and stream
//! let ctx = FunnelContext::new()?;
//! let mut stream = ctx.create_stream(c"MyStream")?;
//! # let vk_instance = std::ptr::null_mut();
//! # let vk_physical_device = std::ptr::null_mut();
//! # let vk_device = std::ptr::null_mut();
//!
//! // Initialize Vulkan integration
//! unsafe { stream.init_vulkan(vk_instance, vk_physical_device, vk_device)?; }
//! stream.vk_set_usage(TRANSFER_DST)?;
//! stream.vk_add_format(FORMAT_B8G8R8A8_SRGB, true, FORMAT_FEATURE_BLIT_DST)?;
//!
//! // Configure stream
//! stream.set_size(1920, 1080)?;
//! stream.set_mode(FunnelMode::Async)?;
//! stream.set_rate(Fraction::VARIABLE, 1.into(), 144.into())?;
//! stream.configure()?;
//! stream.start()?;
//!
//! // Render loop
//! loop {
//!     // Try to get a funnel buffer to stream the frame (may return None in ASYNC mode)
//!     let mut funnel_buffer = stream.dequeue()?;
//!
//!     // Render your application
//!
//!     // If we have a funnel buffer, add blit commands to copy to it
//!     if let Some(buffer) = &mut funnel_buffer {
//!         let vk_image = buffer.vk_get_image()?;
//!         let (acquire_sema, release_sema) = unsafe { buffer.vk_get_semaphores()? };
//!         let fence = unsafe { buffer.vk_get_fence()? };
//!
//!         // Record commands to copy to vk_image
//!     }
//!
//!     // Submit to queue with synchronization
//!     //   wait_semaphores: [..., acquire_sema, ...]
//!     //   signal_semaphores: [..., release_sema, ...]
//!     //   fence: fence
//!     // vkQueueSubmit(queue, &submit_info, fence);
//!
//!     // Enqueue buffer back to stream for PipeWire to send
//!     if let Some(buffer) = funnel_buffer {
//!         unsafe { stream.enqueue(buffer)?; }
//!     }
//! }
//! # }
//! ```
//!
//! # Usage Notes
//!
//! The general design of libfunnel synchronization is roughly that:
//! - Streams may be created and managed by independent threads
//! - Each unique stream must be configured by a single thread (or with external locking)
//! - Stream data processing (dequeing/enqueuing buffers) may happen in a different thread (or multiple threads, in principle)
//! - Stream status (start/stop/skip frame) may also be managed by arbitrary threads
//!
//! Internally, libfunnel uses a single PipeWire thread loop per [`FunnelContext`], and synchronization happens using a context-global lock. Therefore, if your application has multiple completely independent streams that have no relation to each other and are managed by different threads, it may be more efficient to create a whole new [`FunnelContext`] for each thread, and therefore have independent PipeWire daemon connections and thread loops. This is particularly relevant if you are using [`FunnelMode::Synchronous`] mode, since in that mode the PipeWire processing thread is completely blocked while any stream has a buffer dequeued.

use std::{ffi::CStr, marker::PhantomData};

use crate::bindings::{
    VkDevice, VkFence, VkFormat, VkFormatFeatureFlagBits, VkImage, VkImageUsageFlagBits,
    VkInstance, VkPhysicalDevice, VkSemaphore,
};

#[allow(nonstandard_style, dead_code)]
mod bindings;

use bindings::funnel_fraction;

/// Synchronization modes for the frame pacing
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum FunnelMode {
    /// Produce frames asynchronously to the consumer.
    ///
    /// In this mode, libfunnel calls never block and you
    /// must be able to handle the lack of a buffer (by
    /// skipping rendering/copying to it). This mode only
    /// makes sense if your application is FPS-limited by
    /// some other consumer (for example, if it renders to
    /// the screen, usually with VSync). You should configure
    /// the frame rate you expect to produce frames at with
    /// [`FunnelStream::set_rate`].
    ///
    /// This mode essentially behaves like triple buffering.
    /// Whenever the PipeWire cycle runs, the consumer will
    /// receive the frame that was most recently submitted
    /// to [`FunnelStream::enqueue`].
    Async,
    /// Produce frames synchronously to the consumer with
    /// double buffering.
    ///
    /// In this mode, after a frame is produced, it is
    /// queued to be sent out to the consumer in the next
    /// PipeWire process cycle, and you may immediately
    /// dequeue a new buffer to start rendering the next
    /// frame. libfunnel will block at [`FunnelStream::enqueue`]
    /// until the previously queued frame has been consumed.
    /// In this mode, [`FunnelStream::dequeue`] will only
    /// block if there are no free buffers (if the consumer is
    /// not freeing buffers quickly enough).
    ///
    /// This mode effectively adds two frames of latency,
    /// as up to two frames can be rendered ahead of the
    /// PipeWire cycle (one ready to be submitted, and
    /// one blocked at [`FunnelStream::enqueue`]).
    DoubleBuffered,
    /// Produce frames synchronously to the consumer with
    /// single buffering.
    ///
    /// In this mode, after a frame is produced, it is
    /// queued to be sent out to the consumer in the next
    /// PipeWire process cycle. When you are ready to begin
    /// rendering a new frame, libfunnel will block
    /// at [`FunnelStream::dequeue`] until the previous frame
    /// has been sent to the consumer. In this mode,
    /// [`FunnelStream::enqueue`] will never block.
    ///
    /// This mode effectively adds one frame of latency,
    /// as only one frame can be rendered ahead of the
    /// PipeWire cycle.
    SingleBuffered,
    /// Produce frames synchronously with the PipeWire process
    /// cycle.
    ///
    /// In this mode, [`FunnelStream::dequeue`] will wait for
    /// the beginning of a PipeWire process cycle, and the
    /// process cycle will be blocked until the frame is
    /// submitted with [`FunnelStream::enqueue`].
    ///
    /// This mode provides the lowest possible latency, but
    /// is only suitable for applications that do not do much
    /// work to render frames (for example, just a copy), as
    /// the PipeWire graph will be blocked while the buffer
    /// is dequeued. It adds no latency.
    Synchronous,
}

impl From<FunnelMode> for bindings::funnel_mode {
    fn from(mode: FunnelMode) -> Self {
        match mode {
            FunnelMode::Async => bindings::funnel_mode::FUNNEL_ASYNC,
            FunnelMode::DoubleBuffered => bindings::funnel_mode::FUNNEL_DOUBLE_BUFFERED,
            FunnelMode::SingleBuffered => bindings::funnel_mode::FUNNEL_SINGLE_BUFFERED,
            FunnelMode::Synchronous => bindings::funnel_mode::FUNNEL_SYNCHRONOUS,
        }
    }
}

/// Buffer synchronization mode for frames
///
/// See [buffersync](https://libfunnel.readthedocs.io/en/latest/buffersync.html) for more information on sync modes.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum FunnelSync {
    /// Use implicit sync only.
    Implicit,
    /// Use explicit sync only.
    Explicit,
    /// Support both implicit and explicit sync.
    Both,
}

impl From<FunnelSync> for bindings::funnel_sync {
    fn from(sync: FunnelSync) -> Self {
        match sync {
            FunnelSync::Implicit => bindings::funnel_sync::FUNNEL_SYNC_IMPLICIT,
            FunnelSync::Explicit => bindings::funnel_sync::FUNNEL_SYNC_EXPLICIT,
            FunnelSync::Both => bindings::funnel_sync::FUNNEL_SYNC_BOTH,
        }
    }
}

impl From<bindings::funnel_sync> for FunnelSync {
    fn from(sync: bindings::funnel_sync) -> Self {
        match sync {
            bindings::funnel_sync::FUNNEL_SYNC_IMPLICIT => FunnelSync::Implicit,
            bindings::funnel_sync::FUNNEL_SYNC_EXPLICIT => FunnelSync::Explicit,
            bindings::funnel_sync::FUNNEL_SYNC_BOTH => FunnelSync::Both,
        }
    }
}

/// A rational frame rate
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fraction {
    pub num: u32,
    pub den: u32,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub struct Error {
    pub code: i32,
}
impl std::error::Error for Error {}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "funnel error code {}", self.code)
    }
}

fn check(code: i32) -> Result<(), Error> {
    if code >= 0 {
        Ok(())
    } else {
        Err(Error { code })
    }
}

pub struct FunnelContext {
    ctx: *mut bindings::funnel_ctx,
}

impl FunnelContext {
    /// Create a Funnel context.
    ///
    /// As multiple Funnel contexts are completely independent, this function has no
    /// synchronization requirements.
    ///
    /// # Errors
    ///
    /// * `-ECONNREFUSED` - Failed to connect to PipeWire daemon
    pub fn new() -> Result<FunnelContext> {
        let mut ctx = std::ptr::null_mut();
        check(unsafe { bindings::funnel_init(&mut ctx) })?;
        Ok(FunnelContext { ctx })
    }

    /// Shut down a Funnel context.
    pub fn shutdown(self) {
        unsafe {
            bindings::funnel_shutdown(self.ctx);
        }
    }

    /// Create a new stream.
    ///
    /// # Errors
    ///
    /// * `-EIO` - The PipeWire context is invalid (fatal error)
    pub fn create_stream(&self, name: &CStr) -> Result<FunnelStream> {
        let mut stream = std::ptr::null_mut();
        unsafe {
            check(bindings::funnel_stream_create(
                self.ctx,
                name.as_ptr(),
                &mut stream,
            ))?;
        }
        Ok(FunnelStream { stream })
    }
}

pub struct FunnelStream {
    stream: *mut bindings::funnel_stream,
}
impl FunnelStream {
    /// Specify callbacks for buffer creation/destruction.
    pub fn set_buffer_callbacks<CB: BufferCallbacks>(&mut self, data: *mut CB::UserData) {
        unsafe {
            bindings::funnel_stream_set_buffer_callbacks(
                self.stream,
                Some(CB::callback_alloc),
                Some(CB::callback_free),
                data.cast(),
            );
        }
    }

    /// Set the frame dimensions for a stream.
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - Invalid argument
    pub fn set_size(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe { check(bindings::funnel_stream_set_size(self.stream, width, height)) }
    }

    /// Configure the queueing mode for the stream.
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - Invalid argument
    pub fn set_mode(&mut self, mode: FunnelMode) -> Result<()> {
        unsafe { check(bindings::funnel_stream_set_mode(self.stream, mode.into())) }
    }

    /// Configure the synchronization modes for the stream.
    ///
    /// See [buffersync](https://libfunnel.readthedocs.io/en/latest/buffersync.html) for more information on sync modes.
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - The selected sync combination is invalid for this API
    /// * `-EOPNOTSUPP` - The API/driver does not support this sync mode
    pub fn set_sync(&mut self, frontend: FunnelSync, backend: FunnelSync) -> Result<()> {
        unsafe {
            check(bindings::funnel_stream_set_sync(
                self.stream,
                frontend.into(),
                backend.into(),
            ))
        }
    }

    /// Set the frame rate of a stream.
    ///
    /// # Arguments
    ///
    /// * `default` - Default frame rate ([`Fraction::VARIABLE`] for no default or variable)
    /// * `min` - Minimum frame rate ([`Fraction::VARIABLE`] if variable)
    /// * `max` - Maximum frame rate ([`Fraction::VARIABLE`] if variable)
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - Invalid argument
    pub fn set_rate(&mut self, default: Fraction, min: Fraction, max: Fraction) -> Result<()> {
        unsafe {
            check(bindings::funnel_stream_set_rate(
                self.stream,
                default.into(),
                min.into(),
                max.into(),
            ))
        }
    }

    /// Get the currently negotiated frame rate of a stream.
    ///
    /// # Errors
    ///
    /// * `-EINPROGRESS` - The stream is not yet initialized
    pub fn get_rate(&self) -> Result<Fraction> {
        let mut fraction = funnel_fraction { num: 0, den: 0 };
        unsafe {
            check(bindings::funnel_stream_get_rate(
                self.stream,
                &raw mut fraction,
            ))?;
        }
        Ok(Fraction {
            num: fraction.num,
            den: fraction.den,
        })
    }

    /// Clear the supported format list. Used for reconfiguration.
    pub fn clear_formats(&mut self) {
        unsafe { bindings::funnel_stream_clear_formats(self.stream) }
    }

    /// Apply the stream configuration and register the stream with PipeWire.
    ///
    /// If called on an already configured stream, this will update the
    /// configuration.
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - The stream is in an invalid state (missing settings)
    /// * `-EIO` - The PipeWire context is invalid or stream creation failed
    pub fn configure(&mut self) -> Result<()> {
        unsafe { check(bindings::funnel_stream_configure(self.stream)) }
    }

    /// Start running a stream.
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - The stream is in an invalid state (not configured)
    /// * `-EIO` - The PipeWire context is invalid or stream creation failed
    pub fn start(&self) -> Result<()> {
        unsafe { check(bindings::funnel_stream_start(self.stream)) }
    }

    /// Stop running a stream.
    ///
    /// If another thread is blocked on [`FunnelStream::dequeue`], this will unblock it.
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - The stream is not started
    /// * `-EIO` - The PipeWire context is invalid
    pub fn stop(&self) -> Result<()> {
        unsafe { check(bindings::funnel_stream_stop(self.stream)) }
    }

    /// Destroy a stream.
    ///
    /// The stream will be stopped if it is running.
    pub fn destroy(self) {
        unsafe { bindings::funnel_stream_destroy(self.stream) }
    }

    /// Dequeue a buffer from a stream.
    ///
    /// Note that, currently, you may only have one buffer
    /// dequeued at a time.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(buffer))` - A buffer was successfully dequeued
    /// * `Ok(None)` - No buffer is available
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - Stream is in an invalid state
    /// * `-EBUSY` - Attempted to dequeue more than one buffer at once
    /// * `-EIO` - The PipeWire context is invalid
    /// * `-ESHUTDOWN` - Stream is not started
    pub fn dequeue(&self) -> Result<Option<FunnelBuffer<'_>>> {
        let mut buffer = std::ptr::null_mut();
        let ret = unsafe { bindings::funnel_stream_dequeue(self.stream, &mut buffer) };
        match ret {
            0 => Ok(None),
            1 => Ok(Some(FunnelBuffer {
                buffer,
                _marker: PhantomData,
            })),
            code => Err(Error { code }),
        }
    }

    /// Enqueue a buffer to a stream.
    ///
    /// After this call, the buffer is no longer owned by the user and may not be
    /// queued again until it is dequeued.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - The buffer was successfully enqueued
    /// * `Ok(false)` - The buffer was dropped because the stream configuration or state changed
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - Invalid argument, stream is in an invalid state (not yet configured), or buffer requires sync but sync was not handled properly
    /// * `-EIO` - The PipeWire context is invalid
    /// * `-ESHUTDOWN` - Stream is not started
    ///
    /// # Safety
    /// - todo
    // TODO: does buf need to be borrowed from this stream?
    pub unsafe fn enqueue(&self, buffer: FunnelBuffer<'_>) -> Result<bool> {
        let ret = unsafe { bindings::funnel_stream_enqueue(self.stream, buffer.buffer) };
        match ret {
            0 => Ok(false),
            1 => Ok(true),
            code => Err(Error { code }),
        }
    }

    /// Skip a frame for a stream.
    ///
    /// This call forces at least one subsequent call to [`FunnelStream::dequeue`]
    /// to return without a buffer. This is useful to break a thread out of
    /// that function.
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - Stream is in an invalid state (not yet configured)
    pub fn skip_frame(&self) -> Result<()> {
        unsafe { check(bindings::funnel_stream_skip_frame(self.stream)) }
    }

    /// Set up a stream for Vulkan integration.
    ///
    /// # Errors
    ///
    /// * `-EEXIST` - The API was already initialized once
    /// * `-ENOTSUP` - Missing Vulkan extensions
    /// * `-ENODEV` - Could not locate DRM render node, or GBM or Vulkan initialization failed
    ///
    /// # Safety
    /// - todo
    pub unsafe fn init_vulkan(
        &mut self,
        instance: VkInstance,
        phyical_device: VkPhysicalDevice,
        device: VkDevice,
    ) -> Result<()> {
        unsafe {
            check(bindings::funnel_stream_init_vulkan(
                self.stream,
                instance,
                phyical_device,
                device,
            ))
        }
    }

    /// Set the required buffer usage. This will control the usage for
    /// images allocated by libfunnel.
    ///
    /// [`FunnelStream::vk_add_format`] will fail if the requested usages
    /// are not available. In this case, you may reconfigure the usage
    /// and try again.
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - Invalid argument or API is not Vulkan
    pub fn vk_set_usage(&mut self, usage: VkImageUsageFlagBits) -> Result<()> {
        unsafe { check(bindings::funnel_stream_vk_set_usage(self.stream, usage)) }
    }

    /// Add a supported Vulkan format. Must be called in preference order (highest to
    /// lowest). Only some formats are supported by libfunnel:
    ///
    /// - `VK_FORMAT_R8G8B8A8_SRGB`
    /// - `VK_FORMAT_R8G8B8A8_UNORM`
    /// - `VK_FORMAT_B8G8R8A8_SRGB`
    /// - `VK_FORMAT_B8G8R8A8_UNORM`
    ///
    /// The corresponding UNORM variants are also acceptable, and equivalent.
    /// [`FunnelBuffer::vk_get_format`] will always return the SRGB formats. If
    /// you need UNORM (because you are doing sRGB/gamma conversion in your shader),
    /// you can use UNORM constants when you create a VkImageView.
    ///
    /// # Arguments
    ///
    /// * `format` - VkFormat
    /// * `alpha` - Whether alpha is meaningful or ignored
    /// * `feature` - Required VkFormatFeatureFlagBits. Adding a format will fail if the requested features are not available.
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - Invalid argument or API is not Vulkan
    /// * `-ENOTSUP` - VkFormat is not supported by libfunnel
    /// * `-ENOENT` - VkFormat is not supported by the device or not usable
    pub fn vk_add_format(
        &mut self,
        format: VkFormat,
        alpha: bool,
        feature: VkFormatFeatureFlagBits,
    ) -> Result<()> {
        unsafe {
            check(bindings::funnel_stream_vk_add_format(
                self.stream,
                format,
                alpha,
                feature,
            ))
        }
    }
}

pub struct FunnelBuffer<'stream> {
    buffer: *mut bindings::funnel_buffer,
    _marker: PhantomData<&'stream FunnelStream>,
}

impl<'stream> FunnelBuffer<'stream> {
    /// Get the dimensions of a Funnel buffer.
    pub fn get_size(&mut self) -> (u32, u32) {
        let mut width = 0;
        let mut height = 0;
        unsafe { bindings::funnel_buffer_get_size(self.buffer, &mut width, &mut height) }
        (width, height)
    }

    /// Set an arbitrary user data pointer for a buffer.
    ///
    /// The user is responsible for managing the lifetime of this object.
    /// Generally, you should use [`FunnelStream::set_buffer_callbacks`]
    /// to provide buffer creation/destruction callbacks, and set and
    /// release the user data pointer in the alloc and free callback
    /// respectively.
    ///
    /// # Safety
    /// - todo
    pub unsafe fn set_user_data(&mut self, data: *mut ()) {
        unsafe { bindings::funnel_buffer_set_user_data(self.buffer, data.cast()) };
    }

    /// Get an arbitrary user data pointer for a buffer.
    pub fn get_user_data(&mut self) -> *mut () {
        unsafe { bindings::funnel_buffer_get_user_data(self.buffer).cast() }
    }

    /// Check whether a buffer requires explicit synchronization.
    pub fn has_sync(&mut self) -> bool {
        unsafe { bindings::funnel_buffer_has_sync(self.buffer) }
    }

    /// Return whether a buffer is considered efficient for rendering.
    ///
    /// Buffers are considered efficient when they are not using linear tiling
    /// and non-linear tiling is supported by the GPU driver.
    pub fn is_efficient_for_rendering(&mut self) -> bool {
        unsafe { bindings::funnel_buffer_is_efficient_for_rendering(self.buffer) }
    }

    /// Get the VkImage for a Funnel buffer.
    ///
    /// The VkImage is only valid while the buffer is dequeued, or before the destroy
    /// callback is used (if you use buffer callbacks).
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - Invalid argument or API is not Vulkan
    ///
    /// # Safety
    /// -  todo
    pub fn vk_get_image(&mut self) -> Result<VkImage> {
        let mut image = std::ptr::null_mut();
        unsafe {
            check(bindings::funnel_buffer_get_vk_image(
                self.buffer,
                &mut image,
            ))?;
        }
        Ok(image)
    }

    /// Get the VkFormat for a Funnel buffer.
    ///
    /// Returns a tuple of (VkFormat, has_alpha) where has_alpha indicates whether alpha is enabled.
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - Invalid argument or API is not Vulkan
    /// * `-EIO` - Format is unsupported (internal error)
    pub fn vk_get_format(&mut self) -> Result<(VkFormat, bool)> {
        let mut format = VkFormat::default();
        let mut has_alpha = false;
        unsafe {
            check(bindings::funnel_buffer_get_vk_format(
                self.buffer,
                &mut format,
                &mut has_alpha,
            ))?;
        }
        Ok((format, has_alpha))
    }

    /// Get the VkSemaphores for acquiring and releasing the buffer.
    ///
    /// The user must wait on the acquire VkSemaphore object before accessing
    /// the buffer, and signal the release VkSemaphore after accessing the buffer.
    /// These semaphores are valid while the buffer is dequeued.
    ///
    /// Returns a tuple of (acquire, release) semaphores.
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - Invalid argument or API is not Vulkan
    /// * `-EBUSY` - Already called once for this buffer
    /// * `-EIO` - Failed to import acquire semaphore into Vulkan
    ///
    /// # Safety
    /// - todo
    pub unsafe fn vk_get_semaphores(&mut self) -> Result<(VkSemaphore, VkSemaphore)> {
        let mut acquire = std::ptr::null_mut();
        let mut release = std::ptr::null_mut();
        unsafe {
            check(bindings::funnel_buffer_get_vk_semaphores(
                self.buffer,
                &mut acquire,
                &mut release,
            ))?;
        }
        Ok((acquire, release))
    }

    /// Get the VkFence that must be signaled by the queue batch.
    ///
    /// The user must pass this fence to vkQueueSubmit() (or similar),
    /// such that it is signaled when all operations on the buffer
    /// are complete. This fence is valid while the buffer is
    /// dequeued.
    ///
    /// # Errors
    ///
    /// * `-EINVAL` - Invalid argument or API is not Vulkan
    /// * `-EBUSY` - Already called once for this buffer
    ///
    /// # Safety
    /// - todo
    pub unsafe fn vk_get_fence(&mut self) -> Result<VkFence> {
        let mut fence = std::ptr::null_mut();
        unsafe {
            check(bindings::funnel_buffer_get_vk_fence(
                self.buffer,
                &mut fence,
            ))?;
        }
        Ok(fence)
    }
}

impl Fraction {
    /// Indicates that the frame rate is variable
    pub const VARIABLE: Fraction = Fraction { num: 0, den: 1 };

    /// Create a new fraction
    pub const fn new(num: u32, den: u32) -> Self {
        Fraction { num, den }
    }
}

impl From<u32> for Fraction {
    fn from(value: u32) -> Self {
        Fraction { num: value, den: 1 }
    }
}

impl From<Fraction> for funnel_fraction {
    fn from(f: Fraction) -> Self {
        funnel_fraction {
            num: f.num,
            den: f.den,
        }
    }
}

pub trait BufferCallbacks {
    type UserData: Send + Sync + 'static; // TODO: can this be relaxed?

    #[doc(hidden)]
    unsafe extern "C" fn callback_alloc(
        opaque: *mut ::std::os::raw::c_void,
        stream: *mut bindings::funnel_stream,
        buffer: *mut bindings::funnel_buffer,
    ) {
        let stream = FunnelStream { stream };
        let mut buffer = FunnelBuffer {
            buffer,
            _marker: PhantomData,
        };
        Self::on_alloc(opaque.cast(), &stream, &mut buffer);
    }
    #[doc(hidden)]
    unsafe extern "C" fn callback_free(
        opaque: *mut ::std::os::raw::c_void,
        stream: *mut bindings::funnel_stream,
        buffer: *mut bindings::funnel_buffer,
    ) {
        let stream = FunnelStream { stream };
        let mut buffer = FunnelBuffer {
            buffer,
            _marker: PhantomData,
        };
        Self::on_free(opaque.cast(), &stream, &mut buffer);
    }

    fn on_alloc<'stream>(
        userdata: *mut Self::UserData,
        stream: &'stream FunnelStream,
        buffer: &mut FunnelBuffer<'stream>,
    );
    fn on_free<'stream>(
        userdata: *mut Self::UserData,
        stream: &'stream FunnelStream,
        buffer: &mut FunnelBuffer<'stream>,
    );
}

/*
# Lifetimes

For input arguments (the default if not specified):

    borrowed: The object is owned by the caller and borrowed by the function call. After the call, the caller remains responsible for releasing the object at some point.
    borrowed-by user : The object is owned by the caller and borrowed by libfunnel object user. The caller must release user before it releases this object.
    owned: The object ownership is transferred to libfunnel. After the call, the caller must no longer use nor release the object.

For output arguments:

    borrowed-from parent : The object is owned by the parent object parent. Once ownership of parent is transferred back to libfunnel, the borrowed object becomes invalid and may no longer be used.
    owned: The object ownership is transferred to the caller. After the call, the caller must eventually release the object.
    owned-from: The object ownership is transferred to the caller, but it is a child object of parent. The object must be released by the caller before parent is released.

# Lifetime

An object may only be passed to a function as [Lifetime: <b>owned</b>] after all calls that receive it as [Lifetime: <b>borrowed</b>] complete
An object may only be released after all borrows cease and all child objects are released


external: The function must not be called concurrently with other functions also marked external borrowing the same object (typically the first parameter)
internal: The function may be called concurrently with other functions (external and internal) borrowing the same object (typically the first parameter)


*/
