use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use crate::execution_manager::domain::*;
use crate::execution_manager::domain;

use ash::vk;
use ash::vk::{Rect2D, Viewport};
use crate::{BufferView, DescriptorCache, DescriptorSet, DescriptorSetBinding, Device, Error, ExecutionManager, ImageView, PipelineCache};

use anyhow::Result;

/// Trait representing a command buffer that supports graphics commands.
pub trait GraphicsCmdBuffer : TransferCmdBuffer {
    /// Sets the viewport. The equivalent of `vkCmdSetViewport`.
    fn viewport(self, viewport: vk::Viewport) -> Self;
    /// Sets the scissor region. Equivalent of `vkCmdSetScissor`.
    fn scissor(self, scissor: vk::Rect2D) -> Self;
    /// Record a single drawcall. Equivalent of `vkCmdDraw`.
    fn draw(self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> Self;
    /// Bind a graphics pipeline with a given name. This is looked up from the given pipeline cache.
    /// # Errors
    /// This function can report an error in case the pipeline name is not registered in the cache.
    fn bind_graphics_pipeline(self, name: &str, cache: Arc<Mutex<PipelineCache>>) -> Result<Self> where Self: Sized;
    /// Bind a vertex buffer to the given vertex input binding.
    /// Equivalent of `vkCmdBindVertexBuffer`
    fn bind_vertex_buffer(self, binding: u32, buffer: BufferView) -> Self where Self: Sized;
}

/// Trait representing a command buffer that supports transfer commands.
pub trait TransferCmdBuffer {

}

/// Trait representing a command buffer that supports compute commands.
pub trait ComputeCmdBuffer : TransferCmdBuffer {
    
}

/// This struct represents a finished command buffer. This command buffer can't be recorded to anymore.
/// It can only be obtained by calling finish() on an incomplete command buffer;
pub struct CommandBuffer<D: ExecutionDomain> {
    pub(crate) handle: vk::CommandBuffer,
    _domain: PhantomData<D>,
}

/// Completed command buffer
pub trait CmdBuffer {
    /// Delete the command buffer immediately.
    /// This is marked unsafe because there is no guarantee that the command buffer is not in use.
    unsafe fn delete(&mut self, exec: &ExecutionManager) -> Result<()>;
}

/// Incomplete command buffer
pub trait IncompleteCmdBuffer {
    type Domain: ExecutionDomain;

    fn new(device: Arc<Device>, handle: vk::CommandBuffer, flags: vk::CommandBufferUsageFlags) -> Result<Self> where Self: Sized;
    fn finish(self) -> Result<CommandBuffer<Self::Domain>>;
}

/// This struct represents an incomplete command buffer.
/// This is a command buffer that has not been called [`IncompleteCommandBuffer::finish()`] on yet.
/// Calling this method will turn it into an immutable command buffer which can then be submitted
/// to the queue it was allocated from. See also [`ExecutionManager`].
///
/// # Example
/// ```
/// use phobos::{domain, ExecutionManager};
///
/// let exec = ExecutionManager::new(device.clone(), &physical_device);
/// let cmd = exec.on_domain::<domain::All>()?
///               // record commands to this command buffer
///               // ...
///               // convert into a complete command buffer by calling finish().
///               // This allows the command buffer to be submitted.
///               .finish();
/// ```
#[derive(Derivative)]
#[derivative(Debug)]
pub struct IncompleteCommandBuffer<D: ExecutionDomain> {
    #[derivative(Debug="ignore")]
    device: Arc<Device>,
    handle: vk::CommandBuffer,
    current_pipeline_layout: vk::PipelineLayout,
    current_set_layouts: Vec<vk::DescriptorSetLayout>,
    current_bindpoint: vk::PipelineBindPoint, // TODO: Note: technically not correct
    _domain: PhantomData<D>,
}

impl<D: ExecutionDomain> IncompleteCmdBuffer for IncompleteCommandBuffer<D> {
    type Domain = D;

    fn new(device: Arc<Device>, handle: vk::CommandBuffer, flags: vk::CommandBufferUsageFlags) -> Result<Self> {
        unsafe { device.begin_command_buffer(
            handle,
            &vk::CommandBufferBeginInfo::builder().flags(flags))?
        };
        Ok(IncompleteCommandBuffer {
            device: device.clone(),
            handle,
            current_pipeline_layout: vk::PipelineLayout::null(),
            current_set_layouts: vec![],
            current_bindpoint: vk::PipelineBindPoint::default(),
            _domain: PhantomData
        })
    }

    /// Finish recording a command buffer and move its contents into a finished
    /// command buffer that can be submitted
    fn finish(self) -> Result<CommandBuffer<D>> {
        unsafe { self.device.end_command_buffer(self.handle)? }
        Ok(CommandBuffer {
            handle: self.handle,
            _domain: PhantomData,
        })
    }
}

impl<D: ExecutionDomain> CmdBuffer for CommandBuffer<D> {
    unsafe fn delete(&mut self, exec: &ExecutionManager) -> Result<()> {
        let queue = exec.get_queue::<D>().ok_or(Error::NoCapableQueue)?;
        queue.free_command_buffer::<Self>(self.handle)
    }
}

impl<D: ExecutionDomain> IncompleteCommandBuffer<D> {
    /// Obtain a reference to a descriptor set inside the given cache that stores the requested bindings.
    /// This potentially allocates a new descriptor set and writes to it.
    /// # Lifetime
    /// The returned descriptor set lives as long as the cache allows it to live, so it should not be stored across multiple frames.
    /// # Errors
    /// - This function can potentially error if allocating the descriptor set fails. The descriptor cache has builtin ways to handle
    /// a full descriptor pool, but other errors are passed through.
    /// - This function errors if a requested set was not specified in the current pipeline's pipeline layout.
    /// - This function errors if no pipeline is bound.
    pub fn get_descriptor_set<'a>(&mut self, set: u32, mut bindings: DescriptorSetBinding, cache: &'a mut DescriptorCache) -> Result<&'a DescriptorSet> {
        let layout = self.current_set_layouts.get(set as usize).ok_or(Error::NoDescriptorSetLayout)?;
        bindings.layout = layout.clone();
        cache.get_descriptor_set(bindings)
    }

    /// Bind a descriptor set to the command buffer. This descriptor set can be obtained by calling
    /// [`Self::get_descriptor_set`]. Note that the index should be the same as the one used in get_descriptor_set.
    /// To safely do this, prefer using [`Self::bind_new_descriptor_set`] to combine these two functions into one call.
    pub fn bind_descriptor_set(self, index: u32, set: &DescriptorSet) -> Self {
        unsafe {
            self.device.cmd_bind_descriptor_sets(self.handle, self.current_bindpoint, self.current_pipeline_layout,
                                     index,
                                std::slice::from_ref(&set.handle),
                               &[]); }
        self
    }

    /// Obtain a descriptor set from the cache, and immediately bind it to the given index.
    /// # Errors
    /// - This function can error if locking the descriptor cache fails
    /// - This function can error if allocating the descriptor set fails.
    /// # Example
    /// ```
    /// use phobos::{DescriptorSetBuilder, domain, ExecutionManager};
    /// let exec = ExecutionManager::new(device.clone(), &physical_device);    ///
    /// let cmd = exec.on_domain::<domain::All>()?
    ///     .bind_graphics_pipeline("my_pipeline", pipeline_cache.clone())
    ///     .bind_new_descriptor_set(0, descriptor_cache.clone(), DescriptorSetBuilder::new()
    ///         .bind_sampled_image(0, image_view, &sampler)
    ///         .build())
    ///     // ...
    ///     .finish();
    ///
    /// ```
    pub fn bind_new_descriptor_set(mut self, index: u32, cache: Arc<Mutex<DescriptorCache>>, bindings: DescriptorSetBinding, ) -> Result<Self> {
        let mut cache = cache.lock().or_else(|_| Err(anyhow::Error::from(Error::PoisonError)))?;
        let set = self.get_descriptor_set(index, bindings, &mut cache)?;
        Ok(self.bind_descriptor_set(index, set))
    }

    /// Transitions an image layout.
    /// Generally you will not need to call this function manually,
    /// using the render graph api you can do most transitions automatically.
    pub fn transition_image(self, image: &ImageView, src_stage: vk::PipelineStageFlags, dst_stage: vk::PipelineStageFlags,
                            from: vk::ImageLayout, to: vk::ImageLayout,
                            src_access: vk::AccessFlags, dst_access: vk::AccessFlags) -> Self {
        let barrier = vk::ImageMemoryBarrier::builder()
            .image(image.image)
            .subresource_range(image.subresource_range())
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .old_layout(from)
            .new_layout(to)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .build();
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.handle,
                src_stage,
                dst_stage,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                std::slice::from_ref(&barrier)
            );
        }

        self
    }

    /// vkCmdPipelineBarrier2. Prefer using this over regular pipeline barriers if possible, to make
    /// full use of VK_KHR_SYNCHRONIZATION_2.
    pub fn pipeline_barrier_2(self, dependency: &vk::DependencyInfo) -> Self {
        unsafe {
            self.device.cmd_pipeline_barrier2(self.handle, &dependency);
        }

        self
    }

    pub(crate) fn begin_rendering(self, info: &vk::RenderingInfo) -> Self {
        unsafe {
            self.device.cmd_begin_rendering(self.handle, &info);
        }

        self
    }

    pub(crate) fn end_rendering(self) -> Self {
        unsafe {
            self.device.cmd_end_rendering(self.handle);
        }

        self
    }
}

trait GfxSupport : TransferSupport {}
trait TransferSupport {}
trait ComputeSupport : TransferSupport {}

impl GfxSupport for domain::Graphics {}
impl GfxSupport for domain::All {}
impl TransferSupport for domain::Graphics {}
impl TransferSupport for domain::Transfer {}
impl TransferSupport for domain::Compute {}
impl TransferSupport for domain::All {}
impl ComputeSupport for domain::Compute {}
impl ComputeSupport for domain::All {}

impl<D: GfxSupport + ExecutionDomain> GraphicsCmdBuffer for IncompleteCommandBuffer<D> {
    fn viewport(self, viewport: Viewport) -> Self {
        unsafe { self.device.cmd_set_viewport(self.handle, 0, std::slice::from_ref(&viewport)); }
        self
    }

    fn scissor(self, scissor: Rect2D) -> Self {
        unsafe { self.device.cmd_set_scissor(self.handle, 0, std::slice::from_ref(&scissor)); }
        self
    }

    fn draw(self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> Self {
        unsafe { self.device.cmd_draw(self.handle, vertex_count, instance_count, first_vertex, first_instance); }
        self
    }

    fn bind_graphics_pipeline(mut self, name: &str, cache: Arc<Mutex<PipelineCache>>) -> Result<Self> {
        let mut cache = cache.lock().unwrap();
        let pipeline = cache.get_pipeline(name)?;
        unsafe { self.device.cmd_bind_pipeline(self.handle, vk::PipelineBindPoint::GRAPHICS, pipeline.handle); }
        self.current_bindpoint = vk::PipelineBindPoint::GRAPHICS;
        self.current_pipeline_layout = pipeline.layout;
        self.current_set_layouts = pipeline.set_layouts.clone();
        Ok(self)
    }

    fn bind_vertex_buffer(self, binding: u32, buffer: BufferView) -> Self where Self: Sized {
        unsafe { self.device.cmd_bind_vertex_buffers(self.handle, binding, std::slice::from_ref(&buffer.handle), std::slice::from_ref(&buffer.offset)) };
        self
    }
}

impl<D: TransferSupport + ExecutionDomain> TransferCmdBuffer for IncompleteCommandBuffer<D> {
    // Methods for transfer commands
}

impl<D: ComputeSupport + ExecutionDomain> ComputeCmdBuffer for IncompleteCommandBuffer<D> {
    // Methods for compute commands
}