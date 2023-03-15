use std::collections::HashMap;
use crate::{BufferView, Error, ImageView, VirtualResource};

use anyhow::Result;

/// Describes any physical resource handle on the GPU.
#[derive(Debug, Clone)]
pub enum PhysicalResource {
    Image(ImageView),
    Buffer(BufferView),
}

/// Stores bindings from virtual resources to physical resources.
/// # Example usage
/// ```
/// use ash::vk;
/// use phobos::{Error, Image, VirtualResource};
/// use phobos::graph::physical_resource::PhysicalResourceBindings;
///
/// let resource = VirtualResource::new(String::from("image"));
/// let image = Image::new(/*...*/);
/// let view = image.view(vk::ImageAspectFlags::COLOR)?;
/// let mut bindings = PhysicalResourceBindings::new();
/// // Bind the virtual resource to the image
/// bindings.bind_image(String::from("image"), view.clone());
/// // ... Later, lookup the physical image handle from a virtual resource handle
/// let view = bindings.resolve(&resource).ok_or(Error::NoResourceBound)?;
/// ```
#[derive(Debug)]
pub struct PhysicalResourceBindings {
    bindings: HashMap<String, PhysicalResource>
}

impl PhysicalResourceBindings {
    /// Create a new physical resource binding map.
    pub fn new() -> Self {
        PhysicalResourceBindings { bindings: Default::default() }
    }

    /// Bind an image to all virtual resources with `name(+*)` as their uid.
    pub fn bind_image(&mut self, name: impl Into<String>, image: ImageView) {
        self.bindings.insert(name.into(), PhysicalResource::Image(image));
    }

    /// Bind a buffer to all virtual resources with this name as their uid.
    pub fn bind_buffer(&mut self, name: impl Into<String>, buffer: BufferView) { self.bindings.insert(name.into(), PhysicalResource::Buffer(buffer)); }

    /// Alias a resource by giving it an alternative name
    pub fn alias(&mut self, new_name: impl Into<String>, resource: &str) -> Result<()> {
        self.bindings.insert(new_name.into(), self.bindings.get(resource).ok_or(Error::NoResourceBound(resource.to_owned()))?.clone());
        Ok(())
    }

    /// Resolve a virtual resource to a physical resource. Returns `None` if the resource was not found.
    pub fn resolve(&self, resource: &VirtualResource) -> Option<&PhysicalResource> {
        self.bindings.get(&resource.name())
    }
}
