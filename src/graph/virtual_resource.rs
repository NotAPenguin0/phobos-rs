//! Exposes the virtual resource utility used to refer to resources in a graph.

use std::fmt::{Display, Formatter};

use crate::graph::resource::ResourceType;

/// Represents a virtual resource in the system, uniquely identified by a string.
///
/// Note that the resource named `swapchain` is assumed to always be the swapchain resource for presenting.
#[derive(Debug, Default, Clone, Hash, Eq, PartialEq)]
pub struct VirtualResource {
    pub(crate) name: String,
    pub(crate) version: usize,
    ty: ResourceType,
}

/// Holds a hashed resource in the pass graph implementation
#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub struct HashedResource {
    /// Hash of the resource
    pub hash: u64,
}

impl Display for HashedResource {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hash)
    }
}

impl VirtualResource {
    pub(crate) fn final_image(name: impl Into<String>) -> Self {
        VirtualResource {
            name: name.into(),
            version: usize::MAX,
            ty: ResourceType::Image,
        }
    }

    /// Create a new image virtual resource.
    pub fn image(name: impl Into<String>) -> Self {
        VirtualResource {
            name: name.into(),
            version: 0,
            ty: ResourceType::Image,
        }
    }

    /// Create a new buffer virtual resource.
    pub fn buffer(name: impl Into<String>) -> Self {
        VirtualResource {
            name: name.into(),
            version: 0,
            ty: ResourceType::Buffer,
        }
    }

    /// 'Upgrades' the resource to a new version of itself. This is used to obtain the virtual resource name of an input resource after
    /// a task completes.
    pub fn upgrade(&self) -> Self {
        VirtualResource {
            name: self.name.clone(),
            version: self.version + 1,
            ty: self.ty,
        }
    }

    /// Returns the full, original name of the resource
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the version of a resource, the larger this the more recent the version of the resource is.
    pub fn version(&self) -> usize {
        self.version
    }

    /// Returns true if the resource is a source resource, e.g. an instance that does not depend on a previous pass.
    pub fn is_source(&self) -> bool {
        self.version() == 0
    }

    /// Check if these virtual resources refer to the same physical resource
    pub fn is_associated_with(&self, rhs: &VirtualResource) -> bool {
        self.name() == rhs.name()
    }

    /// One virtual resource is older than another if it has less '+' symbols.
    pub fn is_older(lhs: &VirtualResource, rhs: &VirtualResource) -> bool {
        if !lhs.is_associated_with(rhs) {
            return false;
        }
        lhs.version() < rhs.version()
    }

    /// Note that this is not the same as inverting the result of as_older(), for the same exact state of the resource,
    /// both of these functions should return false (they decide whether resources are strictly older or younger than each other).
    pub fn is_younger(lhs: &VirtualResource, rhs: &VirtualResource) -> bool {
        if !lhs.is_associated_with(rhs) {
            return false;
        }
        lhs.version() > rhs.version()
    }

    /// Get the resource type of this virtual resource
    pub fn resource_type(&self) -> ResourceType {
        self.ty
    }

    /// Get the uid of this virtual resource
    pub fn uid(&self) -> String {
        format!("{}", self)
    }
}

impl Display for VirtualResource {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.version == usize::MAX {
            write!(f, "{}_final", self.name())
        } else {
            write!(f, "{}{}", self.name(), String::from_utf8(vec![b'+'; self.version]).unwrap())
        }
    }
}

/// Syntax sugar to easily construct image virtual resources
#[macro_export]
macro_rules! image {
    ($id:literal) => {
        ::phobos::prelude::VirtualResource::image($id)
    };
}

/// Syntax sugar to easily construct buffer virtual resources
#[macro_export]
macro_rules! buffer {
    ($id:literal) => {
        ::phobos::prelude::VirtualResource::buffer($id)
    };
}
