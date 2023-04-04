use crate::graph::resource::ResourceType;

/// Represents a virtual resource in the system, uniquely identified by a string.
#[derive(Debug, Default, Clone, Hash, Eq, PartialEq)]
pub struct VirtualResource {
    pub(crate) uid: String,
    ty: ResourceType,
}

impl VirtualResource {
    /// Create a new image virtual resource. Note that the name should not contain any '+' characters.
    pub fn image(uid: impl Into<String>) -> Self {
        VirtualResource {
            uid: uid.into(),
            ty: ResourceType::Image,
        }
    }

    /// Create a new buffer virtual resource. Note that the name should not contain any '+' characters.
    pub fn buffer(uid: impl Into<String>) -> Self {
        VirtualResource {
            uid: uid.into(),
            ty: ResourceType::Buffer,
        }
    }

    /// 'Upgrades' the resource to a new version of itself. This is used to obtain the virtual resource name of an input resource after
    /// a task completes.
    pub fn upgrade(&self) -> Self {
        VirtualResource {
            uid: self.uid.clone() + "+",
            ty: self.ty,
        }
    }

    /// Returns the full, original name of the resource (without potential version star symbols)
    pub fn name(&self) -> String {
        let mut name = self.uid.clone();
        name.retain(|c| c != '+');
        name
    }

    /// Returns the version of a resource, the larger this the more recent the version of the resource is.
    pub fn version(&self) -> usize {
        self.uid.matches('+').count()
    }

    /// Returns true if the resource is a source resource, e.g. an instance that does not depend on a previous pass.
    pub fn is_source(&self) -> bool {
        // ends_with is a bit more efficient, since we know the '+' is always at the end of a resource uid.
        !self.uid.ends_with('+')
    }

    /// Two virtual resources are associated if and only if their uid's only differ by "+" symbols.
    pub fn is_associated_with(&self, rhs: &VirtualResource) -> bool {
        // Since virtual resource uid's are constructed by appending * symbols, we can simply check whether the largest of the two strings starts with the shorter one
        let larger = if self.uid.len() >= rhs.uid.len() {
            self
        } else {
            rhs
        };
        let smaller = if self.uid.len() < rhs.uid.len() {
            self
        } else {
            rhs
        };
        larger.uid.starts_with(&smaller.uid)
    }

    /// One virtual resource is older than another if it has less '+' symbols.
    pub fn is_older(lhs: &VirtualResource, rhs: &VirtualResource) -> bool {
        if !lhs.is_associated_with(rhs) {
            return false;
        }
        lhs.uid.len() < rhs.uid.len()
    }

    /// Note that this is not the same as inverting the result of as_older(), for the same exact state of the resource,
    /// both of these functions should return false (they decide whether resources are strictly older or younger than each other).
    pub fn is_younger(lhs: &VirtualResource, rhs: &VirtualResource) -> bool {
        if !lhs.is_associated_with(rhs) {
            return false;
        }
        rhs.uid.len() < lhs.uid.len()
    }

    /// Get the resource type of this virtual resource
    pub fn resource_type(&self) -> ResourceType {
        self.ty
    }

    /// Get the uid of this virtual resource
    pub fn uid(&self) -> &String {
        &self.uid
    }
}
