//! Wrappers for acceleration structure triangle geometry data

use ash::vk;

use crate::util::address::DeviceOrHostAddressConst;
use crate::util::to_vk::{AsVulkanType, IntoVulkanType};

/// Triangle data in an acceleration structure
pub struct AccelerationStructureGeometryTrianglesData {
    /// The vertex format
    pub format: vk::Format,
    /// Address of the vertex buffer
    pub vertex_data: DeviceOrHostAddressConst,
    /// Vertex stride in the vertex buffer
    pub stride: vk::DeviceSize,
    /// Highest index of a vertex in the vertex buffer
    pub max_vertex: u32,
    /// Index type, or `vk::IndexType::NONE_KHR` if no index buffer is used.
    pub index_type: vk::IndexType,
    /// Address of the index buffer, this may be null if no index buffer is used
    pub index_data: DeviceOrHostAddressConst,
    /// Address of the buffer with transform data, this may be null
    pub transform_data: DeviceOrHostAddressConst,
    /// Geometry flags
    pub flags: vk::GeometryFlagsKHR,
}

impl AccelerationStructureGeometryTrianglesData {
    /// Create triangle data with default settings
    pub fn default() -> Self {
        Self {
            format: vk::Format::default(),
            vertex_data: DeviceOrHostAddressConst::null_host(),
            stride: 0,
            max_vertex: 0,
            index_type: vk::IndexType::NONE_KHR,
            index_data: DeviceOrHostAddressConst::null_host(),
            transform_data: DeviceOrHostAddressConst::null_host(),
            flags: Default::default(),
        }
    }

    /// Set the vertex data format
    pub fn format(mut self, format: impl Into<vk::Format>) -> Self {
        self.format = format.into();
        self
    }

    /// Set the vertex buffer addres
    pub fn vertex_data(mut self, data: impl Into<DeviceOrHostAddressConst>) -> Self {
        self.vertex_data = data.into();
        self
    }

    /// Set the vertex stride
    pub fn stride(mut self, stride: impl Into<vk::DeviceSize>) -> Self {
        self.stride = stride.into();
        self
    }

    /// Set the highest vertex index
    pub fn max_vertex(mut self, max_vertex: u32) -> Self {
        self.max_vertex = max_vertex;
        self
    }

    /// Set the index data buffer address and its type
    pub fn index_data(
        mut self,
        ty: vk::IndexType,
        data: impl Into<DeviceOrHostAddressConst>,
    ) -> Self {
        self.index_type = ty;
        self.index_data = data.into();
        self
    }

    /// Set the transform data address
    pub fn transform_data(mut self, data: impl Into<DeviceOrHostAddressConst>) -> Self {
        self.transform_data = data.into();
        self
    }

    /// Set the geometry flags
    pub fn flags(mut self, flags: vk::GeometryFlagsKHR) -> Self {
        self.flags = flags;
        self
    }
}

impl IntoVulkanType for AccelerationStructureGeometryTrianglesData {
    type Output = vk::AccelerationStructureGeometryTrianglesDataKHR;

    fn into_vulkan(self) -> Self::Output {
        vk::AccelerationStructureGeometryTrianglesDataKHR {
            s_type: vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            p_next: std::ptr::null(),
            vertex_format: self.format,
            vertex_data: self.vertex_data.as_vulkan(),
            vertex_stride: self.stride,
            max_vertex: self.max_vertex,
            index_type: self.index_type,
            index_data: self.index_data.as_vulkan(),
            transform_data: self.transform_data.as_vulkan(),
        }
    }
}
