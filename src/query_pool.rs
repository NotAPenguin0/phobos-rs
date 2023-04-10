use std::ops::Sub;
use std::time::Duration;

use anyhow::{ensure, Result};
use ash::vk;

use crate::{Device, PipelineStage};

pub trait Query: Clone + Sized {
    const QUERY_TYPE: vk::QueryType;

    type Output;

    fn new(pool: &QueryPoolCreateInfo) -> Self;

    fn size(&self) -> usize;

    fn parse_query(&self, device: &Device, data: &[u64]) -> Self::Output;
}

pub trait ScopedQuery: Query {}

#[derive(Default, Copy, Clone, Debug)]
pub struct Timestamp {
    value: u64,
    // The number of nanoseconds it takes for the timestamp to be incremented
    period: f32,
}

impl Timestamp {
    pub fn raw_value(&self) -> u64 {
        self.value
    }

    pub fn nanoseconds(&self) -> u64 {
        (self.value as f64 * self.period as f64) as u64
    }

    pub fn duration_since_epoch(&self) -> Duration {
        Duration::from_nanos(self.nanoseconds())
    }
}

impl Sub<Timestamp> for Timestamp {
    type Output = Duration;

    fn sub(self, rhs: Timestamp) -> Self::Output {
        Duration::from_nanos(self.nanoseconds() - rhs.nanoseconds())
    }
}

#[derive(Default, Copy, Clone)]
pub struct TimestampQuery {
    valid_bits: u32,
}

impl Query for TimestampQuery {
    const QUERY_TYPE: vk::QueryType = vk::QueryType::TIMESTAMP;

    type Output = Timestamp;

    fn new(_pool: &QueryPoolCreateInfo) -> Self {
        Self::default()
    }

    fn size(&self) -> usize {
        1
    }

    fn parse_query(&self, device: &Device, data: &[u64]) -> Timestamp {
        // We get one value back
        let value = *data.first().unwrap();
        // Now, clear all the invalid bits
        // Algorithm:
        // Suppose 60 bits are valid. Then the upper 64 - 60 = 4 need to be cleared.
        // We can do this by masking with 00001111 ... 1111
        // To construct this mask, we take the max value (1111 ... 1111),
        // and shift it to the right by the number of
        // invalid bits
        let invalid_bits = u64::BITS - self.valid_bits;
        let mask = u64::MAX >> invalid_bits;
        let value = value & mask;
        Timestamp {
            value,
            period: device.properties().limits.timestamp_period,
        }
    }
}

fn num_queries(flags: vk::QueryPipelineStatisticFlags) -> usize {
    flags.as_raw().count_ones() as usize
}

#[derive(Debug, Default, Copy, Clone)]
pub struct PipelineStatistics {
    pub input_assembly_vertices: Option<u64>,
    pub input_assembly_primitives: Option<u64>,
    pub vertex_shader_invocations: Option<u64>,
    pub geometry_shader_invocations: Option<u64>,
    pub geometry_shader_primitives: Option<u64>,
    pub clipping_invocations: Option<u64>,
    pub clipping_primitives: Option<u64>,
    pub fragment_shader_invocations: Option<u64>,
    pub tessellation_control_shader_patches: Option<u64>,
    pub tessellation_evaluation_shader_invocations: Option<u64>,
    pub compute_shader_invocations: Option<u64>,
    pub task_shader_invocations: Option<u64>,
    pub mesh_shader_invocations: Option<u64>,
}

#[derive(Default, Copy, Clone)]
pub struct PipelineStatisticsQuery {
    flags: vk::QueryPipelineStatisticFlags,
}

impl PipelineStatisticsQuery {
    fn read_bit(bit: vk::QueryPipelineStatisticFlags, value: u64, output: &mut PipelineStatistics) {
        match bit {
            vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_VERTICES => {
                output.input_assembly_vertices = Some(value);
            }
            vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_PRIMITIVES => {
                output.input_assembly_primitives = Some(value);
            }
            vk::QueryPipelineStatisticFlags::VERTEX_SHADER_INVOCATIONS => {
                output.vertex_shader_invocations = Some(value);
            }
            vk::QueryPipelineStatisticFlags::GEOMETRY_SHADER_INVOCATIONS => {
                output.geometry_shader_invocations = Some(value);
            }
            vk::QueryPipelineStatisticFlags::GEOMETRY_SHADER_PRIMITIVES => {
                output.geometry_shader_primitives = Some(value);
            }
            vk::QueryPipelineStatisticFlags::CLIPPING_INVOCATIONS => {
                output.clipping_invocations = Some(value);
            }
            vk::QueryPipelineStatisticFlags::CLIPPING_PRIMITIVES => {
                output.clipping_primitives = Some(value);
            }
            vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS => {
                output.fragment_shader_invocations = Some(value);
            }
            vk::QueryPipelineStatisticFlags::TESSELLATION_CONTROL_SHADER_PATCHES => {
                output.tessellation_control_shader_patches = Some(value);
            }
            vk::QueryPipelineStatisticFlags::TESSELLATION_EVALUATION_SHADER_INVOCATIONS => {
                output.tessellation_evaluation_shader_invocations = Some(value);
            }
            vk::QueryPipelineStatisticFlags::COMPUTE_SHADER_INVOCATIONS => {
                output.compute_shader_invocations = Some(value);
            }
            vk::QueryPipelineStatisticFlags::TASK_SHADER_INVOCATIONS_EXT => {
                output.task_shader_invocations = Some(value);
            }
            vk::QueryPipelineStatisticFlags::MESH_SHADER_INVOCATIONS_EXT => {
                output.mesh_shader_invocations = Some(value);
            },
            _ => panic!("Unsupported query pipeline statistic flags"),
        }
    }
}

impl Query for PipelineStatisticsQuery {
    const QUERY_TYPE: vk::QueryType = vk::QueryType::PIPELINE_STATISTICS;
    type Output = PipelineStatistics;

    fn new(pool: &QueryPoolCreateInfo) -> Self {
        Self {
            flags: pool.statistic_flags.unwrap_or_default(),
        }
    }

    fn size(&self) -> usize {
        num_queries(self.flags)
    }

    fn parse_query(&self, _device: &Device, data: &[u64]) -> Self::Output {
        const BITS: [vk::QueryPipelineStatisticFlags; 13] = [
            vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_VERTICES,
            vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_PRIMITIVES,
            vk::QueryPipelineStatisticFlags::VERTEX_SHADER_INVOCATIONS,
            vk::QueryPipelineStatisticFlags::GEOMETRY_SHADER_INVOCATIONS,
            vk::QueryPipelineStatisticFlags::GEOMETRY_SHADER_PRIMITIVES,
            vk::QueryPipelineStatisticFlags::CLIPPING_INVOCATIONS,
            vk::QueryPipelineStatisticFlags::CLIPPING_PRIMITIVES,
            vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS,
            vk::QueryPipelineStatisticFlags::TESSELLATION_CONTROL_SHADER_PATCHES,
            vk::QueryPipelineStatisticFlags::TESSELLATION_EVALUATION_SHADER_INVOCATIONS,
            vk::QueryPipelineStatisticFlags::COMPUTE_SHADER_INVOCATIONS,
            vk::QueryPipelineStatisticFlags::TASK_SHADER_INVOCATIONS_EXT,
            vk::QueryPipelineStatisticFlags::MESH_SHADER_INVOCATIONS_EXT,
            // Unsupported: VK_QUERY_PIPELINE_STATISTIC_CLUSTER_CULLING_SHADER_INVOCATIONS_BIT_HUAWEI
        ];

        let mut output = PipelineStatistics::default();
        let mut current_index = 0;
        for bit in BITS {
            if self.flags.contains(bit) {
                Self::read_bit(bit, *data.get(current_index).unwrap(), &mut output);
                current_index += 1;
            }
        }

        output
    }
}

impl ScopedQuery for PipelineStatisticsQuery {}

#[derive(Default, Clone, Copy)]
pub struct AccelerationStructureCompactedSizeQuery {}

impl Query for AccelerationStructureCompactedSizeQuery {
    const QUERY_TYPE: vk::QueryType = vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    type Output = u64;

    fn new(_pool: &QueryPoolCreateInfo) -> Self {
        Self::default()
    }

    fn size(&self) -> usize {
        1
    }

    fn parse_query(&self, _device: &Device, data: &[u64]) -> Self::Output {
        *data.first().unwrap()
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct QueryPoolCreateInfo {
    pub count: u32,
    pub statistic_flags: Option<vk::QueryPipelineStatisticFlags>,
}

pub struct QueryPool<Q: Query> {
    handle: vk::QueryPool,
    device: Device,
    current: u32,
    count: u32,
    queries: Vec<Q>,
}

impl<Q: Query> QueryPool<Q> {
    /// Create a new query pool with at most `count` entries.
    pub fn new(device: Device, info: QueryPoolCreateInfo) -> Result<Self> {
        let vk_info = vk::QueryPoolCreateInfo {
            s_type: vk::StructureType::QUERY_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            query_type: Q::QUERY_TYPE,
            query_count: info.count,
            pipeline_statistics: info.statistic_flags.unwrap_or_default(),
        };

        let handle = unsafe { device.create_query_pool(&vk_info, None)? };
        #[cfg(feature = "log-objects")]
        trace!("Created VkQueryPool {handle:p}");

        // Every query in the pool must be reset before usage
        unsafe { device.reset_query_pool(handle, 0, info.count) };

        Ok(Self {
            handle,
            device,
            current: 0,
            count: info.count,
            queries: vec![Q::new(&info); info.count as usize],
        })
    }

    /// Advance the query pool to the next query, and return the previous index.
    /// Returns None if the query pool was out of entries.
    pub fn next(&mut self) -> Option<u32> {
        if self.current >= self.count {
            None
        } else {
            let old = self.current;
            self.current += 1;
            Some(old)
        }
    }

    pub fn current(&self) -> u32 {
        if self.current == 0 {
            0
        } else {
            self.current - 1
        }
    }

    pub fn wait_for_results(&mut self, first: u32, count: u32) -> Result<Vec<Q::Output>> {
        ensure!(first < self.count, "Query range out of range of query pool");
        ensure!(first + count <= self.count, "Query range out of range of query pool");

        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT;
        // Assumption: Every query in the pool has the same number of items, this should always be the case.
        let items_per_query = self.queries.first().map(|query| query.size()).unwrap_or_default();
        let mut buffer = vec![u64::default(); count as usize * items_per_query];
        unsafe {
            self.device
                .get_query_pool_results(self.handle, first, count, buffer.as_mut_slice(), flags)?;
        }
        let data = buffer
            .chunks_exact(items_per_query)
            .into_iter()
            .zip(self.queries.iter())
            .map(|(data, query)| query.parse_query(&self.device, data))
            .collect::<Vec<_>>();

        Ok(data)
    }

    pub fn wait_for_all_results(&mut self) -> Result<Vec<Q::Output>> {
        self.wait_for_results(0, self.count)
    }

    pub fn wait_for_single_result(&mut self, index: u32) -> Result<Q::Output> {
        ensure!(index < self.count, "Query range out of range of query pool");
        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT;
        let query = self.queries.get(index as usize).unwrap();
        let num_items = query.size();
        let mut buffer = vec![u64::default(); num_items];
        unsafe {
            self.device
                .get_query_pool_results(self.handle, index, 1, buffer.as_mut_slice(), flags)?;
        }
        let data = query.parse_query(&self.device, buffer.as_slice());
        Ok(data)
    }

    pub fn reset(&mut self) {
        unsafe { self.device.reset_query_pool(self.handle, 0, self.count) };
        self.current = 0;
    }

    pub unsafe fn handle(&self) -> vk::QueryPool {
        self.handle
    }
}

impl QueryPool<TimestampQuery> {
    pub(crate) fn write_timestamp(&mut self, bits: u32, cmd: vk::CommandBuffer, stage: PipelineStage, query: u32) {
        self.queries.get_mut(query as usize).unwrap().valid_bits = bits;
        unsafe {
            self.device.cmd_write_timestamp2(cmd, stage, self.handle, query);
        }
    }
}

impl<Q: Query> Drop for QueryPool<Q> {
    fn drop(&mut self) {
        #[cfg(feature = "log-objects")]
        trace!("Destroying VkQueryPool {:p}", self.handle);

        unsafe {
            self.device.destroy_query_pool(self.handle, None);
        }
    }
}
