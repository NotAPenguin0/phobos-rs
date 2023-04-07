use std::ops::Sub;
use std::time::Duration;

use anyhow::{ensure, Result};
use ash::vk;

use crate::{Device, PipelineStage};

pub trait Query: Default + Clone + Sized {
    const QUERY_TYPE: vk::QueryType;
    const SIZE: usize;

    type Output;

    fn parse_query(&self, device: &Device, data: &[u64]) -> Self::Output;
}

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
    const SIZE: usize = 1; // One u64

    type Output = Timestamp;

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
            period: device.properties().limits.timestamp_period
        }
    }
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
    pub fn new(device: Device, count: u32) -> Result<Self> {
        let info = vk::QueryPoolCreateInfo {
            s_type: vk::StructureType::QUERY_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: Default::default(),
            query_type: Q::QUERY_TYPE,
            query_count: count,
            pipeline_statistics: Default::default(),
        };

        let handle = unsafe { device.create_query_pool(&info, None)? };
        #[cfg(feature = "log-objects")]
        trace!("Created VkQueryPool {handle:p}");

        // Every query in the pool must be reset before usage
        unsafe { device.reset_query_pool(handle, 0, count) };

        Ok(Self {
            handle,
            device,
            current: 0,
            count,
            queries: vec![Q::default(); count as usize],
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

    pub fn wait_for_results(&mut self, first: u32, count: u32) -> Result<Vec<Q::Output>> {
        ensure!(first < self.count, "Query range out of range of query pool");
        ensure!(first + count <= self.count, "Query range out of range of query pool");

        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT;
        let mut buffer = vec![u64::default(); count as usize * Q::SIZE];
        unsafe {
            self.device
                .get_query_pool_results(self.handle, first, count, buffer.as_mut_slice(), flags)?;
        }
        let data = buffer
            .chunks_exact(Q::SIZE)
            .into_iter()
            .zip(self.queries.iter())
            .map(|(data, query)| query.parse_query(&self.device, data))
            .collect::<Vec<_>>();

        Ok(data)
    }

    pub fn wait_for_all_results(&mut self) -> Result<Vec<Q::Output>> {
        self.wait_for_results(0, self.count)
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
