use std::sync::{Arc, Mutex};

use anyhow::Result;

use ash::vk;

use crate::{pipeline::set_layout::DescriptorSetLayoutCreateInfo, util::cache::Resource};

/// The maximum number of resources in a pool
pub static MAX_BINDLESS_COUNT: u32 = 4096;

/// A resource that can be used bindlessly.
pub trait BindlessResource {
    /// Get the [vk::DescriptorType] for this resource.
    fn descriptor_type() -> vk::DescriptorType;

    /// Get the [vk::DescriptorSetLayoutBinding] of this resource.
    fn resource_binding(binding: u32) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: Self::descriptor_type(),
            descriptor_count: MAX_BINDLESS_COUNT,
            stage_flags: vk::ShaderStageFlags::ALL,
            p_immutable_samplers: std::ptr::null(),
        }
    }

    /// Get the [vk::DescriptorImageInfo] of this resource.
    fn descriptor_info(&self) -> vk::DescriptorImageInfo;

    /// Add this resource to `pool` and get a handle to it.
    fn into_bindless(self, pool: &BindlessPool<Self>) -> BindlessHandle<Self>
    where
        Self: Sized
    {
        pool.alloc(self)
    }
}

impl BindlessResource for crate::image::Image {
    fn descriptor_type() -> vk::DescriptorType {
        vk::DescriptorType::STORAGE_IMAGE
    }

    fn descriptor_info(&self) -> vk::DescriptorImageInfo {
        todo!()
    }
}

/// Resource for a combined image sampler.
#[derive(Debug)]
pub struct CombinedImageSampler {
    sampler: Arc<crate::sampler::Sampler>,
    image_view: crate::image::ImageView,
    image_layout: Option<vk::ImageLayout>,
}

impl CombinedImageSampler {
    /// Create a new combined image sampler resource from a sampler and an ImageView.
    /// The default image layout [vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL] will be used.
    pub fn new(sampler: Arc<crate::sampler::Sampler>, image_view: crate::image::ImageView) -> Self {
        Self {
            sampler,
            image_view,
            image_layout: None,
        }
    }

    /// Specify an image layout for this combined image sampler
    pub fn with_layout(self, image_layout: Option<vk::ImageLayout>) -> Self {
        Self { image_layout, .. self }
    }
}

impl BindlessResource for CombinedImageSampler {
    fn descriptor_type() -> vk::DescriptorType {
        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
    }

    fn descriptor_info(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo {
            sampler: unsafe { self.sampler.handle() },
            image_view: unsafe { self.image_view.handle() },
            image_layout: self.image_layout.unwrap_or(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        }
    }
}

/// A handle to a resource contained in a [BindlessPool]
pub struct BindlessHandle<R: BindlessResource> {
    key: u32,
    pool: BindlessPool<R>,
}

impl<R: BindlessResource> Drop for BindlessHandle<R> {
    fn drop(&mut self) {
        self.pool.with(|p| p.take(self.key));
    }
}

impl<R: BindlessResource> BindlessHandle<R> {
    /// Get the index of this resource in the pool's descripor.
    /// This can be sent to a shader through any means to access it.
    pub fn index(&self) -> u32 {
        self.key
    }
}

struct BindlessPoolInner<R> {
    items: Vec<Option<R>>,
    free: Vec<u32>,
    descriptor_set: crate::DescriptorSet,
}

impl<R: BindlessResource> BindlessPoolInner<R> {
    fn update_descriptor_set<'a>(&'a mut self, r: impl Iterator<Item = (u32, &'a R)>) {
        let vk_writes = r
            .map(|(i, r)| {
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: self.descriptor_set.handle,
                    dst_binding: 0,
                    dst_array_element: i,
                    descriptor_count: 1,
                    descriptor_type: R::descriptor_type(),
                    p_image_info: &r.descriptor_info() as *const _,
                    p_buffer_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                }
            })
            .collect::<Vec<_>>();

        unsafe {
            self.descriptor_set.device.update_descriptor_sets(vk_writes.as_slice(), &[]);
        }
    }


    fn take(&mut self, key: u32) -> Option<R> {
        if key <= self.items.len() as _ {
            None
        } else {
            self.items[key as usize].take().and_then(|ob| {
                self.free.push(key);
                Some(ob)
            })
        }
    }
}

/// A bindless pool can hold a number of resources.
/// It will keep a descriptor set up to date that gives access to all resources.
pub struct BindlessPool<R> {
    inner: Arc<Mutex<BindlessPoolInner<R>>>,
}

impl<P: BindlessResource> BindlessPool<P> {
    fn with<F: FnOnce(&mut BindlessPoolInner<P>) -> R, R>(&self, f: F) -> R {
        let mut inner = self.inner.lock().unwrap();
        f(&mut inner)
    }

    /// Allocate a single item from the pool.
    pub fn alloc(&self, item: P) -> BindlessHandle<P> {
        self.with(|p| {
            let key = p.free
                .pop()
                .unwrap_or_else(|| {
                    p.items.push(None);
                    p.items.len() as u32 - 1
                });
            p.update_descriptor_set(std::iter::once((key, &item)));
            p.items[key as usize] = Some(item);
            BindlessHandle {
                key,
                pool: Self { inner: self.inner.clone() }
            }
        })
    }

    /// Allocate a number of items from the pool.
    pub fn alloc_items(&self, items: &[P]) -> impl Iterator<Item = BindlessHandle<P>> {
        self.with(|p| {
            let mut keys = p.free.iter().cloned().rev().take(items.len()).collect::<Vec<_>>();
            if keys.len() < items.len() {
                let old_len = p.items.len() as u32;
                p.items.resize_with(items.len(), || None);
                keys.extend(old_len..p.items.len() as u32);
            }
            p.update_descriptor_set(keys.iter().cloned().zip(items));
            let pool_inner = self.inner.clone();
            keys
                .into_iter()
                .map(move |key| {
                    BindlessHandle {
                        key,
                        pool: Self { inner: pool_inner.clone() }
                    }
                })
        })
    }

    /// Create a new bindless pool
    pub fn new(device: crate::Device) -> Result<Self> {
        let pool_size = vk::DescriptorPoolSize {
            ty: P::descriptor_type(),
            descriptor_count: MAX_BINDLESS_COUNT,
        };
        let pool_create_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &pool_size as *const _,
        };
        let pool = unsafe { device.create_descriptor_pool(&pool_create_info, None)? };
        Self::new_with_pool(device, pool)
    }

    /// Create a new bindless pool that will allocate a descriptor set from `pool`.
    /// The provided pool must be large enough for a [MAX_BINDLESS_COUNT] elements descriptor array
    pub fn new_with_pool(device: crate::Device, pool: vk::DescriptorPool) -> Result<Self> {
        let dsl_info = DescriptorSetLayoutCreateInfo {
            bindings: vec![
                P::resource_binding(0)
            ],
            persistent: true,
            flags: vec![vk::DescriptorBindingFlags::UPDATE_AFTER_BIND, vk::DescriptorBindingFlags::PARTIALLY_BOUND],
        };
        let dsl = crate::pipeline::set_layout::DescriptorSetLayout::create(device.clone(), &dsl_info, ())?;
        let descriptor_set = unsafe { crate::DescriptorSet::new_uninitialized(device, dsl.handle(), pool)? };

        let inner = BindlessPoolInner {
            items: vec![],
            free: vec![],
            descriptor_set,
        };

        Ok(Self { inner: Arc::new(Mutex::new(inner)) })
    }
}
