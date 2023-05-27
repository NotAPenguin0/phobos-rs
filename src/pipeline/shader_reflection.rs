//! Implements shader reflection to generate pipeline layouts automatically

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use anyhow::Result;
use ash::vk;
#[cfg(feature = "shader-reflection")]
use spv_cross::spirv::{Decoration, ExecutionModel, ShaderResources, Type};

use crate::pipeline::pipeline_layout::{PipelineLayoutCreateInfo, PushConstantRange};
use crate::pipeline::set_layout::DescriptorSetLayoutCreateInfo;
use crate::{Error, ShaderCreateInfo};

#[cfg(all(feature = "shader-reflection", not(feature = "hlsl")))]
type Ast = spv_cross::spirv::Ast<spv_cross::glsl::Target>;

#[cfg(all(feature = "shader-reflection", feature = "hlsl"))]
type Ast = spv_cross::spirv::Ast<spv_cross::hlsl::Target>;

#[cfg(feature = "shader-reflection")]
#[derive(Debug, Copy, Clone)]
pub(crate) struct BindingInfo {
    pub set: u32,
    pub binding: u32,
    pub stage: vk::ShaderStageFlags,
    pub count: u32,
    pub ty: vk::DescriptorType,
}

/// Stores reflection information about a pipeline. Can be used to derive a pipeline layout
/// automatically, or access names of descriptor bindings.
#[cfg(feature = "shader-reflection")]
#[derive(Debug)]
pub struct ReflectionInfo {
    pub(crate) bindings: HashMap<String, BindingInfo>,
    pub(crate) push_constants: Vec<PushConstantRange>,
}

#[cfg(feature = "shader-reflection")]
fn get_shader_stage(ast: &Ast) -> Result<vk::ShaderStageFlags> {
    let entry = ast
        .get_entry_points()?
        .first()
        .cloned()
        .ok_or_else(|| Error::NoEntryPoint)?;
    Ok(match entry.execution_model {
        ExecutionModel::Vertex => vk::ShaderStageFlags::VERTEX,
        ExecutionModel::TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        ExecutionModel::TessellationEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        ExecutionModel::Geometry => vk::ShaderStageFlags::GEOMETRY, // EVIL
        ExecutionModel::Fragment => vk::ShaderStageFlags::FRAGMENT,
        ExecutionModel::GlCompute => vk::ShaderStageFlags::COMPUTE,
        ExecutionModel::RayGenerationKHR => vk::ShaderStageFlags::RAYGEN_KHR,
        ExecutionModel::IntersectionKHR => vk::ShaderStageFlags::INTERSECTION_KHR,
        ExecutionModel::AnyHitKHR => vk::ShaderStageFlags::ANY_HIT_KHR,
        ExecutionModel::ClosestHitKHR => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        ExecutionModel::MissKHR => vk::ShaderStageFlags::MISS_KHR,
        ExecutionModel::Kernel => {
            unimplemented!()
        }
    })
}

// Note that aliasing is not supported

#[cfg(feature = "shader-reflection")]
fn find_sampled_images(
    ast: &mut Ast,
    stage: vk::ShaderStageFlags,
    resources: &ShaderResources,
    info: &mut ReflectionInfo,
) -> Result<()> {
    for image in &resources.sampled_images {
        let binding = ast.get_decoration(image.id, Decoration::Binding)?;
        let set = ast.get_decoration(image.id, Decoration::DescriptorSet)?;
        let ty = ast.get_type(image.type_id)?;
        let Type::SampledImage { array, .. } = ty else { unimplemented!() };
        let count = if !array.is_empty() {
            if array[0] == 0 {
                4096 // Max unbounded array size. If this is ever exceeded, I'll fix it.
            } else {
                array[0]
            }
        } else {
            1
        };

        info.bindings.insert(
            ast.get_name(image.id)?,
            BindingInfo {
                set,
                binding,
                stage,
                count,
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            },
        );
    }
    Ok(())
}

#[cfg(feature = "shader-reflection")]
fn find_uniform_buffers(
    ast: &mut Ast,
    stage: vk::ShaderStageFlags,
    resources: &ShaderResources,
    info: &mut ReflectionInfo,
) -> Result<()> {
    for buffer in &resources.uniform_buffers {
        let binding = ast.get_decoration(buffer.id, Decoration::Binding)?;
        let set = ast.get_decoration(buffer.id, Decoration::DescriptorSet)?;
        info.bindings.insert(
            ast.get_name(buffer.id)?,
            BindingInfo {
                set,
                binding,
                stage,
                count: 1,
                ty: vk::DescriptorType::UNIFORM_BUFFER,
            },
        );
    }
    Ok(())
}

#[cfg(feature = "shader-reflection")]
fn find_storage_buffers(
    ast: &mut Ast,
    stage: vk::ShaderStageFlags,
    resources: &ShaderResources,
    info: &mut ReflectionInfo,
) -> Result<()> {
    for buffer in &resources.storage_buffers {
        let binding = ast.get_decoration(buffer.id, Decoration::Binding)?;
        let set = ast.get_decoration(buffer.id, Decoration::DescriptorSet)?;
        info.bindings.insert(
            ast.get_name(buffer.id)?,
            BindingInfo {
                set,
                binding,
                stage,
                count: 1,
                ty: vk::DescriptorType::STORAGE_BUFFER,
            },
        );
    }
    Ok(())
}

#[cfg(feature = "shader-reflection")]
fn find_storage_images(
    ast: &mut Ast,
    stage: vk::ShaderStageFlags,
    resources: &ShaderResources,
    info: &mut ReflectionInfo,
) -> Result<()> {
    for image in &resources.storage_images {
        let binding = ast.get_decoration(image.id, Decoration::Binding)?;
        let set = ast.get_decoration(image.id, Decoration::DescriptorSet)?;
        info.bindings.insert(
            ast.get_name(image.id)?,
            BindingInfo {
                set,
                binding,
                stage,
                count: 1,
                ty: vk::DescriptorType::STORAGE_IMAGE,
            },
        );
    }
    Ok(())
}

#[cfg(feature = "shader-reflection")]
fn find_acceleration_structures(
    ast: &mut Ast,
    stage: vk::ShaderStageFlags,
    resources: &ShaderResources,
    info: &mut ReflectionInfo,
) -> Result<()> {
    for image in &resources.acceleration_structures {
        let binding = ast.get_decoration(image.id, Decoration::Binding)?;
        let set = ast.get_decoration(image.id, Decoration::DescriptorSet)?;
        info.bindings.insert(
            ast.get_name(image.id)?,
            BindingInfo {
                set,
                binding,
                stage,
                count: 1,
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            },
        );
    }
    Ok(())
}

#[cfg(feature = "shader-reflection")]
fn find_push_constants(
    ast: &mut Ast,
    stage: vk::ShaderStageFlags,
    resources: &ShaderResources,
    info: &mut ReflectionInfo,
) -> Result<()> {
    for pc in &resources.push_constant_buffers {
        let ranges = ast.get_active_buffer_ranges(pc.id)?;
        for range in ranges {
            info.push_constants.push(PushConstantRange {
                stage_flags: stage,
                offset: range.offset as u32,
                size: range.range as u32,
            })
        }
    }
    Ok(())
}

#[cfg(feature = "shader-reflection")]
fn reflect_module(module: spv_cross::spirv::Module) -> Result<ReflectionInfo> {
    let mut ast: Ast = Ast::parse(&module)?;
    let resources = ast.get_shader_resources()?;
    let stage = get_shader_stage(&ast)?;

    let mut info = ReflectionInfo {
        bindings: Default::default(),
        push_constants: Default::default(),
    };
    find_sampled_images(&mut ast, stage, &resources, &mut info)?;
    find_uniform_buffers(&mut ast, stage, &resources, &mut info)?;
    find_storage_buffers(&mut ast, stage, &resources, &mut info)?;
    find_push_constants(&mut ast, stage, &resources, &mut info)?;
    find_storage_images(&mut ast, stage, &resources, &mut info)?;
    find_acceleration_structures(&mut ast, stage, &resources, &mut info)?;
    Ok(info)
}

#[cfg(feature = "shader-reflection")]
fn merge_push_constants(reflected_shaders: &[ReflectionInfo]) -> Result<Vec<PushConstantRange>> {
    let mut result = Vec::new();
    for shader in reflected_shaders {
        let mut pc = PushConstantRange {
            stage_flags: Default::default(),
            offset: u32::MAX,
            size: 0,
        };
        for range in &shader.push_constants {
            pc.stage_flags = range.stage_flags;
            pc.offset = pc.offset.min(range.offset);
            pc.size += range.size;
        }
        if pc.size != 0 {
            result.push(pc);
        }
    }

    Ok(result)
}

#[cfg(feature = "shader-reflection")]
pub(crate) fn reflect_shaders(shaders: &[ShaderCreateInfo]) -> Result<ReflectionInfo> {
    let mut reflected_shaders = Vec::new();
    for shader in shaders {
        let module = spv_cross::spirv::Module::from_words(shader.code());
        reflected_shaders.push(reflect_module(module)?);
    }

    Ok(ReflectionInfo {
        bindings: reflected_shaders
            .iter()
            .fold(HashMap::default(), |mut acc, shader| {
                for (name, binding) in &shader.bindings {
                    let entry = acc.entry(name.clone());
                    match entry {
                        // If this entry is already in the map, add its stage.
                        Entry::Occupied(entry) => {
                            let value = entry.into_mut();
                            if value.set != binding.set
                                || value.ty != binding.ty
                                || value.binding != binding.binding
                            {
                                panic!("Aliased descriptor sets used.");
                            }
                            value.stage |= binding.stage;
                        }
                        // Otherwise insert a new binding.
                        Entry::Vacant(entry) => {
                            entry.insert(*binding);
                        }
                    }
                }

                acc
            }),
        push_constants: merge_push_constants(&reflected_shaders)?,
    })
}

#[cfg(feature = "shader-reflection")]
pub(crate) fn build_pipeline_layout(info: &ReflectionInfo) -> PipelineLayoutCreateInfo {
    let mut layout = PipelineLayoutCreateInfo {
        flags: Default::default(),
        set_layouts: vec![],
        push_constants: info.push_constants.clone(),
        persistent: false,
    };

    let mut sets: HashMap<u32, DescriptorSetLayoutCreateInfo> = HashMap::new();
    for binding in info.bindings.values() {
        let entry = sets.entry(binding.set);
        match entry {
            Entry::Occupied(entry) => {
                let set = entry.into_mut();
                set.bindings.push(vk::DescriptorSetLayoutBinding {
                    binding: binding.binding,
                    descriptor_type: binding.ty,
                    descriptor_count: binding.count,
                    stage_flags: binding.stage,
                    p_immutable_samplers: std::ptr::null(),
                });
            }
            Entry::Vacant(entry) => {
                entry.insert(DescriptorSetLayoutCreateInfo {
                    bindings: vec![vk::DescriptorSetLayoutBinding {
                        binding: binding.binding,
                        descriptor_type: binding.ty,
                        descriptor_count: binding.count,
                        stage_flags: binding.stage,
                        p_immutable_samplers: std::ptr::null(),
                    }],
                    persistent: false,
                });
            }
        }
    }

    for i in 0..sets.len() as u32 {
        layout.set_layouts.push(sets.get(&i).unwrap().clone());
    }

    layout
}
