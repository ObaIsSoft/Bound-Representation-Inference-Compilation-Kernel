import tgpu from 'typegpu';

/**
 * BRICK OS: TypeGPU Schema Definitions
 * Defines the memory layout for the VMK simulation.
 */

// Vector types (aliased for clarity)
const vec3 = tgpu.vec3f;

// 1. ToolPath / Component Schema
// Represents a single subtractive/additive operation in the history.
export const ComponentStruct = tgpu.struct({
    pos: vec3,       // Center position
    endPos: vec3,    // End position (for capsules) - Added to support sweeps
    radius: tgpu.f32, // Tool radius
    opType: tgpu.i32, // 0 = Standard, 1 = Subtractive, 2 = Smooth Add
});

// 2. VMK State Schema
// Complete snapshot of the machining context.
export const VmkStateStruct = tgpu.struct({
    stockDims: vec3,
    components: tgpu.array(ComponentStruct, 16), // Fixed size buffer for MVP
    count: tgpu.i32, // Number of active components
});

// Export inferred TS types
export type Component = tgpu.Infer<typeof ComponentStruct>;
export type VmkState = tgpu.Infer<typeof VmkStateStruct>;
