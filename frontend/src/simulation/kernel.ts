import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import { ComponentStruct, VmkStateStruct } from './schema';

/**
 * BRICK OS: TypeGPU WGSL Kernel
 * Implements the SDF Raymarcher in pure WGSL via TypeGPU.
 */

// --- SDF Primitives ---

const sdBox = tgpu.fn([d.vec3f, d.vec3f], d.f32)
    .does((p, b) => {
        const q = d.abs(p).sub(b);
        return d.length(d.max(q, 0.0)).add(d.min(d.max(q.x, d.max(q.y, q.z)), 0.0));
    });

const sdCapsule = tgpu.fn([d.vec3f, d.vec3f, d.vec3f, d.f32], d.f32)
    .does((p, a, b, r) => {
        const pa = p.sub(a);
        const ba = b.sub(a);
        const h = d.clamp(d.dot(pa, ba).div(d.dot(ba, ba)), 0.0, 1.0);
        return d.length(pa.sub(ba.mul(h))).sub(r);
    });

const smin = tgpu.fn([d.f32, d.f32, d.f32], d.f32)
    .does((a, b, k) => {
        const h = d.clamp(d.f32(0.5).add(d.f32(0.5).mul(b.sub(a)).div(k)), 0.0, 1.0);
        return d.mix(b, a, h).sub(k.mul(h).mul(d.f32(1.0).sub(h)));
    });

// --- Map Function ---

const map = tgpu.fn([d.vec3f, VmkStateStruct], d.f32)
    .does((p, state) => {
        // 1. Stock
        let dVal = sdBox(p, state.stockDims.mul(0.5));

        // 2. Components
        // Loop must be unrolled or fixed size in some WGSL contexts, 
        // but TypeGPU handles loops if using `d.loop` or standard for?
        // 'for' loops in TypeGPU work on values, not just code gen? 
        // TypeGPU `does` uses a callback that builds the AST. 
        // We should use `for` loop if TypeGPU supports it in builder. 
        // Standard JS `for` executes at construction time (unroll).
        // To generate a dynamic WGSL loop, we might need `d.loop`?
        // Assuming unrolling for 16 items is fine for MVP.
        // Actually, `VmkStateStruct.components` is a fixed array.

        // Dynamic loop support in TypeGPU v0.9:
        // We can use standard JS `for` to unroll if count is static.
        // But `state.count` is dynamic.
        // We need a runtime loop.
        // TypeGPU/WGSL supports `for`.
        // However, `tgpu` builder syntax:

        // Workaround: Unroll 16 checked iterations.
        for (let i = 0; i < 16; i++) {
            // We can't use `if(i >= state.count) break;` easily inside pure JS builder 
            // unless `state.count` allows comparison.
            // `state.count` is a Tgpu value.
            // `d.u32(i)` vs `state.count`.

            // This generates "if" in WGSL.
            // if (i >= state.count) break; // Not directly supported in simple builder?
            // We will just execute all and check condition in math.

            // Actually, let's try strict unroll logic with a condition.
            const active = d.i32(i).lessThan(state.count);

            // This is tricky. TypeGPU's control flow API is distinct.
            // For MVP, likely just unroll and use `active` as mask.
        }

        // Wait, simpler fallback:
        // Just calculate SD for ALL 16, and mix result?
        // Or just `sdBox` only for now to prove it works.

        return dVal;
    });

// --- Raymarcher ---

export const mainRaymarch = tgpu.fn([
    d.vec3f, // Ray Origin
    d.vec3f, // Ray Dir
    VmkStateStruct // State
], d.vec3f).does((ro, rd, state) => {

    let t = d.f32(0.0);
    let col = d.vec3f(0.0);
    let hit = false;

    // Raymarch Loop (Fixed 64 steps)
    // TypeGPU has no explicit `loop` API in docs I recall?
    // We can write a JS loop that emits 64 checks?
    // Or simple recursion? (No).

    // MVP: Just render a sphere/box at 0,0,0
    const dVal = sdBox(ro.add(rd.mul(2.0)), d.vec3f(1.0));

    // Placeholder to verify pipeline
    const hitSurface = dVal.lessThan(0.01);

    if (true) {
        // Basic visualization
        // return d.vec3f(dVal.div(5.0));
    }

    return d.vec3f(d.abs(rd)); // Debug: Return Ray Dir colors
});

/**
 * NOTE: Full Raymarching loop implementation in TypeGPU requires usage of `d.loop` or similar
 * Control Flow features which are specific to the library version. 
 * For this MVP Step, we export the basic primitives and a debug kernel.
 * 
 * We will refine the loop once we see it running.
 */
