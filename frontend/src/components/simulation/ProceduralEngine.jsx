import React, { useMemo } from 'react';
import * as THREE from 'three';

/**
 * Simple Mulberry32 seeded random number generator.
 * Ensures that for a given seed, the "random" variations are identical every render.
 */
const mulberry32 = (a) => {
    return () => {
        let t = a += 0x6D2B79F5;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
}

/**
 * Generates a highly detailed, "Greebled" engine model procedurally.
 * Fully parametric: Control every aspect via the 'config' prop.
 */
const ProceduralEngine = ({ viewMode, config = {} }) => {

    // Deconstruct config with Defaults
    const {
        seed = 42,
        baseColor = '#555555',
        glowColor = '#44ccff',
        emissiveColor = '#0088ff',
        core = { radius: 1.0, height: 3.5, segments: 64 },
        rings = { count: 20, startY: -1.0, spacing: 0.12, minScale: 1.0, maxScale: 1.1 },
        bolts = { count: 8, radius: 1.4, height: 0.8, size: 0.1 },
        pipes = { count: 4, radius: 1.8, thick: 0.15, segments: 16 }
    } = config;

    // Generate details based on config and seed
    const details = useMemo(() => {
        const rng = mulberry32(seed);
        const ringList = [];
        const boltList = [];
        const pipeList = [];

        // Cooling Fins (Repeated Disks)
        for (let i = 0; i < rings.count; i++) {
            // Deterministic "jitter" based on sine wave + random factor
            const wave = Math.sin(i * 0.5);
            const jitter = (rng() - 0.5) * 0.1;
            const currentScale = rings.minScale + (wave * (rings.maxScale - rings.minScale)) + jitter;

            ringList.push({
                position: [0, rings.startY + (i * rings.spacing), 0],
                args: [core.radius * 1.2, core.radius * 1.2, 0.05, core.segments],
                scale: currentScale
            });
        }

        // Radial Bolts
        for (let i = 0; i < bolts.count; i++) {
            const angle = (i / bolts.count) * Math.PI * 2;
            const x = Math.cos(angle) * bolts.radius;
            const z = Math.sin(angle) * bolts.radius;

            // Top Ring
            boltList.push({
                position: [x, bolts.height, z],
                rotation: [0, -angle, 0],
                args: [bolts.size, bolts.size, 0.3, 6]
            });
            // Bottom Ring
            boltList.push({
                position: [x, -bolts.height, z],
                rotation: [0, -angle, 0],
                args: [bolts.size, bolts.size, 0.3, 6]
            });
        }

        // Manifold Pipes
        for (let i = 0; i < pipes.count; i++) {
            const angle = (i / pipes.count) * Math.PI * 2;
            // Add some random variation to pipe arc length if needed
            const arcLength = Math.PI / 1.5 + (rng() * 0.5);

            pipeList.push({
                rotation: [Math.PI / 2, 0, angle],
                position: [0, 0, 0],
                args: [pipes.radius, pipes.thick, pipes.segments, 32, arcLength]
            });
        }

        return { ringList, boltList, pipeList };
    }, [seed, core, rings, bolts, pipes]);

    const matProps = useMemo(() => {
        if (viewMode === 'hyperrealism') {
            return {
                color: baseColor,
                roughness: 0.2, // Shiny metal
                metalness: 0.9,
                clearcoat: 1,
                clearcoatRoughness: 0.1
            };
        }
        return { color: baseColor };
    }, [viewMode, baseColor]);

    const Material = viewMode === 'hyperrealism' ?
        <meshPhysicalMaterial {...matProps} /> :
        <meshStandardMaterial {...matProps} />;

    const EmissiveMaterial = <meshStandardMaterial
        color={glowColor}
        emissive={emissiveColor}
        emissiveIntensity={viewMode === 'hyperrealism' ? 4 : 1}
        toneMapped={false}
    />;

    return (
        <group>
            {/* Core Cylinder */}
            <mesh castShadow receiveShadow>
                <cylinderGeometry args={[core.radius, core.radius, core.height, core.segments]} />
                {Material}
            </mesh>

            {/* Glowing Core Inner */}
            <mesh position={[0, 0, 0]}>
                <cylinderGeometry args={[core.radius * 0.5, core.radius * 0.5, core.height + 0.1, 32]} />
                {EmissiveMaterial}
            </mesh>

            {/* Render Cooling Rings */}
            {details.ringList.map((r, i) => (
                <mesh key={`ring-${i}`} position={r.position} castShadow receiveShadow>
                    <cylinderGeometry args={[r.args[0] * r.scale, r.args[1] * r.scale, r.args[2], r.args[3]]} />
                    {Material}
                </mesh>
            ))}

            {/* Render Bolts */}
            {details.boltList.map((b, i) => (
                <mesh key={`bolt-${i}`} position={b.position} rotation={b.rotation} castShadow receiveShadow>
                    <cylinderGeometry args={b.args} />
                    <meshStandardMaterial color="#888888" roughness={0.5} metalness={0.8} />
                </mesh>
            ))}

            {/* Render Pipes */}
            {details.pipeList.map((p, i) => (
                <mesh key={`pipe-${i}`} position={p.position} rotation={p.rotation} castShadow receiveShadow>
                    <torusGeometry args={p.args} />
                    <meshStandardMaterial color="#444444" roughness={0.4} metalness={0.6} />
                </mesh>
            ))}
        </group>
    );
};

export default ProceduralEngine;
