import React, { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { OpenSCADMesh } from './OpenSCADMesh';

// Physics Shaders (Phase 18)
import vertexShader from '../../shaders/mesh_physics_vertex.glsl?raw';
import fragmentShader from '../../shaders/mesh_physics_fragment.glsl?raw';

/**
 * Standard Mesh Preview Component
 * 
 * Renders geometry using standard WebGL rasterization (triangles).
 * UPGRADED: Now supports Thermal, Stress, and X-Ray via MeshPhysicsMaterial.
 */
export const StandardMeshPreview = ({
    design,
    viewMode = 'realistic',
    theme,
    physicsData, // Phase 18: Backend Data
    onSDFLoaded
}) => {

    // Shader Material Ref
    const shaderRef = useRef();

    useFrame(({ clock }) => {
        if (shaderRef.current) {
            shaderRef.current.uniforms.uTime.value = clock.elapsedTime;
        }
    });

    // Extract Physics Values Helper
    const physicsUniforms = useMemo(() => {
        let temp = 20.0;
        let stress = 0.0;

        if (physicsData) {
            // Thermal
            const therm = physicsData.metrics?.thermal || physicsData.sub_agent_reports?.thermal;
            if (therm?.equilibrium_temp_c !== undefined) temp = therm.equilibrium_temp_c;

            // Structural
            const struct = physicsData.metrics?.structural || physicsData.sub_agent_reports?.structural;
            // Map Safety Factor to Stress Level (Inverse)
            if (struct?.safety_factor) {
                stress = 1.0 / Math.max(0.1, struct.safety_factor);
            }
        }

        return { temp, stress };
    }, [physicsData]);

    const getMaterial = (color = '#cccccc', metalness = 0.5, roughness = 0.5) => {
        const baseColor = new THREE.Color(color);

        // Advanced Views use Custom Shader
        if (['thermal', 'stress', 'xray', 'heatmap'].includes(viewMode)) {
            let modeInt = 0;
            if (viewMode === 'xray') modeInt = 2;
            if (viewMode === 'thermal' || viewMode === 'heatmap') modeInt = 4;
            if (viewMode === 'stress') modeInt = 7;

            return <shaderMaterial
                ref={shaderRef}
                vertexShader={vertexShader}
                fragmentShader={fragmentShader}
                uniforms={{
                    uViewMode: { value: modeInt },
                    uBaseColor: { value: baseColor },
                    uMetalness: { value: metalness },
                    uRoughness: { value: roughness },
                    uThermalTemp: { value: physicsUniforms.temp },
                    uStressLevel: { value: physicsUniforms.stress },
                    uTime: { value: 0 }
                }}
                transparent={viewMode === 'xray'}
                depthWrite={viewMode !== 'xray'}
                side={THREE.DoubleSide}
            />;
        }

        switch (viewMode) {
            case 'wireframe':
                return <meshBasicMaterial color={theme.colors.accent.primary} wireframe />;
            case 'matte':
                return <meshStandardMaterial color={baseColor} roughness={1.0} metalness={0.0} />;
            case 'solid':
                return <meshStandardMaterial color="#888888" />;
            case 'hidden_line':
                return <meshBasicMaterial color="#222" polygonOffset polygonOffsetFactor={1} />;
            default: // realistic
                return <meshStandardMaterial
                    color={baseColor}
                    metalness={metalness}
                    roughness={roughness}
                />;
        }
    };

    // Parse design content to determine what to render
    const content = useMemo(() => {
        if (!design?.content) return null;
        try {
            if (typeof design.content === 'string') {
                const trimmed = design.content.trim();
                if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
                    return JSON.parse(trimmed);
                }
                // Treat non-JSON strings as OpenSCAD code
                return { type: 'openscad', code: design.content };
            }
            return design.content;
        } catch (e) {
            console.warn("StandardMeshPreview: Failed to parse design content, treating as code", e);
            return { type: 'openscad', code: design.content };
        }
    }, [design]);

    if (!content) return null;

    // 1. OpenSCAD Handling
    if (content.type === 'openscad' || design.type === 'openscad') {
        const code = content.code || (typeof design.content === 'string' ? design.content : '');
        return (
            <OpenSCADMesh
                scadCode={code}
                viewMode={viewMode}
                theme={theme}
                onSDFLoaded={onSDFLoaded}
            />
        );
    }

    // 2. Primitive Handling
    if (content.type === 'primitive') {
        const { geometry, args = [], material = {} } = content;
        const color = material.color || '#cccccc';
        const metalness = material.metalness ?? 0.5;
        const roughness = material.roughness ?? 0.5;

        let GeometryComponent = null;
        let finalArgs = args;

        if (geometry === 'box') {
            GeometryComponent = <boxGeometry args={args.length >= 3 ? args.slice(0, 3) : [1, 1, 1]} />;
        } else if (geometry === 'sphere') {
            GeometryComponent = <sphereGeometry args={[args[0] || 0.5, 32, 32]} />;
        } else if (geometry === 'cylinder') {
            GeometryComponent = <cylinderGeometry args={[args[0] || 0.5, args[0] || 0.5, args[1] || 1, 32]} />;
        }

        if (GeometryComponent) {
            return (
                <mesh>
                    {GeometryComponent}
                    {getMaterial(color, metalness, roughness)}

                    {/* Edges for cleaner look in solid/realistic modes */}
                    {(viewMode === 'solid' || viewMode === 'realistic') && (
                        <lineSegments>
                            <edgesGeometry args={[
                                geometry === 'box' ? new THREE.BoxGeometry(...(args.length >= 3 ? args.slice(0, 3) : [1, 1, 1]))
                                    : (geometry === 'sphere' ? new THREE.SphereGeometry(args[0] || 0.5, 16, 16) // Lower res for edges
                                        : new THREE.CylinderGeometry(args[0] || 0.5, args[0] || 0.5, args[1] || 1, 16))
                            ]} />
                            <lineBasicMaterial color={theme.colors.border.primary || '#444'} transparent opacity={0.4} />
                        </lineSegments>
                    )}
                </mesh>
            );
        }
    }

    return null;
};
