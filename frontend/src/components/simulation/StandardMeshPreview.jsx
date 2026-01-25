import React, { useMemo } from 'react';
import * as THREE from 'three';
import { OpenSCADMesh } from './OpenSCADMesh';

/**
 * Standard Mesh Preview Component
 * 
 * Renders geometry using standard WebGL rasterization (triangles) instead of SDF Raymarching.
 * - Much faster for complex meshes
 * - Supports primitives (Box, Sphere, Cylinder)
 * - Supports OpenSCAD results
 */
export const StandardMeshPreview = ({
    design,
    viewMode = 'realistic',
    theme,
    onSDFLoaded // New callback
}) => {

    // Helper to determine material based on view mode
    const getMaterial = (color = '#cccccc', metalness = 0.5, roughness = 0.5) => {
        const baseColor = new THREE.Color(color);

        switch (viewMode) {
            case 'wireframe':
                return <meshBasicMaterial color={theme.colors.accent.primary} wireframe />;
            case 'matte':
                return <meshStandardMaterial color={baseColor} roughness={1.0} metalness={0.0} />;
            case 'solid':
                return <meshStandardMaterial color="#888888" />;
            case 'xray':
                return <meshStandardMaterial color={baseColor} transparent opacity={0.3} depthWrite={false} />;
            case 'hidden_line':
                return <meshBasicMaterial color="#222" polygonOffset polygonOffsetFactor={1} />; // Simplified
            case 'heatmap':
                return <meshNormalMaterial />; // Placeholder for heatmap
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
