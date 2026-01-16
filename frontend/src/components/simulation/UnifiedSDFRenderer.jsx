import React, { useMemo, useRef, useEffect, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import { useTheme } from '../../contexts/ThemeContext';
import PhysicsOverlays from './PhysicsOverlays';

// Import shaders as raw strings (Vite handles this with ?raw)
import vertexShader from '../../shaders/unified_sdf_vertex.glsl?raw';
import fragmentShader from '../../shaders/unified_sdf.glsl?raw';

/**
 * BRICK OS: Unified SDF Renderer
 * 
 * Single Canvas component that replaces:
 * - DefaultSimulation.jsx
 * - VMKRenderer.jsx
 * - RaymarchScene.jsx
 * - TypeGPURaymarch.jsx
 */

// View Mode Enum (matches shader)
export const VIEW_MODES = {
    REALISTIC: 0,
    WIREFRAME: 1,
    XRAY: 2,
    MATTE: 3,
    HEATMAP: 4,
    CUTAWAY: 5,
    SOLID: 6,
    STRESS: 7,
    HIDDEN_LINE: 8,
    SHADED: 9,
    INTERIOR: 10,
    FLOW: 11
};

// Map string modes to enum
const MODE_MAP = {
    'realistic': VIEW_MODES.REALISTIC,
    'wireframe': VIEW_MODES.WIREFRAME,
    'xray': VIEW_MODES.XRAY,
    'matte': VIEW_MODES.MATTE,
    'heatmap': VIEW_MODES.HEATMAP,
    'cutaway': VIEW_MODES.CUTAWAY,
    'solid': VIEW_MODES.SOLID,
    'stress': VIEW_MODES.STRESS,
    'hidden_line': VIEW_MODES.HIDDEN_LINE,
    'shaded': VIEW_MODES.SHADED,
    'interior': VIEW_MODES.INTERIOR,
    'hyperrealism': VIEW_MODES.REALISTIC,
    'micro': VIEW_MODES.REALISTIC,
    'micro_wgsl': VIEW_MODES.REALISTIC,
    'thermal': VIEW_MODES.HEATMAP,
    'flow': VIEW_MODES.FLOW
};

// Inner kernel component (must be inside Canvas)
const SDFKernel = ({
    viewMode = 'realistic',
    baseShape = 1,
    baseDims = [1, 1, 1],
    baseColor = [0.8, 0.8, 0.8],
    metalness = 0.0,
    roughness = 0.5,
    clipPlane = [0, 1, 0],
    clipOffset = 0,
    clipEnabled = false,
    physicsData = null,
    onShaderError = null,
    theme = 'dark' // Default prop
}) => {
    const meshRef = useRef();
    const materialRef = useRef();
    const { gl, size } = useThree();

    // Convert string viewMode to integer
    const viewModeInt = MODE_MAP[viewMode] ?? VIEW_MODES.REALISTIC;

    // Create uniforms
    const uniforms = useMemo(() => ({
        uTime: { value: 0 },
        uResolution: { value: new THREE.Vector2(800, 600) },
        uCameraPos: { value: new THREE.Vector3(4, 3, 4) },
        uViewMode: { value: viewModeInt },
        uBaseShape: { value: baseShape },
        uBaseDims: { value: new THREE.Vector3(...baseDims) },
        uBaseColor: { value: new THREE.Vector3(...baseColor) },
        uMetalness: { value: metalness },
        uRoughness: { value: roughness },
        uClipPlane: { value: new THREE.Vector3(...clipPlane) },
        uClipOffset: { value: clipOffset },
        uClipEnabled: { value: clipEnabled ? 1.0 : 0.0 },
        uPhysicsDataEnabled: { value: physicsData ? 1.0 : 0.0 },
        uPhysicsTexture: { value: new THREE.Texture() }, // Safety dummy texture
        uComponentCount: { value: 0 },
        uBgColor1: { value: new THREE.Vector3(0.02, 0.03, 0.05) },
        uBgColor2: { value: new THREE.Vector3(0.0, 0.0, 0.0) },
        uGridEnabled: { value: 1.0 },
        uGridColor: { value: new THREE.Vector3(0.1, 0.1, 0.1) }
    }), []);

    // Update uniforms when props change
    useEffect(() => {
        if (materialRef.current) {
            materialRef.current.uniforms.uViewMode.value = viewModeInt;
            materialRef.current.uniforms.uBaseShape.value = baseShape;
            materialRef.current.uniforms.uBaseDims.value.set(...baseDims);
            materialRef.current.uniforms.uBaseColor.value.set(...baseColor);
            materialRef.current.uniforms.uMetalness.value = metalness;
            materialRef.current.uniforms.uRoughness.value = roughness;
            materialRef.current.uniforms.uClipPlane.value.set(...clipPlane);
            materialRef.current.uniforms.uClipOffset.value = clipOffset;
            materialRef.current.uniforms.uClipEnabled.value = clipEnabled ? 1.0 : 0.0;

            // Dynamic Theme Integration
            if (theme && theme.colors) {
                // Use Tertiary for clearer theme identity (lighter sky)
                const bg1 = new THREE.Color(theme.colors.bg.primary);
                const bg2 = new THREE.Color(theme.colors.bg.tertiary || theme.colors.bg.secondary);

                // Use Accent for Grid (e.g. Gold for White theme, Blue for Dark)
                // Fallback to primary border if accent not defined
                const gridColorStr = theme.colors.border.accent || theme.colors.border.primary;
                const grid = new THREE.Color(gridColorStr);

                materialRef.current.uniforms.uBgColor1.value.set(bg1.r, bg1.g, bg1.b);
                materialRef.current.uniforms.uBgColor2.value.set(bg2.r, bg2.g, bg2.b);
                materialRef.current.uniforms.uGridColor.value.set(grid.r, grid.g, grid.b);
            }
        }
    }, [viewModeInt, baseShape, baseDims, baseColor, metalness, roughness, clipPlane, clipOffset, clipEnabled, theme]);

    // Animate time, resolution, and camera position per frame
    useFrame((state) => {
        if (materialRef.current) {
            materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
            materialRef.current.uniforms.uResolution.value.set(size.width, size.height);
            materialRef.current.uniforms.uCameraPos.value.copy(state.camera.position);
        }
    });

    // WebGL 2 detection and error handling
    useEffect(() => {
        const isWebGL2 = gl.capabilities.isWebGL2;
        if (!isWebGL2) {
            console.warn('[UnifiedSDFRenderer] WebGL 2 not available. Some features may not work.');
        }

        // Context loss handling
        const handleContextLost = (e) => {
            e.preventDefault();
            console.warn('[UnifiedSDFRenderer] WebGL context lost, preventing crash.');
        };
        const handleContextRestored = () => {
            console.log('[UnifiedSDFRenderer] WebGL context restored.');
        };

        gl.domElement.addEventListener('webglcontextlost', handleContextLost, false);
        gl.domElement.addEventListener('webglcontextrestored', handleContextRestored, false);

        return () => {
            gl.domElement.removeEventListener('webglcontextlost', handleContextLost);
            gl.domElement.removeEventListener('webglcontextrestored', handleContextRestored);
        };
    }, [gl]);

    return (
        <mesh ref={meshRef}>
            <planeGeometry args={[2, 2]} />
            <shaderMaterial
                ref={materialRef}
                vertexShader={vertexShader}
                fragmentShader={fragmentShader}
                uniforms={uniforms}
                transparent={false}
                depthWrite={true}
                side={THREE.DoubleSide}
            />
        </mesh>
    );
};

// Main exported component with Canvas wrapper
const UnifiedSDFRenderer = ({
    design = null,
    viewMode = 'realistic',
    clipPlane = null,
    physicsData = null,
    className = '',
    style = {}
}) => {
    const { theme, currentTheme } = useTheme();
    const controlsRef = useRef();
    const cameraStateRef = useRef({ position: [4, 3, 4], target: [0, 0, 0] });

    // Extract geometry from design
    const geometryState = useMemo(() => {
        const state = {
            baseShape: 1, // Default to Box
            baseDims: [1, 1, 1],
            baseColor: [0.8, 0.8, 0.8],
            metalness: 0.5,
            roughness: 0.5
        };

        // design is activeTab which has .content as JSON string
        if (design?.content) {
            try {
                const asset = typeof design.content === 'string'
                    ? JSON.parse(design.content)
                    : design.content;

                if (asset.type === 'primitive') {
                    const { geometry, args = [], material = {} } = asset;

                    if (geometry === 'box') {
                        state.baseShape = 1;
                        state.baseDims = args.length >= 3 ? args.slice(0, 3) : [1, 1, 1];
                    } else if (geometry === 'sphere') {
                        state.baseShape = 2;
                        state.baseDims = [args[0] || 0.5, 0, 0];
                    } else if (geometry === 'cylinder') {
                        state.baseShape = 3;
                        state.baseDims = [args[0] || 0.5, args[1] || 1, 0];
                    }

                    if (material.color) {
                        const color = new THREE.Color(material.color);
                        state.baseColor = [color.r, color.g, color.b];
                    }
                    state.metalness = material.metalness ?? 0.5;
                    state.roughness = material.roughness ?? 0.5;
                }
            } catch (e) {
                console.warn('[UnifiedSDFRenderer] Failed to parse design content:', e);
            }
        }

        return state;
    }, [design]);

    // Handle camera state persistence
    const handleCameraChange = useCallback(() => {
        if (controlsRef.current) {
            const controls = controlsRef.current;
            cameraStateRef.current = {
                position: controls.object.position.toArray(),
                target: controls.target.toArray()
            };
        }
    }, []);

    // Error boundary for shader compilation
    const handleShaderError = useCallback((error) => {
        console.error('[UnifiedSDFRenderer] Shader compilation error:', error);
    }, []);

    return (
        <div className={`w-full h-full ${className}`} style={style}>
            <Canvas
                key={design?.id || 'sim-canvas'}
                gl={{
                    antialias: true,
                    powerPreference: 'high-performance',
                    failIfMajorPerformanceCaveat: false,
                    preserveDrawingBuffer: true
                }}
                onCreated={({ gl }) => {
                    // Log WebGL info
                    const info = gl.getContext().getParameter(gl.getContext().VERSION);
                    console.log('[UnifiedSDFRenderer] WebGL:', info);
                }}
            >
                <PerspectiveCamera
                    makeDefault
                    position={cameraStateRef.current.position}
                    fov={35}
                />
                <OrbitControls
                    ref={controlsRef}
                    enableDamping
                    dampingFactor={0.05}
                    target={cameraStateRef.current.target}
                    onEnd={handleCameraChange}
                />
                <SDFKernel
                    viewMode={viewMode}
                    baseShape={geometryState.baseShape}
                    baseDims={geometryState.baseDims}
                    baseColor={geometryState.baseColor}
                    metalness={geometryState.metalness}
                    roughness={geometryState.roughness}
                    clipPlane={clipPlane?.normal || [0, 1, 0]}
                    clipOffset={clipPlane?.offset || 0}
                    clipEnabled={!!clipPlane?.enabled}
                    physicsData={physicsData}
                    onShaderError={handleShaderError}
                    theme={theme}
                />
                <PhysicsOverlays
                    viewMode={viewMode}
                    physicsData={physicsData}
                    baseDims={geometryState.baseDims}
                />
            </Canvas>
        </div>
    );
};

export default UnifiedSDFRenderer;
