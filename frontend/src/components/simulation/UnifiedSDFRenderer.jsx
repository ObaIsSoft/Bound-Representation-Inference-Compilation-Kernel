import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import { useTheme } from '../../contexts/ThemeContext';
import { useSettings } from '../../contexts/SettingsContext';
import { StandardMeshPreview } from './StandardMeshPreview';
import { NeuralSDF } from './NeuralSDF';
import { StrokeCapture } from './StrokeCapture'; // New
import { RegionSelector } from './RegionSelector'; // Phase 8.4
import { RegionInteraction } from './RegionInteraction'; // Phase 8.5
import { useSimulation } from '../../contexts/SimulationContext';
// import PhysicsOverlays from './PhysicsOverlays'; // Disabled temporarily

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
    THERMAL: 4,  // Renamed from HEATMAP (Phase 10)
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
    'thermal': VIEW_MODES.THERMAL,  // Renamed from 'heatmap'
    'cutaway': VIEW_MODES.CUTAWAY,
    'solid': VIEW_MODES.SOLID,
    'stress': VIEW_MODES.STRESS,
    'hidden_line': VIEW_MODES.HIDDEN_LINE,
    'shaded': VIEW_MODES.SHADED,
    'interior': VIEW_MODES.INTERIOR,
    'hyperrealism': VIEW_MODES.REALISTIC,
    'micro': VIEW_MODES.REALISTIC,
    'micro_wgsl': VIEW_MODES.REALISTIC,
    'flow': VIEW_MODES.FLOW
};

// Inner kernel component (must be inside Canvas)
const SDFKernel = ({
    viewMode = 'realistic',
    showGrid = true,
    baseShape = 1,
    baseDims = [1, 1, 1],
    baseColor = [0.8, 0.8, 0.8],
    metalness = 0.0,
    roughness = 0.5,
    clipPlane = [0, 1, 0],
    clipOffset = 0,
    clipEnabled = false,
    physicsData = null,
    contextPhysicsData = null, // Phase 10: Fix scope error
    onShaderError = null,
    theme = 'dark', // Default prop
    meshRenderingMode = 'sdf', // 'sdf' or 'preview'
    design = null, // [FIX] Added missing prop
    sketchPoints = [] // Phase 9.3: Light Pen data
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

        // Mesh SDF Uniforms (Phase 8)
        uMeshSDFEnabled: { value: false },
        uMeshSDFTexture: { value: null },
        uAtlasComponentCount: { value: 0 },
        // We need an array of structs for uComponents. 
        // Three.js handles array of structs via property flattening or careful object construction if UniformsUtils is used.
        // However, raw ShaderMaterial usually expects uComponents[0].transform etc.
        // Best practice in React Three Fiber / raw shader is to initialize with correctly structured objects.
        uComponents: {
            value: new Array(16).fill(0).map(() => ({
                atlasOffset: new THREE.Vector3(),
                atlasScale: new THREE.Vector3(),
                localBounds: [new THREE.Vector3(), new THREE.Vector3()],
                sdfRange: new THREE.Vector2(),
                transform: new THREE.Matrix4()
            }))
        },

        // Phase 9.3: Semantic Sketch Primitives
        uPrimitiveCount: { value: 0 },
        uPrimitives: {
            value: new Array(16).fill(0).map(() => ({
                type: 0,
                p0: new THREE.Vector3(),
                p1: new THREE.Vector3(),
                radius: 0.1,
                blendStrength: 0.2
            }))
        },

        // Phase 9.3: Light Pen Sketching
        uSketchCount: { value: 0 },
        uSketchPoints: { value: new Array(128).fill(0).map(() => new THREE.Vector3()) },

        // Phase 10: Thermal Data
        uThermalTemp: { value: 20.0 }, // Default room temp
        uThermalEnabled: { value: false },

        // Phase 2: Stress Visuals
        uStressLevel: { value: 0.0 },
        uMaxDeflection: { value: 0.0 },

        // Single Mesh Fallback (Wrapped in atlas logic or legacy)
        uMeshBounds: { value: [new THREE.Vector3(0, 0, 0), new THREE.Vector3(1, 1, 1)] },
        uMeshSDF_min: { value: 0.0 },
        uMeshSDF_max: { value: 1.0 },

        uBgColor1: { value: new THREE.Vector3(0.4, 0.5, 0.7) }, // Visible sky blue
        uBgColor2: { value: new THREE.Vector3(0.1, 0.12, 0.18) }, // Dark blue
        uGridEnabled: { value: 1.0 },
        uGridColor: { value: new THREE.Vector3(0.1, 0.1, 0.1) }
    }), []);

    // Update uniforms when props change
    useEffect(() => {
        if (materialRef.current) {
            materialRef.current.uniforms.uViewMode.value = viewModeInt;

            // If in preview mode OR if baseShape is 0 (OpenSCAD/Fallback), render nothing in SDF
            // This allows the fallback mesh to be seen
            materialRef.current.uniforms.uBaseShape.value = (meshRenderingMode === 'preview' || baseShape === 0) ? 0 : baseShape;

            materialRef.current.uniforms.uBaseDims.value.set(...baseDims);
            materialRef.current.uniforms.uBaseColor.value.set(...baseColor);
            materialRef.current.uniforms.uMetalness.value = metalness;
            materialRef.current.uniforms.uRoughness.value = roughness;
            materialRef.current.uniforms.uClipPlane.value.set(...clipPlane);
            materialRef.current.uniforms.uClipOffset.value = clipOffset;
            materialRef.current.uniforms.uClipEnabled.value = clipEnabled ? 1.0 : 0.0;

            // Debug logging for view mode changes
            console.log('[VIEW MODE]', viewMode, '→', viewModeInt, 'Thermal?', viewMode === 'thermal');

            // Dynamic Theme Integration
            if (theme && theme.colors) {
                const parseColor = (c) => {
                    if (!c) return '#4080c0'; // Fallback to visible blue
                    if (c.includes('#')) return c.substring(0, 7);
                    if (c.includes('rgba')) return c.replace('rgba', 'rgb').split(',').slice(0, 3).join(',') + ')';
                    return c;
                };

                const bg1 = new THREE.Color(parseColor(theme.colors.bg?.primary));
                const bg2 = new THREE.Color(parseColor(theme.colors.bg?.tertiary || theme.colors.bg?.secondary));
                const gridColorStr = theme.colors.border?.accent || theme.colors.border?.primary || '#3b82f6';
                const gridColor = new THREE.Color(parseColor(gridColorStr));

                // FIX: Use .set() with r,g,b - Vector3.copy(Color) doesn't work!
                materialRef.current.uniforms.uBgColor1.value.set(bg1.r, bg1.g, bg1.b);
                materialRef.current.uniforms.uBgColor2.value.set(bg2.r, bg2.g, bg2.b);
                materialRef.current.uniforms.uGridColor.value.set(gridColor.r, gridColor.g, gridColor.b);
                materialRef.current.uniforms.uGridEnabled.value = showGrid ? 1.0 : 0.0;
            } else {
                // No theme - use visible defaults
                materialRef.current.uniforms.uBgColor1.value.set(0.4, 0.5, 0.7);
                materialRef.current.uniforms.uBgColor2.value.set(0.1, 0.12, 0.18);
                materialRef.current.uniforms.uGridColor.value.set(0.1, 0.1, 0.1);
                materialRef.current.uniforms.uGridEnabled.value = showGrid ? 1.0 : 0.0;
            }

            // Phase 8.4: Control Mesh SDF Enabled based on settings
            // If data is loaded (checked in other effect), we still respect this switch
            if (materialRef.current && materialRef.current.uniforms.uMeshSDFTexture.value) {
                materialRef.current.uniforms.uMeshSDFEnabled.value = (meshRenderingMode === 'sdf');
            }
            // Note: The specific texture loading effect below also sets it to true, so we need to sync.
        }
    }, [viewModeInt, baseShape, baseDims, baseColor, metalness, roughness, clipPlane, clipOffset, clipEnabled, theme, meshRenderingMode, showGrid]);

    // Phase 10: Update Physics Uniforms (Thermal & Stress)
    useEffect(() => {
        if (!materialRef.current) return;

        // Use context data if available, otherwise fallback to prop
        const data = contextPhysicsData || physicsData;

        // 1. Structural Analysis
        let stressLevel = 0.0;
        let deflection = 0.0;

        if (data && (data.structural || data.sub_agent_reports?.structural)) {
            const structural = data.structural || data.sub_agent_reports?.structural;
            // Safety Factor: < 1.0 is bad. Map SF to 0..1 range for visualization intensity
            // SF = 1.0 -> Stress = 1.0 (Critical)
            // SF = 2.0 -> Stress = 0.5 (Safe)
            const sf = structural.safety_factor || structural.safety_factor_yield || 100.0;
            stressLevel = 1.0 / Math.max(0.1, sf); // Inverted SF
            deflection = structural.max_deflection_m || structural.deflection || 0.0;
        }

        if (materialRef.current.uniforms.uStressLevel) {
            materialRef.current.uniforms.uStressLevel.value = stressLevel;
        }
        if (materialRef.current.uniforms.uMaxDeflection) {
            materialRef.current.uniforms.uMaxDeflection.value = deflection;
        }

        // 2. Thermal Analysis
        let temp = 20.0; // Ambient
        if (data && (data.thermal || data.sub_agent_reports?.thermal)) {
            const thermal = data.thermal || data.sub_agent_reports?.thermal;
            temp = thermal.equilibrium_temp_c || thermal.temp || 20.0;
        }

        if (materialRef.current.uniforms.uThermalTemp) {
            materialRef.current.uniforms.uThermalTemp.value = temp;
        }
        if (materialRef.current.uniforms.uThermalEnabled) {
            // Redundant with viewMode but good for explicit override
            materialRef.current.uniforms.uThermalEnabled.value = (viewMode === 'thermal');
        }

    }, [physicsData, contextPhysicsData, viewMode]);



    // Animate time, resolution, and camera position per frame
    useFrame((state) => {
        if (materialRef.current) {
            materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
            materialRef.current.uniforms.uResolution.value.set(size.width, size.height);
            materialRef.current.uniforms.uCameraPos.value.copy(state.camera.position);

            // Phase 9.3: Sync Sketch Points
            if (sketchPoints && sketchPoints.length > 0) {
                materialRef.current.uniforms.uSketchCount.value = Math.min(sketchPoints.length, 128);
                sketchPoints.forEach((pt, i) => {
                    if (i < 128) materialRef.current.uniforms.uSketchPoints.value[i].copy(pt);
                });
            } else {
                materialRef.current.uniforms.uSketchCount.value = 0;
            }
        }
    });

    // Load Mesh SDF Texture (Phase 8)
    useEffect(() => {
        if (!design?.mesh_sdf_data || !materialRef.current) return;

        try {
            // Decode base64 texture data
            const binary = atob(design.mesh_sdf_data);
            const length = binary.length;
            const array = new Float32Array(length / 4); // Float32 = 4 bytes

            // Convert binary string to Float32Array
            const dataView = new DataView(new ArrayBuffer(length));
            for (let i = 0; i < length; i++) {
                dataView.setUint8(i, binary.charCodeAt(i));
            }
            for (let i = 0; i < array.length; i++) {
                array[i] = dataView.getFloat32(i * 4, true); // true = little-endian
            }

            const resolution = design.sdf_resolution || 64;

            // Create 3D texture
            const texture = new THREE.DataTexture3D(
                array,
                resolution,
                resolution,
                resolution
            );
            texture.format = THREE.RedFormat;
            texture.type = THREE.FloatType;
            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.LinearFilter;
            texture.wrapS = THREE.ClampToEdgeWrapping;
            texture.wrapT = THREE.ClampToEdgeWrapping;
            texture.wrapR = THREE.ClampToEdgeWrapping;
            texture.needsUpdate = true;

            // Update uniforms
            materialRef.current.uniforms.uMeshSDFTexture.value = texture;
            // Respect current mode setting immediately
            materialRef.current.uniforms.uMeshSDFEnabled.value = (meshRenderingMode === 'sdf');

            if (design.sdf_bounds) {
                const [min, max] = design.sdf_bounds;
                materialRef.current.uniforms.uMeshBounds.value[0].set(min[0], min[1], min[2]);
                materialRef.current.uniforms.uMeshBounds.value[1].set(max[0], max[1], max[2]);
            }

            if (design.sdf_range) {
                const [sdf_min, sdf_max] = design.sdf_range;
                materialRef.current.uniforms.uMeshSDF_min.value = sdf_min;
                materialRef.current.uniforms.uMeshSDF_max.value = sdf_max;
            }

            console.log(`[MeshSDF] Loaded ${resolution}³ texture from design`);

            // Phase 8.5: Load Atlas Manifest
            if (design.manifest && Array.isArray(design.manifest)) {
                console.log(`[MeshSDF] Loading Atlas Manifest (${design.manifest.length} components)`);
                materialRef.current.uniforms.uAtlasComponentCount.value = design.manifest.length;

                design.manifest.forEach((comp, i) => {
                    if (i >= 16) return; // limit

                    const uComp = materialRef.current.uniforms.uComponents.value[i];

                    // Atlas UV
                    if (comp.atlas_offset) uComp.atlasOffset.set(...comp.atlas_offset);
                    if (comp.atlas_scale) uComp.atlasScale.set(...comp.atlas_scale);

                    // SDF Range
                    if (comp.sdf_range) uComp.sdfRange.set(...comp.sdf_range);

                    // Local Bounds
                    if (comp.local_bounds) {
                        uComp.localBounds[0].set(...comp.local_bounds[0]);
                        uComp.localBounds[1].set(...comp.local_bounds[1]);
                    }

                    // Transform (Identity for now, but will be wired to Physics/Scatter later)
                    uComp.transform.identity();

                    // If we want to support initial transforms from GLTF, we'd read them here.
                    // But usually we bake centered.
                });
            } else {
                // Fallback for Legacy Single-Mesh (treat as 1 component)
                // If we have data but no manifest, we assume it's a single baked mesh
                console.log('[MeshSDF] No manifest found, assuming single legacy mesh.');
                materialRef.current.uniforms.uAtlasComponentCount.value = 1;
                const uComp = materialRef.current.uniforms.uComponents.value[0];

                // Full Texture usage
                uComp.atlasOffset.set(0, 0, 0);
                uComp.atlasScale.set(1, 1, 1);

                if (design.sdf_range) uComp.sdfRange.set(...design.sdf_range);

                if (design.sdf_bounds) {
                    uComp.localBounds[0].set(...design.sdf_bounds[0]);
                    uComp.localBounds[1].set(...design.sdf_bounds[1]);
                }

                uComp.transform.identity();
            }

        } catch (error) {
            console.error('[MeshSDF] Failed to load texture:', error);
        }



    }, [design?.mesh_sdf_data, design?.sdf_resolution, design?.sdf_bounds, design?.sdf_range, meshRenderingMode]);

    // Phase 9.3: Update Primitives Uniforms
    useEffect(() => {
        if (!materialRef.current || !design?.primitives) return;

        const prims = design.primitives;
        materialRef.current.uniforms.uPrimitiveCount.value = prims.length;

        prims.forEach((p, i) => {
            if (i >= 16) return;
            const uPrim = materialRef.current.uniforms.uPrimitives.value[i];

            uPrim.type = p.type === 'capsule' ? 1 : 0; // 1 = Capsule
            if (p.start) uPrim.p0.set(...p.start);
            if (p.end) uPrim.p1.set(...p.end);
            uPrim.radius = p.radius || 0.1;
            uPrim.blendStrength = p.blend || 0.2;
        });

    }, [design?.primitives]);


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

    // Phase 10: Update Thermal Data from Physics Report
    useEffect(() => {
        if (!materialRef.current) return;

        // MOCK DATA FOR TESTING (Phase 10 verification)
        // TODO: Replace with real data from design.physics_report
        const useMockData = false;

        if (useMockData) {
            // Simulate hot component (200°C) for testing
            materialRef.current.uniforms.uThermalTemp.value = 200.0;
            materialRef.current.uniforms.uThermalEnabled.value = true;
            console.log('[THERMAL] Mock data: 200°C');
        } else {
            // Check if design has thermal data from PhysicsAgent sub_agent_reports
            const thermalData = design?.physics_report?.sub_agent_reports?.thermal;

            if (thermalData && thermalData.equilibrium_temp_c !== undefined) {
                materialRef.current.uniforms.uThermalTemp.value = thermalData.equilibrium_temp_c;
                materialRef.current.uniforms.uThermalEnabled.value = true;
            } else {
                // No thermal data available, fall back to default
                materialRef.current.uniforms.uThermalTemp.value = 20.0;
                materialRef.current.uniforms.uThermalEnabled.value = false;
            }
        }
    }, [design?.physics_report]);

    return (
        <mesh ref={meshRef} frustumCulled={false}>
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

// Recursive ISA: Ghost Mesh for Context Visualization
// Renders a low-opacity wireframe or bounding box to represent
// the "Parent Assembly" when we are focused on a specific "Pod".
const GhostMesh = ({ baseDims, theme, parentDims }) => {
    // Default to a reasonable scale if no parent context is provided
    const dims = parentDims || [baseDims[0] * 2, baseDims[1] * 2, baseDims[2] * 2];

    return (
        <group>
            {/* Context Bounding Box (Ghost) */}
            <mesh>
                <boxGeometry args={dims} />
                <meshBasicMaterial
                    color={theme?.colors?.text?.muted || '#555'}
                    wireframe
                    transparent
                    opacity={0.05} // Lower opacity to be less obtrusive
                />
            </mesh>
            {/* GridHelper Removed to reduce clutter */}
        </group>
    );
};


// Main exported component with Canvas wrapper
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
    const {
        meshRenderingMode,
        showGrid,
        showControlsHelp,
        setShowControlsHelp
    } = useSettings();
    const {
        sketchMode,
        sketchPoints,
        focusedPodId,
        isaTree,
        physicsData: contextPhysicsData // Phase 10: From Context
    } = useSimulation();
    const controlsRef = useRef();
    const cameraStateRef = useRef({ position: [4, 3, 4], target: [0, 0, 0] });

    // Helper: Find parent of focused pod to get dimensions
    const getParentDims = () => {
        if (!focusedPodId || !isaTree) return null;

        const findParent = (node, targetId) => {
            if (!node.children) return null;
            for (let child of node.children) {
                if (child.id === targetId) return node;
                const found = findParent(child, targetId);
                if (found) return found;
            }
            return null;
        };

        const parent = findParent(isaTree, focusedPodId);
        if (parent && parent.constraints) {
            // Heuristic to extract dimensions from constraints
            // e.g. length/width/height or radius
            const c = parent.constraints;
            const l = c.length || c.length_m || 1.0;
            const w = c.width || c.width_m || c.radius || 1.0;
            const h = c.height || c.height_m || c.radius || 1.0;
            return [Number(w), Number(h), Number(l)];
            // Note: Mapping depends on axis orientation, assuming Y-up (h)
        }
        return null;
    };

    const parentDims = getParentDims();


    // Phase 8.4: Region Selection State
    const [showRegionSelector, setShowRegionSelector] = React.useState(false);
    const [selectedRegion, setSelectedRegion] = React.useState(null);

    // Extract geometry from design
    const geometryState = useMemo(() => {
        const state = {
            baseShape: 1, // Default to Box
            baseDims: [1, 1, 1],
            baseColor: [0.8, 0.8, 0.8],
            metalness: 0.5,
            roughness: 0.5
        };

        // STRICT MODE: No heuristics to hide.
        // Instead, try to parse simple geometry to show SOMETHING better than a generic box.

        // Simple Parser for Live SDF Preview
        // Supported: cube(), sphere(), cylinder()
        // This maps the first valid command it finds to the uBaseShape logic for immediate feedback

        if (design?.content && typeof design.content === 'string') {
            const c = design.content;

            // [Phase 17 Fix] Check for Complexity FIRST
            // If the code contains structural keywords, it implies a CSG tree that requires
            // the backend SDF Grid generator. Do NOT try to match primitives here.
            const isComplex = c.includes('module ') ||
                c.includes('difference') ||
                c.includes('union') ||
                c.includes('intersection') ||
                c.includes('hull') ||
                c.includes('import');

            if (isComplex) {
                // Return default state (Box) but let useEffect fetch the real grid
                return state;
            }

            // Sphere Detection
            // Examples: sphere(10); sphere(r=5);
            const sphereMatch = c.match(/sphere\s*\(\s*(?:r\s*=\s*)?([0-9.]+)\s*\)/);
            if (sphereMatch) {
                state.baseShape = 2; // 2 = Sphere
                const r = parseFloat(sphereMatch[1]);
                state.baseDims = [r, r, r];
                return state;
            }

            // Cube Detection
            // Examples: cube([10, 20, 30]); cube(10);
            const cubeMatch = c.match(/cube\s*\(\s*(?:size\s*=\s*)?(?:\[([0-9.,\s]+)\]|([0-9.]+))\s*\)/);
            if (cubeMatch) {
                state.baseShape = 1; // 1 = Box
                if (cubeMatch[1]) {
                    // Vector [x, y, z]
                    const dims = cubeMatch[1].split(',').map(s => parseFloat(s.trim()));
                    state.baseDims = [dims[0] || 1, dims[1] || 1, dims[2] || 1];
                } else if (cubeMatch[2]) {
                    // Scalar length
                    const s = parseFloat(cubeMatch[2]);
                    state.baseDims = [s, s, s];
                }
                return state;
            }

            // Cylinder Detection (Mapped to Capsule for now, or new Cylinder shape id)
            // cylinder(h=10, r=5)
            if (c.includes('cylinder')) {
                state.baseShape = 3; // 3 = Cylinder/Capsule
                // Default dims for cylinder
                state.baseDims = [1, 5, 1]; // r, h, r

                const rMatch = c.match(/r\s*=\s*([0-9.]+)/) || c.match(/r1\s*=\s*([0-9.]+)/);
                const hMatch = c.match(/h\s*=\s*([0-9.]+)/);

                if (rMatch) state.baseDims[0] = parseFloat(rMatch[1]);
                if (hMatch) state.baseDims[1] = parseFloat(hMatch[1]);

                return state;
            }
        }

        return state;
    }, [design?.content, design?.type]);

    // Phase 17: Full SDF Support - Fetch SDF Grid for complex code
    const [localSDFData, setLocalSDFData] = useState(null);
    const [isFetchingSDF, setIsFetchingSDF] = useState(false);

    // Phase 9: Real-Time SDF Baking Support (Moved up to fix ReferenceError)
    const [sdfOverride, setSdfOverride] = useState(null);

    const handleSDFLoaded = useCallback((data) => {
        console.log('[UnifiedSDFRenderer] Received baked SDF data from OpenSCAD:', data);
        setSdfOverride(data);
    }, []);

    useEffect(() => {
        // Reset local data when design changes
        setLocalSDFData(null);
    }, [design?.id]);

    useEffect(() => {
        // Condition: 
        // 1. SDF Mode is Active
        // 2. We have content (code)
        // 3. It's NOT a primitive (baseShape is 1=Box by default or 0)
        // 4. We don't have pre-baked mesh_sdf_data

        if (meshRenderingMode !== 'sdf') return;
        if (!design?.content) return;
        if (design.mesh_sdf_data) return; // Already has it

        // If primitive parser found a shape matches (Sphere=2, Cylinder=3), we likely don't need grid
        // But if it's Box (1), it might be default? 
        // Actually, primitive parser returns 1 for Cube. Default is also 1. 
        // Let's assume if it contains '{' or 'module' it needs Grid.

        const isComplex = typeof design.content === 'string' && (
            design.content.includes('module ') ||
            design.content.includes('difference') ||
            design.content.includes('union') ||
            design.content.includes('intersection') ||
            design.content.includes('import')
        );

        if (!isComplex) return;

        // Debounce fetch
        const timer = setTimeout(async () => {
            if (isFetchingSDF) return;
            setIsFetchingSDF(true);
            console.log('[UnifiedSDF] Fetching SDF Grid for design...');

            try {
                const res = await fetch('http://localhost:8000/api/openscad/compile', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        code: design.content,
                        format: 'sdf_grid',
                        resolution: 64 // Start with 64 for speed
                    })
                });

                const data = await res.json();
                if (data.success && data.sdf_data) {
                    console.log('[UnifiedSDF] Received SDF Grid');
                    setLocalSDFData(data);
                }
            } catch (e) {
                console.error('[UnifiedSDF] Failed to fetch SDF:', e);
            } finally {
                setIsFetchingSDF(false);
            }
        }, 1000);

        return () => clearTimeout(timer);
    }, [design?.content, meshRenderingMode, design?.mesh_sdf_data]);

    // Merge Local Data into Helper for Texture Loading
    const effectiveDesign = useMemo(() => {
        let base = { ...design };

        // Merge fetched SDF (Phase 17)
        if (localSDFData) {
            base = {
                ...base,
                mesh_sdf_data: localSDFData.sdf_data,
                sdf_resolution: localSDFData.resolution,
                sdf_bounds: localSDFData.bounds
            };
        }

        // Merge override (Phase 9 - e.g. from OpenSCADMesh bake)
        if (sdfOverride) {
            base = { ...base, ...sdfOverride };
        }

        return base;
    }, [design, localSDFData, sdfOverride]);

    // Re-use logic for effectiveDesign below...



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

    // Phase 8.4: Show Region Selector on Micro-Machining Mode Entry
    React.useEffect(() => {
        const isMicroMode = viewMode === 'micro' || viewMode === 'micro_wgsl';

        // Reset when leaving only
        if (!isMicroMode) {
            setSelectedRegion(null);
        } else {
            // Show selector if not selected
            setShowRegionSelector(!selectedRegion);
        }
    }, [viewMode, selectedRegion]);
    const handleRegionSelected = React.useCallback((region) => {
        console.log('[MicroMachining] Region selected:', region);
        setSelectedRegion(region);
        setShowRegionSelector(false);
        // TODO: Trigger API call to train neural network for this region
    }, []);

    const handleRegionCancel = React.useCallback(() => {
        console.log('[MicroMachining] Region selection cancelled');
        setShowRegionSelector(false);
        // TODO: Exit Micro-Machining mode or show alternate message
    }, []);

    // (Moved sdfOverride state to top)

    // Merge duplicate effectiveDesign logic
    // We already defined effectiveDesign above (Phase 17), so we don't redeclare it here.
    // However, we need to ensure THAT definition included sdfOverride.
    // Since I cannot edit the previous block easily without context, I will COMMENT OUT this one
    // and rely on the fact that I will update the FIRST definition to include sdfOverride.

    // WAIT: I should just replace this block with NOTHING if I update the first one.
    // Or, better, I will replace THIS block with a comment, and update the FIRST block (in a separate call if needed, or I can try to find where it is).

    // Actually, I should update the FIRST definition to:
    // const effectiveDesign = useMemo(() => { ... merge localSDFData AND sdfOverride ... }, [design, localSDFData, sdfOverride]);

    // Since I can't reach the first definition in this tool call (it's at line ~712), I will delete this redeclaration first.
    // Then I will update the first one.

    // Replacement for lines 804-815:
    // Just keep the state and handler, remove effectiveDesign.

    /* 
       Merged effectiveDesign logic is handled in Phase 17 block above.
       We keep state/handlers here.
    */

    return (
        <div className={`w-full h-full ${className}`} style={style}>
            <Canvas
                key="main-sim-canvas" // STATIC KEY: Critical for MorphPlayer 60fps (Prevents Re-Mount)
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
                frameloop={(viewMode === 'micro' || viewMode === 'micro_wgsl') ? 'demand' : 'always'}
            >
                <PerspectiveCamera
                    makeDefault
                    position={cameraStateRef.current.position}
                    fov={35}
                />
                <OrbitControls
                    makeDefault
                    ref={controlsRef}
                    enableDamping
                    dampingFactor={0.05}
                    enablePan={true}
                    screenSpacePanning={true} // Allow panning up/down (critical for F-22 tail inspection)
                    zoomSpeed={1.2}
                    panSpeed={1.0}
                    rotateSpeed={0.8}
                    target={cameraStateRef.current.target}
                    onEnd={handleCameraChange}
                    enabled={!sketchMode}
                    mouseButtons={{
                        LEFT: THREE.MOUSE.ROTATE,
                        MIDDLE: THREE.MOUSE.DOLLY,
                        RIGHT: THREE.MOUSE.PAN
                    }}
                />

                {/* 0. Sketch Layer */}
                <StrokeCapture enabled={sketchMode} />

                {/* 0.5 Region Selection Layer */}
                {showRegionSelector && (
                    <RegionInteraction
                        enabled={showRegionSelector}
                        onRegionSelected={handleRegionSelected}
                        baseDims={geometryState.baseDims}
                    />
                )}

                {/* 1. SDF Background & Kernel (Always rendered for background/grid) */}
                <SDFKernel
                    viewMode={viewMode}
                    showGrid={showGrid}
                    baseShape={geometryState.baseShape}
                    baseDims={geometryState.baseDims}
                    baseColor={geometryState.baseColor}
                    metalness={geometryState.metalness}
                    roughness={geometryState.roughness}
                    clipPlane={clipPlane?.normal || [0, 1, 0]}
                    clipOffset={clipPlane?.offset || 0}
                    clipEnabled={!!clipPlane?.enabled}
                    physicsData={physicsData}
                    contextPhysicsData={contextPhysicsData} // Phase 10: Pass context data
                    onShaderError={handleShaderError}
                    theme={theme}
                    meshRenderingMode={sdfOverride ? 'sdf' : meshRenderingMode} // Auto-switch to SDF if data exists
                    design={effectiveDesign} // [FIX] Passing design prop
                    sketchPoints={sketchPoints} // Phase 9.3
                />

                {/* 2. Preview Mode Overlays (Standard Mesh Rasterization) */}
                {/* Fallback to Mesh if explicitly requested OR if SDF cannot handle the geometry (baseShape === 0) */}
                {/* AND if we don't have an SDF override yet */}
                {(!sdfOverride && (meshRenderingMode === 'preview' || geometryState.baseShape === 0)) && (
                    <StandardMeshPreview
                        design={design}
                        viewMode={viewMode}
                        theme={theme}
                        physicsData={physicsData || contextPhysicsData} // Phase 18: Pass physics data
                        onSDFLoaded={handleSDFLoaded}
                    />
                )}


                {/* 3. Neural SDF Mode (Micro-Machining) */}
                {(viewMode === 'micro' || viewMode === 'micro_wgsl') && selectedRegion && (
                    <NeuralSDF design={design} region={selectedRegion} />
                )}

                {/* 4. Recursive ISA: Ghost Mode (Context Awareness) */}
                {/* If a pod is focused, we render the 'rest of the world' as a ghost */}
                {focusedPodId && (
                    <GhostMesh
                        baseDims={geometryState.baseDims}
                        theme={theme}
                    />
                )}


                {/* PhysicsOverlays disabled temporarily - will fix later
                <PhysicsOverlays
                    viewMode={viewMode}
                    physicsData={physicsData}
                    baseDims={geometryState.baseDims}
                />
                */}
            </Canvas>

            {/* 4. Region Selector Overlay (Outside Canvas) */}
            {showRegionSelector && (
                <RegionSelector
                    onRegionSelected={handleRegionSelected}
                    onCancel={handleRegionCancel}
                />
            )}

            {/* 4. Region Selector Overlay (Outside Canvas) */}
            {showRegionSelector && (
                <RegionSelector
                    onRegionSelected={handleRegionSelected}
                    onCancel={handleRegionCancel}
                />
            )}

            {/* 5. Interactive Tour / Controls Help */}
            {showControlsHelp && <ControlsTour onClose={() => setShowControlsHelp(false)} />}
        </div>
    );
};

// Simple Help Overlay Component
const ControlsTour = ({ onClose }) => {
    const { theme } = useTheme();

    return (
        <div
            className="absolute top-16 right-4 z-50 w-64 p-4 rounded-lg shadow-xl backdrop-blur-md animate-fade-in origin-top-right"
            style={{
                backgroundColor: theme?.colors?.bg?.secondary ? `${theme.colors.bg.secondary}F2` : '#111827F2', // 95% opacity
                border: `1px solid ${theme?.colors?.border?.primary || '#374151'}`,
                color: theme?.colors?.text?.primary || '#E5E7EB'
            }}
        >
            <div
                className="flex justify-between items-center mb-3 pb-2"
                style={{ borderBottom: `1px solid ${theme?.colors?.border?.primary || '#374151'}` }}
            >
                <h3
                    className="font-bold text-sm"
                    style={{ color: theme?.colors?.accent?.primary || '#60A5FA' }}
                >
                    Navigation Controls
                </h3>
                <button
                    onClick={onClose}
                    className="hover:opacity-80 transition-opacity"
                    style={{ color: theme?.colors?.text?.tertiary || '#9CA3AF' }}
                >
                    ✕
                </button>
            </div>

            <div className="space-y-3 text-xs">
                <div className="flex items-center gap-3">
                    <div
                        className="w-8 h-8 flex items-center justify-center rounded border"
                        style={{
                            backgroundColor: theme?.colors?.bg?.tertiary || '#1F2937',
                            borderColor: theme?.colors?.border?.secondary || '#4B5563',
                            color: theme?.colors?.text?.primary || '#E5E7EB'
                        }}
                    >
                        <span className="font-mono">LMB</span>
                    </div>
                    <div>
                        <p className="font-semibold" style={{ color: theme?.colors?.text?.primary || '#E5E7EB' }}>Rotate</p>
                        <p style={{ color: theme?.colors?.text?.secondary || '#9CA3AF' }}>Orbit around object</p>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    <div
                        className="w-8 h-8 flex items-center justify-center rounded border"
                        style={{
                            backgroundColor: theme?.colors?.bg?.tertiary || '#1F2937',
                            borderColor: theme?.colors?.border?.secondary || '#4B5563',
                            color: theme?.colors?.text?.primary || '#E5E7EB'
                        }}
                    >
                        <span className="font-mono">RMB</span>
                    </div>
                    <div>
                        <p className="font-semibold" style={{ color: theme?.colors?.text?.primary || '#E5E7EB' }}>Pan</p>
                        <p style={{ color: theme?.colors?.text?.secondary || '#9CA3AF' }}>Move camera view</p>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    <div
                        className="w-8 h-8 flex items-center justify-center rounded border"
                        style={{
                            backgroundColor: theme?.colors?.bg?.tertiary || '#1F2937',
                            borderColor: theme?.colors?.border?.secondary || '#4B5563',
                            color: theme?.colors?.text?.primary || '#E5E7EB'
                        }}
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                        </svg>
                    </div>
                    <div>
                        <p className="font-semibold" style={{ color: theme?.colors?.text?.primary || '#E5E7EB' }}>Zoom</p>
                        <p style={{ color: theme?.colors?.text?.secondary || '#9CA3AF' }}>Scroll wheel</p>
                    </div>
                </div>

                <div
                    className="mt-2 pt-2"
                    style={{ borderTop: `1px solid ${theme?.colors?.border?.secondary || '#4B5563'}` }}
                >
                    <p className="italic" style={{ color: theme?.colors?.text?.muted || '#6B7280' }}>
                        Tip: Use <span style={{ color: theme?.colors?.accent?.primary || '#60A5FA' }}>Shift + Drag</span> to Pan without RMB.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default UnifiedSDFRenderer;
