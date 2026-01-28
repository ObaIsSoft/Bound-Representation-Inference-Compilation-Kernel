
/**
 * SDF Grid Renderer - Frontend Component
 * ======================================
 * 
 * Renders signed distance fields using GPU raymarching.
 * Handles both static and progressive streaming modes.
 * 
 * Copied from User-Provided Implementation.
 */

import React, { useRef, useMemo, useState, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';

/**
 * SDFRenderer Component
 */
const SDFRenderer = ({
    sdfGrid,
    bounds,
    colorMode = 'solid',
    isovalue = 0.0,
    stepSize = 1.0,
    opacity = 1.0
}) => {
    const meshRef = useRef();
    const { camera } = useThree();

    // Create 3D texture from SDF grid
    const sdfTexture = useMemo(() => {
        if (!sdfGrid || !Array.isArray(sdfGrid)) {
            console.warn('SDFRenderer: Invalid sdfGrid');
            return null;
        }

        const size = sdfGrid.length;
        // console.log(`Creating 3D texture: ${size}x${size}x${size}`);

        // Flatten 3D grid to 1D array
        const data = new Float32Array(size * size * size);

        let idx = 0;
        for (let z = 0; z < size; z++) {
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    data[idx++] = sdfGrid[x][y][z];
                }
            }
        }

        // Create 3D texture
        const texture = new THREE.Data3DTexture(data, size, size, size);
        texture.format = THREE.RedFormat;
        texture.type = THREE.FloatType;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.wrapR = THREE.ClampToEdgeWrapping;
        texture.needsUpdate = true;

        return texture;
    }, [sdfGrid]);

    // Raymarch shader material
    const material = useMemo(() => {
        if (!sdfTexture || !bounds) return null;

        const minVec = new THREE.Vector3(...bounds.min);
        const maxVec = new THREE.Vector3(...bounds.max);
        const sizeVec = maxVec.clone().sub(minVec);

        return new THREE.ShaderMaterial({
            uniforms: {
                sdfTexture: { value: sdfTexture },
                gridMin: { value: minVec },
                gridMax: { value: maxVec },
                gridSize: { value: sizeVec },
                isovalue: { value: isovalue },
                stepSize: { value: stepSize },
                colorMode: { value: colorMode === 'solid' ? 0 : colorMode === 'distance' ? 1 : colorMode === 'normal' ? 2 : 3 },
                opacity: { value: opacity },
                cameraPos: { value: new THREE.Vector3() }
            },
            vertexShader: `
                varying vec3 vPosition;
                varying vec3 vWorldPosition;
                
                void main() {
                    vPosition = position;
                    vec4 worldPos = modelMatrix * vec4(position, 1.0);
                    vWorldPosition = worldPos.xyz;
                    gl_Position = projectionMatrix * viewMatrix * worldPos;
                }
            `,
            fragmentShader: `
                precision highp float;
                precision highp sampler3D;
                
                uniform sampler3D sdfTexture;
                uniform vec3 gridMin;
                uniform vec3 gridMax;
                uniform vec3 gridSize;
                uniform float isovalue;
                uniform float stepSize;
                uniform int colorMode;
                uniform float opacity;
                uniform vec3 cameraPos;
                
                varying vec3 vPosition;
                varying vec3 vWorldPosition;
                
                // Sample SDF at world position
                float sampleSDF(vec3 pos) {
                    vec3 uvw = (pos - gridMin) / gridSize;
                    
                    // Clamp to valid range
                    if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) {
                        return 1000.0; // Outside bounds
                    }
                    
                    return texture(sdfTexture, uvw).r;
                }
                
                // Compute normal using central differences
                vec3 computeNormal(vec3 pos) {
                    float eps = 0.001;
                    float dx = sampleSDF(pos + vec3(eps, 0, 0)) - sampleSDF(pos - vec3(eps, 0, 0));
                    float dy = sampleSDF(pos + vec3(0, eps, 0)) - sampleSDF(pos - vec3(0, eps, 0));
                    float dz = sampleSDF(pos + vec3(0, 0, eps)) - sampleSDF(pos - vec3(0, 0, eps));
                    return normalize(vec3(dx, dy, dz));
                }
                
                void main() {
                    // Setup ray
                    vec3 rayOrigin = cameraPos;
                    vec3 rayDir = normalize(vWorldPosition - cameraPos);
                    
                    // Raymarch parameters
                    float t = 0.0;
                    float maxDist = length(gridSize) * 2.0;
                    float minStep = length(gridSize) / 256.0; // Minimum step size
                    
                    int maxSteps = 256;
                    bool hit = false;
                    vec3 hitPos;
                    
                    // Raymarch loop
                    for (int i = 0; i < maxSteps; i++) {
                        vec3 p = rayOrigin + rayDir * t;
                        float d = sampleSDF(p);
                        
                        // Check for surface hit
                        if (d < isovalue + 0.001) {
                            hit = true;
                            hitPos = p;
                            break;
                        }
                        
                        // March forward (use distance field for adaptive step size)
                        float marchDist = max(abs(d - isovalue) * stepSize, minStep);
                        t += marchDist;
                        
                        // Early exit if too far
                        if (t > maxDist) break;
                    }
                    
                    if (!hit) {
                        discard; // Ray didn't hit surface
                    }
                    
                    // Compute color based on mode
                    vec3 color;
                    
                    if (colorMode == 0) {
                        // Solid color with simple lighting
                        vec3 normal = computeNormal(hitPos);
                        vec3 lightDir = normalize(vec3(1, 1, 1));
                        float diffuse = max(dot(normal, lightDir), 0.0);
                        float ambient = 0.3;
                        float lighting = ambient + (1.0 - ambient) * diffuse;
                        
                        color = vec3(0.8, 0.2, 0.2) * lighting; // BRICK red
                        
                    } else if (colorMode == 1) {
                        // Distance field visualization
                        float dist = sampleSDF(hitPos);
                        float normalized = (dist + 0.5) / 1.0; // Map [-0.5, 0.5] to [0, 1]
                        normalized = clamp(normalized, 0.0, 1.0);
                        
                        if (normalized < 0.5) {
                            color = mix(vec3(0, 0, 1), vec3(1, 1, 1), normalized * 2.0);
                        } else {
                            color = mix(vec3(1, 1, 1), vec3(1, 0, 0), (normalized - 0.5) * 2.0);
                        }
                        
                    } else if (colorMode == 2) {
                        // Normal map
                        vec3 normal = computeNormal(hitPos);
                        color = normal * 0.5 + 0.5; // Map [-1,1] to [0,1]
                        
                    } else {
                        // Depth visualization
                        float depth = t / maxDist;
                        color = vec3(1.0 - depth);
                    }
                    
                    gl_FragColor = vec4(color, opacity);
                }
            `,
            side: THREE.FrontSide,
            transparent: opacity < 1.0,
            depthWrite: opacity >= 1.0
        });
    }, [sdfTexture, bounds, isovalue, stepSize, colorMode, opacity]);

    // Update camera position uniform every frame
    useFrame(() => {
        if (material && camera) {
            material.uniforms.cameraPos.value.copy(camera.position);
        }
    });

    if (!material || !bounds) {
        return null;
    }

    // Bounding box dimensions
    const width = bounds.max[0] - bounds.min[0];
    const height = bounds.max[1] - bounds.min[1];
    const depth = bounds.max[2] - bounds.min[2];

    // Center position
    const centerX = (bounds.min[0] + bounds.max[0]) / 2;
    const centerY = (bounds.min[1] + bounds.max[1]) / 2;
    const centerZ = (bounds.min[2] + bounds.max[2]) / 2;

    return (
        <mesh
            ref={meshRef}
            material={material}
            position={[centerX, centerY, centerZ]}
        >
            <boxGeometry args={[width, height, depth]} />
        </mesh>
    );
};


/**
 * Progressive SDF Loader Component
 * 
 * Handles streaming SDF data from backend and progressive rendering.
 */
const ProgressiveSDFRenderer = ({
    scadCode,
    resolution = 64,
    ...rendererProps
}) => {
    const { camera, controls } = useThree(); // Access camera and controls
    const [sdfGrid, setSDFGrid] = useState(null);
    const [bounds, setBounds] = useState(null);
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState(null);

    // Auto-Fit Camera when bounds are loaded
    useEffect(() => {
        if (bounds) {
            const min = new THREE.Vector3(...bounds.min);
            const max = new THREE.Vector3(...bounds.max);

            // Calculate Center and Size
            const center = new THREE.Vector3().addVectors(min, max).multiplyScalar(0.5);
            const size = new THREE.Vector3().subVectors(max, min);
            const maxDim = Math.max(size.x, size.y, size.z);

            console.log('[SDF AutoZoom] Fitting camera to object:', { center, maxDim });

            // Move Camera back
            if (camera) {
                // Determine a nice viewing distance (e.g. 2x the object size)
                const distance = maxDim * 2.0;

                // Position camera at an isometric-like angle
                camera.position.set(
                    center.x + distance * 0.5,
                    center.y + distance * 0.5,
                    center.z + distance * 0.5
                );
                camera.lookAt(center);

                camera.updateProjectionMatrix();
            }

            // Update OrbitControls target if available
            if (controls) {
                controls.target.copy(center);
                controls.update();
            }
        }
    }, [bounds, camera, controls]);

    useEffect(() => {
        if (!scadCode) return;

        // AbortController to cancel pending requests on unmount/re-render
        const controller = new AbortController();
        const signal = controller.signal;

        const loadSDF = async () => {
            setLoading(true);
            setError(null);
            setProgress(0);

            try {
                // Use the new compile-stream endpoint
                const response = await fetch('http://localhost:8000/api/openscad/compile-stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        code: scadCode,
                        resolution,
                        use_winding_number: true
                    }),
                    signal // Pass abort signal
                });

                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}`);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                let buffer = '';
                // Initialize empty grid (Zeroed)
                const grid = new Array(resolution).fill(null).map(() =>
                    new Array(resolution).fill(null).map(() =>
                        new Float32Array(resolution).fill(10.0) // Init with 'outside' distance > 0
                    )
                );

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });

                    // Process SSE events
                    const lines = buffer.split('\n\n');
                    buffer = lines.pop(); // Keep incomplete event

                    for (const line of lines) {
                        const eventMatch = line.match(/event: (\w+)/);
                        const dataMatch = line.match(/data: (.+)/);

                        if (eventMatch && dataMatch) {
                            const eventType = eventMatch[1];
                            const data = JSON.parse(dataMatch[1]);

                            if (eventType === 'start') {
                                console.log('[SDF Stream] Started:', data);
                                setBounds(data.bounds);

                            } else if (eventType === 'slice') {
                                // Add slice to grid
                                const z = data.slice_index;
                                for (let x = 0; x < resolution; x++) {
                                    for (let y = 0; y < resolution; y++) {
                                        grid[x][y][z] = data.slice_data[x][y];
                                    }
                                }

                                setProgress(data.progress);

                                // Update grid every 5 slices (more frequent updates)
                                if (z % 5 === 0) {
                                    setSDFGrid([...grid]); // Clone to trigger re-render
                                }

                            } else if (eventType === 'complete') {
                                console.log('[SDF Stream] Complete:', data.metadata);
                                setSDFGrid([...grid]);
                                setProgress(1.0);
                                setLoading(false);

                            } else if (eventType === 'error') {
                                throw new Error(data.error);
                            }
                        }
                    }
                }

            } catch (err) {
                if (err.name === 'AbortError') {
                    console.log('[SDF Stream] Aborted');
                } else {
                    console.error('[SDF Stream] Failed:', err);
                    setError(err.message);
                }
                setLoading(false);
            }
        };

        loadSDF();

        return () => {
            controller.abort();
        };
    }, [scadCode, resolution]);

    if (error) {
        // Suppress specific "No geometry" errors which are expected for blank files
        if (error.includes('No compilable geometry')) {
            return null; // Render nothing (clean slate)
        }

        return (
            <Html center>
                <div className="flex flex-col items-center justify-center p-4 bg-red-900/90 rounded-lg backdrop-blur-md border border-red-500/30">
                    <div className="text-red-400 font-bold text-xs uppercase tracking-wider mb-1">SDF Stream Error</div>
                    <div className="text-white text-xs font-mono">{error}</div>
                </div>
            </Html>
        );
    }

    // Show partial result if available
    if (loading && progress > 0) {
        return (
            <>
                {sdfGrid && bounds && (
                    <SDFRenderer
                        sdfGrid={sdfGrid}
                        bounds={bounds}
                        opacity={0.5} // Ghostly during load
                        {...rendererProps}
                    />
                )}
                <Html center>
                    <div className="bg-black/50 p-2 rounded text-white text-xs font-mono backdrop-blur-sm">
                        Loading SDF: {(progress * 100).toFixed(0)}%
                    </div>
                </Html>
            </>
        );
    }

    if (!sdfGrid || !bounds) {
        return null;
    }

    return (
        <SDFRenderer
            sdfGrid={sdfGrid}
            bounds={bounds}
            {...rendererProps}
        />
    );
};


export { SDFRenderer, ProgressiveSDFRenderer };
export default ProgressiveSDFRenderer;
