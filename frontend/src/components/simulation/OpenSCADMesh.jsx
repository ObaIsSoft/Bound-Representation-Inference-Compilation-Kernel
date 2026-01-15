import React, { useState, useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';
import { Html } from '@react-three/drei';

/**
 * OpenSCAD Mesh Renderer
 * Compiles OpenSCAD code via backend API and renders the resulting STL mesh
 */
export const OpenSCADMesh = ({ scadCode, viewMode = 'realistic', theme }) => {
    const [geometry, setGeometry] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const { gl } = useThree();

    // Refs for debouncing and request cancellation
    const abortControllerRef = React.useRef(null);
    const debounceTimeoutRef = React.useRef(null);

    useEffect(() => {
        // Clear previous timeout
        if (debounceTimeoutRef.current) clearTimeout(debounceTimeoutRef.current);

        // Cancel any ongoing request
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        if (!scadCode || scadCode.trim() === '') return;

        // Debounce compilation by 1500ms to allow typing to finish
        debounceTimeoutRef.current = setTimeout(() => {
            const compileAndLoad = async () => {
                setLoading(true);
                setError(null);

                // Create new AbortController for this request
                abortControllerRef.current = new AbortController();

                try {
                    console.log('OpenSCADMesh: Starting compilation for code length:', scadCode.length);

                    // Call backend OpenSCAD compiler
                    const response = await fetch('http://localhost:8000/api/openscad/compile', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ code: scadCode }),
                        signal: abortControllerRef.current.signal
                    });

                    const data = await response.json();
                    console.log('OpenSCAD backend response:', data);

                    if (!data.success) {
                        console.error('OpenSCAD compilation failed:', data.error);
                        setError(data.error);
                        setLoading(false);
                        return;
                    }

                    // Create Three.js geometry from vertices and faces
                    const geo = new THREE.BufferGeometry();

                    // Convert vertices array to Float32Array
                    const vertices = new Float32Array(data.vertices.flat());
                    geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

                    // Convert faces to indices
                    const indices = new Uint32Array(data.faces.flat());
                    geo.setIndex(new THREE.BufferAttribute(indices, 1));

                    // Compute normals for proper lighting
                    geo.computeVertexNormals();

                    // Auto-scale to fit in viewport (convert mm to reasonable scene units)
                    geo.computeBoundingBox();
                    const bbox = geo.boundingBox;
                    const size = new THREE.Vector3();
                    bbox.getSize(size);
                    const maxDim = Math.max(size.x, size.y, size.z);

                    // Scale to fit within ~2 units (good for default camera distance)
                    const targetSize = 2;
                    const scale = maxDim > 0 ? targetSize / maxDim : 1;
                    geo.scale(scale, scale, scale);

                    // Center the geometry
                    geo.center();

                    setGeometry(geo);
                    setLoading(false);

                } catch (err) {
                    if (err.name === 'AbortError') {
                        console.log('OpenSCAD compilation aborted');
                        return;
                    }
                    console.error('OpenSCAD compilation error:', err);
                    setError(`Compilation failed: ${err.message}`);
                    setLoading(false);
                }
            };

            compileAndLoad();
        }, 1500); // 1.5s debounce

        // Cleanup on unmount or before next effect run
        return () => {
            if (debounceTimeoutRef.current) clearTimeout(debounceTimeoutRef.current);
            if (abortControllerRef.current) abortControllerRef.current.abort();
        };
    }, [scadCode]);

    const renderMaterial = () => {
        const baseProps = {
            color: viewMode === 'solid' ? '#888888' : theme?.colors?.text?.muted || '#cccccc',
            transparent: viewMode === 'xray',
            opacity: viewMode === 'xray' ? 0.3 : 1,
            side: viewMode === 'interior' ? THREE.BackSide : THREE.DoubleSide
        };

        switch (viewMode) {
            case 'realistic':
                return <meshStandardMaterial {...baseProps} roughness={0.5} metalness={0.5} />;
            case 'matte':
                return <meshStandardMaterial {...baseProps} roughness={1} metalness={0} />;
            case 'wireframe':
                return <meshBasicMaterial {...baseProps} wireframe color="#00ff88" />;
            case 'heatmap':
                return <meshNormalMaterial />;
            case 'hyperrealism':
                return <meshPhysicalMaterial {...baseProps} roughness={0} metalness={1} clearcoat={1} clearcoatRoughness={0} />;
            case 'solid':
                return <meshStandardMaterial {...baseProps} color="#888888" />;
            case 'interior':
                return <meshStandardMaterial {...baseProps} color="#ffaa00" side={THREE.BackSide} />;
            case 'shaded':
                return <meshPhongMaterial {...baseProps} shininess={100} />;
            case 'xray':
                return <meshStandardMaterial {...baseProps} transparent opacity={0.3} depthWrite={false} />;
            case 'hidden_line':
                return <meshBasicMaterial color="#1e1e1e" polygonOffset polygonOffsetFactor={1} />;
            // Physics Modes
            case 'stress':
                return <meshStandardMaterial color="#ef4444" wireframe={true} emissive="#7f1d1d" />;
            case 'flow':
                return <meshPhysicalMaterial color="#3b82f6" transmission={0.6} clearcoat={1} />;
            case 'micro':
                return <meshStandardMaterial {...baseProps} roughness={0.8} metalness={0.2} />;
            default:
                return <meshStandardMaterial {...baseProps} />;
        }
    };

    if (loading) {
        return (
            <group>
                <mesh>
                    <boxGeometry args={[1, 1, 1]} />
                    <meshBasicMaterial wireframe color={theme?.colors?.text?.tertiary || '#444'} transparent opacity={0.2} />
                </mesh>
                <Html position={[0, -1.2, 0]} center>
                    <div className="px-3 py-1.5 rounded-lg backdrop-blur-md bg-black/50 border border-white/10 flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
                        <span className="text-xs font-mono text-white/80">
                            {geometry ? 'Updating...' : 'Compiling OpenSCAD...'}
                        </span>
                    </div>
                </Html>
            </group>
        );
    }

    if (error) {
        return null; // Error is handled in SimulationBay UI
    }

    if (!geometry) return null;

    return (
        <group>
            <mesh geometry={geometry}>
                {renderMaterial()}
            </mesh>
            {/* Edges for cleaner look */}
            {(viewMode === 'solid' || viewMode === 'realistic') && (
                <lineSegments>
                    <edgesGeometry args={[geometry, 60]} />
                    <lineBasicMaterial color={theme?.colors?.border?.primary || '#444'} transparent opacity={0.4} />
                </lineSegments>
            )}
        </group>
    );
};
