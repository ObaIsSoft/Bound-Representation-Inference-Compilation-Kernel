import React, { useState, useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';
import { Html } from '@react-three/drei';
import { AlertTriangle, X } from 'lucide-react';

/**
 * OpenSCAD Mesh Renderer
 * Compiles OpenSCAD code via backend API and renders the resulting STL mesh
 * Supports progressive assembly rendering for complex models
 */
export const OpenSCADMesh = ({ scadCode, viewMode = 'realistic', theme, onSDFLoaded }) => {
    const [geometry, setGeometry] = useState(null);
    const [parts, setParts] = useState([]); // For progressive rendering
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [progress, setProgress] = useState(0);
    const [progressiveMode, setProgressiveMode] = useState(false);
    const { gl } = useThree();

    // Refs for debouncing and request cancellation
    const abortControllerRef = React.useRef(null);
    const debounceTimeoutRef = React.useRef(null);

    // Check if progressive rendering is supported
    useEffect(() => {
        fetch('http://localhost:8000/api/openscad/info')
            .then(res => res.json())
            .then(data => {
                setProgressiveMode(data.progressive_rendering === true);
            })
            .catch(() => setProgressiveMode(false));
    }, []);

    useEffect(() => {
        // Clear previous timeout
        if (debounceTimeoutRef.current) clearTimeout(debounceTimeoutRef.current);

        // Cancel any ongoing request or stream
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        if (!scadCode || scadCode.trim() === '') return;

        // Debounce compilation by 1500ms to allow typing to finish
        debounceTimeoutRef.current = setTimeout(() => {
            // Decide between progressive and monolithic compilation
            if (progressiveMode && scadCode.includes('module ')) {
                // Use progressive rendering for assemblies
                compileProgressive();
            } else {
                // Use monolithic rendering for simple models
                compileMonolithic();
            }
        }, 1500); // 1.5s debounce

        // Cleanup on unmount or before next effect run
        return () => {
            if (debounceTimeoutRef.current) clearTimeout(debounceTimeoutRef.current);
            if (abortControllerRef.current) abortControllerRef.current.abort();
        };
    }, [scadCode, progressiveMode]);

    const compileProgressive = async () => {
        console.log('OpenSCADMesh: Starting progressive compilation...');
        setLoading(true);
        setError(null);
        setParts([]);
        setProgress(0);

        try {
            // Use fetch with ReadableStream for SSE (EventSource doesn't support POST)
            const response = await fetch('http://localhost:8000/api/openscad/compile-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code: scadCode })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            // Read stream
            while (true) {
                const { done, value } = await reader.read();

                if (done) break;

                // Decode chunk
                buffer += decoder.decode(value, { stream: true });

                // Process complete SSE messages
                const lines = buffer.split('\n\n');
                buffer = lines.pop() || ''; // Keep incomplete message in buffer

                for (const message of lines) {
                    if (!message.trim()) continue;

                    // Parse SSE format: "event: type\ndata: json"
                    const eventMatch = message.match(/event:\s*(\w+)/);
                    const dataMatch = message.match(/data:\s*(.+)/s);

                    if (!eventMatch || !dataMatch) continue;

                    const eventType = eventMatch[1];
                    const eventData = JSON.parse(dataMatch[1]);

                    // Handle events
                    if (eventType === 'start') {
                        console.log('Progressive compilation started:', eventData);
                    } else if (eventType === 'part') {
                        console.log('Part received:', eventData.part_id, eventData.progress);

                        // Validate part data
                        if (!eventData.vertices || !eventData.faces) {
                            console.error('Invalid part data - missing vertices or faces:', eventData);
                            setError(`Part ${eventData.part_id || 'unknown'} has invalid geometry data`);
                            continue;
                        }

                        try {
                            // Create Three.js geometry from part data
                            const geo = new THREE.BufferGeometry();
                            const vertices = new Float32Array(eventData.vertices.flat());
                            geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

                            const indices = new Uint32Array(eventData.faces.flat());
                            geo.setIndex(new THREE.BufferAttribute(indices, 1));
                            geo.computeVertexNormals();

                            // Add part to collection
                            setParts(prev => [...prev, {
                                id: eventData.part_id,
                                geometry: geo,
                                depth: eventData.depth || 0
                            }]);

                            setProgress(eventData.progress || 0);
                        } catch (err) {
                            console.error('Error creating geometry for part:', eventData.part_id, err);
                            setError(`Failed to create geometry for ${eventData.part_id}: ${err.message}`);
                        }
                    } else if (eventType === 'complete') {
                        console.log('Progressive compilation complete:', eventData);
                        setLoading(false);
                    } else if (eventType === 'error' || eventType === 'part_error') {
                        console.error('Compilation error:', eventData);
                        setError(eventData.error || 'Compilation failed');
                    }
                }
            }

            setLoading(false);

        } catch (err) {
            console.error('Progressive compilation error:', err);
            setError(`Streaming compilation failed: ${err.message}`);
            setLoading(false);
        }
    };

    const compileMonolithic = async () => {
        setLoading(true);
        setError(null);
        setParts([]);

        // Create new AbortController for this request
        abortControllerRef.current = new AbortController();

        try {
            console.log('OpenSCADMesh: Starting monolithic compilation...');

            // Call backend OpenSCAD compiler
            const response = await fetch('http://localhost:8000/api/openscad/compile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code: scadCode }),
                signal: abortControllerRef.current.signal
            });

            const data = await response.json();

            if (!data.success) {
                setError(data.error);
                setLoading(false);
                return;
            }

            // Create Three.js geometry from vertices and faces
            const geo = new THREE.BufferGeometry();
            const vertices = new Float32Array(data.vertices.flat());
            geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

            const indices = new Uint32Array(data.faces.flat());
            geo.setIndex(new THREE.BufferAttribute(indices, 1));
            geo.computeVertexNormals();

            // Auto-scale and center
            geo.computeBoundingBox();
            const bbox = geo.boundingBox;
            const size = new THREE.Vector3();
            bbox.getSize(size);
            const maxDim = Math.max(size.x, size.y, size.z);
            const targetSize = 2;
            const scale = maxDim > 0 ? targetSize / maxDim : 1;
            geo.scale(scale, scale, scale);
            geo.center();

            setGeometry(geo);

            // Propagate SDF Data if available
            if (data.sdf_data && onSDFLoaded) {
                onSDFLoaded({
                    mesh_sdf_data: data.sdf_data,
                    sdf_resolution: data.sdf_resolution,
                    sdf_bounds: data.sdf_bounds,
                    sdf_range: data.sdf_range
                });
            }

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
            case 'xray':
                return <meshStandardMaterial {...baseProps} />;
            case 'interior':
                return <meshStandardMaterial {...baseProps} />;
            case 'solid':
                return <meshStandardMaterial {...baseProps} />;
            default:
                return <meshStandardMaterial {...baseProps} />;
        }
    };

    // Render progressive parts or monolithic geometry
    if (loading) {
        return (
            <Html center>
                <div style={{
                    background: 'rgba(0,0,0,0.8)',
                    color: '#fff',
                    padding: '20px',
                    borderRadius: '8px',
                    textAlign: 'center'
                }}>
                    <div>Compiling OpenSCAD...</div>
                    {progress > 0 && (
                        <div style={{ marginTop: '10px' }}>
                            <div style={{
                                width: '200px',
                                height: '4px',
                                background: '#333',
                                borderRadius: '2px',
                                overflow: 'hidden'
                            }}>
                                <div style={{
                                    width: `${progress * 100}%`,
                                    height: '100%',
                                    background: theme?.colors?.accent || '#00ff88',
                                    transition: 'width 0.3s'
                                }} />
                            </div>
                            <div style={{ marginTop: '5px', fontSize: '12px' }}>
                                {Math.round(progress * 100)}%
                            </div>
                        </div>
                    )}
                </div>
            </Html>
        );
    }

    if (error) {
        return (
            <Html center>
                <div style={{
                    background: theme?.colors?.background?.elevated || '#1a1a1a',
                    border: `1px solid ${theme?.colors?.error || '#ef4444'}`,
                    borderRadius: '12px',
                    padding: '0',
                    maxWidth: '500px',
                    minWidth: '320px',
                    maxHeight: '400px',
                    display: 'flex',
                    flexDirection: 'column',
                    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.4)',
                    fontFamily: theme?.fonts?.mono || 'monospace'
                }}>
                    {/* Header */}
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        padding: '16px 20px',
                        borderBottom: `1px solid ${theme?.colors?.border?.subtle || '#333'}`,
                        background: `linear-gradient(135deg, ${theme?.colors?.error || '#ef4444'}15, transparent)`
                    }}>
                        <AlertTriangle
                            size={24}
                            color={theme?.colors?.error || '#ef4444'}
                            strokeWidth={2}
                        />
                        <h3 style={{
                            margin: 0,
                            fontSize: '16px',
                            fontWeight: 600,
                            color: theme?.colors?.text?.primary || '#ffffff',
                            flex: 1
                        }}>
                            Compilation Error
                        </h3>
                        <button
                            onClick={() => setError(null)}
                            style={{
                                background: 'transparent',
                                border: 'none',
                                cursor: 'pointer',
                                padding: '4px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                borderRadius: '4px',
                                transition: 'background 0.2s'
                            }}
                            onMouseEnter={(e) => e.target.style.background = theme?.colors?.background?.hover || '#333'}
                            onMouseLeave={(e) => e.target.style.background = 'transparent'}
                        >
                            <X size={18} color={theme?.colors?.text?.muted || '#888'} />
                        </button>
                    </div>

                    {/* Error Content - Scrollable */}
                    <div style={{
                        padding: '20px',
                        overflowY: 'auto',
                        maxHeight: '300px',
                        fontSize: '13px',
                        lineHeight: '1.6',
                        color: theme?.colors?.text?.secondary || '#cccccc',
                        wordBreak: 'break-word',
                        fontFamily: theme?.fonts?.mono || 'monospace'
                    }}>
                        <pre style={{
                            margin: 0,
                            whiteSpace: 'pre-wrap',
                            fontFamily: 'inherit',
                            fontSize: 'inherit',
                            color: theme?.colors?.error || '#ef4444',
                            background: `${theme?.colors?.error || '#ef4444'}10`,
                            padding: '12px',
                            borderRadius: '6px',
                            border: `1px solid ${theme?.colors?.error || '#ef4444'}30`
                        }}>
                            {error}
                        </pre>
                    </div>

                    {/* Footer with action hint */}
                    <div style={{
                        padding: '12px 20px',
                        borderTop: `1px solid ${theme?.colors?.border?.subtle || '#333'}`,
                        fontSize: '12px',
                        color: theme?.colors?.text?.muted || '#888',
                        textAlign: 'center'
                    }}>
                        Check your OpenSCAD syntax and try again
                    </div>
                </div>
            </Html>
        );
    }

    // Render progressive parts
    if (parts.length > 0) {
        return (
            <group>
                {parts.map(part => (
                    <mesh key={part.id} geometry={part.geometry}>
                        {renderMaterial()}
                    </mesh>
                ))}
            </group>
        );
    }

    // Render monolithic geometry
    if (!geometry) return null;

    return (
        <mesh geometry={geometry}>
            {renderMaterial()}
        </mesh>
    );
};
