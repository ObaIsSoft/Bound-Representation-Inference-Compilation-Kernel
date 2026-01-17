import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Line, Cone } from '@react-three/drei';
import { useSettings } from '../../contexts/SettingsContext';
import { QUALITY_PRESETS } from '../../utils/visualizationConfig';

/**
 * BRICK OS: Physics Visualization Overlays
 * Renders data-driven overlays (trails, force vectors, orbits)
 * on top of the Unified SDF Renderer.
 */

// Proper Arrow Component using Cone + Line
const ForceArrow = ({ origin = [0, 0, 0], direction = [0, 1, 0], length = 1, color = '#ff0000', label = '' }) => {
    const dir = useMemo(() => new THREE.Vector3(...direction).normalize(), [direction]);
    const end = useMemo(() => {
        const o = new THREE.Vector3(...origin);
        return o.add(dir.clone().multiplyScalar(length));
    }, [origin, dir, length]);

    // Arrow shaft line
    const shaftEnd = useMemo(() => {
        const o = new THREE.Vector3(...origin);
        return o.add(dir.clone().multiplyScalar(length * 0.7));
    }, [origin, dir, length]);

    // Cone rotation to point in direction
    const rotation = useMemo(() => {
        const quaternion = new THREE.Quaternion();
        quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
        const euler = new THREE.Euler().setFromQuaternion(quaternion);
        return [euler.x, euler.y, euler.z];
    }, [dir]);

    return (
        <group>
            {/* Shaft */}
            <Line
                points={[origin, [shaftEnd.x, shaftEnd.y, shaftEnd.z]]}
                color={color}
                lineWidth={3}
            />
            {/* Arrowhead */}
            <Cone
                args={[0.08, 0.2, 8]}
                position={[end.x, end.y, end.z]}
                rotation={rotation}
            >
                <meshBasicMaterial color={color} />
            </Cone>
        </group>
    );
};

const PhysicsOverlays = ({ viewMode, physicsData, baseDims = [1, 1, 1] }) => {
    const groupRef = useRef();
    const time = useRef(0);
    const { visualizationQuality } = useSettings();

    // Get quality preset
    const quality = QUALITY_PRESETS[visualizationQuality] || QUALITY_PRESETS.HIGH;

    // Animate time for procedural effects
    useFrame((state, delta) => {
        time.current += delta;
    });

    // 1. Model Position Tracking (from physics data)
    const modelPosition = useMemo(() => {
        if (physicsData?.state?.position) {
            const pos = physicsData.state.position;
            return new THREE.Vector3(pos.x || 0, pos.y || pos.altitude || 0, pos.z || 0);
        }
        return new THREE.Vector3(0, 0, 0);
    }, [physicsData?.state?.position, physicsData?.state?.altitude]);

    // 2. Live Motion Trail with velocity-based coloring
    const motionTrailData = useMemo(() => {
        if (physicsData?.motionTrail && physicsData.motionTrail.length > 2) {
            const maxTrailLength = quality.trailLength;
            const trail = physicsData.motionTrail.slice(-maxTrailLength);

            const points = trail.map(p => new THREE.Vector3(p.x || 0, p.y || 0, p.z || 0));

            // Compute velocity-based colors (blue → green → red)
            const colors = trail.map((p, i) => {
                const velocity = p.velocity || physicsData.state?.velocity || 0;
                const normalizedV = Math.min(Math.abs(velocity) / 100, 1); // Normalize to 0-1

                // Blue (slow) → Green (medium) → Red (fast)
                if (normalizedV < 0.5) {
                    const t = normalizedV * 2;
                    return new THREE.Color().lerpColors(
                        new THREE.Color('#0099ff'),
                        new THREE.Color('#00ff88'),
                        t
                    );
                } else {
                    const t = (normalizedV - 0.5) * 2;
                    return new THREE.Color().lerpColors(
                        new THREE.Color('#00ff88'),
                        new THREE.Color('#ff4444'),
                        t
                    );
                }
            });

            return { points, colors };
        }
        return null;
    }, [physicsData?.motionTrail, quality.trailLength, physicsData?.state?.velocity]);

    // 3. Force Vectors (gravity, thrust, drag, net)
    const forceVectors = useMemo(() => {
        if (!physicsData?.state?.force_vectors) return null;

        const vectors = physicsData.state.force_vectors;
        const scale = 0.01; // Scale down force magnitude for visualization

        return {
            gravity: {
                direction: [0, -1, 0],
                magnitude: (vectors.gravity?.magnitude || 9.81) * scale,
                color: '#9C27B0' // Purple
            },
            thrust: {
                direction: vectors.thrust ?
                    [vectors.thrust.x || 0, vectors.thrust.y || 1, vectors.thrust.z || 0] :
                    [0, 1, 0],
                magnitude: (vectors.thrust?.magnitude || 0) * scale,
                color: '#4CAF50' // Green
            },
            drag: {
                direction: vectors.drag ?
                    [vectors.drag.x || 0, vectors.drag.y || -1, vectors.drag.z || 0] :
                    [0, -1, 0],
                magnitude: (vectors.drag?.magnitude || 0) * scale,
                color: '#2196F3' // Blue
            },
            net: {
                direction: vectors.net ?
                    [vectors.net.x || 0, vectors.net.y || 1, vectors.net.z || 0] :
                    [0, 1, 0],
                magnitude: (vectors.net?.magnitude || 0) * scale,
                color: '#FF9800' // Orange
            }
        };
    }, [physicsData?.state?.force_vectors]);

    // 4. Procedural Orbit Path (fallback)
    const proceduralOrbit = useMemo(() => {
        const points = [];
        const segments = quality.fieldLineDensity || 32;
        for (let i = 0; i <= segments; i++) {
            const angle = (i / segments) * Math.PI * 2;
            const x = Math.cos(angle) * (baseDims[0] * 1.2);
            const z = Math.sin(angle) * (baseDims[2] * 1.2);
            points.push(new THREE.Vector3(x, baseDims[1] * 0.3, z));
        }
        return points;
    }, [baseDims, quality.fieldLineDensity]);

    // Determine what to show based on view mode
    const showOverlays = viewMode === 'stress' || viewMode === 'flow' || viewMode === 'heatmap';
    const showForceVectors = viewMode === 'stress'; // Always show in stress mode (fallback to gravity)
    const showOrbit = showOverlays;
    const hasLiveTrail = motionTrailData && motionTrailData.points.length > 2;

    if (!showOverlays) return null;

    return (
        <group ref={groupRef}>
            {/* Motion Trail with Velocity Gradient */}
            {showOrbit && hasLiveTrail && (
                <Line
                    points={motionTrailData.points}
                    vertexColors={motionTrailData.colors}
                    lineWidth={4}
                    opacity={0.9}
                    transparent
                />
            )}

            {/* Procedural Orbit (fallback) */}
            {showOrbit && !hasLiveTrail && (
                <Line
                    points={proceduralOrbit}
                    color="#00ffff"
                    lineWidth={2}
                    dashed
                    dashScale={10}
                    dashSize={0.5}
                    gapSize={0.2}
                    opacity={0.6}
                    transparent
                />
            )}

            {/* Force Vectors */}
            {showForceVectors && forceVectors && (
                <group position={modelPosition}>
                    {/* Gravity (always shown in fallback mode) */}
                    {forceVectors.gravity && forceVectors.gravity.magnitude > 0 && (
                        <ForceArrow
                            origin={[0, 0, 0]}
                            direction={forceVectors.gravity.direction}
                            length={forceVectors.gravity.magnitude}
                            color={forceVectors.gravity.color}
                            label="G"
                        />
                    )}

                    {/* Thrust (only if present) */}
                    {forceVectors.thrust && forceVectors.thrust.magnitude > 0 && (
                        <ForceArrow
                            origin={[0, 0, 0]}
                            direction={forceVectors.thrust.direction}
                            length={forceVectors.thrust.magnitude}
                            color={forceVectors.thrust.color}
                            label="T"
                        />
                    )}

                    {/* Drag (only if present) */}
                    {forceVectors.drag && forceVectors.drag.magnitude > 0 && (
                        <ForceArrow
                            origin={[0, 0, 0]}
                            direction={forceVectors.drag.direction}
                            length={forceVectors.drag.magnitude}
                            color={forceVectors.drag.color}
                            label="D"
                        />
                    )}

                    {/* Net Force (only if present) */}
                    {forceVectors.net && forceVectors.net.magnitude > 0 && (
                        <ForceArrow
                            origin={[0, 0, 0]}
                            direction={forceVectors.net.direction}
                            length={forceVectors.net.magnitude}
                            color={forceVectors.net.color}
                            label="F_net"
                        />
                    )}
                </group>
            )}

            {/* Flow Particles (quality-aware count) */}
            {viewMode === 'flow' && (
                <group>
                    {[...Array(Math.floor(quality.particleCount / 20))].map((_, i) => {
                        const angle = (i / (quality.particleCount / 20)) * Math.PI * 2 + time.current * 0.5;
                        const r = baseDims[0] * 0.8;
                        return (
                            <mesh key={i} position={[Math.cos(angle) * r, 0, Math.sin(angle) * r]}>
                                <sphereGeometry args={[0.03, 8, 8]} />
                                <meshBasicMaterial color="#00ffff" />
                            </mesh>
                        );
                    })}
                </group>
            )}
        </group>
    );
};

export default PhysicsOverlays;

