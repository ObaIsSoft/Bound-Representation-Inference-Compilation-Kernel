import React, { useRef, useState, useCallback } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { Line } from '@react-three/drei';
import * as THREE from 'three';
import { useSimulation } from '../../contexts/SimulationContext';

/**
 * StrokeCapture
 * 
 * Captures 2D pointer movements projected onto a 3D plane.
 * Used for Semantic Sketching.
 */
export const StrokeCapture = ({ enabled }) => {
    const { camera, raycaster, pointer } = useThree();
    const { sketchPoints, setSketchPoints } = useSimulation();
    const [currentStrokePoints, setCurrentStrokePoints] = useState([]);
    const [isDrawing, setIsDrawing] = useState(false);
    const planeRef = useRef();

    // We use a temporary vector to avoid garbage collection
    const intersectionPoint = useRef(new THREE.Vector3());

    const handlePointerDown = useCallback((e) => {
        if (!enabled) return;
        // Don't propagate to orbit controls
        e.stopPropagation();
        setIsDrawing(true);
        setCurrentStrokePoints([]); // Start new stroke (local only)
        // DON'T clear global sketchPoints - we want accumulation
    }, [enabled]);

    const handlePointerMove = useCallback((e) => {
        if (!enabled || !isDrawing) return;

        // Raycast against our invisible plane
        if (planeRef.current) {
            // e.point is the intersection point in world space provided by r3f
            const point = e.point;
            setCurrentStrokePoints(prev => {
                const newStroke = [...prev, point.clone()];
                // Append to global accumulated sketches
                // Combine previous strokes with current stroke
                const allPoints = [...(sketchPoints || []), ...newStroke];
                setSketchPoints(allPoints.slice(-128)); // Keep last 128 points total
                return newStroke;
            });
        }
    }, [enabled, isDrawing, sketchPoints, setSketchPoints]);

    const handlePointerUp = useCallback(async (e) => {
        if (!isDrawing) return;
        setIsDrawing(false);

        // Log stroke
        console.log("Stroke Captured:", currentStrokePoints.length, "points");

        // Send to backend for reification (geometry conversion)
        if (currentStrokePoints.length > 1) {
            try {
                const response = await fetch('http://localhost:8000/api/orchestrator/reify_stroke', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        points: currentStrokePoints.map(p => [p.x, p.y, p.z]),
                        optimize: true // Enable Smart Snap (Phase 11)
                    })
                });

                if (response.ok) {
                    const result = await response.json();
                    console.log('[SKETCH] Backend acknowledged:', result);
                    // TODO: Update design state with returned primitive
                } else {
                    console.warn('[SKETCH] Backend reification failed');
                }
            } catch (err) {
                console.error('[SKETCH] Failed to send to backend:', err);
            }
        }

    }, [isDrawing, currentStrokePoints]);

    if (!enabled) return null;

    return (
        <group>
            {/* Invisible Catch Plane (facing camera or XY plane) */}
            {/* For now, we use a fixed XY plane at Z=0 for sketching on the "floor" */}
            <mesh
                ref={planeRef}
                visible={false} // Invisible but raycastable
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerLeave={handlePointerUp}
                rotation={[-Math.PI / 2, 0, 0]} // Horizontal Plane
                position={[0, 0, 0]}
            >
                <planeGeometry args={[100, 100]} />
                <meshBasicMaterial color="red" wireframe />
            </mesh>

            {/* Visual Feedback Line */}
            {currentStrokePoints.length > 1 && (
                <Line
                    points={currentStrokePoints}
                    color="#facc15" // Yellow/Gold
                    lineWidth={1}
                    dashed={false}
                />
            )}
        </group>
    );
};
