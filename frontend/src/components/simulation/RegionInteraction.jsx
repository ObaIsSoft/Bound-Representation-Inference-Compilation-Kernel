
import React, { useRef, useState, useMemo } from 'react';
import * as THREE from 'three';
import { useTheme } from '../../contexts/ThemeContext';

export const RegionInteraction = ({ onRegionSelected, enabled, baseDims = [1, 1, 1] }) => {
    const [startPoint, setStartPoint] = useState(null);
    const [endPoint, setEndPoint] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const { theme } = useTheme();

    const handlePointerDown = (e) => {
        if (!enabled) return;
        e.stopPropagation(); // Stop orbit controls (if they respect this)

        // Capture 3D point
        const point = e.point.clone();
        setStartPoint(point);
        setEndPoint(point);
        setIsDragging(true);

        // Disable controls via event propagation capture? 
        // OrbitControls usually listens to domElement.
        // We might need to disable controls explicitly in parent.
    };

    const handlePointerMove = (e) => {
        if (!enabled || !isDragging) return;
        e.stopPropagation();
        setEndPoint(e.point.clone());
    };

    const handlePointerUp = (e) => {
        if (!enabled || !isDragging) return;
        e.stopPropagation();
        setIsDragging(false);

        if (startPoint && endPoint) {
            // Calculate Min/Max
            const min = new THREE.Vector3(
                Math.min(startPoint.x, endPoint.x),
                Math.min(startPoint.y, endPoint.y),
                Math.min(startPoint.z, endPoint.z)
            );
            const max = new THREE.Vector3(
                Math.max(startPoint.x, endPoint.x),
                Math.max(startPoint.y, endPoint.y),
                Math.max(startPoint.z, endPoint.z)
            );

            // Validate Size (> 1mm)
            if (min.distanceTo(max) > 0.001) {
                onRegionSelected({ min: min.toArray(), max: max.toArray() });
            }
        }
    };

    // Derived state for box visual
    const { boxPos, boxArgs } = useMemo(() => {
        if (!startPoint || !endPoint) return { boxPos: [0, 0, 0], boxArgs: [0, 0, 0] };

        const width = Math.abs(endPoint.x - startPoint.x);
        const height = Math.abs(endPoint.y - startPoint.y);
        const depth = Math.abs(endPoint.z - startPoint.z);

        const x = (startPoint.x + endPoint.x) / 2;
        const y = (startPoint.y + endPoint.y) / 2;
        const z = (startPoint.z + endPoint.z) / 2;

        return { boxPos: [x, y, z], boxArgs: [width, height, depth] };
    }, [startPoint, endPoint]);

    // Theme colors
    const selectionColor = theme?.colors?.accent?.primary || '#ffc107';

    return (
        <group>
            {/* Interaction Volume: Invisible Box matching Base Shape */}
            {/* We make it slightly larger to allow easier selection around edges */}
            <mesh
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                visible={false}
            >
                <boxGeometry args={[baseDims[0], baseDims[1], baseDims[2]]} />
                <meshBasicMaterial />
            </mesh>

            {/* Visual Feedback Box */}
            {(isDragging || (startPoint && endPoint)) && (
                <group position={boxPos}>
                    <mesh>
                        <boxGeometry args={boxArgs} />
                        <meshBasicMaterial
                            color={selectionColor}
                            opacity={0.3}
                            transparent
                            depthTest={false} // Always show on top
                        />
                    </mesh>
                    <lineSegments>
                        <edgesGeometry args={[new THREE.BoxGeometry(...boxArgs)]} />
                        <lineBasicMaterial color={selectionColor} />
                    </lineSegments>
                </group>
            )}
        </group>
    );
};
