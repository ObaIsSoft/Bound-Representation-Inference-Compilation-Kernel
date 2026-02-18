/**
 * Ghost Geometry - Non-destructive AI Suggestion Visualization
 * 
 * Displays proposed geometry as translucent overlays before user confirmation.
 * Core principle: AI suggestions are ghosts until accepted.
 */

import React, { useState, useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';

/**
 * Ghost Geometry Component
 */
export default function GhostGeometry({ data, onAccept, onReject, index, theme }) {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  
  // Pulse animation
  useFrame((state) => {
    if (!meshRef.current) return;
    const pulse = Math.sin(state.clock.elapsedTime * 2 + index) * 0.05 + 1;
    meshRef.current.scale.setScalar(pulse);
  });
  
  // Gold ghost color from theme accent
  const ghostColor = theme.colors.accent.primary;
  
  return (
    <group>
      {/* Ghost mesh with transparent material */}
      <mesh
        ref={meshRef}
        geometry={data.geometry}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <meshPhysicalMaterial
          color={ghostColor}
          transparent
          opacity={0.3}
          transmission={0.6}
          roughness={0.2}
          metalness={0.1}
          side={THREE.DoubleSide}
        />
      </mesh>
      
      {/* Hover tooltip */}
      {hovered && (
        <Html position={[0, 1.5, 0]} center>
          <div 
            className="rounded-lg p-3 min-w-[200px]"
            style={{
              backgroundColor: theme.colors.bg.secondary + 'EE',
              backdropFilter: 'blur(8px)',
              border: `1px solid ${theme.colors.accent.primary}60`,
              color: theme.colors.text.primary
            }}
          >
            <div 
              className="text-xs font-bold uppercase tracking-wider mb-1"
              style={{ color: theme.colors.accent.primary }}
            >
              AI Suggestion
            </div>
            <div className="text-sm mb-3">{data.reason || 'Proposed geometry change'}</div>
            <div className="flex gap-2">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onAccept();
                }}
                className="flex-1 px-3 py-1.5 rounded text-xs font-bold transition-colors"
                style={{
                  backgroundColor: theme.colors.status.success,
                  color: theme.colors.bg.primary
                }}
              >
                Accept
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onReject();
                }}
                className="flex-1 px-3 py-1.5 rounded text-xs font-bold transition-colors"
                style={{
                  backgroundColor: theme.colors.bg.tertiary,
                  color: theme.colors.text.primary,
                  border: `1px solid ${theme.colors.border.primary}`
                }}
              >
                Reject
              </button>
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}
