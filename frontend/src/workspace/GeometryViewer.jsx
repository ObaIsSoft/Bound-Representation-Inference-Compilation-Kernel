/**
 * Geometry Viewer - 3D Model Display
 * 
 * Displays geometry from various sources:
 * - GLB/GLTF models
 * - SDF shaders
 * - Procedural meshes
 * 
 * Supports multiple render modes and interactions.
 */

import React, { useRef, useMemo, useEffect, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { useGLTF, Edges, Box, Sphere } from '@react-three/drei';
import * as THREE from 'three';
import SDFMaterial from './SDFMaterial';

/**
 * Geometry Viewer Component
 */
export function GeometryViewer({ 
  data, 
  renderMode = 'solid', // solid, wireframe, sdf, ghost
  selected = false,
  onClick,
  onHover,
  ...props 
}) {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  
  // Parse geometry data
  const geometry = useMemo(() => {
    if (!data) return null;
    
    // If it's already a Three.js geometry
    if (data instanceof THREE.BufferGeometry) {
      return data;
    }
    
    // If it's GLB data (ArrayBuffer)
    if (data instanceof ArrayBuffer || data instanceof Uint8Array) {
      // Would use GLTFLoader here - simplified for now
      return null;
    }
    
    // If it's a primitive definition
    if (data.type === 'box') {
      return new THREE.BoxGeometry(
        data.params?.width || 1,
        data.params?.height || 1,
        data.params?.depth || 1
      );
    }
    
    if (data.type === 'sphere') {
      return new THREE.SphereGeometry(
        data.params?.radius || 0.5,
        32, 32
      );
    }
    
    if (data.type === 'cylinder') {
      return new THREE.CylinderGeometry(
        data.params?.radius || 0.5,
        data.params?.radius || 0.5,
        data.params?.height || 1,
        32
      );
    }
    
    return null;
  }, [data]);
  
  // Determine material based on render mode
  const material = useMemo(() => {
    switch (renderMode) {
      case 'wireframe':
        return (
          <meshBasicMaterial
            color="#3b82f6"
            wireframe
            transparent
            opacity={0.5}
          />
        );
      
      case 'ghost':
        return (
          <meshPhysicalMaterial
            color="#ffb700"
            transparent
            opacity={0.3}
            transmission={0.6}
            roughness={0.2}
            metalness={0.1}
          />
        );
      
      case 'sdf':
        return <SDFMaterial />;
      
      case 'solid':
      default:
        return (
          <meshStandardMaterial
            color={selected ? '#3b82f6' : '#64748b'}
            roughness={0.4}
            metalness={0.3}
          />
        );
    }
  }, [renderMode, selected]);
  
  // Hover animation
  useFrame(() => {
    if (!meshRef.current) return;
    
    const targetScale = hovered ? 1.05 : 1;
    meshRef.current.scale.lerp(
      new THREE.Vector3(targetScale, targetScale, targetScale),
      0.1
    );
  });
  
  if (!geometry) {
    return null;
  }
  
  return (
    <group {...props}>
      <mesh
        ref={meshRef}
        geometry={geometry}
        onClick={onClick}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          onHover?.(true);
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          setHovered(false);
          onHover?.(false);
        }}
      >
        {material}
      </mesh>
      
      {/* Selection outline */}
      {selected && (
        <Edges geometry={geometry} threshold={15}>
          <lineBasicMaterial color="#3b82f6" linewidth={2} />
        </Edges>
      )}
      
      {/* Hover highlight */}
      {hovered && !selected && (
        <Edges geometry={geometry} threshold={15}>
          <lineBasicMaterial color="#60a5fa" linewidth={1} transparent opacity={0.5} />
        </Edges>
      )}
    </group>
  );
}

/**
 * Bounding Box Visualizer
 */
export function BoundingBox({ min, max, color = '#22d3ee' }) {
  const size = [
    max[0] - min[0],
    max[1] - min[1],
    max[2] - min[2]
  ];
  
  const center = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2
  ];
  
  return (
    <Box args={size} position={center}>
      <meshBasicMaterial
        color={color}
        wireframe
        transparent
        opacity={0.3}
      />
    </Box>
  );
}

/**
 * Origin Marker
 */
export function OriginMarker({ size = 0.5 }) {
  return (
    <group>
      {/* X axis - Red */}
      <mesh position={[size / 2, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.03, size / 4, 8]} />
        <meshBasicMaterial color="#ef4444" />
      </mesh>
      <mesh position={[size / 2, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.01, 0.01, size, 8]} />
        <meshBasicMaterial color="#ef4444" />
      </mesh>
      
      {/* Y axis - Green */}
      <mesh position={[0, size / 2, 0]}>
        <coneGeometry args={[0.03, size / 4, 8]} />
        <meshBasicMaterial color="#22c55e" />
      </mesh>
      <mesh position={[0, size / 2, 0]}>
        <cylinderGeometry args={[0.01, 0.01, size, 8]} />
        <meshBasicMaterial color="#22c55e" />
      </mesh>
      
      {/* Z axis - Blue */}
      <mesh position={[0, 0, size / 2]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.03, size / 4, 8]} />
        <meshBasicMaterial color="#3b82f6" />
      </mesh>
      <mesh position={[0, 0, size / 2]} rotation={[-Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.01, 0.01, size, 8]} />
        <meshBasicMaterial color="#3b82f6" />
      </mesh>
      
      {/* Origin point */}
      <mesh>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshBasicMaterial color="#ffffff" />
      </mesh>
    </group>
  );
}

export default GeometryViewer;
