/**
 * Flow Visualization - Airshaper-style CFD Rendering
 * 
 * Visualizes CFD results as:
 * - Streamlines/particle trails
 * - Pressure heatmaps on geometry surface
 * - Velocity vector arrows
 * - Turbulence indicators
 */

import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line } from '@react-three/drei';
import * as THREE from 'three';

/**
 * Flow Visualization Component
 */
export default function FlowVisualization({ data, theme }) {
  if (!data?.streamlines && !data?.velocityField) return null;
  
  return (
    <group>
      {/* Streamlines */}
      {data.streamlines?.map((streamline, i) => (
        <Streamline key={i} data={streamline} theme={theme} />
      ))}
      
      {/* Velocity Vectors */}
      {data.velocityField?.map((field, i) => (
        <VelocityVector key={i} data={field} theme={theme} />
      ))}
      
      {/* Flow Particles */}
      <FlowParticles data={data} theme={theme} />
    </group>
  );
}

/**
 * Streamline Component
 */
function Streamline({ data, theme }) {
  const points = useMemo(() => {
    return data.points.map(p => new THREE.Vector3(p[0], p[1], p[2]));
  }, [data.points]);
  
  // Color based on velocity using theme status colors
  const color = useMemo(() => {
    const speed = data.velocity || 1;
    if (speed < 5) return theme.colors.status.info;
    if (speed < 15) return theme.colors.status.success;
    if (speed < 30) return theme.colors.status.warning;
    return theme.colors.status.error;
  }, [data.velocity, theme]);
  
  return (
    <Line
      points={points}
      color={color}
      lineWidth={2}
      transparent
      opacity={0.6}
    />
  );
}

/**
 * Velocity Vector Component
 */
function VelocityVector({ data, theme }) {
  const meshRef = useRef();
  
  const { position, direction, magnitude } = data;
  
  // Scale arrow by velocity magnitude
  const scale = Math.min(magnitude * 0.1, 2);
  
  // Color by magnitude using theme colors
  const color = useMemo(() => {
    if (magnitude < 5) return theme.colors.status.info;
    if (magnitude < 15) return theme.colors.status.success;
    if (magnitude < 30) return theme.colors.status.warning;
    return theme.colors.status.error;
  }, [magnitude, theme]);
  
  // Calculate rotation to align with direction
  const rotation = useMemo(() => {
    const dir = new THREE.Vector3(direction[0], direction[1], direction[2]).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const axis = new THREE.Vector3().crossVectors(up, dir).normalize();
    const angle = Math.acos(up.dot(dir));
    return [axis.x * angle, axis.y * angle, axis.z * angle];
  }, [direction]);
  
  return (
    <group position={position} rotation={rotation}>
      <mesh ref={meshRef} scale={[scale, scale, scale]}>
        <coneGeometry args={[0.05, 0.2, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.7} />
      </mesh>
    </group>
  );
}

/**
 * Flow Particles Component
 */
function FlowParticles({ data, theme }) {
  const particlesRef = useRef();
  const particleCount = 100;
  
  // Initialize particle positions
  const positions = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 10;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 10;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 10;
    }
    return pos;
  }, []);
  
  // Animate particles
  useFrame((state) => {
    if (!particlesRef.current) return;
    
    const posArray = particlesRef.current.geometry.attributes.position.array;
    
    for (let i = 0; i < particleCount; i++) {
      const idx = i * 3;
      
      // Move particles in flow direction (simplified)
      posArray[idx] += 0.05;
      posArray[idx + 1] += Math.sin(state.clock.elapsedTime + i) * 0.01;
      
      // Reset if out of bounds
      if (posArray[idx] > 5) {
        posArray[idx] = -5;
        posArray[idx + 1] = (Math.random() - 0.5) * 10;
        posArray[idx + 2] = (Math.random() - 0.5) * 10;
      }
    }
    
    particlesRef.current.geometry.attributes.position.needsUpdate = true;
  });
  
  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particleCount}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.05}
        color={theme.colors.status.info}
        transparent
        opacity={0.6}
        sizeAttenuation
      />
    </points>
  );
}
