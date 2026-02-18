/**
 * Motion Trail - Physics Trajectory Visualization
 * 
 * Visualizes physics simulation results as:
 * - Trajectory paths
 * - Ghost frames at time intervals
 * - Velocity/acceleration vectors
 * - Collision prediction markers
 */

import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line, Html } from '@react-three/drei';
import * as THREE from 'three';

/**
 * Motion Trail Component
 */
export default function MotionTrail({ points, color, showGhosts = true, theme }) {
  if (!points || points.length < 2) return null;
  
  const trailPoints = useMemo(() => {
    return points.map(p => new THREE.Vector3(p[0], p[1], p[2]));
  }, [points]);
  
  // Sample ghost frame positions (every Nth point)
  const ghostFrames = useMemo(() => {
    if (!showGhosts) return [];
    const interval = Math.max(1, Math.floor(points.length / 10));
    return points.filter((_, i) => i % interval === 0);
  }, [points, showGhosts]);
  
  // Use provided color or theme accent
  const trailColor = color || theme.colors.accent.primary;
  
  return (
    <group>
      {/* Main trajectory line */}
      <Line
        points={trailPoints}
        color={trailColor}
        lineWidth={3}
        transparent
        opacity={0.6}
      />
      
      {/* Ghost frames */}
      {ghostFrames.map((point, i) => (
        <GhostFrame
          key={i}
          position={point}
          index={i}
          total={ghostFrames.length}
          color={trailColor}
          theme={theme}
        />
      ))}
      
      {/* Start/End markers */}
      <StartMarker position={points[0]} theme={theme} />
      <EndMarker position={points[points.length - 1]} theme={theme} />
    </group>
  );
}

/**
 * Ghost Frame - Shows object position at a point in time
 */
function GhostFrame({ position, index, total, color, theme }) {
  const meshRef = useRef();
  
  // Fade opacity based on position in timeline
  const opacity = (index + 1) / total * 0.5;
  
  // Scale down for older frames
  const scale = 0.5 + (index / total) * 0.5;
  
  return (
    <mesh ref={meshRef} position={position} scale={[scale, scale, scale]}>
      <boxGeometry args={[0.2, 0.2, 0.2]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={opacity}
        wireframe
      />
    </mesh>
  );
}

/**
 * Start Marker
 */
function StartMarker({ position, theme }) {
  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[0.1, 16, 16]} />
        <meshBasicMaterial color={theme.colors.status.success} />
      </mesh>
      <Html distanceFactor={10}>
        <div 
          className="px-2 py-1 rounded font-bold text-xs"
          style={{
            backgroundColor: theme.colors.status.success,
            color: theme.colors.bg.primary
          }}
        >
          START
        </div>
      </Html>
    </group>
  );
}

/**
 * End Marker
 */
function EndMarker({ position, theme }) {
  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[0.1, 16, 16]} />
        <meshBasicMaterial color={theme.colors.status.error} />
      </mesh>
      <Html distanceFactor={10}>
        <div 
          className="px-2 py-1 rounded font-bold text-xs"
          style={{
            backgroundColor: theme.colors.status.error,
            color: theme.colors.text.primary
          }}
        >
          END
        </div>
      </Html>
    </group>
  );
}

/**
 * Velocity Vector - Shows instantaneous velocity
 */
export function VelocityVector({ position, velocity, scale = 1, theme }) {
  const magnitude = Math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2);
  
  // Calculate rotation to align with velocity
  const rotation = useMemo(() => {
    const dir = new THREE.Vector3(velocity[0], velocity[1], velocity[2]).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const axis = new THREE.Vector3().crossVectors(up, dir).normalize();
    const angle = Math.acos(up.dot(dir));
    return [axis.x * angle, axis.y * angle, axis.z * angle];
  }, [velocity]);
  
  const arrowScale = magnitude * scale;
  
  return (
    <group position={position} rotation={rotation}>
      {/* Arrow shaft */}
      <mesh position={[0, arrowScale / 2, 0]}>
        <cylinderGeometry args={[0.02, 0.02, arrowScale, 8]} />
        <meshBasicMaterial color={theme.colors.status.info} />
      </mesh>
      
      {/* Arrow head */}
      <mesh position={[0, arrowScale, 0]}>
        <coneGeometry args={[0.05, 0.1, 8]} />
        <meshBasicMaterial color={theme.colors.status.info} />
      </mesh>
    </group>
  );
}

/**
 * Force Vector - Shows applied forces
 */
export function ForceVector({ position, force, label, theme }) {
  const magnitude = Math.sqrt(force[0]**2 + force[1]**2 + force[2]**2);
  
  if (magnitude < 0.001) return null;
  
  const rotation = useMemo(() => {
    const dir = new THREE.Vector3(force[0], force[1], force[2]).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const axis = new THREE.Vector3().crossVectors(up, dir).normalize();
    const angle = Math.acos(up.dot(dir));
    return [axis.x * angle, axis.y * angle, axis.z * angle];
  }, [force]);
  
  const scale = Math.min(magnitude * 0.1, 2);
  
  return (
    <group position={position} rotation={rotation}>
      <mesh position={[0, scale / 2, 0]}>
        <cylinderGeometry args={[0.03, 0.03, scale, 8]} />
        <meshBasicMaterial color={theme.colors.status.warning} />
      </mesh>
      <mesh position={[0, scale, 0]}>
        <coneGeometry args={[0.06, 0.12, 8]} />
        <meshBasicMaterial color={theme.colors.status.warning} />
      </mesh>
      
      {label && (
        <Html position={[0, scale + 0.2, 0]} center distanceFactor={10}>
          <div 
            className="px-2 py-1 rounded whitespace-nowrap text-xs"
            style={{
              backgroundColor: theme.colors.status.warning,
              color: theme.colors.bg.primary
            }}
          >
            {label}: {magnitude.toFixed(2)} N
          </div>
        </Html>
      )}
    </group>
  );
}
