/**
 * Omniviewport - Unified Spatial Workspace for BRICK OS
 * 
 * Core Principle: The user types intent, the AI manifests geometry in situ,
 * physics runs continuously in the background (like a game engine), and 
 * corrections happen as ghost overlays before confirmation.
 * 
 * Features:
 * - SDF Raymarching (Primary geometry rendering)
 * - Ghost Mode (AI suggestions as translucent overlays)
 * - CFD/Stress Visualization (Simulation overlays)
 * - Lightpen Annotations (3D drawing/comments)
 * - Motion Timeline (Physics simulation preview)
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  Grid, 
  Environment
} from '@react-three/drei';
import * as THREE from 'three';
import { useTheme } from '../contexts/ThemeContext';
import { useSpatialState } from '../hooks/useSpatialState';
import useWebSocket from '../hooks/useWebSocket';
import GhostGeometry from './GhostGeometry';
import SDFViewport from './SDFViewport';
import FlowVisualization from './FlowVisualization';
import MotionTrail from './MotionTrail';

/**
 * Main Omniviewport Component
 * Replaces the traditional Workspace with a unified 3D canvas
 */
export default function Omniviewport({ projectId }) {
  const { theme } = useTheme();
  const viewportRef = useRef();
  const [mode, setMode] = useState('design'); // design | simulate | animate | review
  
  // Real-time connection to backend orchestrator
  const { 
    isConnected,
    messages,
    thoughts,
    sendMessage,
  } = useWebSocket({ projectId });
  
  // Extract geometry and simulation data from messages
  const geometryStream = messages.find(m => m.type === 'geometry')?.payload || null;
  const simulationResults = messages.find(m => m.type === 'simulation')?.payload || null;
  const agentThoughts = thoughts || [];
  
  const connectionStatus = isConnected ? 'connected' : 'disconnected';
  
  // Mock active agents for display
  const activeAgents = agentThoughts.length > 0 ? ['GeometryAgent', 'PhysicsAgent'] : [];
  
  const acceptSuggestion = useCallback((id) => {
    console.log('[Omniviewport] Accepting suggestion:', id);
    sendMessage({ type: 'accept_suggestion', suggestion_id: id });
  }, [sendMessage]);
  
  const rejectSuggestion = useCallback((id) => {
    console.log('[Omniviewport] Rejecting suggestion:', id);
    sendMessage({ type: 'reject_suggestion', suggestion_id: id });
  }, [sendMessage]);

  // Use existing spatial state hook
  const {
    annotations,
    addAnnotation,
  } = useSpatialState();

  // Handle pointer interactions based on current mode
  const handlePointerDown = useCallback((event) => {
    if (mode === 'annotate' && event.button === 2) {
      event.stopPropagation();
    }
  }, [mode]);

  // Get canvas background color from theme
  const canvasBackground = theme.colors.bg.primary;

  return (
    <div 
      className="w-full h-full relative overflow-hidden touch-none"
      style={{ backgroundColor: canvasBackground }}
      onWheel={(e) => e.stopPropagation()}
    >
      {/* Connection Status Indicator */}
      <ConnectionStatus status={connectionStatus} theme={theme} />
      
      {/* 3D Canvas - Everything happens here */}
      <Canvas
        ref={viewportRef}
        camera={{ position: [5, 5, 5], fov: 50 }}
        gl={{ 
          antialias: true, 
          alpha: true,
          powerPreference: "high-performance"
        }}
        onPointerDown={handlePointerDown}
        dpr={[1, 2]}
        style={{ background: canvasBackground }}
      >
        <color attach="background" args={[canvasBackground]} />
        <SceneContents 
          geometry={geometryStream}
          simulation={simulationResults}
          thoughts={agentThoughts}
          mode={mode}
          theme={theme}
        />
      </Canvas>
    </div>
  );
}

/**
 * Internal Scene Component
 * Handles all 3D rendering
 */
function SceneContents({ 
  geometry, 
  simulation, 
  thoughts, 
  mode,
  theme
}) {
  const groupRef = useRef();
  
  // Animation frame updates
  useFrame((state, delta) => {
    if (mode === 'animate' && simulation?.motionData) {
      updateKinematics(groupRef.current, simulation.motionData, delta);
    }
  });

  // Grid colors from theme
  const gridSectionColor = theme.colors.border.secondary;
  const gridCellColor = theme.colors.bg.tertiary;

  return (
    <group ref={groupRef}>
      {/* Base Grid - Infinite reference plane */}
      <Grid 
        infiniteGrid 
        fadeDistance={50} 
        fadeStrength={5}
        cellSize={1}
        sectionSize={10}
        sectionColor={gridSectionColor}
        cellColor={gridCellColor}
      />
      
      {/* SDF Raymarched Geometry (Primary) */}
      {geometry?.sdfShader && (
        <SDFViewport 
          shaderCode={geometry.sdfShader}
          bounds={geometry.bounds}
          theme={theme}
        />
      )}
      
      {/* Traditional Mesh Fallback (if SDF not available) */}
      {geometry?.mesh && !geometry?.sdfShader && (
        <GeometryMesh geometryData={geometry} theme={theme} />
      )}
      
      {/* Simulation Visualization (CFD/Stress) */}
      {mode === 'simulate' && simulation?.type === 'CFD' && (
        <FlowVisualization data={simulation} theme={theme} />
      )}
      
      {mode === 'simulate' && simulation?.type === 'STRESS' && (
        <StressHeatmap data={simulation} theme={theme} />
      )}
      
      {/* Motion Trails (Animation Mode) */}
      {mode === 'animate' && simulation?.motionData?.trajectory && (
        <MotionTrail 
          points={simulation.motionData.trajectory}
          color={theme.colors.status.success}
          theme={theme}
        />
      )}
      
      {/* Default Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
      <directionalLight position={[-10, -10, -5]} intensity={0.3} />
      
      {/* Environment */}
      <Environment preset="studio" />
      
      {/* Camera Controls */}
      <OrbitControls 
        makeDefault 
        enableDamping
        dampingFactor={0.05}
        minDistance={1}
        maxDistance={50}
        enableZoom={true}
        zoomSpeed={0.5}
        enablePan={true}
        panSpeed={0.8}
        enableRotate={true}
        rotateSpeed={0.8}
        mouseButtons={{
          LEFT: THREE.MOUSE.ROTATE,
          MIDDLE: THREE.MOUSE.DOLLY,
          RIGHT: THREE.MOUSE.PAN
        }}
      />
    </group>
  );
}

/**
 * Connection Status Indicator
 */
function ConnectionStatus({ status, theme }) {
  const getStatusColor = () => {
    switch (status) {
      case 'connected': return theme.colors.status.success;
      case 'connecting': return theme.colors.status.warning;
      case 'error': return theme.colors.status.error;
      default: return theme.colors.status.error;
    }
  };

  return (
    <div 
      className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 rounded-full z-50"
      style={{ 
        backgroundColor: theme.colors.bg.secondary + 'CC',
        backdropFilter: 'blur(8px)',
        border: `1px solid ${theme.colors.border.primary}`
      }}
    >
      <div 
        className="w-2 h-2 rounded-full animate-pulse"
        style={{ backgroundColor: getStatusColor() }}
      />
      <span 
        className="text-xs font-medium uppercase tracking-wider"
        style={{ color: theme.colors.text.secondary }}
      >
        {status}
      </span>
    </div>
  );
}

/**
 * Update kinematics for animation mode
 */
function updateKinematics(group, motionData, delta) {
  if (!group || !motionData) return;
  
  const { position, rotation, scale } = motionData;
  
  if (position) {
    group.position.set(position[0], position[1], position[2]);
  }
  if (rotation) {
    group.rotation.set(rotation[0], rotation[1], rotation[2]);
  }
  if (scale) {
    group.scale.set(scale[0], scale[1], scale[2]);
  }
}

/**
 * Geometry Mesh Component (Fallback for non-SDF geometry)
 */
function GeometryMesh({ geometryData, theme }) {
  const meshRef = useRef();
  
  useEffect(() => {
    if (geometryData.mesh && meshRef.current) {
      console.log('[GeometryMesh] Loading mesh:', geometryData.format);
    }
  }, [geometryData]);

  return (
    <mesh ref={meshRef}>
      <meshStandardMaterial 
        color={theme.colors.text.tertiary}
        roughness={0.4}
        metalness={0.6}
      />
    </mesh>
  );
}

/**
 * Stress Heatmap Component
 */
function StressHeatmap({ data, theme }) {
  return (
    <group>
      {/* Stress visualization implementation */}
    </group>
  );
}
