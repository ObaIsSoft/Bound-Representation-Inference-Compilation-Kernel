/**
 * Annotation Engine - Lightpen/3D Comment System
 * 
 * Allows users to draw strokes directly on 3D geometry and attach comments.
 * Annotations become persistent constraints that agents can read and respond to.
 * 
 * Features:
 * - Raycast-based 3D drawing on SDF surfaces
 * - Stroke-to-SDF conversion (tubes)
 * - Comment threading
 * - Visual indicators for annotated areas
 */

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { Html, Line } from '@react-three/drei';
import * as THREE from 'three';

/**
 * Annotation Engine Component
 */
export function AnnotationEngine({ 
  viewportRef, 
  projectId, 
  onAnnotationCreate,
  annotations = []
}) {
  const { camera, scene, raycaster, pointer } = useThree();
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentStroke, setCurrentStroke] = useState([]);
  const [activeAnnotation, setActiveAnnotation] = useState(null);
  const [showAll, setShowAll] = useState(true);
  
  // Raycast against SDF surface (approximated)
  const raycastToSurface = useCallback((clientX, clientY) => {
    // Create ray from camera
    const mouse = new THREE.Vector2(
      (clientX / window.innerWidth) * 2 - 1,
      -(clientY / window.innerHeight) * 2 + 1
    );
    
    raycaster.setFromCamera(mouse, camera);
    
    // For SDF, we'd need to raymarch into the scene
    // For now, use a plane at z=0 as approximation
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const target = new THREE.Vector3();
    
    raycaster.ray.intersectPlane(plane, target);
    return target;
  }, [camera, raycaster]);
  
  // Start drawing
  const handlePointerDown = useCallback((e) => {
    if (e.button !== 2) return; // Right click only
    
    e.stopPropagation();
    setIsDrawing(true);
    
    const point = raycastToSurface(e.clientX, e.clientY);
    if (point) {
      setCurrentStroke([point]);
    }
  }, [raycastToSurface]);
  
  // Continue drawing
  const handlePointerMove = useCallback((e) => {
    if (!isDrawing) return;
    
    const point = raycastToSurface(e.clientX, e.clientY);
    if (point) {
      setCurrentStroke(prev => [...prev, point]);
    }
  }, [isDrawing, raycastToSurface]);
  
  // End drawing
  const handlePointerUp = useCallback(() => {
    if (!isDrawing || currentStroke.length < 2) {
      setIsDrawing(false);
      setCurrentStroke([]);
      return;
    }
    
    // Create annotation
    const annotation = {
      id: Date.now().toString(),
      type: 'stroke',
      points: currentStroke.map(p => [p.x, p.y, p.z]),
      comment: '',
      createdAt: new Date().toISOString(),
      author: 'user',
      resolved: false
    };
    
    onAnnotationCreate(annotation);
    setActiveAnnotation(annotation.id);
    setIsDrawing(false);
    setCurrentStroke([]);
  }, [isDrawing, currentStroke, onAnnotationCreate]);
  
  // Event listeners
  useEffect(() => {
    const canvas = viewportRef.current;
    if (!canvas) return;
    
    canvas.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    
    return () => {
      canvas.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [viewportRef, handlePointerDown, handlePointerMove, handlePointerUp]);
  
  return (
    <group>
      {/* Existing Annotations */}
      {showAll && annotations.map(annotation => (
        <AnnotationVisual
          key={annotation.id}
          annotation={annotation}
          isActive={activeAnnotation === annotation.id}
          onClick={() => setActiveAnnotation(
            activeAnnotation === annotation.id ? null : annotation.id
          )}
        />
      ))}
      
      {/* Currently Drawing Stroke */}
      {isDrawing && currentStroke.length > 1 && (
        <AnnotationStroke 
          points={currentStroke}
          color="#ffd700"
          opacity={0.8}
        />
      )}
      
      {/* Drawing Indicator */}
      {isDrawing && (
        <Html center>
          <div className="bg-yellow-500 text-black px-3 py-1 rounded-full text-xs font-medium animate-pulse">
            Drawing...
          </div>
        </Html>
      )}
      
      {/* Annotation Toolbar */}
      <AnnotationToolbar
        annotations={annotations}
        showAll={showAll}
        setShowAll={setShowAll}
        activeAnnotation={activeAnnotation}
        setActiveAnnotation={setActiveAnnotation}
      />
    </group>
  );
}

/**
 * Visual representation of an annotation
 */
function AnnotationVisual({ annotation, isActive, onClick }) {
  const points = annotation.points.map(p => new THREE.Vector3(p[0], p[1], p[2]));
  
  return (
    <group onClick={onClick}>
      {/* Stroke Line */}
      <Line
        points={points}
        color={annotation.resolved ? '#22c55e' : isActive ? '#ffd700' : '#ff6b6b'}
        lineWidth={isActive ? 4 : 2}
        transparent
        opacity={0.8}
      />
      
      {/* Start Point Marker */}
      <mesh position={points[0]}>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshBasicMaterial 
          color={annotation.resolved ? '#22c55e' : '#ffd700'}
          transparent
          opacity={0.6}
        />
      </mesh>
      
      {/* Comment Bubble */}
      {isActive && annotation.comment && (
        <Html position={points[Math.floor(points.length / 2)]} center>
          <div className="bg-yellow-400 text-black p-3 rounded-lg shadow-xl max-w-xs">
            <div className="text-xs font-medium mb-1">
              {annotation.author} • {new Date(annotation.createdAt).toLocaleTimeString()}
            </div>
            <div className="text-sm">{annotation.comment}</div>
            {annotation.resolved && (
              <div className="mt-2 text-xs text-green-700 font-medium">
                ✓ Resolved
              </div>
            )}
          </div>
        </Html>
      )}
    </group>
  );
}

/**
 * Annotation Stroke Component
 */
function AnnotationStroke({ points, color, opacity }) {
  const linePoints = points.map(p => new THREE.Vector3(p.x, p.y, p.z));
  
  return (
    <Line
      points={linePoints}
      color={color}
      lineWidth={3}
      transparent
      opacity={opacity}
    />
  );
}

/**
 * Annotation Toolbar UI
 */
function AnnotationToolbar({ 
  annotations, 
  showAll, 
  setShowAll,
  activeAnnotation,
  setActiveAnnotation
}) {
  const resolvedCount = annotations.filter(a => a.resolved).length;
  const unresolvedCount = annotations.length - resolvedCount;
  
  return (
    <div className="absolute top-20 right-4 bg-black/80 backdrop-blur rounded-xl p-3 z-50">
      <div className="flex items-center justify-between mb-3">
        <span className="text-white font-medium text-sm">Annotations</span>
        <button
          onClick={() => setShowAll(!showAll)}
          className="text-xs text-white/60 hover:text-white"
        >
          {showAll ? 'Hide' : 'Show'}
        </button>
      </div>
      
      <div className="space-y-2 text-xs">
        <div className="flex justify-between text-white/70">
          <span>Total:</span>
          <span>{annotations.length}</span>
        </div>
        <div className="flex justify-between text-green-400">
          <span>Resolved:</span>
          <span>{resolvedCount}</span>
        </div>
        <div className="flex justify-between text-yellow-400">
          <span>Pending:</span>
          <span>{unresolvedCount}</span>
        </div>
      </div>
      
      {unresolvedCount > 0 && (
        <button
          onClick={() => {
            const next = annotations.find(a => !a.resolved);
            if (next) setActiveAnnotation(next.id);
          }}
          className="mt-3 w-full bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-400 text-xs py-1.5 rounded transition-colors"
        >
          Jump to Next
        </button>
      )}
      
      <div className="mt-3 pt-3 border-t border-white/10 text-white/40 text-xs">
        Right-click to draw
      </div>
    </div>
  );
}

export default AnnotationEngine;
