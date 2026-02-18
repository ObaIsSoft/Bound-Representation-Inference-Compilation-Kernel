/**
 * Workspace Components - Omniviewport 3D Workspace
 * 
 * Exports all workspace components for the unified 3D environment:
 * - Omniviewport: Main unified workspace component
 * - GhostGeometry: Non-destructive AI suggestion visualization
 * - FlowVisualization: Airshaper-style CFD rendering
 * - MotionTrail: Physics trajectory visualization
 * - AnnotationEngine: Lightpen annotation system
 * - SpatialToolbar: Mode-switching toolbar
 * - GeometryViewer: 3D model display component
 * - SDFMaterial: Raymarched SDF shader material
 */

// Core workspace
export { default as Omniviewport } from './Omniviewport';
export { default as GhostGeometry } from './GhostGeometry';
export { default as FlowVisualization } from './FlowVisualization';
export { default as MotionTrail } from './MotionTrail';
export { default as AnnotationEngine } from './AnnotationEngine';
export { default as SpatialToolbar } from './SpatialToolbar';
export { default as SimulationOverlay } from './SimulationOverlay';
export { default as CommandGhost } from './CommandGhost';
export { default as AgentPresenceOverlay } from './AgentPresenceOverlay';

// Geometry and Rendering
export { default as GeometryViewer, BoundingBox, OriginMarker } from './GeometryViewer';
export { default as SDFMaterial } from './SDFMaterial';
export { default as SDFViewport } from './SDFViewport';

// Constants
export const VIEWPORT_MODES = {
  DESIGN: 'design',
  SIMULATE: 'simulate',
  ANIMATE: 'animate',
  REVIEW: 'review',
};

export const GHOST_COLORS = {
  SUGGESTION: '#ffb700',
  WARNING: '#ef4444',
  INFO: '#3b82f6',
  SUCCESS: '#22c55e',
};

export const SIMULATION_TYPES = {
  CFD: 'cfd',
  FEA: 'fea',
  PHYSICS: 'physics',
  THERMAL: 'thermal',
};

// Version
export const WORKSPACE_VERSION = '1.0.0';
