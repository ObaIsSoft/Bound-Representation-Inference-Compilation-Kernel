# Omniviewport - Unified 3D Workspace Implementation

## Overview

The Omniviewport is a unified spatial workspace for BRICK OS that replaces the traditional Workspace with a 3D-first environment. It provides an immersive interface where users can design, simulate, animate, and review hardware projects in a single unified view.

## Components

### Core Components (14 files, ~3,000 lines)

| Component | Purpose | Size |
|-----------|---------|------|
| `Omniviewport.jsx` | Main workspace container with WebSocket integration | 13,181 bytes |
| `GhostGeometry.jsx` | Non-destructive AI suggestion visualization | 9,150 bytes |
| `CommandGhost.jsx` | Spatial command input (text commands in 3D space) | 8,875 bytes |
| `AnnotationEngine.jsx` | Lightpen annotation system for 3D markup | 8,307 bytes |
| `SDFViewport.jsx` | SDF raymarching viewport | 7,780 bytes |
| `GeometryViewer.jsx` | 3D model display with multiple render modes | 5,765 bytes |
| `MotionTrail.jsx` | Physics trajectory visualization | 5,455 bytes |
| `SimulationOverlay.jsx` | Simulation results HUD overlay | 5,383 bytes |
| `FlowVisualization.jsx` | Airshaper-style CFD rendering | 4,427 bytes |
| `SDFMaterial.jsx` | Raymarched SDF shader material | 4,983 bytes |
| `AgentPresenceOverlay.jsx` | AI agent activity visualization | 2,950 bytes |
| `SpatialToolbar.jsx` | Mode-switching toolbar (design/simulate/animate/review) | 2,685 bytes |
| `useWebSocketStream.js` | Real-time backend connection hook | 3,725 bytes |
| `index.js` | Module exports and constants | 1,862 bytes |

## Features

### 1. SDF Raymarching (Primary Geometry Rendering)
- Real-time signed distance field rendering via shaders
- Supports primitives: sphere, box, cylinder
- Boolean operations: union, subtract, intersect, smooth blend
- Hardware-accelerated on GPU

### 2. Ghost Mode (AI Suggestions)
- Translucent gold overlays for proposed geometry changes
- Non-destructive workflow - user must accept changes
- Shows AI reasoning on hover
- Accept/Reject actions for user confirmation

### 3. CFD/Stress Visualization
- Flow streamlines with velocity-based coloring
- Pressure heatmaps on geometry surface
- Velocity vector arrows
- Particle flow animation
- Safety factor display

### 4. Lightpen Annotations
- Right-click to draw on 3D surfaces
- Raycasting against mesh for stroke attachment
- Persistent annotations with timestamps
- Comment attachments

### 5. Motion Timeline
- Physics trajectory visualization
- Ghost frames at time intervals
- Start/end markers
- Velocity and force vectors

### 6. Spatial Command Input
- Text commands in 3D space
- AI agent suggestions
- Natural language intent processing

## Workspace Modes

| Mode | Description |
|------|-------------|
| **design** | Standard 3D editing with SDF/mesh rendering |
| **simulate** | CFD/FEA visualization with simulation overlays |
| **animate** | Physics timeline with motion trails and ghost frames |
| **review** | Annotation mode with markup and comments |

## Color Scheme

| Purpose | Color | Hex |
|---------|-------|-----|
| Ghost Suggestion | Gold | `#ffb700` |
| Warning | Red | `#ef4444` |
| Info | Blue | `#3b82f6` |
| Success | Green | `#22c55e` |
| Flow (slow) | Blue | `#3b82f6` |
| Flow (medium) | Cyan | `#22d3ee` |
| Flow (fast) | Yellow | `#facc15` |
| Flow (very fast) | Red | `#ef4444` |

## Backend Integration

The Omniviewport connects to the BRICK OS backend via WebSocket at:
```
ws://localhost:8000/ws/{projectId}
```

### Message Types
- `GEOMETRY` - New geometry data
- `SIMULATION` - Simulation results
- `AGENT_THOUGHT` - AI agent reasoning stream
- `ERROR` - Server error messages
- `CLEAR_THOUGHTS` - Reset agent thoughts

### Command Types
- `INTENT` - Natural language command
- `ACCEPT_GHOST` - Accept suggestion
- `REJECT_GHOST` - Reject suggestion

## Integration with Workspace Page

The `Workspace.jsx` page now uses the Omniviewport component:

```jsx
import { Omniviewport } from '../workspace';

export default function Workspace() {
    // ...
    return (
        <div className="flex h-screen">
            <LockedSidebar ... />
            <div className="flex-1 flex flex-col relative">
                <Omniviewport projectId={currentProject} />
            </div>
        </div>
    );
}
```

## Build Verification

```
✓ 2738 modules transformed
✓ built in 18.72s
```

All components successfully integrated and build passes without errors.

## Dependencies

- `@react-three/fiber` - React Three.js renderer
- `@react-three/drei` - Useful helpers (Grid, OrbitControls, etc.)
- `@react-three/postprocessing` - Post-processing effects
- `three` - Three.js core library
- `framer-motion` - Animation library

## Next Steps

1. **Backend WebSocket Endpoint** - Implement `/ws/{projectId}` in FastAPI
2. **SDF Shader Generation** - Connect HWC.to_glsl() output to SDFMaterial
3. **Physics Integration** - Connect physics_agent simulation results to MotionTrail
4. **Ghost Mode Backend** - Implement AI suggestion generation and approval flow
5. **Annotation Persistence** - Store annotations in Supabase
6. **CFD Integration** - Connect Airshaper-style CFD results to FlowVisualization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Omniviewport                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              React Three Fiber Canvas                  │  │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌─────────┐  │  │
│  │  │ SDFViewport │ FlowVisualization │ MotionTrail │ GhostGeometry │  │  │
│  │  └─────────┘  └──────────┘  └─────────┘  └─────────┘  │  │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐                │  │
│  │  │ Grid    │  │ Lighting │  │ Controls│                │  │
│  │  └─────────┘  └──────────┘  └─────────┘                │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ CommandGhost │ AnnotationEngine │ SimulationOverlay │ AgentPresenceOverlay │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ WebSocket
┌─────────────────────────────────────────────────────────────┐
│                    BRICK OS Backend                          │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ 91 Agents   │  │ HWC Compiler │  │ Physics Kernel   │   │
│  │             │  │              │  │                  │   │
│  │ - Geometry  │  │ - SDF→GLSL   │  │ - 6-DOF dynamics │   │
│  │ - Physics   │  │ - CSG ops    │  │ - CFD solver     │   │
│  │ - Materials │  │ - Mesh export│  │ - Stress analysis│   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```
