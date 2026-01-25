# Implementation Plan: Analytic Geometry Evaluator

## Goal
Implement a pure Python variable-aware geometry evaluator to estimate the **Bounding Box** and **Center of Gravity** of an OpenSCAD script *instantly* (<100ms), without invoking the OpenSCAD binary. This resolves the "auto-centering" latency and timeout issues.

## User Constraints
- Must work on complex models (F-22) where OpenSCAD `compile` times out.
- Must be "Intelligent" (handle variables, loops, modules).
- Must correct the camera/grid positioning automatically.

## Architecture

### 1. `GeometryEstimator` Class
A new class in `backend/agents/geometry_estimator.py` (or integrated into parser).
- **Input**: List of `ASTNode` (from Parser) + Global Variables.
- **Output**: `AABB` (Axis Aligned Bounding Box) `[min_x, min_y, min_z, max_x, max_y, max_z]`.

### 2. Logic (Recursive Traversal)
- **Primitives**:
    - `cube([x,y,z])`: Bounds are `0..x, 0..y, 0..z` (unless centered).
    - `sphere(r)`: Bounds ` -r..r`.
    - `cylinder(h, r)`: Bounds `-r..r, 0..h, -r..r`.
- **Transforms**:
    - `translate([x,y,z])`: Shift child bounds by vector.
    - `rotate([...])`: Apply rotation matrix to child's *8 corner points* and re-compute AABB.
    - `scale([x,y,z])`: Multiply child bounds.
- **Booleans**:
    - `union/hull/minkowski`: Merge child bounds (Union of AABBs).
    - `difference`: **Crucial Heuristic**: The bounds of `A - B` are roughly the bounds of `A`. We ignore B for bounding box purposes (safe, conservative).
    - `intersection`: Intersection of AABBs (or just A for safety).
- **Modules**:
    - Inline dimensions using the parser's existing variable resolution.

### 3. Integration
- OpenSCADAgent calls `parser.parse()`.
- Passes AST to `GeometryEstimator`.
- Gets `center` vector.
- Injects `translate([-cx, -cy, -cz])`.

## files

#### [NEW] [geometry_estimator.py](file:///Users/obafemi/Documents/dev/brick/backend/agents/geometry_estimator.py)
- `estimate_bounds(nodes, variables)`

#### [MODIFY] [openscad_agent.py](file:///Users/obafemi/Documents/dev/brick/backend/agents/openscad_agent.py)
- Replace "Pre-Scan" logic with "Analytic Estimation".

## Verification
- Run `test_progressive_f22.py`.
- Expect `[Intelligence] Analytic Scan: 0.05s`.
- Expect correct centering.
