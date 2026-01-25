# BRICK Intelligence: How It "Knows"

You asked four critical questions about the nature of this system. Here is how BRICK thinks.

## 1. "How does it know where to put what?"
BRICK does not "guess" positions. It solves **Constraints**.

*   **The Recursive ISA (Instruction Set Architecture)**:
    Instead of a flat list of parts, BRICK understands the design as a **Semantic Tree**.
    *   *Human Way*: "Put the engine at x=150."
    *   *BRICK Way*: "The Engine `mounts_to` the Wing mount-point."
*   **The Constraint Solver**:
    When you say "Add an engine," the **SystemsAgent** looks at the `Wing` component, finds the node labeled `hardpoint_0`, triggers the **GeometryEstimator** to find that point's global `(x,y,z)`, and snaps the Engine there.
*   **Collision Avoidance**: The **PhysicsAgent** constantly runs a background check. If two bounding boxes overlap, it flagging a `[CLASH]` error, prompting the LLM to "Move it aft by 50mm".

## 2. "How does it know what unheard-of projects look like?"
BRICK does not rely on "Training Data" of existing designs. It relies on **First Principles**.

*   **Function Over Form**:
    If you ask for an "Interstellar Ion Thruster" (which doesn't exist yet), BRICK doesn't look for a JPEG. It looks for the **Physics Equation**:
    $$ F = \dot{m} v_e $$
    It knows it needs:
    1.  A propellant source (Tank).
    2.  An ionization chamber (Magnetic Coils).
    3.  A nozzle (Electric Field accelerator).
*   It "hallucinates" the geometry based on these functional requirements. The coil *must* wrap around the chamber. The nozzle *must* be at the back. The "look" emerges from the *function*.

## 3. "I have a project with 2 million parts. How does it handle that?"
**Recursive Level-of-Detail (LOD)**.

*   **The "Tree", not the "Bucket"**:
    Traditional CAD loads all 2 million parts into RAM. It crashes.
    BRICK loads only the **Root Nodes** (e.g., "Fuselage", "Wing L", "Wing R").
    *   Inside "Wing L", there are 10,000 rivets. BRICK represents them as a single "Texture" or "SDF Block" until you **Zoom In**.
    *   As you zoom, the **UnifiedSDFRenderer** dynamically requests the sub-nodes.
    *   This is how video games render infinite worlds. BRICK applies this to Hardware Engineering.
*   **Parallel Compilation**:
    The 2 million parts are not compiled endlessly. If 100,000 rivets are identical, they are compiled *once* (cached) and instanced 100,000 times on the GPU.

---

# UI Polish: Brainstorming

## 4. The Animated Intro ("Boot Sequence")
We want the user to feel they are accessing a high-power terminal, not just a website.

*   **Concept**: "Hardware Kernel Initialization".
*   **Visuals**:
    1.  **Black Screen**.
    2.  **Terminal Text Scrolling Fast**:
        `> MOUNTING PHYSICS ORACLES... [OK]`
        `> CALIBRATING SDF ENGINE... [OK]`
        `> ESTABLISHING NEURAL LINK... [OK]`
    3.  **The Flash**: A bright "CRT turn on" flash.
    4.  **The Reveal**: The Grid fades in from the center out. The UI slides in from the edges.
*   **Sound**: A subtle "Hum" or "Click-Thrummm".

## 5. Optional Gridlines
A simple, elegant toggle.

*   **Location**: Bottom-Right "View settings" floating island (near the axis indicator).
*   **Behavior**:
    *   Toggle Switch: "Grid".
    *   When OFF: The infinite void. Just the model floating in space (great for screenshots).
    *   When ON: The precision measurement floor.
