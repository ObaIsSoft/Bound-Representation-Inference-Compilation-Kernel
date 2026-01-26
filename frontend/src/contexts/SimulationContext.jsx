import React, { createContext, useContext, useState } from 'react';

const SimulationContext = createContext();

export const useSimulation = () => {
    const context = useContext(SimulationContext);
    if (!context) {
        throw new Error('useSimulation must be used within a SimulationProvider');
    }
    return context;
};

export const SimulationProvider = ({ children }) => {
    const [isRunning, setIsRunning] = useState(false);
    const [paused, setPaused] = useState(false);

    // Test Scenarios
    // 'none' = Standard Physics
    // 'thermal' = Temperature Stress Test
    // 'structural' = Load/Stress Test
    // 'propulsion' = Thrust/Efficiency Test
    // 'aerodynamics' = High-Fidelity CFD Simulation
    // 'space' = Orbital Dynamics & Radiation
    // 'hydrodynamics' = Underwater/Marine Environment
    const [testScenario, setTestScenario] = useState('none');

    // Interaction Modes
    const [sketchMode, setSketchMode] = useState(false); // Phase 9.3
    const [viewMode, setViewMode] = useState('realistic'); // Phase 10: thermal/stress views
    const [sketchPoints, setSketchPoints] = useState([]); // Array of Vectors

    // Environmental Parameters (Test Inputs)
    const [testParams, setTestParams] = useState({
        temperature: 20,      // Celsius
        windSpeed: 0,         // m/s
        gravity: 9.81,        // m/s^2
        load: 0,              // kg
        rpm: 0,               // Rotations per minute
        // Aerodynamics-specific
        airDensity: 1.225,    // kg/m³ (sea level)
        attackAngle: 0,       // degrees
        // Space-specific
        orbitAltitude: 400,   // km (ISS altitude)
        solarRadiation: 1361, // W/m² (solar constant)
        // Hydrodynamics-specific
        waterDepth: 10,       // meters
        waterDensity: 1025,   // kg/m³ (seawater)
        currentSpeed: 0       // m/s
    });

    const [metrics, setMetrics] = useState({
        cpu: 12,
        memory: 24,
        network: { aero: 12, thermal: 4, geo: 142 },
        graph: { nodes: 1242, score: 0.998 }
    });
    const [kernelLogs, setKernelLogs] = useState([
        '[SYSTEM] vHIL Kernel Initialized',
        '[INFO] Agent Swarm: READY'
    ]);

    // State for Physics Engine
    const [physState, setPhysState] = useState({
        velocity: 0,
        altitude: 0,
        temperature: 20,
        fuel: 100,
        // 3D Position (for locomotive motion)
        position: { x: 0, y: 0, z: 0 },
        // Orientation (Euler angles in radians)
        orientation: { yaw: 0, pitch: 0, roll: 0 },
        // Acceleration for smooth transitions
        acceleration: 0
    });

    // Motion trail for visualization (last 100 positions)
    const [motionTrail, setMotionTrail] = useState([]);

    // State for Chemistry Engine
    const [chemState, setChemState] = useState({
        integrity: 1.0,
        corrosion_depth: 0.0,
        mass_loss: 0.0
    });

    const [kclCode, setKclCode] = useState('// KCL Source Code will appear here after compilation');
    const [compilationResult, setCompilationResult] = useState(null); // Stores full AST/ASM/Metrics from backend

    // Solver Status (ARES/LDP)
    const [solverStatus, setSolverStatus] = useState({
        ares: 'INIT',
        ldp: 'IDLE'
    });

    // Recursive ISA: Focused Pod ID for Ghost Mode
    // Recursive ISA: Focused Pod ID for Ghost Mode
    const [focusedPodId, setFocusedPodId] = useState(null);
    const [isaTree, setIsaTree] = useState(null); // Shared ISA Tree State

    // Critique / Reasoning Stream (Agent Feedback)
    const [reasoningStream, setReasoningStream] = useState([]);

    // Phase 10: Physics Analysis State
    const [physicsData, setPhysicsData] = useState({
        thermal: { temp: 20.0, status: 'nominal' },
        structural: { safety_factor: 10.0, stress: 0.0, deflection: 0.0 }
    });

    // Phase 14: Generative Design State
    const [morphSequence, setMorphSequence] = useState(null);

    // Phase 10: Trigger Analysis
    const runPhysicsAnalysis = React.useCallback(async (design) => {
        if (!design) return;

        console.log("Requesting Physics Analysis (Backend) for:", design.id);

        try {
            // Prepare Payload for Backend PhysicsAgent
            let geometryTree = [];
            let designParams = {};

            // Extract content (handling both JSON and raw string)
            let contentObj = {};
            if (typeof design.content === 'string') {
                try {
                    contentObj = JSON.parse(design.content);
                } catch (e) {
                    contentObj = { raw: design.content };
                }
            } else {
                contentObj = design.content;
            }

            // Simple geometry tree construction from design content
            if (contentObj.type === 'primitive') {
                geometryTree.push({
                    name: "main_body",
                    params: {
                        shape: contentObj.geometry,
                        // Map args to standard dimensions
                        width: contentObj.args?.[0] || 1,
                        height: contentObj.args?.[1] || 1,
                        length: contentObj.args?.[2] || 1,
                        radius: contentObj.args?.[0] || 0.5
                    },
                    material: { name: contentObj.material || "Generic" }
                });
            } else {
                // Fallback for custom code / unknown structure
                geometryTree.push({ name: "unknown_geometry", params: {} });
            }

            // Extract Params/Keywords
            const contentStr = typeof design.content === 'string' ? design.content : JSON.stringify(design.content);
            designParams = {
                power_watts: contentStr.includes("power") ? 500.0 : 50.0,
                g_loading: 3.0,
                physics_domain: contentStr.includes("nuclear") ? "NUCLEAR" : null
            };

            const res = await fetch('http://localhost:8000/api/physics/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    geometry_tree: geometryTree,
                    design_params: designParams,
                    environment: {
                        gravity: 9.81,
                        temperature: 20.0,
                        regime: "GROUND"
                    }
                })
            });

            if (!res.ok) throw new Error("Backend Analysis Failed");

            const realResult = await res.json();
            console.log("Physics Backend Result:", realResult);

            // Map Backend Result to Frontend State Structure
            // Backend returns: { physics_predictions, validation_flags, sub_agent_reports: { thermal, structural } }

            const mappedData = {
                thermal: realResult.sub_agent_reports?.thermal || { temp: 20.0, status: 'nominal' },
                structural: realResult.sub_agent_reports?.structural || { safety_factor: 10.0, stress: 0.0 },
                predictions: realResult.physics_predictions
            };

            setPhysicsData(mappedData);
            return mappedData;

        } catch (e) {
            console.error("Physics Analysis Failed:", e);
            return null;
        }
    }, []);

    // Helper to refresh ISA tree
    const refreshIsaTree = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/isa/tree');
            if (res.ok) {
                const data = await res.json();
                setIsaTree(data.tree);
            }
        } catch (err) {
            console.error("Failed to fetch ISA tree:", err);
        }
    };

    // Helper: Trigger Critique Loop
    const triggerCritique = async (geometry, sketch) => {
        try {
            const res = await fetch('http://localhost:8000/api/critique', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ geometry, sketch })
            });
            if (res.ok) {
                const data = await res.json();
                const critiques = data.critiques || [];

                if (critiques.length > 0) {
                    setReasoningStream(prev => {
                        const newLogs = critiques.map(c => ({
                            time: new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit' }),
                            agent: c.agent.toUpperCase(),
                            thought: c.message
                        }));
                        return [...prev.slice(-50), ...newLogs];
                    });
                }
            }
        } catch (e) {
            console.error("Critique Loop Failed:", e);
        }
    };


    // Real vHIL Loop (API Polling)
    React.useEffect(() => {
        // Initial Fetch
        refreshIsaTree();

        if (!isRunning) return;

        // Physics Loop (200ms instead of 100ms to prevent browser resource exhaustion)
        const physInterval = setInterval(async () => {
            // ... (rest of physics loop)

            try {
                // Build environment config based on active scenario
                let environment = {
                    gravity: testParams.gravity,
                    temperature: physState.temperature
                };

                // Scenario-specific environment configuration
                if (testScenario === 'aerodynamics') {
                    environment = {
                        ...environment,
                        regime: 'AERIAL',
                        fluid_density: testParams.airDensity,
                        wind_speed: testParams.windSpeed,
                        attack_angle: testParams.attackAngle
                    };
                } else if (testScenario === 'space') {
                    environment = {
                        ...environment,
                        regime: 'ORBITAL',
                        orbit_altitude_km: testParams.orbitAltitude,
                        solar_radiation_W_m2: testParams.solarRadiation,
                        gravity: 3.986004418e14 / Math.pow((6371 + testParams.orbitAltitude) * 1000, 2) // Orbital gravity
                    };
                } else if (testScenario === 'hydrodynamics') {
                    environment = {
                        ...environment,
                        regime: 'MARINE',
                        fluid_density: testParams.waterDensity,
                        depth_m: testParams.waterDepth,
                        current_speed: testParams.currentSpeed,
                        pressure_Pa: 101325 + (testParams.waterDensity * 9.81 * testParams.waterDepth) // Hydrostatic pressure
                    };
                } else {
                    environment.regime = 'AERIAL'; // Default
                    environment.fluid_density = 1.225;
                }

                const inputs = {
                    thrust: testParams.rpm * 50,
                    gravity: environment.gravity,
                    mass: testParams.load + 10.0,
                    environment: environment
                };

                const res = await fetch('http://localhost:8000/api/physics/step', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ state: physState, inputs, dt: 0.2 })
                });
                if (res.ok) {
                    const data = await res.json();

                    // Update position based on velocity integration (if not provided by backend)
                    const newState = data.state;
                    if (!newState.position && newState.velocity) {
                        const dt = 0.1;
                        const vx = newState.velocity * Math.sin(physState.orientation?.yaw || 0);
                        const vz = newState.velocity * Math.cos(physState.orientation?.yaw || 0);

                        newState.position = {
                            x: (physState.position?.x || 0) + vx * dt,
                            y: newState.altitude || physState.altitude || 0,
                            z: (physState.position?.z || 0) + vz * dt
                        };
                    }

                    // Preserve orientation if not updated by backend
                    if (!newState.orientation) {
                        newState.orientation = physState.orientation || { yaw: 0, pitch: 0, roll: 0 };
                    }

                    setPhysState(newState);

                    // Update motion trail (keep last 100 points) with velocity data
                    if (newState.position) {
                        const trailPoint = {
                            ...newState.position,
                            velocity: newState.velocity || 0
                        };
                        setMotionTrail(prev => [...prev.slice(-99), trailPoint]);
                    }

                    // Update CPU/Memory from Physics
                    setMetrics(prev => ({
                        ...prev,
                        cpu: Math.min(100, Math.abs(data.state.acceleration) * 2 + 10),
                        memory: 24 + (data.state.altitude * 0.01)
                    }));

                    // Appemd Logs from Physics Engine (if any)
                    if (data.state.logs && data.state.logs.length > 0) {
                        setKernelLogs(prev => [...prev, ...data.state.logs].slice(-20));
                    }

                    // Local Synthetic Logs (Legacy/Fallback)
                    if (data.state.altitude > 0 && physState.altitude === 0) {
                        setKernelLogs(prev => [...prev, `[INFO] LIFTOFF. Vel: ${data.state.velocity.toFixed(1)} m/s`].slice(-20));
                    }
                } else {
                    throw new Error('Physics Kernel Fault');
                }
            } catch (err) {
                console.error("Physics Fault:", err);
                setKernelLogs(prev => [...prev, `[ERR] Physics Kernel Fault: ${err.message}`].slice(-20));
            }
        }, 200); // 200ms interval to prevent browser resource exhaustion

        // Chemistry Loop (Slower - 500ms)
        const chemInterval = setInterval(async () => {
            // Only run if we are in a relevant scenario, or just always run background corrosion
            try {
                const inputs = {
                    ph: 6.5, // Slightly acidic 
                    temperature: physState.temperature, // Coupled specific physics temp
                    humidity: 0.8,
                    material_type: testParams.material || "steel"
                };
                const res = await fetch('http://localhost:8000/api/chemistry/step', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ state: chemState, inputs, dt: 0.5 })
                });
                if (res.ok) {
                    const data = await res.json();
                    setChemState(data.state);

                    // Update Graph Score based on integrity
                    setMetrics(prev => ({
                        ...prev,
                        graph: { ...prev.graph, score: data.state.integrity }
                    }));

                    if (data.state.integrity < 0.9 && chemState.integrity >= 0.9) {
                        setKernelLogs(prev => [...prev, `[WARN] STRUCTURAL DEGRADATION DETECTED. Integrity: ${(data.state.integrity * 100).toFixed(1)}%`].slice(-20));
                    }
                } else {
                    throw new Error('Chemistry Kernel Fault');
                }
            } catch (err) {
                console.error("Chemistry Fault:", err);
                setKernelLogs(prev => [...prev, `[ERR] Chemistry Kernel Fault: ${err.message}`].slice(-20));
            }
        }, 500);

        // System Status Polling (2s)
        const statusInterval = setInterval(async () => {
            try {
                const res = await fetch('http://localhost:8000/api/system/status');
                if (res.ok) {
                    const data = await res.json();
                    setSolverStatus({
                        ares: data.ares,
                        ldp: data.ldp
                    });
                }
            } catch (err) {
                // Silent fail for status checks
            }
        }, 2000);



        return () => {
            clearInterval(physInterval);
            clearInterval(chemInterval);
            clearInterval(statusInterval);
        };
    }, [isRunning, physState, chemState, testParams]);

    // Autosave (Debounced 5s) also triggers Critique
    React.useEffect(() => {
        const timer = setTimeout(() => {
            if (isaTree || (sketchPoints && sketchPoints.length > 0)) {
                saveInternal('autosave.brick');
                // Trigger Critique Loop asynchronously
                triggerCritique(isaTree ? [isaTree] : [], sketchPoints);
            }
        }, 5000);
        return () => clearTimeout(timer);
    }, [isaTree, sketchPoints, testParams, physState]);

    const saveInternal = async (filename) => {
        try {
            const projectData = {
                geometry: isaTree,
                physics: {
                    state: physState,
                    params: testParams,
                    scenario: testScenario
                },
                sketch: sketchPoints,
                timestamp: new Date().toISOString()
            };

            await fetch('http://localhost:8000/api/project/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ data: projectData, filename })
            });
            // console.log("Autosaved to", filename);
        } catch (e) {
            console.error("Autosave failed", e);
        }
    };


    const updateTestParam = (key, value) => {
        setTestParams(prev => ({
            ...prev,
            [key]: value
        }));
    };

    const value = {
        isRunning,
        setIsRunning,
        paused,
        setPaused,
        testScenario,
        setTestScenario,
        testParams,
        updateTestParam,
        // Telemetry Data
        metrics,
        kernelLogs,
        physState,
        chemState,
        kclCode,
        setKclCode,
        compilationResult,
        setCompilationResult,
        // Motion Data
        motionTrail,
        // Interaction Modes
        sketchMode,
        setSketchMode,
        viewMode,
        setViewMode,
        sketchPoints,
        sketchPoints,
        setSketchPoints,
        // System Status
        solverStatus,
        // Recursive ISA
        focusedPodId,
        setFocusedPodId,
        isaTree,
        refreshIsaTree,
        updateIsaNode: (nodeId, updates) => {
            setIsaTree(prevTree => {
                if (!prevTree) return prevTree;
                const updateRecursive = (node) => {
                    if (node.id === nodeId) {
                        return { ...node, ...updates };
                    }
                    if (node.children) {
                        return { ...node, children: node.children.map(updateRecursive) };
                    }
                    return node;
                };
                return updateRecursive(prevTree);
            });
        },
        saveProject: async (filename = "save.brick") => {
            try {
                const projectData = {
                    geometry: isaTree,
                    physics: {
                        state: physState,
                        params: testParams,
                        scenario: testScenario
                    },
                    sketch: sketchPoints,
                    timestamp: new Date().toISOString()
                };

                const res = await fetch('http://localhost:8000/api/project/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ data: projectData, filename })
                });

                if (!res.ok) throw new Error("Save failed");
                return await res.json();
            } catch (e) {
                console.error("Project Save Error:", e);
                throw e;
            }
        },
        loadProject: async (filename = "save.brick") => {
            try {
                const res = await fetch(`http://localhost:8000/api/project/load?filename=${filename}`);
                if (!res.ok) throw new Error("Load failed");
                const { data } = await res.json();

                if (data.geometry) setIsaTree(data.geometry);
                if (data.physics) {
                    if (data.physics.state) setPhysState(data.physics.state);
                    if (data.physics.params) setTestParams(data.physics.params);
                    if (data.physics.scenario) setTestScenario(data.physics.scenario);
                }
                if (data.sketch) setSketchPoints(data.sketch);

                return data;
            } catch (e) {
                console.error("Project Load Error:", e);
                throw e;
            }
        },
        // Phase 4: Intelligent Agents
        reasoningStream,
        setReasoningStream,
        triggerCritique,
        // Phase 10
        physicsData,
        runPhysicsAnalysis,
        // Phase 14
        morphSequence,
        setMorphSequence,
        // Helper to update geometry tree directly from MorphPlayer
        setGeometryTree: (newNodes) => {
            // newNodes is a flat list of node objects from the genome
            // The ISA Tree expects a hierarchical structure or at least a standard format.
            // If we just want to visualize, we might need a `visualizationOverride` state.
            // But for now, let's try to map it to the ISA tree if possible.
            // Actually, UnifiedSDFRenderer takes `geometryTree` as prop.
            // If `App.jsx` passes `isaTree` to Renderer, then updating `isaTree` here works.

            // Convert flat list to minimal tree for rendering
            // This is a simplification for the MVP Morph Player
            const root = newNodes.find(n => n.id === 'root') || newNodes[0];
            if (root) {
                // Construct a synthetic tree for the renderer
                const syntheticTree = {
                    id: root.id,
                    name: root.type,
                    params: Object.fromEntries(
                        Object.entries(root.params).map(([k, v]) => [k, v.value])
                    ),
                    transform: root.transform,
                    children: [] // MVP: Only morphing root for now or need full reconstruction
                };
                setIsaTree(syntheticTree);
            }
        },
    };

    return (
        <SimulationContext.Provider value={value}>
            {children}
        </SimulationContext.Provider>
    );
};
