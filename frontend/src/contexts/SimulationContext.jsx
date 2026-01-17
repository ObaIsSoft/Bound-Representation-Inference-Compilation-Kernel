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

    // Real vHIL Loop (API Polling)
    React.useEffect(() => {
        if (!isRunning) return;

        // Physics Loop (200ms instead of 100ms to prevent browser resource exhaustion)
        const physInterval = setInterval(async () => {
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

        return () => {
            clearInterval(physInterval);
            clearInterval(chemInterval);
        };
    }, [isRunning, physState, chemState, testParams]);

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
        // Motion Data
        motionTrail
    };

    return (
        <SimulationContext.Provider value={value}>
            {children}
        </SimulationContext.Provider>
    );
};
