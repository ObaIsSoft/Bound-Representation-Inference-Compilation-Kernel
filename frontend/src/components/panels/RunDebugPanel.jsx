import React, { useState } from 'react';
import { Play, Square, Bug, Thermometer, Wind, Gauge, Activity, ListChecks, Waves, Rocket, CloudRain } from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';
import { useSimulation } from '../../contexts/SimulationContext';

const RunDebugPanel = ({ width }) => {
    const { theme } = useTheme();
    const {
        isRunning, setIsRunning,
        testScenario, setTestScenario,
        testParams, updateTestParam
    } = useSimulation();

    const [debugMode, setDebugMode] = useState(false);

    if (width <= 0) return null;

    const renderParamInput = (icon, label, key, unit, min, max, step) => (
        <div className="flex items-center justify-between p-2 rounded" style={{ backgroundColor: theme.colors.bg.primary }}>
            <div className="flex items-center gap-2">
                {React.createElement(icon, { size: 14, style: { color: theme.colors.text.secondary } })}
                <span className="text-[10px] font-mono" style={{ color: theme.colors.text.tertiary }}>{label}</span>
            </div>
            <div className="flex items-center gap-1">
                <input
                    type="range"
                    min={min} max={max} step={step}
                    value={testParams[key]}
                    onChange={(e) => updateTestParam(key, parseFloat(e.target.value))}
                    className="w-16 h-1 rounded-lg appearance-none cursor-pointer"
                    style={{ backgroundColor: theme.colors.bg.elevated }}
                    disabled={isRunning}
                />
                <input
                    type="number"
                    value={testParams[key]}
                    onChange={(e) => updateTestParam(key, parseFloat(e.target.value))}
                    className="w-10 bg-transparent text-right text-[10px] font-mono outline-none border-b border-transparent focus:border-current"
                    style={{ color: theme.colors.text.primary }}
                />
                <span className="text-[10px] font-mono w-4" style={{ color: theme.colors.text.muted }}>{unit}</span>
            </div>
        </div>
    );

    const toggleSimulation = async () => {
        const newStatus = !isRunning;
        setIsRunning(newStatus); // Optimistic

        try {
            await fetch('http://localhost:8000/api/simulation/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    command: newStatus ? 'START' : 'STOP',
                    scenario: testScenario,
                    params: testParams
                })
            });
        } catch (e) {
            console.error("Simulation control failed:", e);
            setIsRunning(!newStatus); // Revert
        }
    };

    return (
        <aside
            className="h-full flex flex-col shrink-0 overflow-hidden"
            style={{
                width,
                backgroundColor: theme.colors.bg.secondary,
                borderRight: `1px solid ${theme.colors.border.primary}`
            }}
        >
            <PanelHeader title="Run and Debug" icon={Play} />

            <div className="p-3 space-y-4">
                {/* 1. Configuration / Scenario Selector */}
                <div>
                    <label className="text-[9px] uppercase font-mono mb-2 block font-semibold" style={{ color: theme.colors.text.muted }}>
                        Test Scenario
                    </label>
                    <select
                        value={testScenario}
                        onChange={(e) => setTestScenario(e.target.value)}
                        className="w-full rounded p-2 text-xs font-mono outline-none mb-2"
                        disabled={isRunning}
                        style={{
                            backgroundColor: theme.colors.bg.primary,
                            border: `1px solid ${theme.colors.border.primary}`,
                            color: theme.colors.text.primary
                        }}
                    >
                        <option value="none">Standard Physics</option>
                        <option value="thermal">Thermal Stress Test</option>
                        <option value="structural">Structural Load Test</option>
                        <option value="propulsion">Propulsion Efficiency</option>
                        <option value="wind_tunnel">Wind Tunnel</option>
                        <option value="aerodynamics">High-Fidelity CFD</option>
                        <option value="space">Orbital Dynamics</option>
                        <option value="hydrodynamics">Marine Environment</option>
                    </select>

                    <p className="text-[9px] leading-tight" style={{ color: theme.colors.text.muted }}>
                        {testScenario === 'none' && 'Default environmental physics.'}
                        {testScenario === 'thermal' && 'Simulate extreme temperature variations to test material expansion and failure points.'}
                        {testScenario === 'structural' && 'Apply gravitational and kinetic loads to verify structural integrity.'}
                        {testScenario === 'propulsion' && 'Test engine thrust curves and fuel consumption rates.'}
                        {testScenario === 'wind_tunnel' && 'Simulate aerodynamics with variable wind speed and direction.'}
                        {testScenario === 'aerodynamics' && 'High-fidelity CFD with real-time drag, lift, and pressure calculations.'}
                        {testScenario === 'space' && 'Orbital mechanics with microgravity, radiation, and thermal vacuum conditions.'}
                        {testScenario === 'hydrodynamics' && 'Underwater dynamics with buoyancy, drag, and hydrostatic pressure.'}
                    </p>
                </div>

                {/* 2. Run Controls */}
                <div className="flex gap-2">
                    <button
                        onClick={toggleSimulation}
                        className="flex-1 flex items-center justify-center gap-2 py-2 rounded font-mono text-xs font-bold transition-all shadow-lg"
                        style={{
                            backgroundColor: isRunning ? theme.colors.status.error + '22' : theme.colors.status.success + '22',
                            border: `1px solid ${isRunning ? theme.colors.status.error : theme.colors.status.success}44`,
                            color: isRunning ? theme.colors.status.error : theme.colors.status.success,
                            boxShadow: isRunning ? `0 0 10px ${theme.colors.status.error}22` : `0 0 10px ${theme.colors.status.success}22`
                        }}
                    >
                        {isRunning ? <><Square size={14} fill="currentColor" /> ABORT TEST</> : <><Play size={14} fill="currentColor" /> RUN TEST</>}
                    </button>
                    <button
                        onClick={() => setDebugMode(!debugMode)}
                        className="px-3 py-2 rounded transition-all"
                        style={{
                            backgroundColor: debugMode ? theme.colors.accent.primary + '1A' : theme.colors.bg.tertiary,
                            border: `1px solid ${debugMode ? theme.colors.accent.primary : theme.colors.border.primary}`,
                            color: debugMode ? theme.colors.accent.primary : theme.colors.text.tertiary
                        }}
                        title="Toggle Debug Mode"
                    >
                        <Bug size={14} />
                    </button>
                </div>

                <div className="h-px w-full" style={{ backgroundColor: theme.colors.border.primary + '40' }} />

                {/* 3. Test Parameters (Context Aware) */}
                <div className="space-y-2">
                    <label className="text-[9px] uppercase font-mono block font-semibold" style={{ color: theme.colors.text.muted }}>
                        <ListChecks size={10} className="inline mr-1" />
                        Test Parameters
                    </label>

                    {/* Always show basic env params or specific ones based on scenario */}
                    {testScenario === 'none' && (
                        <div className="text-[10px] font-mono opacity-50 italic pl-1" style={{ color: theme.colors.text.muted }}>No active test override.</div>
                    )}

                    {(testScenario === 'thermal' || testScenario === 'propulsion') &&
                        renderParamInput(Thermometer, 'Ambient Temp', 'temperature', '°C', -50, 150, 1)
                    }

                    {(testScenario === 'wind_tunnel' || testScenario === 'propulsion') &&
                        renderParamInput(Wind, 'Wind Speed', 'windSpeed', 'm/s', 0, 100, 1)
                    }

                    {testScenario === 'structural' &&
                        renderParamInput(Gauge, 'Gravity Load', 'gravity', 'g', 0, 10, 0.1)
                    }

                    {testScenario === 'propulsion' &&
                        renderParamInput(Activity, 'Throttle', 'rpm', '%', 0, 100, 1)
                    }

                    {/* New Aerodynamics Parameters */}
                    {testScenario === 'aerodynamics' && (
                        <>
                            {renderParamInput(Wind, 'Wind Speed', 'windSpeed', 'm/s', 0, 100, 1)}
                            {renderParamInput(CloudRain, 'Air Density', 'airDensity', 'kg/m³', 0.5, 1.5, 0.01)}
                            {renderParamInput(Gauge, 'Attack Angle', 'attackAngle', '°', -20, 40, 1)}
                        </>
                    )}

                    {/* Space Parameters */}
                    {testScenario === 'space' && (
                        <>
                            {renderParamInput(Rocket, 'Orbit Altitude', 'orbitAltitude', 'km', 200, 2000, 10)}
                            {renderParamInput(Thermometer, 'Solar Radiation', 'solarRadiation', 'W/m²', 0, 1500, 10)}
                        </>
                    )}

                    {/* Hydrodynamics Parameters */}
                    {testScenario === 'hydrodynamics' && (
                        <>
                            {renderParamInput(Waves, 'Water Depth', 'waterDepth', 'm', 0, 500, 1)}
                            {renderParamInput(CloudRain, 'Water Density', 'waterDensity', 'kg/m³', 997, 1050, 1)}
                            {renderParamInput(Wind, 'Current Speed', 'currentSpeed', 'm/s', 0, 5, 0.1)}
                        </>
                    )}
                </div>

                {debugMode && (
                    <div className="mt-4 p-2 rounded border" style={{ backgroundColor: theme.colors.status.error + '1A', borderColor: theme.colors.status.error + '40' }}>
                        <div className="text-[9px] font-mono mb-1" style={{ color: theme.colors.status.error }}>Debugger Active</div>
                        <div className="text-[9px] font-mono" style={{ color: theme.colors.text.tertiary }}>Listening on port 9229...</div>
                    </div>
                )}
            </div>
        </aside>
    );
};

export default RunDebugPanel;
