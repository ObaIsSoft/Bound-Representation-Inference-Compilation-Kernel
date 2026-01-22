import React, { useEffect, useState } from 'react';
import { CheckCircle, AlertTriangle, XCircle, Activity, Thermometer, Wind, Gauge, ArrowRight } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';
import { useSimulation } from '../../contexts/SimulationContext';

const TestResultsTab = () => {
    const { theme } = useTheme();
    const { isRunning, testScenario, testParams, kernelLogs, physState, chemState, compilationResult } = useSimulation();
    const [results, setResults] = useState(null);

    // Auto-scroll logs
    const logContainerRef = React.useRef(null);
    useEffect(() => {
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [kernelLogs]);

    // Handle Simulation End -> Generate Report
    useEffect(() => {
        if (!isRunning && kernelLogs.length > 0 && !results) {
            generateReport();
        } else if (isRunning) {
            setResults(null);
        }
    }, [isRunning, kernelLogs]);

    const generateReport = () => {
        let report = { passed: true, metrics: [] };

        // 1. Prioritize Backend Verification (Orchestrator Truth)
        if (compilationResult && compilationResult.physics_predictions) {
            const preds = compilationResult.physics_predictions;
            const flags = compilationResult.validation_flags || {};

            // Map backend keys to UI metrics
            const backendMetrics = [];

            if (preds.drag_N) backendMetrics.push({ label: 'Drag Force (Sim)', value: `${preds.drag_N.toFixed(0)} N`, status: preds.drag_N < 2000 ? 'success' : 'warning' });
            if (preds.lift_N) backendMetrics.push({ label: 'Lift Force (Sim)', value: `${preds.lift_N.toFixed(0)} N`, status: preds.lift_N > (testParams.load || 10) * 9.81 ? 'success' : 'critical' });
            if (preds.max_temp_C) backendMetrics.push({ label: 'Peak Temp', value: `${preds.max_temp_C.toFixed(1)}°C`, status: preds.max_temp_C < 1000 ? 'success' : 'critical' });
            if (preds.stress_MPa) backendMetrics.push({ label: 'Max Stress', value: `${preds.stress_MPa.toFixed(1)} MPa`, status: preds.stress_MPa < 500 ? 'success' : 'warning' });

            // If we have backend data, blindly use it + validation flags
            if (backendMetrics.length > 0) {
                report.metrics = backendMetrics;
                report.passed = flags.physics_safe !== false; // Default to true if missing
                setResults(report);
                return;
            }
        }

        // 2. Fallback to Client-Side Heuristics (Visual Truth)
        switch (testScenario) {
            case 'thermal':
                const maxTemp = physState.temperature;
                const stress = Math.min(100, Math.max(0, (maxTemp - 20) * 0.8));
                report.metrics = [
                    { label: 'Peak Temperature', value: `${maxTemp.toFixed(1)}°C`, status: maxTemp < 100 ? 'success' : 'critical' },
                    { label: 'Thermal Stress', value: `${stress.toFixed(1)}%`, status: stress > 80 ? 'critical' : 'info' },
                    { label: 'Material Integrity', value: `${(chemState.integrity * 100).toFixed(1)}%`, status: chemState.integrity > 0.9 ? 'success' : 'warning' }
                ];
                report.passed = stress < 90 && chemState.integrity > 0.8;
                break;

            case 'propulsion':
                report.metrics = [
                    { label: 'Max Velocity', value: `${physState.velocity.toFixed(1)} m/s`, status: 'info' },
                    { label: 'Vertical Displacement', value: `${physState.altitude.toFixed(1)} m`, status: 'info' },
                    { label: 'Energy Reserves', value: `${physState.fuel.toFixed(1)}%`, status: physState.fuel > 10 ? 'success' : 'warning' }
                ];
                report.passed = physState.altitude > 10;
                break;

            case 'structural':
            case 'chemical': //  New scenario
                const corrosion = chemState.corrosion_depth;
                report.metrics = [
                    { label: 'Start Integrity', value: '100%', status: 'success' },
                    { label: 'End Integrity', value: `${(chemState.integrity * 100).toFixed(2)}%`, status: chemState.integrity > 0.9 ? 'success' : 'critical' },
                    { label: 'Corrosion Depth', value: `${(corrosion * 1000).toFixed(2)}µm`, status: corrosion < 0.001 ? 'success' : 'warning' }
                ];
                report.passed = chemState.integrity > 0.9;
                break;

            case 'aerodynamics':
                const velocity = Math.abs(physState.velocity) || testParams.windSpeed || 40;
                const dragForce = 0.5 * testParams.airDensity * velocity * velocity * 3.14; // Rough calculation
                const liftCoeff = ((velocity > 20) ? 1.2 : 0.8) + (testParams.attackAngle / 100);
                report.metrics = [
                    { label: 'Drag Force', value: `${dragForce.toFixed(0)} N`, status: dragForce < 1500 ? 'success' : 'warning' },
                    { label: 'Lift Coefficient', value: liftCoeff.toFixed(2), status: liftCoeff > 1.0 ? 'success' : 'critical' },
                    { label: 'Reynolds Number', value: `${((testParams.airDensity * velocity * 2.4) / 1.81e-5 / 1e6).toFixed(1)}M`, status: 'info' },
                    { label: 'Flow Regime', value: velocity > 30 ? 'Turbulent' : 'Laminar', status: 'info' }
                ];
                report.passed = dragForce < 2000 && liftCoeff > 1.0;
                break;

            case 'space':
                const orbitVel = Math.sqrt(3.986004418e14 / ((6371 + testParams.orbitAltitude) * 1000));
                const orbitalPeriod = (2 * Math.PI * (6371 + testParams.orbitAltitude) * 1000 / orbitVel / 60).toFixed(0);
                const gravityLevel = (3.986004418e14 / Math.pow((6371 + testParams.orbitAltitude) * 1000, 2)).toFixed(3);
                report.metrics = [
                    { label: 'Orbital Velocity', value: `${(orbitVel / 1000).toFixed(2)} km/s`, status: 'info' },
                    { label: 'Orbital Period', value: `${orbitalPeriod} min`, status: 'info' },
                    { label: 'Microgravity', value: `${gravityLevel} m/s²`, status: parseFloat(gravityLevel) < 0.1 ? 'success' : 'warning' },
                    { label: 'Solar Radiation', value: `${testParams.solarRadiation} W/m²`, status: testParams.solarRadiation < 1400 ? 'success' : 'critical' }
                ];
                report.passed = parseFloat(gravityLevel) < 0.1;
                break;

            case 'hydrodynamics':
                const waterVel = testParams.currentSpeed || 1;
                const hydroDrag = 0.5 * testParams.waterDensity * waterVel * waterVel * 2.0;
                const pressure = (101325 + (testParams.waterDensity * 9.81 * testParams.waterDepth)) / 1000; // kPa
                const buoyancy = testParams.waterDensity * 9.81 * 5; // Assuming 5m³ volume
                report.metrics = [
                    { label: 'Hydrostatic Pressure', value: `${pressure.toFixed(1)} kPa`, status: pressure < 5000 ? 'success' : 'warning' },
                    { label: 'Drag Force', value: `${hydroDrag.toFixed(0)} N`, status: 'info' },
                    { label: 'Buoyant Force', value: `${(buoyancy / 1000).toFixed(1)} kN`, status: 'info' },
                    { label: 'Depth', value: `${testParams.waterDepth} m`, status: testParams.waterDepth < 200 ? 'success' : 'critical' }
                ];
                report.passed = pressure < 10000;
                break;

            default:
                report.metrics = [
                    { label: 'Execution Time', value: 'Live', status: 'info' },
                    { label: 'Final Velocity', value: `${physState.velocity.toFixed(2)} m/s`, status: 'info' }
                ];
        }
        setResults(report);
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case 'success': return <CheckCircle size={14} style={{ color: theme.colors.status.success }} />;
            case 'warning': return <AlertTriangle size={14} style={{ color: theme.colors.status.warning }} />;
            case 'critical': return <XCircle size={14} style={{ color: theme.colors.status.error }} />;
            default: return <Activity size={14} style={{ color: theme.colors.text.tertiary }} />;
        }
    };

    return (
        <div className="h-full flex gap-4">
            {/* Live Logs */}
            <div className="w-1/2 flex flex-col font-mono text-xs">
                <div className="mb-2 uppercase font-bold text-[10px]" style={{ color: theme.colors.text.muted }}>Live Telemetry</div>
                <div className="flex-1 overflow-y-auto p-2 rounded space-y-1" style={{ backgroundColor: theme.colors.bg.primary, border: `1px solid ${theme.colors.border.primary}` }} ref={logContainerRef}>
                    {isRunning || kernelLogs.length > 0 ? (
                        kernelLogs.map((log, i) => (
                            <div key={i} className="flex gap-2 opacity-80">
                                <span style={{ color: theme.colors.text.tertiary }}>{log.startsWith('[') ? log.split(']')[0] + ']' : '>'}</span>
                                <span style={{ color: theme.colors.text.primary }}>{log.startsWith('[') ? log.split(']').slice(1).join(']') : log}</span>
                            </div>
                        ))
                    ) : (
                        results ? (
                            <div className="flex flex-col items-center justify-center h-full opacity-50 space-y-2">
                                <CheckCircle size={24} style={{ color: theme.colors.status.success }} />
                                <span>Test Cycle Complete</span>
                            </div>
                        ) : (
                            <div className="flex flex-col items-center justify-center h-full opacity-30">
                                <span>Ready to start simulation...</span>
                            </div>
                        )
                    )}
                </div>
            </div>

            {/* Final Report */}
            <div className="w-1/2 flex flex-col">
                <div className="mb-2 uppercase font-bold text-[10px]" style={{ color: theme.colors.text.muted }}>Test Report</div>
                <div className="flex-1 p-4 rounded border flex flex-col" style={{
                    backgroundColor: theme.colors.bg.primary,
                    borderColor: theme.colors.border.primary
                }}>
                    {results ? (
                        <>
                            <div className="flex items-center gap-3 mb-6 pb-4 border-b" style={{ borderColor: theme.colors.border.primary }}>
                                {results.passed ?
                                    <CheckCircle size={32} style={{ color: theme.colors.status.success }} /> :
                                    <XCircle size={32} style={{ color: theme.colors.status.error }} />
                                }
                                <div>
                                    <div className="text-sm font-bold" style={{ color: theme.colors.text.primary }}>
                                        {results.passed ? 'TEST PASSED' : 'TEST FAILED'}
                                    </div>
                                    <div className="text-xs opacity-70" style={{ color: theme.colors.text.secondary }}>
                                        Scenario: {testScenario.toUpperCase()}
                                    </div>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                {results.metrics.map((metric, i) => (
                                    <div key={i} className="p-3 rounded" style={{ backgroundColor: theme.colors.bg.secondary }}>
                                        <div className="flex items-center justify-between mb-1">
                                            <span className="text-[10px] uppercase font-bold" style={{ color: theme.colors.text.muted }}>{metric.label}</span>
                                            {getStatusIcon(metric.status)}
                                        </div>
                                        <div className="text-lg font-mono font-bold" style={{ color: theme.colors.text.primary }}>
                                            {metric.value}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </>
                    ) : (
                        <div className="flex-1 flex flex-col items-center justify-center opacity-30 gap-2">
                            <Activity size={32} />
                            <div className="text-xs font-mono">No results available</div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default TestResultsTab;
