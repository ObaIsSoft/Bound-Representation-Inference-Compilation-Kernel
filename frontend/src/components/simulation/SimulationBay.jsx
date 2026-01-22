import React, { useState, useMemo, useEffect, useRef } from 'react';
import { Box, Layers, Eye, Grid, Activity, Sun, Square, LayoutTemplate, Zap, Ghost, FileDigit, Thermometer, Droplet, BoxSelect, Monitor } from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';
import { useDesign } from '../../contexts/DesignContext';
import { useSettings } from '../../contexts/SettingsContext';
import { useSimulation } from '../../contexts/SimulationContext';

// Unified SDF Renderer (replaces DefaultSimulation, RaymarchScene, TypeGPURaymarch)
import UnifiedSDFRenderer from './UnifiedSDFRenderer';

// Legacy imports kept for gradual migration (can be removed after verification)
// import DefaultSimulation from './DefaultSimulation';
// import RaymarchScene from './RaymarchScene';
// import TypeGPURaymarch from './TypeGPURaymarch';

// View Modes Definition (Stable)
const ALL_VIEW_MODES = [
    // Standard Modes (Always visible)
    { id: 'realistic', label: 'Realistic', icon: Eye, category: 'standard' },
    { id: 'matte', label: 'Matte', icon: Droplet, category: 'standard' },
    { id: 'wireframe', label: 'Wireframe', icon: Grid, category: 'standard' },
    { id: 'thermal', label: 'Thermal', icon: Activity, category: 'standard' }, // Phase 10: Renamed from 'heatmap'
    { id: 'hyperrealism', label: 'Hyperrealism', icon: Sun, category: 'standard' },
    { id: 'solid', label: 'Solid', icon: Square, category: 'standard' },
    { id: 'interior', label: 'Interior', icon: BoxSelect, category: 'standard' },
    { id: 'shaded', label: 'Shaded', icon: Monitor, category: 'standard' },
    { id: 'xray', label: 'X-Ray', icon: Ghost, category: 'standard' },
    { id: 'hidden_line', label: 'Hidden Line', icon: FileDigit, category: 'standard' },

    // Physics Modes (Run & Debug only)
    { id: 'stress', label: 'Stress Map', icon: Layers, category: 'physics' },
    { id: 'flow', label: 'Flow Dynamics', icon: Zap, category: 'physics' },
    { id: 'micro', label: 'Micro-Machining (GLSL)', icon: Monitor, category: 'physics' },
    { id: 'micro_wgsl', label: 'Micro-Machining (WebGPU)', icon: Activity, category: 'physics' }
];

const SimulationBay = ({ activeActivity }) => {
    const { theme } = useTheme();
    const { activeTab } = useDesign();
    const { showTemperatureSensor } = useSettings();
    const { isRunning, testParams, updateTestParam } = useSimulation();
    const [viewMode, setViewMode] = useState('wireframe');
    const [showGrid, setShowGrid] = useState(true);
    const [isExploded, setIsExploded] = useState(false);
    const [viewportMenuOpen, setViewportMenuOpen] = useState(false);
    const [showViewMenu, setShowViewMenu] = useState(false);
    const viewportMenuRef = useRef(null);

    // 3D Thermometer Logic
    const isUntitled = activeTab?.name?.startsWith('Untitled Design');
    const showTempControl = showTemperatureSensor && isUntitled;

    // Physics State (Lifted for Visualization)
    const [physicsResult, setPhysicsResult] = useState(null);
    const { physState, motionTrail, metrics } = useSimulation();

    // Combine live simulation data
    const livePhysicsData = {
        state: physState,
        motionTrail: motionTrail,
        metrics: metrics
    };

    // Simply use all modes always. 
    // Restriction logic was preventing users from inspecting static models.
    const visibleModes = ALL_VIEW_MODES;


    return (
        <main className="flex-1 flex flex-col min-w-0 relative">
            <PanelHeader title="Simulation Bay" icon={Box}>
                <div className="flex items-center gap-2">
                    <div className="flex items-center gap-2 px-2 py-1 rounded-sm border" style={{
                        backgroundColor: theme.colors.bg.tertiary,
                        borderColor: theme.colors.border.primary
                    }}>
                        <div className={`w-1.5 h-1.5 rounded-full ${isRunning ? 'animate-pulse' : ''}`} style={{
                            backgroundColor: isRunning ? theme.colors.status.success : theme.colors.status.warning
                        }} />
                        <span className="text-[10px] font-mono uppercase" style={{ color: theme.colors.text.secondary }}>
                            {isRunning ? 'RUNNING' : 'STANDBY'}
                        </span>
                    </div>
                </div>
            </PanelHeader>

            <div className="flex-1 relative overflow-hidden flex flex-col items-center justify-center" style={{ backgroundColor: theme.colors.bg.primary }}>
                {!activeTab ? (
                    <div className="flex flex-col items-center gap-4 z-10 opacity-40 pointer-events-none select-none">
                        <BoxSelect size={48} style={{ color: theme.colors.text.muted }} strokeWidth={1} />
                        <span className="text-xs font-bold tracking-[0.2em] uppercase" style={{ color: theme.colors.text.muted }}>
                            Select a file to start designing
                        </span>
                    </div>
                ) : (
                    <UnifiedSDFRenderer
                        design={activeTab}
                        viewMode={viewMode}
                        physicsData={livePhysicsData || physicsResult}
                    />
                )}

                {/* Overlay UI - Combined FPS + Buffer */}
                <div className="absolute top-2 left-2 z-10 pointer-events-none">
                    <div className="backdrop-blur-sm p-1 rounded border pointer-events-auto"
                        style={{
                            backgroundColor: theme.colors.bg.secondary + '66',
                            borderColor: theme.colors.border.secondary
                        }}
                    >
                        <div className="flex items-center gap-1.5">
                            <div className="flex flex-col">
                                <div className="text-[5px] font-mono uppercase leading-tight" style={{ color: theme.colors.text.muted }}>FPS</div>
                                <div className="text-[10px] font-mono font-bold leading-tight" style={{ color: theme.colors.status.success }}>60</div>
                            </div>
                            <div className="w-px h-4" style={{ backgroundColor: theme.colors.border.secondary }} />
                            <div className="flex items-center gap-0.5">
                                <div className={`w-0.5 h-0.5 rounded-full ${isRunning ? 'animate-pulse' : ''}`}
                                    style={{ backgroundColor: isRunning ? theme.colors.status.success : theme.colors.status.warning }}
                                />
                                <span className="text-[5px] font-mono" style={{ color: theme.colors.text.secondary }}>
                                    {isRunning ? 'Act' : 'Sta'}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* View Mode Toolbar */}
                <div className="absolute top-2 right-2 z-10 flex flex-col gap-1 pointer-events-auto">
                    <div className="relative">
                        <button
                            onClick={() => setShowViewMenu(!showViewMenu)}
                            className="backdrop-blur-sm p-1 rounded border flex items-center gap-1 transition-colors"
                            style={{
                                backgroundColor: theme.colors.bg.secondary + '66',
                                borderColor: theme.colors.border.secondary,
                                color: theme.colors.text.primary
                            }}
                        >
                            {ALL_VIEW_MODES.find(m => m.id === viewMode)?.icon &&
                                React.createElement(ALL_VIEW_MODES.find(m => m.id === viewMode).icon, {
                                    size: 10,
                                    style: { color: theme.colors.text.primary }
                                })
                            }
                            <span className="text-[8px] font-mono leading-none">
                                {ALL_VIEW_MODES.find(m => m.id === viewMode)?.label}
                            </span>
                        </button>

                        {showViewMenu && (
                            <div className="absolute top-full right-0 mt-2 w-48 backdrop-blur-md border rounded overflow-hidden shadow-xl"
                                style={{
                                    maxHeight: 'calc(100vh - 200px)',
                                    overflowY: 'auto',
                                    backgroundColor: theme.colors.bg.secondary + 'F2',
                                    borderColor: theme.colors.border.primary
                                }}>
                                <div className="grid grid-cols-1 gap-px">
                                    {visibleModes.map(mode => (
                                        <button
                                            key={mode.id}
                                            onClick={() => {
                                                setViewMode(mode.id);
                                                setShowViewMenu(false);
                                            }}
                                            className="flex items-center gap-3 px-3 py-2 text-xs font-mono transition-colors text-left"
                                            style={{
                                                backgroundColor: viewMode === mode.id ? theme.colors.bg.tertiary : 'transparent',
                                                color: viewMode === mode.id ? theme.colors.text.primary : theme.colors.text.secondary
                                            }}
                                        >
                                            <mode.icon size={14} style={{ color: viewMode === mode.id ? theme.colors.accent.primary : theme.colors.text.muted }} />
                                            {mode.label}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* View Controls Overlay */}
                <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-1 p-0.5 rounded-full border backdrop-blur-sm transition-all duration-300 group z-10"
                    style={{
                        backgroundColor: theme.colors.bg.secondary + '99',
                        borderColor: theme.colors.border.primary,
                        boxShadow: '0 4px 16px rgba(0,0,0,0.3)'
                    }}
                >
                    {/* Scatter Toggle */}
                    <button
                        onClick={() => setIsExploded(!isExploded)}
                        className="flex items-center gap-1 px-1.5 py-0.5 rounded-full transition-all border"
                        style={{
                            backgroundColor: isExploded ? theme.colors.accent.primary + '33' : 'transparent',
                            borderColor: isExploded ? theme.colors.accent.primary : 'transparent',
                            color: isExploded ? theme.colors.accent.primary : theme.colors.text.muted
                        }}
                        title={isExploded ? "Assemble" : "Scatter"}
                    >
                        <Layers size={8} />
                        <span className="text-[7px] font-bold uppercase tracking-wider hidden sm:block">
                            {isExploded ? "Assemble" : "Scatter"}
                        </span>
                    </button>

                    <div className="w-px h-2" style={{ backgroundColor: theme.colors.border.primary }} />

                    <div className="relative" ref={viewportMenuRef}></div>
                </div>

                {/* Temperature Sensor Input - SYNCED WITH CONTEXT */}
                {showTempControl && (
                    <div
                        className="absolute bottom-2 right-2 z-20 flex items-center gap-1 p-1 rounded border shadow-lg backdrop-blur-sm"
                        style={{
                            backgroundColor: theme.colors.bg.secondary + '99',
                            borderColor: theme.colors.border.primary
                        }}
                    >
                        <Thermometer size={8} style={{ color: theme.colors.accent.primary }} />
                        <div className="flex flex-col items-end">
                            <label className="text-[5px] font-bold uppercase tracking-wider leading-tight" style={{ color: theme.colors.text.secondary }}>
                                Temp
                            </label>
                            <div className="flex items-center gap-0.5">
                                <input
                                    type="number"
                                    value={testParams.temperature}
                                    onChange={(e) => updateTestParam('temperature', Number(e.target.value))}
                                    className="w-8 bg-transparent text-[8px] font-mono outline-none text-right border-b border-transparent focus:border-current transition-colors"
                                    style={{ color: theme.colors.text.primary }}
                                />
                                <span className="text-[6px] font-mono" style={{ color: theme.colors.text.muted }}>°C</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* PHYSICS TERMINAL OVERLAY */}
            <PhysicsOverlay theme={theme} active={showViewMenu} onResult={setPhysicsResult} />
        </main>
    );
};

const PhysicsOverlay = ({ theme, active, onResult }) => {
    const [domain, setDomain] = useState('NUCLEAR');
    const [query, setQuery] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    // Auto-expand if result is present
    const [expanded, setExpanded] = useState(false);

    const runPhysics = async () => {
        setLoading(true);
        try {
            const res = await fetch('http://localhost:8000/api/physics/solve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query || "Simulate",
                    domain: domain,
                    params: {} // Populate based on form inputs later
                })
            });
            const data = await res.json();
            setResult(data);
            if (onResult) onResult({ ...data, domain }); // Pass up
            setExpanded(true);
        } catch (e) {
            setResult({ error: e.message });
        }
        setLoading(false);
    };

    if (!active) return null; // Or control visibility via a separate state

    return (
        <div className="absolute top-4 left-4 z-50 w-80 border rounded-lg p-4 shadow-2xl backdrop-blur-xl"
            style={{
                backgroundColor: theme.colors.bg.secondary + 'F2', // 95% opacity
                borderColor: theme.colors.border.primary
            }}
        >
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-xs font-bold uppercase tracking-wider" style={{ color: theme.colors.text.primary }}>Physics Oracle</h3>
                <div className={`w-2 h-2 rounded-full ${loading ? 'animate-ping' : ''}`}
                    style={{ backgroundColor: loading ? theme.colors.status.warning : theme.colors.status.success }}
                />
            </div>

            <div className="space-y-3">
                <div>
                    <label className="text-[10px] uppercase" style={{ color: theme.colors.text.muted }}>Domain</label>
                    <select
                        value={domain}
                        onChange={e => setDomain(e.target.value)}
                        className="w-full text-xs p-1 rounded mt-1 border"
                        style={{
                            backgroundColor: theme.colors.bg.tertiary,
                            borderColor: theme.colors.border.secondary,
                            color: theme.colors.text.primary
                        }}
                    >
                        <option value="NUCLEAR">NUCLEAR</option>
                        <option value="ASTROPHYSICS">ASTROPHYSICS</option>
                        <option value="THERMODYNAMICS">THERMODYNAMICS</option>
                        <option value="OPTICS">OPTICS</option>
                        <option value="FLUID">FLUID</option>
                        <option value="CIRCUIT">CIRCUIT</option>
                        <option value="MECHANICS">MECHANICS</option>
                        <option value="ELECTROMAGNETISM">ELECTROMAGNETISM</option>
                        <option value="QUANTUM">QUANTUM</option>
                        <option value="ACOUSTICS">ACOUSTICS</option>
                        <option value="MATERIALS">MATERIALS</option>
                        <option value="PLASMA">PLASMA</option>
                        <option value="RELATIVITY">RELATIVITY</option>
                        <option value="GEOPHYSICS">GEOPHYSICS</option>
                    </select>
                </div>

                <div>
                    <label className="text-[10px] uppercase" style={{ color: theme.colors.text.muted }}>Input</label>
                    <input
                        type="text"
                        value={query}
                        onChange={e => setQuery(e.target.value)}
                        placeholder="e.g. Calculate Critical Mass"
                        className="w-full text-xs p-1 rounded mt-1 border"
                        style={{
                            backgroundColor: theme.colors.bg.tertiary,
                            borderColor: theme.colors.border.secondary,
                            color: theme.colors.text.primary
                        }}
                    />
                </div>

                <button
                    onClick={runPhysics}
                    disabled={loading}
                    className="w-full text-xs py-1.5 rounded uppercase font-bold transition-colors"
                    style={{
                        backgroundColor: theme.colors.accent.primary,
                        color: theme.colors.bg.primary
                    }}
                >
                    {loading ? 'Solving...' : 'Compute'}
                </button>

                {result && expanded && (
                    <div className="mt-4 p-2 rounded border max-h-60 overflow-y-auto"
                        style={{
                            backgroundColor: theme.colors.bg.primary,
                            borderColor: theme.colors.border.secondary
                        }}
                    >
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-[10px]" style={{ color: theme.colors.status.success }}>RESULT</span>
                            <button onClick={() => setExpanded(false)} className="text-[10px] hover:text-white" style={{ color: theme.colors.text.muted }}>CLOSE</button>
                        </div>
                        <pre className="text-[10px] font-mono whitespace-pre-wrap" style={{ color: theme.colors.text.secondary }}>
                            {JSON.stringify(result, null, 2)}
                        </pre>
                    </div>
                )}
            </div>
        </div>
    );
};

export default SimulationBay;
