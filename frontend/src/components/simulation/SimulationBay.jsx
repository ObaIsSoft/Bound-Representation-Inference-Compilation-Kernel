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
    const { isRunning, setIsRunning, paused, setPaused, setTestScenario, runPhysicsAnalysis } = useSimulation();
    const [viewMode, setViewMode] = useState('wireframe');
    const [isExploded, setIsExploded] = useState(false);
    const [viewportMenuOpen, setViewportMenuOpen] = useState(false);
    const [showViewMenu, setShowViewMenu] = useState(false);
    const viewportMenuRef = useRef(null);

    // Phase 10: Trigger Physics Analysis on Mode Switch
    useEffect(() => {
        if (viewMode === 'thermal' || viewMode === 'stress') {
            if (activeTab) {
                runPhysicsAnalysis(activeTab);
            }
        }
    }, [viewMode, activeTab, runPhysicsAnalysis]);

    // 3D Thermometer Logic
    const isUntitled = activeTab?.name?.startsWith('Untitled Design');
    const showTempControl = showTemperatureSensor && isUntitled;

    // Physics State (Lifted for Visualization)
    const [physicsResult, setPhysicsResult] = useState(null);
    const { physState, motionTrail, metrics } = useSimulation();

    // Combine live simulation data with static analysis results
    const combinedPhysicsData = useMemo(() => {
        return {
            state: physState,
            motionTrail: motionTrail,
            metrics: metrics,
            ...(physicsResult || {}) // Merge static analysis (structural, etc)
        };
    }, [physState, motionTrail, metrics, physicsResult]);

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
                        physicsData={combinedPhysicsData}
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
                            <div className="flex flex-col">
                                <div className="text-[5px] font-mono uppercase leading-tight" style={{ color: theme.colors.text.muted }}>Mass</div>
                                <div className="text-[10px] font-mono font-bold leading-tight" style={{ color: theme.colors.text.primary }}>
                                    {combinedPhysicsData.metrics?.mass ? combinedPhysicsData.metrics.mass.toFixed(1) : '-'} kg
                                </div>
                            </div>
                            <div className="w-px h-4" style={{ backgroundColor: theme.colors.border.secondary }} />
                            <div className="flex flex-col">
                                <div className="text-[5px] font-mono uppercase leading-tight" style={{ color: theme.colors.text.muted }}>Stress</div>
                                <div className="text-[10px] font-mono font-bold leading-tight" style={{ color: theme.colors.text.primary }}>
                                    {combinedPhysicsData.metrics?.max_stress ? (combinedPhysicsData.metrics.max_stress / 1e6).toFixed(1) : '0.0'} MPa
                                </div>
                            </div>
                            <div className="w-px h-4" style={{ backgroundColor: theme.colors.border.secondary }} />
                            <div className="flex flex-col">
                                <div className="text-[5px] font-mono uppercase leading-tight" style={{ color: theme.colors.text.muted }}>Drag</div>
                                <div className="text-[10px] font-mono font-bold leading-tight" style={{ color: theme.colors.text.primary }}>
                                    {combinedPhysicsData.metrics?.drag_force ? combinedPhysicsData.metrics.drag_force.toFixed(1) : '0.0'} N
                                </div>
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
            <PhysicsOverlay theme={theme} active={true} onResult={setPhysicsResult} activeTab={activeTab} />
        </main>
    );
};

const PhysicsOverlay = ({ theme, active, onResult, activeTab }) => {
    const [mode, setMode] = useState('VALIDATE');
    const [domain, setDomain] = useState('NUCLEAR');
    const [query, setQuery] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [expanded, setExpanded] = useState(false);

    // State for Dynamic Validation Context
    const [valContext, setValContext] = useState({ material: 'Generic', load: '0 N' });

    const runValidation = React.useCallback(async (overrideContext) => {
        const ctx = overrideContext || valContext;
        // Don't validate if no material/geometry
        if (!ctx.material || ctx.material === 'Unknown') return;

        setLoading(true);
        try {
            // Re-parse for the API call to ensure latest state
            let geometry = { type: 'box', dims: { length: 1, width: 1, height: 1 } };
            let material = ctx.material;
            // Default load if none provided
            let loads = { force: 1000 };

            if (activeTab?.content) {
                try {
                    const asset = typeof activeTab.content === 'string' ? JSON.parse(activeTab.content) : activeTab.content;
                    if (asset.type === 'primitive') {
                        if (asset.geometry === 'box') {
                            geometry = {
                                type: 'box',
                                dims: {
                                    length: Number(asset.args?.[0] || 1),
                                    width: Number(asset.args?.[1] || 1),
                                    height: Number(asset.args?.[2] || 1)
                                }
                            };
                        } else if (asset.geometry === 'cylinder') {
                            geometry = {
                                type: 'cylinder',
                                dims: {
                                    radius: Number(asset.args?.[0] || 0.5),
                                    height: Number(asset.args?.[1] || 1)
                                }
                            };
                        }
                    }
                } catch (e) { }
            }

            const res = await fetch('http://localhost:8000/api/physics/validate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ geometry, material, loads })
            });
            const data = await res.json();
            setResult(data);
            if (onResult && data.valid) onResult({ ...data, type: 'validation' });
            // Auto-expand only if there are warnings or errors to be less intrusive
            if (data.warnings?.length > 0 || !data.valid) setExpanded(true);
        } catch (e) {
            setResult({ error: e.message });
            setExpanded(true);
        }
        setLoading(false);
    }, [activeTab, valContext, onResult]);

    // Update context AND trigger validation when activeTab changes
    useEffect(() => {
        if (activeTab?.content) {
            try {
                const asset = typeof activeTab.content === 'string' ? JSON.parse(activeTab.content) : activeTab.content;
                const mat = asset.material ? (typeof asset.material === 'string' ? asset.material : asset.material.name) : 'Generic';
                const load = asset.loads ? 'Custom' : 'Self-weight';

                const newContext = { material: mat, load };
                setValContext(newContext);

                // Auto-trigger validation if in VALIDATE mode
                if (mode === 'VALIDATE') {
                    // Small timeout to allow state to settle/render
                    const timer = setTimeout(() => {
                        runValidation(newContext);
                    }, 500);
                    return () => clearTimeout(timer);
                }
            } catch (e) {
                setValContext({ material: 'Unknown', load: '-' });
            }
        }
    }, [activeTab, mode]); // runValidation is stable or re-created with activeTab, so we omit it to avoid loops or use it carefully.
    // Actually, we should refactor runValidation to NOT depend on valContext if we pass it in.
    // I updated runValidation to accept overrideContext.

    if (!active) return null;

    return (
        <div className="absolute bottom-12 left-4 z-50 w-72 border rounded-lg shadow-xl backdrop-blur-md transition-all duration-300 overflow-hidden"
            style={{
                backgroundColor: theme.colors.bg.secondary + 'E6', // High opacity
                borderColor: theme.colors.border.primary
            }}
        >
            {/* Header / Mode Switch - Compact */}
            <div className="flex items-center justify-between p-2 border-b" style={{ borderColor: theme.colors.border.secondary }}>
                <div className="flex gap-2 bg-black/20 p-0.5 rounded-md">
                    {['VALIDATE'].map(m => (
                        <button
                            key={m}
                            onClick={() => { setMode(m); setResult(null); }}
                            className="text-[9px] px-2 py-0.5 rounded font-bold transition-all"
                            style={{
                                backgroundColor: mode === m ? theme.colors.bg.primary : 'transparent',
                                color: mode === m ? theme.colors.accent.primary : theme.colors.text.muted,
                                boxShadow: mode === m ? '0 1px 2px rgba(0,0,0,0.1)' : 'none'
                            }}
                        >
                            {m}
                        </button>
                    ))}
                </div>

                {loading && <Activity size={12} className="animate-spin" style={{ color: theme.colors.accent.primary }} />}
            </div>

            <div className="p-2 space-y-2">
                {mode === 'VALIDATE' && (
                    <div>
                        <div className="grid grid-cols-2 gap-2 text-[9px] mb-2 font-mono">
                            <div className="flex flex-col">
                                <span style={{ color: theme.colors.text.muted }}>Material</span>
                                <span className="truncate font-bold" style={{ color: theme.colors.text.primary }} title={valContext.material}>
                                    {valContext.material}
                                </span>
                            </div>
                            <div className="flex flex-col text-right">
                                <span style={{ color: theme.colors.text.muted }}>Est. Load</span>
                                <span style={{ color: theme.colors.text.primary }}>{valContext.load}</span>
                            </div>
                        </div>
                        <button
                            onClick={runValidation}
                            disabled={loading}
                            className="w-full py-1 rounded text-[9px] font-bold uppercase tracking-wider flex items-center justify-center gap-1 hover:brightness-110"
                            style={{
                                backgroundColor: theme.colors.accent.primary,
                                color: theme.colors.bg.primary
                            }}
                        >
                            {loading ? 'Analyzing...' : 'Run Analysis'}
                        </button>
                    </div>
                )}

                {/* Inline Result Display - Compact */}
                {result && expanded && (
                    <div className="mt-2 pt-2 border-t text-[9px]" style={{ borderColor: theme.colors.border.secondary }}>
                        <div className="flex justify-between items-center mb-1">
                            <span className="font-bold uppercase" style={{ color: theme.colors.status.success }}>Results</span>
                            <button onClick={() => setExpanded(false)} style={{ color: theme.colors.text.muted }}>✕</button>
                        </div>
                        {result.metrics ? (
                            <div className="grid grid-cols-2 gap-x-2 gap-y-1 font-mono">
                                {Object.entries(result.metrics).map(([k, v]) => (
                                    <React.Fragment key={k}>
                                        <span style={{ color: theme.colors.text.secondary }}>{k.split('_')[0]}</span>
                                        <span className="text-right" style={{ color: theme.colors.text.primary }}>{v}</span>
                                    </React.Fragment>
                                ))}
                            </div>
                        ) : (
                            <div className="font-mono opacity-80 max-h-20 overflow-y-auto">
                                {result.answer || JSON.stringify(result)}
                            </div>
                        )}
                        {result.warnings?.length > 0 && (
                            <div className="mt-1 text-red-400 font-bold">
                                ⚠ {result.warnings.length} Warning(s)
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default SimulationBay;
