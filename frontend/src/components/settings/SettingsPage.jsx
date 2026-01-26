import React from 'react';
import { Settings, Palette, Code2, Terminal, Zap, Globe, Shield, Bell, Save, Command } from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';
import { useSettings } from '../../contexts/SettingsContext';

const SettingsPage = () => {
    const { theme, currentTheme, changeTheme } = useTheme();
    const {
        fontSize,
        setFontSize,
        autoSave,
        setAutoSave,
        formatOnSave,
        setFormatOnSave,
        notifications,
        setNotifications,
        physicsKernel,
        setPhysicsKernel,
        compilerOptimization,
        setCompilerOptimization,
        showTemperatureSensor,
        setShowTemperatureSensor,
        show3DThermometer,
        setShow3DThermometer,
        meshRenderingMode,
        setMeshRenderingMode,
        visualizationQuality,
        setVisualizationQuality,
        showGrid,
        setShowGrid,
        showControlsHelp,
        setShowControlsHelp,
        resetToDefaults
    } = useSettings();

    const settingsSections = [
        {
            title: 'Appearance',
            icon: Palette,
            settings: [
                {
                    label: 'Theme',
                    type: 'select',
                    value: currentTheme,
                    onChange: changeTheme,
                    options: [
                        { value: 'blue', label: 'Blue' },
                        { value: 'white-gold', label: 'White and Gold' },
                        { value: 'high-contrast', label: 'High Contrast' },
                        { value: 'black-gold', label: 'Black and Gold' }
                    ]
                },
                {
                    label: 'Editor Font Size',
                    type: 'select',
                    value: fontSize,
                    onChange: setFontSize,
                    options: [
                        { value: '10', label: '10px' },
                        { value: '12', label: '12px' },
                        { value: '14', label: '14px' },
                        { value: '16', label: '16px' }
                    ]
                }
            ]
        },
        {
            title: 'Editor',
            icon: Code2,
            settings: [
                {
                    label: 'Auto Save',
                    type: 'toggle',
                    value: autoSave,
                    onChange: setAutoSave,
                    description: 'Automatically save files after editing'
                },
                {
                    label: 'Format on Save',
                    type: 'toggle',
                    value: formatOnSave,
                    onChange: setFormatOnSave,
                    description: 'Automatically format code when saving'
                }
            ]
        },
        {
            title: 'Physics Kernel',
            icon: Globe,
            settings: [
                {
                    label: 'Default Environment',
                    type: 'select',
                    value: physicsKernel,
                    onChange: setPhysicsKernel,
                    options: [
                        { value: 'EARTH_AERO', label: 'Earth Aerodynamics' },
                        { value: 'EARTH_MARINE', label: 'Earth Marine' },
                        { value: 'ORBITAL_LEO', label: 'Low Earth Orbit' },
                        { value: 'ORBITAL_GEO', label: 'Geostationary Orbit' }
                    ]
                },
                {
                    label: 'Simulation Frequency',
                    type: 'select',
                    value: '1000',
                    onChange: () => { },
                    options: [
                        { value: '100', label: '100 Hz' },
                        { value: '500', label: '500 Hz' },
                        { value: '1000', label: '1000 Hz' },
                        { value: '2000', label: '2000 Hz' }
                    ]
                },
                {
                    label: 'Manual Temperature Input',
                    type: 'toggle',
                    value: showTemperatureSensor,
                    onChange: setShowTemperatureSensor,
                    description: 'Enables manual temperature override controls in simulation view.'
                },
                {
                    label: '3D Thermometer',
                    type: 'toggle',
                    value: show3DThermometer,
                    onChange: setShow3DThermometer,
                    description: 'Visualizes temperature with a 3D model in the scene.'
                }
            ]
        },
        {
            title: 'Rendering (Phase 8)',
            icon: Zap,
            settings: [
                {
                    label: 'Show Grid',
                    type: 'toggle',
                    value: showGrid || false,
                    onChange: setShowGrid,
                    description: 'Toggle visibility of the 3D reference grid'
                },
                {
                    label: 'Show Controls Help',
                    type: 'toggle',
                    value: showControlsHelp || false,
                    onChange: setShowControlsHelp,
                    description: 'Show overlay with navigation controls'
                },
                {
                    label: 'Mesh Rendering Mode',
                    type: 'select',
                    value: meshRenderingMode,
                    onChange: setMeshRenderingMode,
                    options: [
                        { value: 'sdf', label: 'SDF Mode (Boolean Ops)' },
                        { value: 'preview', label: 'Preview Mode (Fast)' }
                    ],
                    description: 'SDF mode enables boolean operations, Preview mode uses fast mesh rendering'
                },
                {
                    label: 'Visualization Quality (Bake Resolution)',
                    type: 'select',
                    value: visualizationQuality,
                    onChange: setVisualizationQuality,
                    options: [
                        { value: 'LOW', label: 'Low (32³ - Fast)' },
                        { value: 'MEDIUM', label: 'Medium (64³ - Balanced)' },
                        { value: 'HIGH', label: 'High (128³ - Detailed)' },
                        { value: 'ULTRA', label: 'Ultra (256³ - Precision)' }
                    ],
                    description: 'Controls the precision of SDF generation. Higher quality = longer bake time.'
                }
            ]
        },
        {
            title: 'Compiler',
            icon: Zap,
            settings: [
                {
                    label: 'Optimization Level',
                    type: 'select',
                    value: compilerOptimization,
                    onChange: setCompilerOptimization,
                    options: [
                        { value: 'debug', label: 'Debug (No optimization)' },
                        { value: 'balanced', label: 'Balanced' },
                        { value: 'performance', label: 'Performance' },
                        { value: 'size', label: 'Size' }
                    ]
                },
                {
                    label: 'Incremental Compilation',
                    type: 'toggle',
                    value: true,
                    onChange: () => { },
                    description: 'Only recompile changed modules'
                }
            ]
        },
        {
            title: 'Security',
            icon: Shield,
            settings: [
                {
                    label: 'Secure Boot',
                    type: 'toggle',
                    value: true,
                    onChange: () => { },
                    description: 'Verify kernel integrity on startup'
                },
                {
                    label: 'Agent Sandboxing',
                    type: 'toggle',
                    value: true,
                    onChange: () => { },
                    description: 'Run AI agents in isolated environment'
                }
            ]
        },
        {
            title: 'Notifications',
            icon: Bell,
            settings: [
                {
                    label: 'Enable Notifications',
                    type: 'toggle',
                    value: notifications,
                    onChange: setNotifications,
                    description: 'Show desktop notifications'
                },
                {
                    label: 'Agent Proposals',
                    type: 'toggle',
                    value: true,
                    onChange: () => { },
                    description: 'Notify when agents suggest changes'
                }
            ]
        }
    ];

    return (
        <div
            className="flex-1 h-full flex flex-col overflow-hidden"
            style={{ backgroundColor: theme.colors.bg.primary }}
        >
            <PanelHeader title="Kernel Settings" icon={Settings} />

            <div className="flex-1 overflow-y-auto p-6">
                <div className="max-w-4xl mx-auto space-y-8">
                    {/* Header */}
                    <div className="space-y-2">
                        <h1 className="text-2xl font-bold font-mono" style={{ color: theme.colors.accent.primary }}>
                            Settings
                        </h1>
                        <p className="text-sm" style={{ color: theme.colors.text.tertiary }}>
                            Configure BRICK IDE preferences and kernel parameters
                        </p>
                    </div>

                    {/* Settings Sections */}
                    {settingsSections.map((section, idx) => (
                        <div key={idx} className="space-y-4">
                            <div className="flex items-center gap-2 font-mono" style={{ color: theme.colors.text.secondary }}>
                                <section.icon size={16} style={{ color: theme.colors.accent.primary }} />
                                <h2 className="text-sm font-bold uppercase tracking-wider">{section.title}</h2>
                            </div>

                            <div
                                className="rounded-lg p-4 space-y-4"
                                style={{
                                    backgroundColor: theme.colors.bg.secondary,
                                    border: `1px solid ${theme.colors.border.primary}`
                                }}
                            >
                                {section.settings.map((setting, settingIdx) => (
                                    <div key={settingIdx} className="space-y-2">
                                        <div className="flex items-center justify-between">
                                            <div className="space-y-1">
                                                <label className="text-xs font-mono font-semibold" style={{ color: theme.colors.text.primary }}>
                                                    {setting.label}
                                                </label>
                                                {setting.description && (
                                                    <p className="text-[10px]" style={{ color: theme.colors.text.muted }}>
                                                        {setting.description}
                                                    </p>
                                                )}
                                            </div>

                                            {setting.type === 'toggle' ? (
                                                <button
                                                    onClick={() => setting.onChange(!setting.value)}
                                                    className="relative w-12 h-6 rounded-full transition-colors"
                                                    style={{
                                                        backgroundColor: setting.value ? theme.colors.accent.primary : theme.colors.bg.tertiary
                                                    }}
                                                >
                                                    <div
                                                        className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${setting.value ? 'translate-x-6' : 'translate-x-0'
                                                            }`}
                                                    />
                                                </button>
                                            ) : (
                                                <select
                                                    value={setting.value}
                                                    onChange={(e) => setting.onChange(e.target.value)}
                                                    className="rounded px-3 py-1.5 text-xs font-mono outline-none"
                                                    style={{
                                                        backgroundColor: theme.colors.bg.primary,
                                                        border: `1px solid ${theme.colors.border.secondary}`,
                                                        color: theme.colors.text.primary
                                                    }}
                                                >
                                                    {setting.options.map((opt) => (
                                                        <option key={opt.value} value={opt.value}>
                                                            {opt.label}
                                                        </option>
                                                    ))}
                                                </select>
                                            )}
                                        </div>

                                        {settingIdx < section.settings.length - 1 && (
                                            <div style={{ borderBottom: `1px solid ${theme.colors.border.primary}50` }} />
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}



                    {/* Command Reference Cheat Sheet */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 font-mono" style={{ color: theme.colors.text.secondary }}>
                            <Command size={16} style={{ color: theme.colors.accent.primary }} />
                            <h2 className="text-sm font-bold uppercase tracking-wider">Command Reference</h2>
                        </div>

                        <div className="rounded-lg border overflow-hidden" style={{ borderColor: theme.colors.border.primary, backgroundColor: theme.colors.bg.secondary }}>
                            {/* Shell */}
                            {/* Shell */}
                            <div className="p-4 border-b" style={{ borderColor: theme.colors.border.primary }}>
                                <div className="flex items-center gap-2 text-sm font-bold mb-3 uppercase tracking-wider" style={{ color: theme.colors.status.warning }}>
                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: theme.colors.status.warning }} /> Shell
                                </div>
                                <div className="space-y-2">
                                    {[
                                        { cmd: 'ls [-l]', desc: 'List directory contents (mass, material, lock).' },
                                        { cmd: 'cd <path>', desc: 'Change current working directory.' },
                                        { cmd: 'pwd', desc: 'Print working directory path.' },
                                        { cmd: 'find <name>', desc: 'Search for nodes by name recursively.' },
                                        { cmd: 'inspect <node>', desc: 'View properties (mass, material).' },
                                        { cmd: 'set <key> <val>', desc: 'Mutate a state parameter (requires lock).' },
                                        { cmd: 'lock / unlock', desc: 'Acquire/Release write lock.' },
                                        { cmd: 'mkdir <name>', desc: 'Create a new component node.' },
                                        { cmd: 'units [set <type>]', desc: 'View or set unit system.' },
                                    ].map((item, i) => (
                                        <div key={i} className="flex flex-col sm:flex-row sm:items-baseline gap-2 text-xs font-mono">
                                            <div className="sm:w-[180px] shrink-0 font-bold" style={{ color: theme.colors.status.warning }}>{item.cmd}</div>
                                            <div style={{ color: theme.colors.text.secondary }}>{item.desc}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* PVC */}
                            <div className="p-4 border-b" style={{ borderColor: theme.colors.border.primary }}>
                                <div className="flex items-center gap-2 text-sm font-bold mb-3 uppercase tracking-wider" style={{ color: theme.colors.status.info }}>
                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: theme.colors.status.info }} /> PVC (Physical Version Control)
                                </div>
                                <div className="space-y-2">
                                    {[
                                        { cmd: 'brick status', desc: 'Show modified files and staging area.' },
                                        { cmd: 'brick diff', desc: 'Compare Working Tree vs HEAD.' },
                                        { cmd: 'brick commit -m <msg>', desc: 'Record changes to repository.' },
                                        { cmd: 'brick branch <name>', desc: 'Create branch operations.' },
                                        { cmd: 'brick checkout <ref>', desc: 'Switch branches/restore files.' },
                                        { cmd: 'brick merge <branch>', desc: 'Merge with physics conflict resolution.' },
                                        { cmd: 'brick blame <key>', desc: 'Show provenance for parameter mutation.' },
                                    ].map((item, i) => (
                                        <div key={i} className="flex flex-col sm:flex-row sm:items-baseline gap-2 text-xs font-mono">
                                            <div className="sm:w-[180px] shrink-0 font-bold" style={{ color: theme.colors.status.info }}>{item.cmd}</div>
                                            <div style={{ color: theme.colors.text.secondary }}>{item.desc}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* vHIL */}
                            <div className="p-4 border-b" style={{ borderColor: theme.colors.border.primary }}>
                                <div className="flex items-center gap-2 text-sm font-bold mb-3 uppercase tracking-wider" style={{ color: theme.colors.status.success }}>
                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: theme.colors.status.success }} /> vHIL (Runtime Kernel)
                                </div>
                                <div className="space-y-2">
                                    {[
                                        { cmd: 'run vhil', desc: 'Boot Physics Kernel.' },
                                        { cmd: 'halt / reboot', desc: 'Stop/Restart virtual machine.' },
                                        { cmd: 'reload <p>=<v>', desc: 'Hot-swap control parameters.' },
                                        { cmd: 'telemetry', desc: 'Stream real-time sensor data.' },
                                        { cmd: 'tune <p> <v>', desc: 'Live parameter tuning.' },
                                        { cmd: 'inject <fault>', desc: 'Simulate hardware failures.' },
                                        { cmd: 'probe <signal>', desc: 'Oscilloscope monitoring of signals.' },
                                    ].map((item, i) => (
                                        <div key={i} className="flex flex-col sm:flex-row sm:items-baseline gap-2 text-xs font-mono">
                                            <div className="sm:w-[180px] shrink-0 font-bold" style={{ color: theme.colors.status.success }}>{item.cmd}</div>
                                            <div style={{ color: theme.colors.text.secondary }}>{item.desc}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* 3D Navigation */}
                            <div className="p-4 border-b" style={{ borderColor: theme.colors.border.primary }}>
                                <div className="flex items-center gap-2 text-sm font-bold mb-3 uppercase tracking-wider" style={{ color: theme.colors.accent.primary }}>
                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: theme.colors.accent.primary }} /> 3D Navigation
                                </div>
                                <div className="space-y-2">
                                    {[
                                        { cmd: 'LMB (Drag)', desc: 'Rotate / Orbit around object.' },
                                        { cmd: 'RMB (Drag)', desc: 'Pan camera view.' },
                                        { cmd: 'Scroll', desc: 'Zoom in / out.' },
                                        { cmd: 'Shift + Drag', desc: 'Alternative Pan (without RMB).' },
                                        { cmd: 'Double Click', desc: 'Focus on point.' },
                                    ].map((item, i) => (
                                        <div key={i} className="flex flex-col sm:flex-row sm:items-baseline gap-2 text-xs font-mono">
                                            <div className="sm:w-[180px] shrink-0 font-bold" style={{ color: theme.colors.accent.primary }}>{item.cmd}</div>
                                            <div style={{ color: theme.colors.text.secondary }}>{item.desc}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* System - Mapped to Accent Secondary as 'functional' color */}
                            <div className="p-4">
                                <div className="flex items-center gap-2 text-sm font-bold mb-3 uppercase tracking-wider" style={{ color: theme.colors.accent.secondary }}>
                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: theme.colors.accent.secondary }} /> System & Build
                                </div>
                                <div className="space-y-2">
                                    {[
                                        { cmd: 'doctor', desc: 'Run system-wide health checks.' },
                                        { cmd: 'brick compile', desc: 'Synthesize geometry.' },
                                        { cmd: 'brick mfg export', desc: 'Export CAD/CAM data (STL, STEP).' },
                                        { cmd: 'brick optimize', desc: 'Run genetic algorithms.' },
                                        { cmd: 'brick verify', desc: 'Formal strict verification.' },
                                        { cmd: 'brick audit', desc: 'Cryptographic state validation.' },
                                    ].map((item, i) => (
                                        <div key={i} className="flex flex-col sm:flex-row sm:items-baseline gap-2 text-xs font-mono">
                                            <div className="sm:w-[180px] shrink-0 font-bold" style={{ color: theme.colors.accent.secondary }}>{item.cmd}</div>
                                            <div style={{ color: theme.colors.text.secondary }}>{item.desc}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="flex justify-end gap-3 pt-4" style={{ borderTop: `1px solid ${theme.colors.border.primary}` }}>
                        <button
                            onClick={resetToDefaults}
                            className="px-4 py-2 rounded text-xs font-mono transition-colors"
                            style={{
                                backgroundColor: theme.colors.bg.tertiary,
                                color: theme.colors.text.secondary
                            }}
                        >
                            Reset to Defaults
                        </button>
                        <button
                            className="px-4 py-2 rounded text-xs font-mono font-bold transition-colors flex items-center gap-2"
                            style={{
                                background: `linear-gradient(to right, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`,
                                color: theme.colors.bg.primary
                            }}
                        >
                            <Save size={14} />
                            Save Settings
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SettingsPage;
