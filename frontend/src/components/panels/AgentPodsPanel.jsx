import React, { useState, useEffect } from 'react';
import {
    Brain, Zap, Factory, Terminal, Activity, Cpu, Database, Network,
    Shield, TrendingUp, DollarSign, AlertTriangle, ChevronDown, ChevronRight, Power,
    Globe, Box, Scale, Layers, CheckCircle, PenTool, Compass, Star, LayoutTemplate, Download,
    Thermometer, Navigation, ShieldAlert, Mountain, Scissors, Ruler, Beaker, FlaskConical,
    Triangle, Droplet, Map, CircuitBoard, Shuffle, GitBranch, Stethoscope, MonitorPlay,
    FileText, ClipboardCheck, Eye, BookOpen, Settings, MessageSquare, Wifi, GraduationCap,
    Lock, Plus
} from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';
import { useSimulation } from '../../contexts/SimulationContext';

// ... (Icons exist)

const AGENT_REGISTRY = [
    // --- Core Agents ---
    { id: 'environment', name: 'Environment Agent', role: 'Context & Regime', category: 'Core', icon: Globe, color: '#3b82f6' },
    { id: 'geometry', name: 'Geometry Agent', role: 'KCL Generation', category: 'Core', icon: Box, color: '#8b5cf6' },
    { id: 'surrogate_physics', name: 'Surrogate Physics', role: 'Fast Prediction', category: 'Core', icon: Zap, color: '#eab308' },
    { id: 'mass_properties', name: 'Mass Properties', role: 'Inertia & CoG', category: 'Core', icon: Scale, color: '#f59e0b' },
    { id: 'manifold', name: 'Manifold Agent', role: 'Watertight Check', category: 'Core', icon: Layers, color: '#10b981' },
    { id: 'validator', name: 'Validator Agent', role: 'Constraint Check', category: 'Core', icon: CheckCircle, color: '#ef4444' },

    // --- Design & Planning ---
    { id: 'designer', name: 'Designer Agent', role: 'Aesthetics & Style', category: 'Design', icon: PenTool, color: '#ec4899' },
    { id: 'design_exploration', name: 'Design Exploration', role: 'Parametric Search', category: 'Design', icon: Compass, color: '#f97316' },
    { id: 'design_quality', name: 'Design Quality', role: 'Fidelity Enhancer', category: 'Design', icon: Star, color: '#d946ef' },
    { id: 'template_design', name: 'Template Agent', role: 'Pattern Library', category: 'Design', icon: LayoutTemplate, color: '#6366f1' },
    { id: 'asset_sourcing', name: 'Asset Sourcing', role: 'Catalog Search', category: 'Design', icon: Download, color: '#06b6d4' },

    // --- Analysis ---
    { id: 'thermal', name: 'Thermal Agent', role: 'Heat Analysis', category: 'Analysis', icon: Thermometer, color: '#f43f5e' },
    { id: 'dfm', name: 'DfM Agent', role: 'Manufacturability', category: 'Analysis', icon: Factory, color: '#84cc16' },
    { id: 'cps', name: 'CPS Agent', role: 'Control Systems', category: 'Analysis', icon: Cpu, color: '#3b82f6' },
    { id: 'gnc', name: 'GNC Agent', role: 'Guidance & Nav', category: 'Analysis', icon: Navigation, color: '#a855f7' },
    { id: 'mitigation', name: 'Mitigation Agent', role: 'Fix Proposer', category: 'Analysis', icon: ShieldAlert, color: '#22c55e' },
    { id: 'topological', name: 'Topological Agent', role: 'Terrain & Mode', category: 'Analysis', icon: Mountain, color: '#78716c' },

    // --- Manufacturing ---
    { id: 'manufacturing', name: 'Manufacturing Agent', role: 'BOM & Cost', category: 'Manufacturing', icon: Factory, color: '#10b981' },
    { id: 'slicer', name: 'Slicer Agent', role: 'G-Code Gen', category: 'Manufacturing', icon: Scissors, color: '#f59e0b' },
    { id: 'tolerance', name: 'Tolerance Agent', role: 'Fit Analysis', category: 'Manufacturing', icon: Ruler, color: '#64748b' },

    // --- Material ---
    { id: 'material', name: 'Material Agent', role: 'Properties', category: 'Material', icon: Beaker, color: '#14b8a6' },
    { id: 'chemistry', name: 'Chemistry Agent', role: 'Composition', category: 'Material', icon: FlaskConical, color: '#0ea5e9' },

    // --- Structural/Arch ---
    { id: 'structural_load', name: 'Structural Load', role: 'Static Analysis', category: 'Structural', icon: Triangle, color: '#f97316' },
    { id: 'mep', name: 'MEP Agent', role: 'Systems Routing', category: 'Structural', icon: Droplet, color: '#3b82f6' },
    { id: 'zoning', name: 'Zoning Agent', role: 'Compliance', category: 'Structural', icon: Map, color: '#8b5cf6' },

    // --- Electronics ---
    { id: 'electronics', name: 'Electronics Agent', role: 'Power & PCB', category: 'Electronics', icon: CircuitBoard, color: '#ef4444' },

    // --- Advanced ---
    { id: 'multi_mode', name: 'Multi-Mode Agent', role: 'Transition Logic', category: 'Advanced', icon: Shuffle, color: '#ec4899' },
    { id: 'nexus', name: 'Nexus Agent', role: 'Tree Nav', category: 'Advanced', icon: Network, color: '#06b6d4' },
    { id: 'pvc', name: 'PVC Agent', role: 'Version Control', category: 'Advanced', icon: GitBranch, color: '#f59e0b' },
    { id: 'doctor', name: 'Doctor Agent', role: 'Diagnostics', category: 'Advanced', icon: Stethoscope, color: '#22c55e' },
    { id: 'shell', name: 'Shell Agent', role: 'CLI Exec', category: 'Advanced', icon: Terminal, color: '#64748b' },
    { id: 'vhil', name: 'VHIL Agent', role: 'Hardware Sim', category: 'Advanced', icon: MonitorPlay, color: '#8b5cf6' },

    // --- Documentation ---
    { id: 'documentation', name: 'Doc Agent', role: 'Reporting', category: 'Docs & QA', icon: FileText, color: '#6b7280' },
    { id: 'diagnostic', name: 'Diagnostic Agent', role: 'Root Cause', category: 'Docs & QA', icon: Activity, color: '#ef4444' },
    { id: 'verification', name: 'Verification Agent', role: 'Testing', category: 'Docs & QA', icon: ClipboardCheck, color: '#10b981' },
    { id: 'visual_validator', name: 'Visual Validator', role: 'Aesthetic Check', category: 'Docs & QA', icon: Eye, color: '#ec4899' },

    // --- Specialized ---
    { id: 'standards', name: 'Standards Agent', role: 'ISO/ANSI', category: 'Specialized', icon: BookOpen, color: '#1d4ed8' },
    { id: 'component', name: 'Component Agent', role: 'Parts Selection', category: 'Specialized', icon: Settings, color: '#db2777' },
    { id: 'conversational', name: 'Conversational', role: 'NLP Interface', category: 'Specialized', icon: MessageSquare, color: '#8b5cf6' },
    { id: 'remote', name: 'Remote Agent', role: 'Collaboration', category: 'Specialized', icon: Wifi, color: '#0ea5e9' },

    // --- Training ---
    { id: 'physics_trainer', name: 'Physics Trainer', role: 'Model Learning', category: 'Training', icon: GraduationCap, color: '#f59e0b' },
];

// Helper to group agents
const groupAgentsByCategory = (agents) => {
    return agents.reduce((acc, agent) => {
        const cat = agent.category || 'Other';
        if (!acc[cat]) acc[cat] = [];
        acc[cat].push(agent);
        return acc;
    }, {});
};

// Helper to get color based on category and theme
const getCategoryColor = (category, theme) => {
    switch (category) {
        case 'Core': return theme.colors.status.info;
        case 'Design': return theme.colors.accent.primary;
        case 'Analysis': return theme.colors.status.warning;
        case 'Manufacturing': return theme.colors.status.success;
        case 'Material': return theme.colors.status.info;
        case 'Structural': return theme.colors.accent.secondary;
        case 'Electronics': return theme.colors.status.error;
        case 'Advanced': return theme.colors.accent.primary;
        case 'Docs & QA': return theme.colors.text.muted;
        case 'Specialized': return theme.colors.status.info;
        case 'Training': return theme.colors.accent.secondary;
        default: return theme.colors.text.primary;
    }
};

const AgentRow = ({ agent, isActive, onToggle, runStatus, logs, computedStats, isEssential }) => {
    const { theme } = useTheme();
    const [isExpanded, setIsExpanded] = useState(false);

    const agentColor = getCategoryColor(agent.category, theme);

    return (
        <div
            className="flex flex-col rounded border transition-all duration-200"
        // ... (same styling)
        >
            <div
                className="flex items-center p-3 gap-3 cursor-pointer hover:bg-white/5 transition-colors group"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                {/* ... (Chevron & Icon) */}
                <button
                    className="transition-colors p-1 rounded hover:bg-white/10"
                    style={{ color: theme.colors.text.tertiary }}
                    onClick={(e) => {
                        e.stopPropagation();
                        setIsExpanded(!isExpanded);
                    }}
                >
                    {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                </button>

                <div className="p-1.5 rounded-md relative" style={{
                    backgroundColor: isActive ? agentColor + '15' : theme.colors.bg.tertiary,
                    color: isActive ? agentColor : theme.colors.text.muted
                }}>
                    <agent.icon size={16} />
                    {isEssential && (
                        <div className="absolute -top-1 -right-1 bg-gray-800 rounded-full p-0.5 border border-gray-600">
                            <Lock size={8} className="text-yellow-500" />
                        </div>
                    )}
                </div>

                {/* ... (Label & Stats) */}
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                        <h3 className="text-xs font-bold font-mono truncate" style={{ color: theme.colors.text.primary }}>
                            {agent.name}
                        </h3>
                        {isActive && <div className="w-1 h-1 rounded-full animate-pulse" style={{ backgroundColor: agentColor }} />}
                    </div>
                </div>

                {isActive && !isExpanded && (
                    <div className="flex gap-2 text-[9px] font-mono opacity-60" style={{ color: theme.colors.text.secondary }}>
                        <span>LOAD:{computedStats?.cpu ? computedStats.cpu.toFixed(1) : 0}%</span>
                        <span>RUNS:{computedStats?.runs || 0}</span>
                    </div>
                )}

                {/* Toggle Button (Disabled if essential) */}
                {isEssential ? (
                    <div className="p-1.5 opacity-50 cursor-not-allowed" title="Essential Agent">
                        <Lock size={12} style={{ color: theme.colors.text.muted }} />
                    </div>
                ) : (
                    <button
                        onClick={(e) => { e.stopPropagation(); onToggle(); }}
                        className={`p-1.5 rounded-full transition-all opacity-0 group-hover:opacity-100 ${isActive ? 'opacity-100 bg-green-500/10 text-green-500' : 'bg-white/5 hover:bg-white/10'}`}
                        style={!isActive ? { color: theme.colors.text.muted } : {}}
                    >
                        <Power size={12} />
                    </button>
                )}
            </div>

            {/* Same Expanded View ... */}
            {isExpanded && (
                <div className="border-t p-3 pl-10" style={{ borderColor: theme.colors.border.primary, backgroundColor: theme.colors.bg.tertiary }}>
                    <div className="flex items-center gap-4 mb-2 text-[10px] font-mono" style={{ color: theme.colors.text.muted }}>
                        <span>ROLE: {agent.role}</span>
                        <span>LOAD: {computedStats?.cpu ? computedStats.cpu.toFixed(2) : 0}%</span>
                        <span>RUNS: {computedStats?.runs || 0}</span>
                        <span>LAST: {computedStats?.last_active || "Never"}</span>
                        {runStatus && <span style={{ color: runStatus === 'success' ? theme.colors.status.success : theme.colors.status.warning }}>[{runStatus.toUpperCase()}]</span>}
                    </div>
                    {/* ... output logs */}
                </div>
            )}
        </div>
    );
};

const CategorySection = ({ title, agents, activeAgents, agentStates, toggleAgent, realMetrics, essentials }) => {
    // ... (same open logic)
    const { theme } = useTheme();
    const [isOpen, setIsOpen] = useState(true);

    return (
        <div className="mb-4">
            {/* ... header */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center gap-2 px-2 py-1 mb-2 text-xs font-bold uppercase tracking-wider hover:text-white transition-colors"
                style={{ color: theme.colors.text.tertiary }}
            >
                {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                {title} ({agents.length})
            </button>

            {isOpen && (
                <div className="flex flex-col gap-1 pl-2 border-l" style={{ borderColor: theme.colors.border.secondary }}>
                    {agents.map(agent => {
                        // Use REAL Metrics if available, else 0
                        const metric = realMetrics[agent.id] || { cpu: 0, runs: 0, last_active: null };

                        return (
                            <AgentRow
                                key={agent.id}
                                agent={agent}
                                isActive={activeAgents.includes(agent.id)}
                                onToggle={() => toggleAgent(agent.id)}
                                runStatus={agentStates[agent.id]?.status}
                                logs={agentStates[agent.id]?.logs || agent.logs}
                                computedStats={metric}
                                isEssential={essentials.includes(agent.id)}
                            />
                        );
                    })}
                </div>
            )}
        </div>
    );
};

const AgentPodsPanel = ({ width }) => {
    const { theme } = useTheme();

    // Core Data
    const [activeAgents, setActiveAgents] = useState([]);
    const [agentStates, setAgentStates] = useState({});
    const [realMetrics, setRealMetrics] = useState({}); // Stores /api/agents/metrics
    const [essentials, setEssentials] = useState([]);

    // Profile Data
    const [profiles, setProfiles] = useState([]);
    const [currentProfile, setCurrentProfile] = useState(null);
    const [isCreatingProfile, setIsCreatingProfile] = useState(false);

    const groupedAgents = groupAgentsByCategory(AGENT_REGISTRY);

    // Initial Fetch
    useEffect(() => {
        const fetchSystemData = async () => {
            // 1. Essentials
            try {
                const res = await fetch('http://localhost:8000/api/system/profiles/essentials');
                const data = await res.json();
                if (data.essentials) {
                    setEssentials(data.essentials);
                    // Ensure essentials are active initially
                    setActiveAgents(prev => [...new Set([...prev, ...data.essentials])]);
                }
            } catch (e) { console.error(e); }

            // 2. Profiles
            try {
                const res = await fetch('http://localhost:8000/api/system/profiles');
                const data = await res.json();
                if (data.profiles) setProfiles(data.profiles);
            } catch (e) { console.error(e); }
        };
        fetchSystemData();
    }, []);

    // Metric Polling
    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const res = await fetch('http://localhost:8000/api/agents/metrics');
                const data = await res.json();
                if (data.metrics) setRealMetrics(data.metrics);
            } catch (e) { }
        };
        fetchMetrics();
        const interval = setInterval(fetchMetrics, 2000); // 2s polling
        return () => clearInterval(interval);
    }, []);


    const applyProfile = async (profileId) => {
        try {
            const res = await fetch(`http://localhost:8000/api/system/profiles/${profileId}`);
            const data = await res.json();
            if (data.active_agents) {
                // Ensure essentials are merged in case the profile is old
                const merged = [...new Set([...data.active_agents, ...essentials])];
                setActiveAgents(merged);
                setCurrentProfile(data.name);
            }
        } catch (e) { console.error(e); }
    };

    const handleCreateProfile = async () => {
        const name = prompt("Enter profile name:");
        if (!name) return;

        try {
            const res = await fetch('http://localhost:8000/api/system/profiles/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, agents: activeAgents })
            });
            const data = await res.json();
            if (data.success) {
                // Refresh list
                const lRes = await fetch('http://localhost:8000/api/system/profiles');
                const lData = await lRes.json();
                setProfiles(lData.profiles || []);
                alert("Profile created!");
                setCurrentProfile(name);
            }
        } catch (e) { alert("Failed: " + e.message); }
    };

    // ... (runAgent same as before)
    const runAgent = async (id) => {
        // ... (same implementation)
    };

    const toggleAgent = (id) => {
        if (essentials.includes(id)) return; // Lock check

        const isActive = activeAgents.includes(id);
        if (isActive) {
            setActiveAgents(prev => prev.filter(a => a !== id));
        } else {
            setActiveAgents(prev => [...prev, id]);
            // runAgent(id); // Optional auto-run
        }
    };

    return (
        <div className="flex flex-col min-w-0 h-full border-r relative" style={{ width, borderColor: theme.colors.border.primary, backgroundColor: theme.colors.bg.primary }}>
            <PanelHeader title="Agent Pods" icon={Network} />

            {/* Profile Selector */}
            <div className="p-2 border-b" style={{ borderColor: theme.colors.border.primary }}>
                <div className="flex items-center justify-between text-[10px] mb-2 opacity-70">
                    <span className="font-mono">SYSTEM PROFILE</span>
                    <button
                        onClick={handleCreateProfile}
                        className="flex items-center gap-1 hover:text-white transition-colors"
                        title="Save Current agents as Profile"
                    >
                        <Plus size={10} /> CREATE
                    </button>
                </div>
                <div className="flex flex-wrap gap-1">
                    {profiles.map(p => (
                        <button
                            key={p.id}
                            onClick={() => applyProfile(p.id)}
                            className="px-2 py-1 rounded text-[9px] font-bold uppercase transition-all border flex items-center gap-1"
                            style={{
                                borderColor: theme.colors.border.secondary,
                                backgroundColor: currentProfile === p.name ? theme.colors.accent.primary : 'transparent',
                                color: currentProfile === p.name ? '#000' : theme.colors.text.secondary
                            }}
                        >
                            {p.name}
                        </button>
                    ))}
                </div>
            </div>

            <div className="flex-1 p-3 overflow-y-auto scrollbar-thin">
                {Object.keys(groupedAgents).map(category => (
                    <CategorySection
                        key={category}
                        title={category}
                        agents={groupedAgents[category]}
                        activeAgents={activeAgents}
                        agentStates={agentStates}
                        toggleAgent={toggleAgent}
                        realMetrics={realMetrics}
                        essentials={essentials}
                    />
                ))}
            </div>

            {/* Footer metrics now aggregations of real metrics */}
            <div className="h-8 border-t flex items-center px-4 gap-4 text-[10px] font-mono shrink-0" style={{ borderColor: theme.colors.border.primary, backgroundColor: theme.colors.bg.secondary, color: theme.colors.text.tertiary }}>
                {/* ... Same footer structure but maybe sum up realMetrics values? */}
                {/* For now keeping static or simple sum */}
                <div className="flex items-center gap-2">
                    <Activity size={12} className={activeAgents.length > 0 ? "animate-pulse" : ""} style={{ color: activeAgents.length > 0 ? theme.colors.status.success : theme.colors.status.error }} />
                    <span className="hidden sm:inline">{activeAgents.length} ACTIVE</span>
                </div>
            </div>
        </div>
    );
};

export default AgentPodsPanel;

