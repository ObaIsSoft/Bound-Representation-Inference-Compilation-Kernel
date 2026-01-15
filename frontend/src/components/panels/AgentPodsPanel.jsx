import React, { useState, useEffect } from 'react';
import {
    Brain, Zap, Factory, Terminal, Activity, Cpu, Database, Network,
    Shield, TrendingUp, DollarSign, AlertTriangle, ChevronDown, ChevronRight, Power,
    Globe, Box, Scale, Layers, CheckCircle, PenTool, Compass, Star, LayoutTemplate, Download,
    Thermometer, Navigation, ShieldAlert, Mountain, Scissors, Ruler, Beaker, FlaskConical,
    Triangle, Droplet, Map, CircuitBoard, Shuffle, GitBranch, Stethoscope, MonitorPlay,
    FileText, ClipboardCheck, Eye, BookOpen, Settings, MessageSquare, Wifi, GraduationCap
} from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';

const AGENT_REGISTRY = [
    // --- Core Agents ---
    { id: 'environment', name: 'Environment Agent', role: 'Context & Regime', category: 'Core', icon: Globe, color: '#3b82f6', stats: { cpu: 12, memory: 40 }, logs: ["Detecting regime: AERO", "Gravity set to 9.81 m/s²"] },
    { id: 'geometry', name: 'Geometry Agent', role: 'KCL Generation', category: 'Core', icon: Box, color: '#8b5cf6', stats: { cpu: 65, memory: 120 }, logs: ["Compiling B-Rep...", "Generating GLTF mesh"] },
    { id: 'surrogate_physics', name: 'Surrogate Physics', role: 'Fast Prediction', category: 'Core', icon: Zap, color: '#eab308', stats: { cpu: 45, memory: 60 }, logs: ["Estimating drag coefficient...", "Lift/Drag ratio: 14.2"] },
    { id: 'mass_properties', name: 'Mass Properties', role: 'Inertia & CoG', category: 'Core', icon: Scale, color: '#f59e0b', stats: { cpu: 5, memory: 10 }, logs: ["Calculating CoG...", "Inertia tensor updated"] },
    { id: 'manifold', name: 'Manifold Agent', role: 'Watertight Check', category: 'Core', icon: Layers, color: '#10b981', stats: { cpu: 15, memory: 30 }, logs: ["Checking edge connectivity...", "Mesh is watertight"] },
    { id: 'validator', name: 'Validator Agent', role: 'Constraint Check', category: 'Core', icon: CheckCircle, color: '#ef4444', stats: { cpu: 8, memory: 20 }, logs: ["Validating KCL syntax...", "0 constraints violated"] },

    // --- Design & Planning ---
    { id: 'designer', name: 'Designer Agent', role: 'Aesthetics & Style', category: 'Design', icon: PenTool, color: '#ec4899', stats: { cpu: 22, memory: 50 }, logs: ["Generating matte finish...", "Applying color scheme 'Orbital'"] },
    { id: 'design_exploration', name: 'Design Exploration', role: 'Parametric Search', category: 'Design', icon: Compass, color: '#f97316', stats: { cpu: 88, memory: 90 }, logs: ["Sampling design space...", "Ranking candidate #42"] },
    { id: 'design_quality', name: 'Design Quality', role: 'Fidelity Enhancer', category: 'Design', icon: Star, color: '#d946ef', stats: { cpu: 30, memory: 45 }, logs: ["Adding panel lines...", "Refining mesh density"] },
    { id: 'template_design', name: 'Template Agent', role: 'Pattern Library', category: 'Design', icon: LayoutTemplate, color: '#6366f1', stats: { cpu: 10, memory: 25 }, logs: ["Loading airfoil template...", "Parametric features extracted"] },
    { id: 'asset_sourcing', name: 'Asset Sourcing', role: 'Catalog Search', category: 'Design', icon: Download, color: '#06b6d4', stats: { cpu: 4, memory: 15 }, logs: ["Querying NASA 3D Catalog...", "Asset 'Thruster_v2' found"] },

    // --- Analysis ---
    { id: 'thermal', name: 'Thermal Agent', role: 'Heat Analysis', category: 'Analysis', icon: Thermometer, color: '#f43f5e', stats: { cpu: 55, memory: 70 }, logs: ["Calculating heat dissipation...", "Warning: Hotspot detected"] },
    { id: 'dfm', name: 'DfM Agent', role: 'Manufacturability', category: 'Analysis', icon: Factory, color: '#84cc16', stats: { cpu: 20, memory: 35 }, logs: ["Checking wall thickness...", "Tool access verified"] },
    { id: 'cps', name: 'CPS Agent', role: 'Control Systems', category: 'Analysis', icon: Cpu, color: '#3b82f6', stats: { cpu: 40, memory: 50 }, logs: ["Simulating PID loop...", "Sensor placement optimized"] },
    { id: 'gnc', name: 'GNC Agent', role: 'Guidance & Nav', category: 'Analysis', icon: Navigation, color: '#a855f7', stats: { cpu: 35, memory: 40 }, logs: ["Checking stability margins...", "Control authority: Nominal"] },
    { id: 'mitigation', name: 'Mitigation Agent', role: 'Fix Proposer', category: 'Analysis', icon: ShieldAlert, color: '#22c55e', stats: { cpu: 12, memory: 20 }, logs: ["Analyzing error report...", "Proposed fix: Thicken wall"] },
    { id: 'topological', name: 'Topological Agent', role: 'Terrain & Mode', category: 'Analysis', icon: Mountain, color: '#78716c', stats: { cpu: 18, memory: 30 }, logs: ["Classifying terrain...", "Mode set to: KINETIC"] },

    // --- Manufacturing ---
    { id: 'manufacturing', name: 'Manufacturing Agent', role: 'BOM & Cost', category: 'Manufacturing', icon: Factory, color: '#10b981', stats: { cpu: 25, memory: 40 }, logs: ["Generating BOM...", "Unit cost estimated: $124.50"] },
    { id: 'slicer', name: 'Slicer Agent', role: 'G-Code Gen', category: 'Manufacturing', icon: Scissors, color: '#f59e0b', stats: { cpu: 75, memory: 80 }, logs: ["Generating layers...", "Print time: 4h 12m"] },
    { id: 'tolerance', name: 'Tolerance Agent', role: 'Fit Analysis', category: 'Manufacturing', icon: Ruler, color: '#64748b', stats: { cpu: 15, memory: 20 }, logs: ["Checking ISO fits...", "Clearance: 0.05mm"] },

    // --- Material ---
    { id: 'material', name: 'Material Agent', role: 'Properties', category: 'Material', icon: Beaker, color: '#14b8a6', stats: { cpu: 8, memory: 15 }, logs: ["Material: Al-6061-T6", "Yield strength check: OK"] },
    { id: 'chemistry', name: 'Chemistry Agent', role: 'Composition', category: 'Material', icon: FlaskConical, color: '#0ea5e9', stats: { cpu: 5, memory: 10 }, logs: ["Checking galvanic compatibility...", "No hazards found"] },

    // --- Structural/Arch ---
    { id: 'structural_load', name: 'Structural Load', role: 'Static Analysis', category: 'Structural', icon: Triangle, color: '#f97316', stats: { cpu: 45, memory: 50 }, logs: ["Solving load paths...", "Safety factor: 2.1"] },
    { id: 'mep', name: 'MEP Agent', role: 'Systems Routing', category: 'Structural', icon: Droplet, color: '#3b82f6', stats: { cpu: 30, memory: 40 }, logs: ["Routing HVAC ducts...", "Checking electrical load"] },
    { id: 'zoning', name: 'Zoning Agent', role: 'Compliance', category: 'Structural', icon: Map, color: '#8b5cf6', stats: { cpu: 10, memory: 20 }, logs: ["Checking setbacks...", "Height limit: Compliant"] },

    // --- Electronics ---
    { id: 'electronics', name: 'Electronics Agent', role: 'Power & PCB', category: 'Electronics', icon: CircuitBoard, color: '#ef4444', stats: { cpu: 25, memory: 30 }, logs: ["Calculating power budget...", "Selecting ESCs..."] },

    // --- Advanced ---
    { id: 'multi_mode', name: 'Multi-Mode Agent', role: 'Transition Logic', category: 'Advanced', icon: Shuffle, color: '#ec4899', stats: { cpu: 55, memory: 45 }, logs: ["Simulation transition: AERO -> GROUND", "Locking control surfaces"] },
    { id: 'nexus', name: 'Nexus Agent', role: 'Tree Nav', category: 'Advanced', icon: Network, color: '#06b6d4', stats: { cpu: 10, memory: 15 }, logs: ["Context: /geometry/fuselage", "Listening for commands"] },
    { id: 'pvc', name: 'PVC Agent', role: 'Version Control', category: 'Advanced', icon: GitBranch, color: '#f59e0b', stats: { cpu: 12, memory: 25 }, logs: ["Snapshot taken: 'Wing_v2'", "Diffing state..."] },
    { id: 'doctor', name: 'Doctor Agent', role: 'Diagnostics', category: 'Advanced', icon: Stethoscope, color: '#22c55e', stats: { cpu: 5, memory: 10 }, logs: ["System health: 98%", "All systems nominal"] },
    { id: 'shell', name: 'Shell Agent', role: 'CLI Exec', category: 'Advanced', icon: Terminal, color: '#64748b', stats: { cpu: 2, memory: 5 }, logs: ["Ready.", "_"] },
    { id: 'vhil', name: 'VHIL Agent', role: 'Hardware Sim', category: 'Advanced', icon: MonitorPlay, color: '#8b5cf6', stats: { cpu: 80, memory: 95 }, logs: ["Emulating sensor inputs...", "Real-time loop active"] },

    // --- Documentation ---
    { id: 'documentation', name: 'Doc Agent', role: 'Reporting', category: 'Docs & QA', icon: FileText, color: '#6b7280', stats: { cpu: 15, memory: 30 }, logs: ["Generating PDF report...", "Compiling specs"] },
    { id: 'diagnostic', name: 'Diagnostic Agent', role: 'Root Cause', category: 'Docs & QA', icon: Activity, color: '#ef4444', stats: { cpu: 10, memory: 15 }, logs: ["Monitoring error stream...", "No active faults"] },
    { id: 'verification', name: 'Verification Agent', role: 'Testing', category: 'Docs & QA', icon: ClipboardCheck, color: '#10b981', stats: { cpu: 40, memory: 50 }, logs: ["Running test suite A...", "Pass rate: 100%"] },
    { id: 'visual_validator', name: 'Visual Validator', role: 'Aesthetic Check', category: 'Docs & QA', icon: Eye, color: '#ec4899', stats: { cpu: 30, memory: 60 }, logs: ["Checking render quality...", "No artifacts detected"] },

    // --- Specialized ---
    { id: 'standards', name: 'Standards Agent', role: 'ISO/ANSI', category: 'Specialized', icon: BookOpen, color: '#1d4ed8', stats: { cpu: 5, memory: 10 }, logs: ["Looking up ISO-9001...", "Reqs loaded"] },
    { id: 'component', name: 'Component Agent', role: 'Parts Selection', category: 'Specialized', icon: Settings, color: '#db2777', stats: { cpu: 15, memory: 25 }, logs: ["Matching motor specs...", "Selected: T-Motor U8"] },
    { id: 'conversational', name: 'Conversational', role: 'NLP Interface', category: 'Specialized', icon: MessageSquare, color: '#8b5cf6', stats: { cpu: 35, memory: 50 }, logs: ["Parsing intent...", "Response generated"] },
    { id: 'remote', name: 'Remote Agent', role: 'Collaboration', category: 'Specialized', icon: Wifi, color: '#0ea5e9', stats: { cpu: 10, memory: 20 }, logs: ["Syncing session...", "User connected"] },

    // --- Training ---
    { id: 'physics_trainer', name: 'Physics Trainer', role: 'Model Learning', category: 'Training', icon: GraduationCap, color: '#f59e0b', stats: { cpu: 95, memory: 90 }, logs: ["Training epoch 42...", "Loss: 0.043"] },
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
        case 'Material': return theme.colors.status.info; // Cyan-ish usually, mapping to Info
        case 'Structural': return theme.colors.accent.secondary;
        case 'Electronics': return theme.colors.status.error;
        case 'Advanced': return theme.colors.accent.primary;
        case 'Docs & QA': return theme.colors.text.muted;
        case 'Specialized': return theme.colors.status.info;
        case 'Training': return theme.colors.accent.secondary;
        default: return theme.colors.text.primary;
    }
};

const AgentRow = ({ agent, isActive, onToggle, runStatus, logs }) => {
    const { theme } = useTheme();
    const [isExpanded, setIsExpanded] = useState(false);
    const [localCpu, setLocalCpu] = useState(agent.stats.cpu);

    const agentColor = getCategoryColor(agent.category, theme);

    // Simulate life only if active
    useEffect(() => {
        if (!isActive) {
            setLocalCpu(0);
            return;
        }
        const interval = setInterval(() => {
            const base = agent.stats.cpu === 0 ? 5 : agent.stats.cpu;
            setLocalCpu(prev => Math.max(1, Math.min(99, base + (Math.random() * 10 - 5))));
        }, 1200 + Math.random() * 500);
        return () => clearInterval(interval);
    }, [isActive, agent.stats.cpu]);

    return (
        <div
            className="flex flex-col rounded border transition-all duration-200"
            style={{
                backgroundColor: theme.colors.bg.secondary,
                borderColor: isActive ? agentColor + '40' : theme.colors.border.primary
            }}
        >
            <div
                className="flex items-center p-3 gap-3 cursor-pointer hover:bg-white/5 transition-colors group"
                onClick={() => setIsExpanded(!isExpanded)}
            >
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

                <div className="p-1.5 rounded-md" style={{
                    backgroundColor: isActive ? agentColor + '15' : theme.colors.bg.tertiary,
                    color: isActive ? agentColor : theme.colors.text.muted
                }}>
                    <agent.icon size={16} />
                </div>

                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                        <h3 className="text-xs font-bold font-mono truncate" style={{ color: theme.colors.text.primary }}>
                            {agent.name}
                        </h3>
                        {isActive && <div className="w-1 h-1 rounded-full animate-pulse" style={{ backgroundColor: agentColor }} />}
                    </div>
                </div>

                {/* Power Toggle */}
                <button
                    onClick={(e) => { e.stopPropagation(); onToggle(); }}
                    className={`p-1.5 rounded-full transition-all opacity-0 group-hover:opacity-100 ${isActive ? 'opacity-100 bg-green-500/10 text-green-500' : 'bg-white/5 hover:bg-white/10'}`}
                    style={!isActive ? { color: theme.colors.text.muted } : {}}
                >
                    <Power size={12} />
                </button>
            </div>

            {isExpanded && (
                <div className="border-t p-3 pl-10" style={{ borderColor: theme.colors.border.primary, backgroundColor: theme.colors.bg.tertiary }}>
                    <div className="flex items-center gap-4 mb-2 text-[10px] font-mono" style={{ color: theme.colors.text.muted }}>
                        <span>ROLE: {agent.role}</span>
                        <span>CPU: {Math.round(localCpu)}%</span>
                        <span>MEM: {agent.stats.memory}MB</span>
                        {runStatus && <span style={{ color: runStatus === 'success' ? theme.colors.status.success : theme.colors.status.warning }}>[{runStatus.toUpperCase()}]</span>}
                    </div>
                    <div className="font-mono text-[10px] space-y-1 opacity-80" style={{ color: theme.colors.text.secondary }}>
                        {logs && logs.length > 0 ? (
                            logs.map((log, i) => (
                                <div key={i} className="flex gap-2">
                                    <span style={{ color: agentColor }}>{`>`}</span>
                                    <span>{log}</span>
                                </div>
                            ))
                        ) : (
                            <div className="italic" style={{ color: theme.colors.text.muted }}>Agent suspended. Click toggle to run.</div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

const CategorySection = ({ title, agents, activeAgents, agentStates, toggleAgent }) => {
    const { theme } = useTheme();
    const [isOpen, setIsOpen] = useState(true);

    return (
        <div className="mb-4">
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
                    {agents.map(agent => (
                        <AgentRow
                            key={agent.id}
                            agent={agent}
                            isActive={activeAgents.includes(agent.id)}
                            onToggle={() => toggleAgent(agent.id)}
                            runStatus={agentStates[agent.id]?.status}
                            logs={agentStates[agent.id]?.logs || agent.logs}
                        />
                    ))}
                </div>
            )}
        </div>
    );
};

const AgentPodsPanel = ({ width }) => {
    const { theme } = useTheme();
    // Start with empty active agents, letting user drive them
    const [activeAgents, setActiveAgents] = useState([]);
    const [agentStates, setAgentStates] = useState({});
    const groupedAgents = groupAgentsByCategory(AGENT_REGISTRY);

    const runAgent = async (id) => {
        // Construct plausible params based on ID
        // In a real app, this comes from the central DesignContext/Store
        const payload = {
            // Common
            project_id: "demo-project",

            // Thermal/Structural
            power_watts: 450.0,
            surface_area: 1.2,
            mass_kg: 12.5,
            g_loading: 4.5,

            // Electronics
            components: ["mcu", "lidar", "radio", "motor_driver"],
            motor_count: 6,

            // Cost/Mfg
            volume_cm3: 1250.0,
            infill_percent: 35,
            material_cost_per_kg: 24.50,

            // Compliance
            regime: "AERIAL",

            // Designer
            style: "cyberpunk"
        };

        try {
            const res = await fetch(`http://localhost:8000/api/agents/${id}/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (data.status === 'success' && data.result) {
                setAgentStates(prev => ({
                    ...prev,
                    [id]: {
                        status: 'success',
                        logs: data.result.logs || ["Analysis complete."]
                    }
                }));
            }
        } catch (e) {
            console.error(`Failed to run agent ${id}`, e);
            setAgentStates(prev => ({
                ...prev,
                [id]: {
                    status: 'error',
                    logs: [`Error: ${e.message}`, "Check connection."]
                }
            }));
        }
    };

    const toggleAgent = (id) => {
        const isActive = activeAgents.includes(id);
        if (isActive) {
            setActiveAgents(prev => prev.filter(a => a !== id));
            // Maybe clear logs? Keep them for history.
        } else {
            setActiveAgents(prev => [...prev, id]);
            // Trigger run immediately
            runAgent(id);
        }
    };

    return (
        <div
            className="flex flex-col min-w-0 h-full border-r relative"
            style={{
                width: width,
                borderColor: theme.colors.border.primary,
                backgroundColor: theme.colors.bg.primary
            }}
        >
            <PanelHeader title="Agent Pods" icon={Network} />

            <div className="flex-1 p-3 overflow-y-auto scrollbar-thin">
                {Object.keys(groupedAgents).map(category => (
                    <CategorySection
                        key={category}
                        title={category}
                        agents={groupedAgents[category]}
                        activeAgents={activeAgents}
                        agentStates={agentStates}
                        toggleAgent={toggleAgent}
                    />
                ))}
            </div>

            {/* Status Footer */}
            <div className="h-8 border-t flex items-center px-4 gap-4 text-[10px] font-mono shrink-0 overflow-hidden whitespace-nowrap"
                style={{
                    borderColor: theme.colors.border.primary,
                    color: theme.colors.text.tertiary,
                    backgroundColor: theme.colors.bg.secondary
                }}
            >
                <div className="flex items-center gap-2">
                    <Activity size={12} className={activeAgents.length > 0 ? "animate-pulse" : ""} style={{ color: activeAgents.length > 0 ? theme.colors.status.success : theme.colors.status.error }} />
                    <span className="hidden sm:inline">{activeAgents.length} ACTIVE</span>
                </div>
            </div>
        </div>
    );
};

export default AgentPodsPanel;
