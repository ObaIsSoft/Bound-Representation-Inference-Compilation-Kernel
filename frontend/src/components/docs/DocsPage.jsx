import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
    BookOpen, Cpu, Zap, Globe, Shield, Settings, Terminal, ChevronRight,
    Box, Activity, Code, HardDrive, Layers, Menu, X, ExternalLink,
    Search, Brain, Share2, Database, FileText, Dna, Rocket, Waves,
    Thermometer, ClipboardCheck, Microscope, AlertCircle, Lightbulb,
    Info, CheckCircle2, Copy, Network, MessageSquare, Wifi, GraduationCap,
    Lock, Plus, Factory, TrendingUp, DollarSign, AlertTriangle, Power,
    Scale, Compass, Star, LayoutTemplate, Download, Navigation,
    ShieldAlert, Mountain, Scissors, Ruler, Beaker, FlaskConical,
    Triangle, Droplet, Map, CircuitBoard, Shuffle, GitBranch,
    Stethoscope, MonitorPlay, Eye, ClipboardList, Smile, Users, Heart,
    Wand2, Glasses
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTheme } from '../../contexts/ThemeContext';
import PanelHeader from '../shared/PanelHeader';

/**
 * BRICK OS DOCUMENTATION - THE DUAL-MODE REFERENCE (V2 - THEMED)
 * Fixed: Sidebar navigation logic and theme-compliance.
 */

const DocsPage = () => {
    const { theme } = useTheme();
    const [activeSection, setActiveSection] = useState('welcome');
    const [searchTerm, setSearchTerm] = useState('');
    const [displayMode, setDisplayMode] = useState('layman'); // 'layman' | 'expert'
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const [copyFeedback, setCopyFeedback] = useState(null);
    const contentRef = useRef(null);

    // --- ENHANCED DATA STORE WITH DIRECT IDS ---
    const KNOWLEDGE_BASE = useMemo(() => ([
        {
            id: 'welcome',
            title: { layman: 'Hello! (Start Here)', expert: 'System Entry & Initialization' },
            icon: Smile,
            summary: { layman: 'An easy intro to BRICK OS.', expert: 'Architectural overview and kernel philosophy.' },
            content: [
                {
                    subtitle: { layman: 'What is BRICK?', expert: 'Kernel Identity (The BRICK Objective)' },
                    layman: 'Think of BRICK as a "magic Lego set" that understands physics. You say what you want (like a drone), and it finds the pieces and makes sure it won\'t break before you build it.',
                    expert: 'BRICK (Bound-Representation Inference-Compilation Kernel) is a domain-specific hardware compiler. It maps high-level semantic intent into a 5-tier reactive constraint graph. Unlike CAD (drawing), BRICK is Invariant-Driven Synthesis: the geometry is a side-effect of satisfying physical bounds.',
                    tags: ['philosophy', 'intro']
                },
                {
                    subtitle: { layman: 'Why it\'s special', expert: 'Strict Compliance & Semantic Physics' },
                    layman: 'Most computers just draw shapes. BRICK "understands" that metal is heavy and air is windy. It prevents mistakes by knowing how the real world works.',
                    expert: 'Traditional tools are post-hoc (design then test). BRICK is real-time: every operation is validated by the Unified Physics Kernel (UPK) against conservation laws. It uses a 64-bit unit-enforced ISA to ensure dimensional consistency across all 8 physics domains.',
                    tags: ['validation', 'compliance']
                }
            ]
        },
        {
            id: 'agents',
            title: { layman: 'The Team (Agents)', expert: 'Tiered Agent Swarm (60+ Entities)' },
            icon: Users,
            summary: { layman: 'Meet the 60+ experts inside.', expert: 'Low-level specifications for autonomous nodes.' },
            content: [
                {
                    id: 'tier-a',
                    subtitle: { layman: 'The Builders (Physics & Geo)', expert: 'Tier A: The Physical Providers' },
                    layman: '• **Nature Expert**: Sets the weather and gravity.\n• **Shape Maker**: Draws the 3D models.\n• **Physics Judge**: Makes sure the design won\'t break.',
                    expert: '• **PhysicsAgent**: The principal validator. Orchestrates 6-DOF simulation loops and interfaces with the multi-fidelity router.\n• **EnvironmentAgent**: Selects planetary constants (G, ρ_atm) and operational regimes (AERO/STN/SPACE).\n• **GeometryAgent**: Transpiles semantic parameters into KCL (KittyCAD Language) and OpenSCAD intermediate representations.\n• **MaterialAgent**: Deep-integration with NIST/MP data; handles temperature-dependent properties ρ(T) and E(T).',
                    tags: ['agents', 'tier-a', 'physics']
                },
                {
                    id: 'tier-b',
                    subtitle: { layman: 'The Inventors (Optimization)', expert: 'Tier B: Evolutionary Design Kernels' },
                    layman: '• **The Dreamer**: Tries thousands of shapes to find the best one.\n• **The Artist**: Makes it look beautiful.',
                    expert: '• **OptimizationAgent**: Implements Multi-Objective Genetic Algorithms (NSGA-II) for topology optimization.\n• **LatticeSynthesisAgent**: Generates internally complex micro-geometries (Triply Periodic Minimal Surfaces) for weight reduction.\n• **VisualValidatorAgent**: LLM-based aesthetic critic using CLIP-inspired embeddings to align design with brand identity.',
                    tags: ['agents', 'tier-b', 'gen-ai']
                },
                {
                    id: 'tier-c',
                    subtitle: { layman: 'The Factory Crew', expert: 'Tier C: Manufacturing & Logistics' },
                    layman: '• **Shop Manager**: Figures out cost and build steps.\n• **Safety Inspector**: Makes sure a factory can actually build it.',
                    expert: '• **ManufacturingAgent**: Synthesizes a Bill of Process (BoP). Calculates machining toolpaths and subtractive SDF boolean steps.\n• **DfMAgent**: Performs wall-thickness verification and draft-angle analysis at 1nm precision.\n• **SourcingAgent**: Real-time component matching via web-scraped inventory (DigiKey/Mouser APIs).',
                    tags: ['agents', 'tier-c', 'mfg']
                },
                {
                    id: 'tier-de',
                    subtitle: { layman: 'The Brains (Cyber-Physical)', expert: 'Tier D/E: Systems & Meta-Operations' },
                    layman: '• **The Pilot**: Writes the "brain" for your vehicle.\n• **The Electrician**: Fixes wires and batteries.\n• **The Librarian**: Writes the manual.',
                    expert: '• **GncAgent**: Stability derivative synthesis and law control (PID/LQR/MPC).\n• **ElectronicsAgent**: Genetic Trace Routing for PCB synthesis and power-budget thermal analysis.\n• **DevOpsAgent**: Orchestrates Docker-based build-environments and system-level PVC (Physical Version Control) commits.',
                    tags: ['agents', 'tier-d', 'robotics']
                }
            ]
        },
        {
            id: 'cockpit',
            title: { layman: 'The Cockpit (UI)', expert: 'Frontend Interaction & Panels' },
            icon: Rocket,
            summary: { layman: 'A guide to every button.', expert: 'State management and component specifications.' },
            content: [
                {
                    subtitle: { layman: 'The Activity Bar', expert: 'Task-Centric Routing (ActivityBar.jsx)' },
                    layman: 'Your main remote control. Each icon opens a different work-bench (Search, Design, Run, Swarm).',
                    expert: 'A sticky sidebar managing the `activeActivity` state. Each tab corresponds to a lazy-loaded context which mounts specialized panels into the Flexible Panel Layout.',
                    tags: ['ui', 'routing']
                },
                {
                    subtitle: { layman: 'The Terminal', expert: 'Introspection Multiplexer (IntegratedTerminal)' },
                    layman: 'The "Under the Hood" view. **Reasoning Tab** shows AI thoughts. **vHIL Tab** shows virtual sensor signals.',
                    expert: 'A bottom-docked multiplexer for system-wide streams. Includes:\n• **Reasoning**: Raw LLM Thought-Trace.\n• **PVC**: Physical Version Control (Git CLI).\n• **KCL**: Live intermediate representation of geometry bytecode.',
                    tags: ['ui', 'terminal', 'logic']
                }
            ]
        },
        {
            id: 'backend',
            title: { layman: 'How It Thinks (Backend)', expert: 'Kernel Orchestration & Logic' },
            icon: Brain,
            summary: { layman: 'Understanding the Project Manager inside.', expert: 'LangGraph, NetworkX, and SSE specifications.' },
            content: [
                {
                    subtitle: { layman: 'The Project Manager', expert: 'Graph Orchestration (LangGraph)' },
                    layman: 'Think of it as a flowchart that makes sure the "Material Expert" talks to the "Physics Judge" at the right time.',
                    expert: 'Utilizes a cyclical StateGraph. Nodes represent individual agents (Environment, Geometry, etc.). Conditional edges perform "router logic" to handle design failures by looping back to optimization nodes.',
                    code: 'graph = StateGraph(AgentState)\ngraph.add_node("physics", physics_agent)\ngraph.add_edge("physics", "optimization")\ngraph.add_conditional_edges("opt", check_done)',
                    tags: ['backend', 'graph', 'orchestrator']
                },
                {
                    subtitle: { layman: 'The Spider Web', expert: 'Reactive Dependency Mappings (NetworkX)' },
                    layman: 'Every part of your design is connected. If you change the wing size, the whole web "vibrates" and updates the weight and battery automatically.',
                    expert: 'A DAG (Directed Acyclic Graph) of physical parameters. Root nodes (inputs) propagate invalidations through edges based on mathematical relationships defined in the UPK. Uses `networkx` for topological sorting of re-computations.',
                    tags: ['backend', 'networkx', 'logic_core']
                },
                {
                    subtitle: { layman: 'Fast Streaming', expert: 'SSE Streaming Pipeline' },
                    layman: 'It doesn\'t wait to finish before showing work. It "streams" parts to you one by one, like a movie.',
                    expert: 'Server-Sent Events (SSE) bridge the Backend/Frontend gap. As the agent swarm bakes individual sub-components (pods), they are serialized to JSON/GLB and streamed to the UI `EventSource` for immediate side-loading into the 3D scene.',
                    tags: ['backend', 'sse', 'realtime']
                }
            ]
        },
        {
            id: 'physics',
            title: { layman: 'Rules of Reality (Physics)', expert: 'Unified Physics Kernel (UPK)' },
            icon: Activity,
            summary: { layman: 'Why designs in BRICK actually work.', expert: 'Multiphysics numerical integration & PDE logic.' },
            content: [
                {
                    subtitle: { layman: 'Real World Science', expert: 'Domain Provider Architecture' },
                    layman: 'BRICK knows that metal melts and air is thick. It uses the same math as NASA and Boeing.',
                    expert: 'A decoupled kernel containing 8 domains (Thermodynamics, Fluids, Mechanics, Electromagnetism, Nuclear, etc.). It routes queries to multi-fidelity providers including SymPy (Symbolic), SciPy (Numerical), and FEniCS (FEA).',
                    tags: ['physics', 'science', 'math']
                },
                {
                    subtitle: { layman: 'The Student & Teacher', expert: 'PINN (Physics-Informed Neural Nets)' },
                    layman: 'A "Teacher" judge checks the math, while a "Student" explorer tries to find better ways to build.',
                    expert: 'Neural surrogates trained on solver residuals. We use a Teacher (High-Fidelity Solver) to supervise a Student (ML Model). This allows the system to predict complex thermal/structural behavior 100x faster than traditional FEA during the "Search" phase.',
                    tags: ['physics', 'ai', 'surrogate']
                }
            ]
        },
        {
            id: 'isa',
            title: { layman: 'The Unit Rules (ISA)', expert: 'ISA 64-Bit Logic Specification' },
            icon: Cpu,
            summary: { layman: 'How we keep units consistent.', expert: 'Dimensional analysis and unit-type enforcement.' },
            content: [
                {
                    subtitle: { layman: 'Unit Safety', expert: 'PhysicalValue & UnitDimension' },
                    layman: 'BRICK is very strict about units. You can\'t accidentally add "inches" to "kilograms". It catches these mistakes instantly.',
                    expert: 'Every parameter is wrapped in a `PhysicalValue` object containing: Magnitude, Unit Symbol, Dimensional Signature, and Tolerance. Operations (+, -, *, /) perform a dimensional check (`L^1 M^0 T^-2`) at the parser level.',
                    code: 'class PhysicalValue(BaseModel):\n    magnitude: float\n    unit: UnitDimension\n    tolerance: float = 0.001\n    locked: bool = False',
                    tags: ['isa', 'units', 'safety']
                }
            ]
        }
    ]), []);

    // --- Search Engine ---
    const searchResults = useMemo(() => {
        if (!searchTerm) return KNOWLEDGE_BASE;
        const lowTerm = searchTerm.toLowerCase();

        return KNOWLEDGE_BASE.map(section => {
            const matchedFeatures = section.content.filter(f =>
                f.subtitle[displayMode].toLowerCase().includes(lowTerm) ||
                (f[displayMode] && f[displayMode].toLowerCase().includes(lowTerm)) ||
                (displayMode === 'expert' && f.expert && f.expert.toLowerCase().includes(lowTerm)) ||
                f.tags.some(t => t.toLowerCase().includes(lowTerm))
            );

            if (section.title[displayMode].toLowerCase().includes(lowTerm) || matchedFeatures.length > 0) {
                return { ...section, content: matchedFeatures.length > 0 ? matchedFeatures : section.content, matched: true };
            }
            return null;
        }).filter(Boolean);
    }, [searchTerm, displayMode, KNOWLEDGE_BASE]);

    const handleCopy = (text, id) => {
        navigator.clipboard.writeText(text);
        setCopyFeedback(id);
        setTimeout(() => setCopyFeedback(null), 2000);
    };

    const handleSectionSelect = (id) => {
        setActiveSection(id);
        setSearchTerm(''); // Clear search so the active section actually shows
        setIsMobileMenuOpen(false);
        if (contentRef.current) contentRef.current.scrollTop = 0;
    };

    const Highlight = ({ children }) => {
        if (!searchTerm || typeof children !== 'string') return children;
        const parts = children.split(new RegExp(`(${searchTerm})`, 'gi'));
        return parts.map((part, i) =>
            part.toLowerCase() === searchTerm.toLowerCase()
                ? <mark key={i} className="rounded px-0.5" style={{ backgroundColor: theme.colors.accent.primary + '60', color: theme.colors.text.primary }}>{part}</mark>
                : part
        );
    };

    return (
        <div className="flex-1 h-full flex flex-col overflow-hidden"
            style={{ backgroundColor: theme.colors.bg.primary }}>

            <PanelHeader title="SYSTEM BIBLE" icon={BookOpen} />

            {/* --- Mode Toggle & Search Header --- */}
            <header className="px-6 py-4 border-b flex flex-col md:flex-row items-center justify-between gap-6 z-20"
                style={{ borderColor: theme.colors.border.primary, backgroundColor: theme.colors.bg.secondary }}>

                <div className="flex p-1.5 rounded-2xl border relative"
                    style={{ backgroundColor: theme.colors.bg.tertiary, borderColor: theme.colors.border.primary }}>
                    <button
                        onClick={() => setDisplayMode('layman')}
                        className={`flex items-center gap-2 px-6 py-2 rounded-xl text-xs font-bold transition-all z-10 ${displayMode === 'layman' ? '' : 'opacity-40 hover:opacity-100'}`}
                        style={{ color: displayMode === 'layman' ? theme.colors.bg.primary : theme.colors.text.primary }}
                    >
                        <Smile size={14} /> Friendly Mode
                    </button>
                    <button
                        onClick={() => setDisplayMode('expert')}
                        className={`flex items-center gap-2 px-6 py-2 rounded-xl text-xs font-bold transition-all z-10 ${displayMode === 'expert' ? '' : 'opacity-40 hover:opacity-100'}`}
                        style={{ color: displayMode === 'expert' ? theme.colors.bg.primary : theme.colors.text.primary }}
                    >
                        <Glasses size={14} /> Expert Mode
                    </button>
                    <motion.div
                        layoutId="mode-pill"
                        initial={false}
                        animate={{ x: displayMode === 'layman' ? 0 : '100%' }}
                        className="absolute top-1.5 left-1.5 w-[calc(50%-6px)] h-[calc(100%-12px)] rounded-xl"
                        style={{ backgroundColor: theme.colors.accent.primary }}
                    />
                </div>

                <div className="flex items-center gap-4 w-full md:w-auto">
                    <div className="relative group w-full md:w-80">
                        <Search size={14} className="absolute left-4 top-1/2 -translate-y-1/2 opacity-30 group-focus-within:opacity-100 transition-opacity"
                            style={{ color: theme.colors.text.primary }} />
                        <input
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            placeholder={displayMode === 'layman' ? "Ask about LEGO or Experts..." : "Search ISA, KCL, or PINN logic..."}
                            className="w-full rounded-2xl py-3 pl-10 pr-10 text-xs font-medium outline-none border transition-all"
                            style={{
                                backgroundColor: theme.colors.bg.tertiary,
                                borderColor: theme.colors.border.primary,
                                color: theme.colors.text.primary,
                                ':focus': { borderColor: theme.colors.accent.primary }
                            }}
                        />
                        {searchTerm && (
                            <button onClick={() => setSearchTerm('')} className="absolute right-3 top-1/2 -translate-y-1/2 opacity-50 hover:opacity-100" style={{ color: theme.colors.text.primary }}>
                                <X size={14} />
                            </button>
                        )}
                    </div>
                </div>
            </header>

            <div className="flex-1 flex overflow-hidden">
                {/* --- Sidebar (Navigation) --- */}
                <aside className="hidden lg:flex flex-col w-72 border-r shrink-0"
                    style={{ borderColor: theme.colors.border.primary, backgroundColor: theme.colors.bg.secondary }}>
                    <nav className="flex-1 overflow-y-auto p-4 space-y-2 custom-scrollbar">
                        <div className="px-3 py-2 text-[10px] font-bold uppercase tracking-widest opacity-30" style={{ color: theme.colors.text.primary }}>Knowledge Domains</div>
                        {searchResults.map((section) => (
                            <button
                                key={section.id}
                                onClick={() => handleSectionSelect(section.id)}
                                className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl text-xs font-bold transition-all relative
                                    ${activeSection === section.id ? 'translate-x-1' : 'opacity-40 hover:opacity-100'}`}
                                style={{
                                    backgroundColor: activeSection === section.id ? theme.colors.accent.primary : 'transparent',
                                    color: activeSection === section.id ? theme.colors.bg.primary : theme.colors.text.primary
                                }}
                            >
                                <section.icon size={16} />
                                <div className="text-left leading-tight truncate">
                                    {section.title[displayMode]}
                                </div>
                                {activeSection === section.id && (
                                    <motion.div layoutId="nav-line" className="absolute left-0 w-1 h-1/2 rounded-full" style={{ backgroundColor: theme.colors.bg.primary }} />
                                )}
                            </button>
                        ))}
                    </nav>
                </aside>

                {/* --- Main Content --- */}
                <main
                    ref={contentRef}
                    className="flex-1 overflow-y-auto relative scroll-smooth"
                    style={{
                        backgroundImage: `radial-gradient(${theme.colors.accent.primary}08 2px, transparent 2px)`,
                        backgroundSize: '40px 40px'
                    }}
                >
                    <div className="max-w-4xl mx-auto p-6 md:p-12 pb-60 space-y-24">
                        {searchResults.length === 0 ? (
                            <div className="text-center py-40">
                                <Search size={48} className="mx-auto opacity-10 mb-4" style={{ color: theme.colors.text.primary }} />
                                <h3 className="text-xl font-bold opacity-30" style={{ color: theme.colors.text.primary }}>No intelligence found matching "{searchTerm}"</h3>
                                <button onClick={() => setSearchTerm('')} className="mt-4 font-bold text-xs underline" style={{ color: theme.colors.accent.primary }}>Clear Filters</button>
                            </div>
                        ) : (
                            searchResults.map((section) => (
                                <section key={section.id} id={section.id}
                                    className={`space-y-12 animate-in fade-in slide-in-from-bottom-6 duration-700 ${activeSection !== section.id && !searchTerm ? 'hidden' : ''}`}>

                                    <div className="relative">
                                        <div className="flex items-center gap-5 mb-4">
                                            <div className="p-4 rounded-[2rem] shadow-xl" style={{ backgroundColor: theme.colors.accent.primary + '15', color: theme.colors.accent.primary }}>
                                                <section.icon size={32} strokeWidth={1.5} />
                                            </div>
                                            <div>
                                                <h2 className="text-4xl md:text-5xl font-bold tracking-tighter" style={{ color: theme.colors.text.primary }}>
                                                    <Highlight>{section.title[displayMode]}</Highlight>
                                                </h2>
                                                <p className="text-lg opacity-50 font-medium" style={{ color: theme.colors.text.secondary }}>{section.summary[displayMode]}</p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-1 gap-16 md:ml-20">
                                        {section.content.map((module, mIdx) => (
                                            <div key={mIdx} className="space-y-6">
                                                <h3 className="text-2xl font-bold font-mono tracking-tight" style={{ color: theme.colors.text.primary }}>
                                                    <Highlight>{module.subtitle[displayMode]}</Highlight>
                                                </h3>

                                                <div className="text-base leading-relaxed opacity-80" style={{ color: theme.colors.text.secondary }}>
                                                    {module[displayMode].split('\n').map((line, lIdx) => (
                                                        <p key={lIdx} className={line.startsWith('•') ? "mt-2 mb-1" : "mb-4"}>
                                                            <Highlight>{line}</Highlight>
                                                        </p>
                                                    ))}
                                                </div>

                                                {module.code && (
                                                    <div className="relative group rounded-3xl overflow-hidden border" style={{ backgroundColor: theme.colors.bg.tertiary, borderColor: theme.colors.border.primary }}>
                                                        <div className="px-5 py-3 border-b flex justify-between items-center text-[10px] font-mono opacity-40 uppercase tracking-widest" style={{ borderColor: theme.colors.border.primary, color: theme.colors.text.primary }}>
                                                            <span>{displayMode === 'expert' ? 'Technical Specification' : 'Source Logic'}</span>
                                                            <Code size={12} />
                                                        </div>
                                                        <pre className="p-6 overflow-x-auto text-xs font-mono leading-relaxed" style={{ color: theme.colors.text.primary }}>
                                                            <code>{module.code}</code>
                                                        </pre>
                                                        <button
                                                            onClick={() => handleCopy(module.code, `${section.id}-${mIdx}`)}
                                                            className="absolute bottom-4 right-4 p-2 rounded-xl scale-90 md:group-hover:scale-100 transition-all md:opacity-0 md:group-hover:opacity-100 shadow-xl"
                                                            style={{ backgroundColor: theme.colors.accent.primary, color: theme.colors.bg.primary }}
                                                        >
                                                            {copyFeedback === `${section.id}-${mIdx}` ? <CheckCircle2 size={16} /> : <Copy size={16} />}
                                                        </button>
                                                    </div>
                                                )}

                                                {module.tags && (
                                                    <div className="flex flex-wrap gap-2">
                                                        {module.tags.map(tag => (
                                                            <span key={tag} className="px-3 py-1 rounded-full text-[9px] font-bold uppercase tracking-tighter border opacity-40 hover:opacity-100 transition-opacity cursor-help"
                                                                style={{ backgroundColor: theme.colors.bg.secondary, borderColor: theme.colors.border.primary, color: theme.colors.text.secondary }}>
                                                                #{tag}
                                                            </span>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </section>
                            ))
                        )}
                    </div>
                </main>
            </div>

            {/* --- Mobile Overlay --- */}
            <AnimatePresence>
                {isMobileMenuOpen && (
                    <motion.div
                        initial={{ x: '-100%' }} animate={{ x: 0 }} exit={{ x: '-100%' }}
                        className="fixed inset-0 z-[100] p-6 flex flex-col lg:hidden"
                        style={{ backgroundColor: theme.colors.bg.primary }}
                    >
                        <div className="flex justify-between items-center mb-10">
                            <span className="font-bold text-2xl tracking-tighter" style={{ color: theme.colors.text.primary }}>DOMAIN EXPLORER</span>
                            <X size={32} onClick={() => setIsMobileMenuOpen(false)} style={{ color: theme.colors.text.primary }} />
                        </div>
                        <div className="flex-1 space-y-4 overflow-y-auto">
                            {searchResults.map((section) => (
                                <button key={section.id} onClick={() => handleSectionSelect(section.id)}
                                    className="w-full flex items-center gap-6 p-6 rounded-3xl border text-xl font-bold"
                                    style={{
                                        backgroundColor: activeSection === section.id ? theme.colors.accent.primary : theme.colors.bg.secondary,
                                        color: activeSection === section.id ? theme.colors.bg.primary : theme.colors.text.primary,
                                        borderColor: theme.colors.border.primary
                                    }}>
                                    <section.icon size={28} />
                                    {section.title[displayMode]}
                                </button>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default DocsPage;
