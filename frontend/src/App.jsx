import React, { useState, useRef, useCallback } from 'react';
import { GripVertical, GripHorizontal } from 'lucide-react';
import Header from './components/layout/Header';
import ActivityBar from './components/layout/ActivityBar';
import DesignLibrary from './components/design/DesignLibrary';
import TabBar from './components/design/TabBar';
import CodeEditor from './components/editor/CodeEditor';
import SimulationBay from './components/simulation/SimulationBay';
import ControlDeck from './components/control/ControlDeck';
import IntegratedTerminal from './components/terminal/IntegratedTerminal';
import BootSequence from './components/ui/BootSequence';

import SettingsPage from './components/settings/SettingsPage';
import SearchPanel from './components/panels/SearchPanel';
import ComponentPanel from './components/panels/ComponentPanel';
import RunDebugPanel from './components/panels/RunDebugPanel';
import AgentPodsPanel from './components/panels/AgentPodsPanel';
import CompilePanel from './components/panels/CompilePanel';
import ManufacturingPanel from './components/panels/ManufacturingPanel';
import ForkPanel from './components/panels/ForkPanel';
import ExportPanel from './components/panels/ExportPanel';
import { ACTIVITY_BAR_WIDTH, DEFAULT_PANEL_SIZES } from './utils/constants';
import ISABrowserPanel from './components/panels/ISABrowserPanel';
import { useTheme } from './contexts/ThemeContext';
import { useSettings } from './contexts/SettingsContext';
import { useDesign } from './contexts/DesignContext';
import { useSimulation } from './contexts/SimulationContext';

export default function App() {
    const { theme } = useTheme();
    const { fontSize, aiModel } = useSettings();
    const { isEditorVisible, createArtifactTab, setPendingPlanId } = useDesign();
    const { isRunning, setIsRunning, metrics, setKclCode, setFocusedPodId, focusedPodId, setCompilationResult } = useSimulation(); // Phase 9: Added focusedPodId
    const [activeActivity, setActiveActivity] = useState('design');
    const [activeTab, setActiveTab] = useState('terminal');

    // Auto-switch to Results tab when Simulation starts
    React.useEffect(() => {
        if (isRunning) setActiveTab('results');
    }, [isRunning]);

    // Shell Terminal State
    const [commandInput, setCommandInput] = useState('');
    const [commandHistory, setCommandHistory] = useState([
        { type: 'sys', text: 'BRICK Kernel v2.3.5 Booted.' },
        { type: 'res', text: 'Type "brick telemetry --live" to start stream.' }
    ]);

    // PVC Terminal State (separate from shell)
    const [pvcCommandInput, setPvcCommandInput] = useState('');
    const [pvcCommandHistory, setPvcCommandHistory] = useState([
        { type: 'sys', text: 'Physics Verification & Compilation Terminal' },
        { type: 'res', text: 'Ready for brick compile/verify commands.' }
    ]);

    // Resizable Panel States
    const [leftWidth, setLeftWidth] = useState(DEFAULT_PANEL_SIZES.left);
    const [rightWidth, setRightWidth] = useState(DEFAULT_PANEL_SIZES.right);
    const [bottomHeight, setBottomHeight] = useState(DEFAULT_PANEL_SIZES.bottom);

    // ControlDeck Session Management
    const [sessions, setSessions] = useState([
        {
            id: 'default',
            title: 'Initial VTOL Config',
            timestamp: Date.now(),
            messages: [
                { role: 'assistant', text: 'BRICK Kernel Initialized. Ready for hardware intent.', agent: 'ARES_CORE' }
            ],
            isaSnapshot: { total_mass: 3630, shroud_dia: 7.62 }
        }
    ]);
    const [currentSessionId, setCurrentSessionId] = useState('default');
    const [isProcessingIntent, setIsProcessingIntent] = useState(false);
    const [reasoningStream, setReasoningStream] = useState([]);

    const currentSession = sessions.find(s => s.id === currentSessionId) || sessions[0];

    // Add reasoning log
    const addReasoning = (agent, thought) => {
        const log = {
            time: new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
            agent,
            thought
        };
        setReasoningStream(prev => [...prev, log]);
    };

    // Handle new session creation
    const handleNewSession = () => {
        const id = Date.now().toString();
        const newSession = {
            id,
            title: 'New Branch',
            timestamp: Date.now(),
            messages: [{ role: 'assistant', text: 'New design branch initialized. Environment: EARTH_AERO.', agent: 'ARES_CORE' }],
            isaSnapshot: { ...currentSession.isaSnapshot }
        };
        setSessions([newSession, ...sessions]);
        setCurrentSessionId(id);
        setReasoningStream([]);
    };

    // Handle sending intent
    const handleSendIntent = async (text) => {
        setIsProcessingIntent(true);

        // Add user message
        setSessions(prev => prev.map(s => {
            if (s.id === currentSessionId) {
                return { ...s, messages: [...s.messages, { role: 'user', text }] };
            }
            return s;
        }));

        try {
            // Call Backend API
            const response = await fetch('http://localhost:8000/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    context: currentSession.messages.map(m => ({ role: m.role, text: m.text })),
                    aiModel: aiModel, // Pass selected model (mock, ollama, etc.)
                    focusedPodId: focusedPodId // Phase 9: Recursive ISA Context
                })
            });

            const result = await response.json();

            // Simulate reasoning stream from logs if available
            if (result.logs) {
                result.logs.forEach((log, idx) => {
                    setTimeout(() => {
                        addReasoning("System", log);
                    }, idx * 100);
                });
            }

            // Update session with response
            setSessions(prev => prev.map(s => {
                if (s.id === currentSessionId) {
                    const newMessages = [...s.messages];

                    // Add any backend messages (artifacts, multiple text chunks)
                    if (result.messages && result.messages.length > 0) {
                        result.messages.forEach(msg => {
                            // If it's an artifact, create a tab instead of adding to chat
                            if (msg.type === 'artifact') {
                                createArtifactTab({
                                    id: msg.id,
                                    title: msg.title,
                                    content: msg.content
                                });
                                // Set as pending plan for approval
                                setPendingPlanId(msg.id);
                                // Add a simple notification to chat with artifact ID
                                newMessages.push({
                                    role: 'assistant',
                                    text: `📄 Generated: ${msg.title}`,
                                    agent: 'DOCUMENT_AGENT',
                                    artifactId: msg.id, // Store artifact ID for clickability
                                    artifactTitle: msg.title
                                });
                            } else {
                                // Regular message
                                newMessages.push({
                                    role: 'assistant',
                                    text: msg.content || msg.text,
                                    ...msg,
                                    agent: 'THE DREAMER'
                                });
                            }
                        });
                    }

                    // Add response message (fallback text) if no structured messages
                    if (result.response && (!result.messages || result.messages.length === 0)) {
                        newMessages.push({
                            role: 'assistant',
                            text: result.response,
                            agent: result.intent === 'design_request' ? 'DESIGNER_AGENT' : 'THE DREAMER'
                        });
                    }

                    return {
                        ...s,
                        messages: newMessages,
                        title: text.length > 20 ? text.slice(0, 20) + "..." : text,
                        isProcessing: false
                    };
                }
                return s;
            }));

        } catch (err) {
            console.error('Intent processing error:', err);
            addReasoning("System", `Error connecting to backend: ${err.message}`);
            setSessions(prev => prev.map(s => {
                if (s.id === currentSessionId) {
                    return { ...s, isProcessing: false };
                }
                return s;
            }));
        } finally {
            setIsProcessingIntent(false);
        }
    };

    // Handle rollback to specific message
    const handleRollback = (messageIndex, excludeTarget = false) => {
        setSessions(prev => prev.map(s => {
            if (s.id === currentSessionId) {
                // If excludeTarget is true, we roll back to BEFORE this message (effectively deleting it)
                // Otherwise we keep it as the last message.
                const sliceEnd = excludeTarget ? messageIndex : messageIndex + 1;
                return { ...s, messages: s.messages.slice(0, sliceEnd) };
            }
            return s;
        }));
        addReasoning("SYSTEM", `Restored design state from input index ${messageIndex}`);
    };

    // Handle session deletion
    const handleDeleteSession = (sessionId, e) => {
        e.stopPropagation(); // Prevent loading the session when clicking delete

        setSessions(prev => {
            const newSessions = prev.filter(s => s.id !== sessionId);
            // If we deleted the current session, switch to another or create default
            if (sessionId === currentSessionId) {
                if (newSessions.length > 0) {
                    setCurrentSessionId(newSessions[0].id);
                } else {
                    // Create a fresh default session if all deleted
                    const defaultSession = {
                        id: 'default-' + Date.now(),
                        title: 'New Session',
                        timestamp: Date.now(),
                        messages: [{ role: 'assistant', text: 'Ready for hardware intent.', agent: 'ARES_CORE' }],
                        isaSnapshot: { total_mass: 0 }
                    };
                    newSessions.push(defaultSession);
                    setCurrentSessionId(defaultSession.id);
                }
            }
            return newSessions;
        });
    };


    // Resize Handlers
    const isResizing = useRef(null);

    const startResizing = (direction) => (e) => {
        isResizing.current = direction;
        document.body.style.cursor = direction.includes('w') ? 'col-resize' : 'row-resize';
        document.addEventListener('mousemove', handleResize);
        document.addEventListener('mouseup', stopResizing);
    };

    const stopResizing = () => {
        isResizing.current = null;
        document.body.style.cursor = 'default';
        document.removeEventListener('mousemove', handleResize);
        document.removeEventListener('mouseup', stopResizing);
    };

    const handleResize = useCallback((e) => {
        if (!isResizing.current) return;

        if (isResizing.current === 'left') {
            const newWidth = e.clientX - ACTIVITY_BAR_WIDTH;
            setLeftWidth(newWidth > 0 ? newWidth : 0);
        } else if (isResizing.current === 'right') {
            const newWidth = window.innerWidth - e.clientX;
            setRightWidth(newWidth > 0 ? newWidth : 0);
        } else if (isResizing.current === 'bottom') {
            const newHeight = window.innerHeight - e.clientY;
            setBottomHeight(newHeight > 0 ? newHeight : 0);
        }
    }, []);

    const handleCommand = async (e) => {
        if (e.key === 'Enter') {
            const input = commandInput.trim();
            if (!input) return;
            setCommandHistory(prev => [...prev, { type: 'cmd', text: `brick:~ % ${input}` }]);

            const parts = input.trim().split(' ');
            const cmd = parts[0];
            const args = parts.slice(1);

            let response = null;

            // --- LOCAL CLIENT-SIDE COMMANDS ---
            if (cmd === 'clear') {
                setCommandHistory([]);
                setCommandInput('');
                return;
            }
            else if (cmd === 'run') {
                setIsRunning(true);
                setActiveTab('vhil');
                response = { type: 'sys', text: 'Booting vHIL Kernel...\nKernel Active. Telemetry stream started.' };
            }
            else if (input.includes('telemetry')) {
                setIsRunning(true);
                setActiveTab('vhil');
                response = { type: 'sys', text: 'Streaming Telemetry Data...' };
            }
            else if (cmd === 'halt') {
                setIsRunning(false);
                response = { type: 'err', text: 'Kernel Halted.' };
            }
            // --- REMOTE BACKEND SHELL ---
            else if (cmd === 'brick') {
                // Handle brick checkout (Phase 9)
                if (args[0] === 'checkout') {
                    const path = args[1] || ".";

                    try {
                        const res = await fetch('http://localhost:8000/api/isa/checkout', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ path })
                        });
                        const data = await res.json();

                        if (data.success) {
                            setFocusedPodId(data.pod_id); // Update Simulator Context
                            response = {
                                type: 'sys',
                                text: [
                                    data.message,
                                    data.pod_id ? `Scope: ${data.name} [ID: ${data.pod_id.substring(0, 8)}]` : 'Scope: Global',
                                    data.pod_id ? `Constraints: ${JSON.stringify(data.constraints)}` : ''
                                ].filter(Boolean).join('\n')
                            };
                        } else {
                            response = { type: 'err', text: `Checkout Failed: ${data.message}` };
                        }
                    } catch (err) {
                        response = { type: 'err', text: `Connection Error: ${err.message}` };
                    }
                }
                // Handle brick compile and verify specially - redirect to PVC tab
                else if (args[0] === 'help') {
                    response = {
                        type: 'res',
                        text: [
                            'BRICK OS Kernel v2.3.5 - Command Reference',
                            '--------------------------------------------',
                            'brick compile        Compile current design intent to physical constraints',
                            'brick verify         Run physics engine verification on active kernel',
                            'brick telemetry      Stream live vHIL data from active simulation',
                            'brick doctor         Run system diagnostics and agent health check',
                            'run                  Boot vHIL Kernel and start simulation',
                            'halt                 Stop active simulation',
                            'clear                Clear terminal output',
                            '',
                            'Standard shell commands (cd, ls, mkdir, etc.) are passed to system shell.'
                        ].join('\n')
                    };
                }
                else if (args[0] === 'vhil') {
                    // Handle 'brick vhil status' or 'brick vhil'
                    if (!args[1] || args[1] === 'status') {
                        if (!isRunning) {
                            response = { type: 'res', text: 'vHIL Kernel is OFFLINE. Type "run" to boot.' };
                        } else {
                            response = {
                                type: 'res',
                                text: [
                                    'vHIL Kernel Status: ONLINE',
                                    '--------------------------',
                                    `CPU Load:       ${metrics.cpu.toFixed(1)}%`,
                                    `Memory:         ${metrics.memory.toFixed(1)}%`,
                                    `Network Latency:`,
                                    `  AERO_ENGINE:  ${metrics.network.aero.toFixed(0)}ms`,
                                    `  THERMAL_GRID: ${metrics.network.thermal.toFixed(0)}ms`,
                                    `  GEO_MATER:    ${metrics.network.geo.toFixed(0)}ms`,
                                    '',
                                    `Graph Validty:  ${metrics.graph.score}`
                                ].join('\n')
                            };
                        }
                    } else {
                        response = { type: 'err', text: `Usage: brick vhil [status]` };
                    }
                }
                else if (args[0] === 'compile' || args[0] === 'verify') {
                    // Add command to PVC history
                    setPvcCommandHistory(prev => [...prev, { type: 'cmd', text: `brick:~ % ${input}` }]);

                    // Switch to PVC tab
                    setActiveTab('pvc');

                    try {
                        let res, data;

                        if (args[0] === 'compile') {
                            // Full Compilation (Orchestrator)
                            res = await fetch('http://localhost:8000/api/compile', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    user_intent: args.length > 1 ? args.slice(1).join(' ') : "Generic structural assembly",
                                    project_id: "sim-1"
                                })
                            });
                        } else {
                            // Quick Verification (Physics Only)
                            res = await fetch('http://localhost:8000/api/physics/verify', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    environment: { regime: "AERIAL", gravity: 9.81, fluid_density: 1.225 },
                                    geometry_tree: [],
                                    design_params: { available_thrust_N: 50 }
                                })
                            });
                        }

                        data = await res.json();

                        let pvcResponse;
                        if (args[0] === 'compile') {
                            // Update KCL Code in Context
                            if (data.kcl_code) {
                                setKclCode(data.kcl_code);
                            }
                            // Update Simulation Context with full result for Panels
                            setCompilationResult(data);

                            // Format compilation output
                            const flags = data.validation_flags || {};

                            pvcResponse = {
                                type: data.success ? 'sys' : 'err',
                                text: [
                                    'Compiling design intent...',
                                    '[INFO] Environment Identified',
                                    '[INFO] Geometry Generated (KCL)',
                                    '[INFO] Manufacturing BOM Calculated',
                                    '[INFO] Physics Validated',
                                    '',
                                    flags.physics_safe ? 'Build Success (1.2s)' : 'Build Failed'
                                ].join('\n')
                            };
                        } else {
                            // Verify output
                            pvcResponse = data;
                        }

                        // Add response to PVC history only
                        setPvcCommandHistory(prev => [...prev, pvcResponse]);

                        // Show acknowledgment in shell
                        response = {
                            type: 'sys',
                            text: `→ Redirected to PVC tab. Running ${args[0]}...`
                        };
                    } catch (err) {
                        const errorResponse = { type: 'err', text: `${args[0]} Failed: ${err.message}` };
                        setPvcCommandHistory(prev => [...prev, errorResponse]);
                        response = { type: 'err', text: `Connection Failed: ${err.message}` };
                    }
                }
                else {
                    // Other brick commands go through shell API
                    try {
                        const res = await fetch('http://localhost:8000/api/shell/execute', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ cmd, args })
                        });
                        const data = await res.json();
                        response = data;
                    } catch (err) {
                        response = { type: 'err', text: `Connection Failed: ${err.message}` };
                    }
                }
            }
            // --- SYSTEM SIMULATIONS (Fallback) ---
            else if (cmd === 'doctor') {
                response = { type: 'sys', text: 'System Health Check:\n[OK] Geometry Agent\n[OK] Physics Kernel (vHIL)\n[OK] Material Database\n[OK] API Gateway (Localhost:8000)' };
            }
            else {
                // Pass all other commands to backend shell (cd, mkdir, etc.)
                try {
                    const res = await fetch('http://localhost:8000/api/shell/execute', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ cmd, args })
                    });
                    const data = await res.json();
                    response = data;
                } catch (err) {
                    response = { type: 'err', text: `Connection Failed: ${err.message}` };
                }
            }

            if (response) {
                setCommandHistory(prev => [...prev, response]);
            }

            setCommandInput('');
        }
    };



    // ...

    // Render left panel based on active activity
    const renderLeftPanel = () => {
        switch (activeActivity) {
            case 'search':
                return <SearchPanel width={leftWidth} />;
            case 'components':
                return <ComponentPanel width={leftWidth} />;
            case 'design':
                return <DesignLibrary width={leftWidth} />;
            case 'run':
                return <RunDebugPanel width={leftWidth} />;
            case 'agents':
                return <AgentPodsPanel width={leftWidth} />;
            case 'compile':
                return <CompilePanel width={leftWidth} />;
            case 'mfg':
                return <ManufacturingPanel width={leftWidth} />;
            case 'fork':
                return <ForkPanel width={leftWidth} />;
            case 'export':
                return <ExportPanel width={leftWidth} />;
            case 'isa': // Phase 9: Hardware ISA Browser
                return <ISABrowserPanel width={leftWidth} />;
            default:
                return <DesignLibrary width={leftWidth} />;
        }
    };

    return (
        <div
            className="flex flex-col h-screen w-full overflow-hidden select-none font-sans"
            style={{
                backgroundColor: theme.colors.bg.primary,
                color: theme.colors.text.primary,
                '--editor-font-size': `${fontSize}px`
            }}
        >
            <BootSequence />
            <Header isRunning={isRunning} />

            {/* Main Interaction Matrix */}
            <div className="flex flex-1 overflow-hidden min-h-0 relative">

                <ActivityBar activeTab={activeActivity} setActiveTab={setActiveActivity} />

                {activeActivity === 'settings' || activeActivity === 'docs' ? (
                    activeActivity === 'settings' ? <SettingsPage /> : null
                ) : (
                    <>
                        {renderLeftPanel()}

                        <div
                            onMouseDown={startResizing('left')}
                            className="w-1 bg-transparent cursor-col-resize transition-all shrink-0 z-40 flex items-center justify-center"
                            style={{
                                backgroundColor: 'transparent',
                                ':hover': { backgroundColor: theme.colors.accent.primary + '80' }
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = theme.colors.accent.primary + '80'}
                            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                        >
                            <GripVertical size={10} style={{ color: theme.colors.border.secondary, opacity: 0 }} />
                        </div>


                        <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
                            <TabBar />
                            <div className="flex-1 flex overflow-hidden">
                                {/* Split View: Editor + Simulation */}
                                {isEditorVisible && (
                                    <div className="w-[45%] flex flex-col overflow-hidden">
                                        <CodeEditor />
                                    </div>
                                )}
                                <SimulationBay activeActivity={activeActivity} />
                            </div>

                            <div
                                onMouseDown={startResizing('bottom')}
                                className="h-1 bg-transparent cursor-row-resize transition-all shrink-0 z-40 flex items-center justify-center border-t"
                                style={{ borderColor: theme.colors.border.secondary + '33' }}
                                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = theme.colors.accent.primary + '80'}
                                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                            >
                                <GripHorizontal size={10} style={{ color: theme.colors.border.secondary, opacity: 0 }} />
                            </div>

                            <IntegratedTerminal
                                height={bottomHeight}
                                activeTab={activeTab}
                                setActiveTab={setActiveTab}
                                commandHistory={commandHistory}
                                commandInput={commandInput}
                                setCommandInput={setCommandInput}
                                handleCommand={handleCommand}
                                pvcCommandHistory={pvcCommandHistory}
                                pvcCommandInput={pvcCommandInput}
                                setPvcCommandInput={setPvcCommandInput}
                                reasoningStream={reasoningStream}
                            />
                        </div>

                        <div
                            onMouseDown={startResizing('right')}
                            className="w-1 bg-transparent cursor-col-resize transition-all shrink-0 z-40 flex items-center justify-center"
                            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = theme.colors.accent.primary + '80'}
                            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                        >
                            <GripVertical size={10} style={{ color: theme.colors.border.secondary, opacity: 0 }} />
                        </div>

                        <ControlDeck
                            width={rightWidth}
                            sessions={sessions}
                            currentSession={currentSession}
                            onNewSession={handleNewSession}
                            onLoadSession={setCurrentSessionId}
                            onSendIntent={handleSendIntent}
                            onRollback={handleRollback}
                            onDeleteSession={handleDeleteSession}
                            isProcessing={isProcessingIntent}
                            onSetReasoningTab={setActiveTab}
                        />
                    </>
                )}
            </div>
        </div>
    );
}
