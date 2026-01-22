import React, { useState, useEffect, useRef } from 'react';
import {
    Play, RotateCw, FileCode, Cpu, Layers, Terminal, AlertCircle, CheckCircle,
    Settings, ChevronDown, ChevronRight, Hash, Database
} from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';
import { useSimulation } from '../../contexts/SimulationContext';

const CompilePanel = ({ width }) => {
    const { theme } = useTheme();
    const { compilationResult, setCompilationResult, setKclCode } = useSimulation();

    const [activeTab, setActiveTab] = useState('ast');
    const [isBuilding, setIsBuilding] = useState(false);
    const [buildStatus, setBuildStatus] = useState('idle'); // idle, building, success, error
    const [logs, setLogs] = useState([
        { type: 'info', msg: 'KCL Compiler v2.4.0 ready.' },
        { type: 'info', msg: 'Target: BRICK-RISC-V64 (GLSL)' }
    ]);
    const logsEndRef = useRef(null);

    // Sync status with context updates
    useEffect(() => {
        if (compilationResult) {
            setBuildStatus('success');
            setLogs(prev => [...prev, { type: 'success', msg: 'Updated from Context.' }]);
        }
    }, [compilationResult]);

    const scrollToBottom = () => {
        logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [logs]);

    const handleBuild = async () => {
        if (isBuilding) return;
        setIsBuilding(true);
        setBuildStatus('building');
        setLogs([{ type: 'info', msg: 'Starting build process...' }]);

        try {
            setLogs(prev => [...prev, { type: 'info', msg: 'Sending User Intent to Logic Core...' }]);

            // Invoke Backend
            const res = await fetch('http://localhost:8000/api/compile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_intent: "Recompile current state", // In a real flow, we'd grab from Editor or Intent Input
                    project_id: "sim-1"
                })
            });

            const data = await res.json();

            if (data.success) {
                setCompilationResult(data);
                if (data.kcl_code) setKclCode(data.kcl_code);

                setLogs(prev => [
                    ...prev,
                    { type: 'info', msg: 'Geometry Graph [OK]' },
                    { type: 'info', msg: 'Physics Validation [OK]' },
                    { type: 'success', msg: 'Build SUCCESS.' }
                ]);
                setBuildStatus('success');
            } else {
                throw new Error("Compilation returned success:false");
            }

        } catch (err) {
            setLogs(prev => [...prev, { type: 'error', msg: `Build Failed: ${err.message}` }]);
            setBuildStatus('error');
        } finally {
            setIsBuilding(false);
        }
    };

    // Recursive render of JSON Geometry Tree (AST)
    const renderAST = (node, depth = 0) => {
        if (!node) return null;

        // Handle list of nodes (top level)
        if (Array.isArray(node)) {
            return node.map((child, i) => <div key={i}>{renderAST(child, depth)}</div>);
        }

        if (typeof node === 'object') {
            const isLeaf = !node.children && !node.body;
            return (
                <div style={{ marginLeft: depth * 12 }} className="font-mono text-xs my-0.5">
                    <div>
                        <span style={{ color: theme.colors.accent.secondary }}>{node.type || 'Node'}</span>
                        {node.id && <span style={{ color: theme.colors.text.secondary }}> #{node.id}</span>}
                        {node.operation && <span style={{ color: theme.colors.status.success }}> {node.operation}</span>}

                        {/* Params */}
                        {node.params && (
                            <span style={{ color: theme.colors.text.muted }}>
                                ({Object.entries(node.params).map(([k, v]) => `${k}=${v}`).join(', ')})
                            </span>
                        )}

                        {/* Recursion */}
                        {node.children && <div>{renderAST(node.children, depth + 1)}</div>}
                    </div>
                </div>
            );
        }
        return null;
    };

    // Data extraction
    const astData = compilationResult?.geometry_tree || [];
    const asmData = compilationResult?.glsl_code || "// No GLSL Assembly generated yet.";

    return (
        <div
            className="flex flex-col min-w-0 h-full border-r"
            style={{
                width: width,
                borderColor: theme.colors.border.primary,
                backgroundColor: theme.colors.bg.primary
            }}
        >
            <PanelHeader title="Compile ISA" icon={Cpu} />

            {/* Controls */}
            <div className="p-3 border-b space-y-3" style={{ borderColor: theme.colors.border.primary }}>
                <div className="flex gap-2">
                    <div className="flex-1 rounded px-2 py-1 text-xs flex items-center justify-between" style={{ backgroundColor: theme.colors.bg.tertiary }}>
                        <span style={{ color: theme.colors.text.muted }}>TARGET</span>
                        <span className="font-mono" style={{ color: theme.colors.status.warning }}>GLSL-SDF</span>
                    </div>
                    <div className="flex-1 rounded px-2 py-1 text-xs flex items-center justify-between" style={{ backgroundColor: theme.colors.bg.tertiary }}>
                        <span style={{ color: theme.colors.text.muted }}>OPT</span>
                        <span className="font-mono" style={{ color: theme.colors.status.success }}>O2</span>
                    </div>
                </div>
                <button
                    onClick={handleBuild}
                    disabled={isBuilding}
                    className="w-full py-2 rounded flex items-center justify-center gap-2 text-xs font-bold transition-all"
                    style={{
                        backgroundColor: isBuilding ? theme.colors.bg.tertiary : theme.colors.accent.primary,
                        color: isBuilding ? theme.colors.text.muted : theme.colors.bg.primary,
                        cursor: isBuilding ? 'wait' : 'pointer'
                    }}
                >
                    {isBuilding ? <RotateCw size={14} className="animate-spin" /> : <Play size={14} fill="currentColor" />}
                    {isBuilding ? 'COMPILING...' : 'REBUILD KERNEL'}
                </button>
            </div>

            {/* Inspection Tabs */}
            <div className="flex border-b" style={{ borderColor: theme.colors.border.primary }}>
                {['ast', 'asm', 'symbols'].map(tab => (
                    <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className="flex-1 py-2 text-[10px] font-bold uppercase tracking-wider border-b-2 transition-colors"
                        style={{
                            borderColor: activeTab === tab ? theme.colors.accent.primary : 'transparent',
                            color: activeTab === tab ? theme.colors.text.primary : theme.colors.text.muted,
                            backgroundColor: activeTab === tab ? theme.colors.bg.tertiary : 'transparent'
                        }}
                    >
                        {tab}
                    </button>
                ))}
            </div>

            {/* Main Content Area */}
            <div className="flex-1 overflow-auto p-4" style={{ backgroundColor: theme.colors.bg.elevated }}>
                {activeTab === 'ast' && (
                    <div className="space-y-1">
                        {astData.length > 0 ? renderAST(astData) : <span className="italic text-xs text-gray-500">Tree Empty</span>}
                    </div>
                )}
                {activeTab === 'asm' && (
                    <pre className="font-mono text-[9px] whitespace-pre leading-relaxed" style={{ color: theme.colors.status.info }}>
                        {asmData}
                    </pre>
                )}
                {activeTab === 'symbols' && (
                    <div className="font-mono text-xs space-y-2">
                        <div className="flex text-[10px] pb-1 border-b" style={{ borderColor: theme.colors.border.secondary, color: theme.colors.text.muted }}>
                            <span className="w-16">ID</span>
                            <span className="flex-1">TYPE</span>
                        </div>
                        {Array.isArray(astData) && astData.map((node, i) => (
                            <div key={i} className="flex" style={{ color: theme.colors.text.primary }}>
                                <span className="w-16" style={{ color: theme.colors.status.info }}>{node.id || i}</span>
                                <span className="flex-1">{node.type}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Terminal Log */}
            <div className="h-1/3 border-t flex flex-col min-h-[150px]" style={{ borderColor: theme.colors.border.primary, backgroundColor: theme.colors.bg.secondary }}>
                <div className="px-3 py-1.5 border-b flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider" style={{ borderColor: theme.colors.border.primary, color: theme.colors.text.muted }}>
                    <Terminal size={10} />
                    <span>Compiler Log</span>
                </div>
                <div className="flex-1 p-3 overflow-y-auto font-mono text-[10px] space-y-1">
                    {logs.map((log, i) => (
                        <div key={i} className="flex gap-2" style={{
                            color: log.type === 'success' ? theme.colors.status.success :
                                log.type === 'error' ? theme.colors.status.error : theme.colors.text.secondary
                        }}>
                            <span>{'>'}</span>
                            <span>{log.msg}</span>
                        </div>
                    ))}
                    <div ref={logsEndRef} />
                </div>
            </div>

            {/* Status Footer */}
            <div className="h-8 border-t flex items-center px-4 gap-4 text-[10px] font-mono shrink-0"
                style={{
                    backgroundColor: theme.colors.bg.secondary,
                    borderColor: theme.colors.border.primary,
                    color: theme.colors.text.tertiary
                }}
            >
                <div className="flex items-center gap-2">
                    {buildStatus === 'building' && <RotateCw size={12} className="animate-spin" style={{ color: theme.colors.status.warning }} />}
                    {buildStatus === 'success' && <CheckCircle size={12} style={{ color: theme.colors.status.success }} />}
                    {buildStatus === 'error' && <AlertCircle size={12} style={{ color: theme.colors.status.error }} />}
                    {buildStatus === 'idle' && <Terminal size={12} />}
                    <span className="uppercase">STATUS: {buildStatus}</span>
                </div>
            </div>
        </div>
    );
};

export default CompilePanel;
