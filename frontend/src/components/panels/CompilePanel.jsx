import React, { useState, useEffect, useRef } from 'react';
import {
    Play, RotateCw, FileCode, Cpu, Layers, Terminal, AlertCircle, CheckCircle,
    Settings, ChevronDown, ChevronRight, Hash, Database
} from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';

const MOCK_AST = {
    type: 'Program',
    body: [
        { type: 'Import', module: 'std.geometry' },
        {
            type: 'VariableDeclaration',
            name: 'main_body',
            value: {
                type: 'Extrude',
                args: ['sketch_1', '25mm']
            }
        },
        {
            type: 'FunctionCall',
            name: 'fillet',
            args: ['main_body.edges', '2mm']
        }
    ]
};

const MOCK_ASM = `
; TARGET: BRICK-RISC-V64
; OPT: O2

_start:
    LOAD  R1, [0x4000]  ; Load sketch_1
    MOV   R2, #25       ; Height
    CALL  EXTRUDE       ; Extrude(R1, R2) -> R0
    STORE R0, [0x5000]  ; Store main_body

    LOAD  R1, [0x5000]  ; Load main_body
    CALL  GET_EDGES     ; -> R1
    MOV   R2, #2        ; Radius
    CALL  FILLET        ; Fillet(R1, R2)
    
    RET
`;

const CompilePanel = ({ width }) => {
    const { theme } = useTheme();
    const [activeTab, setActiveTab] = useState('ast');
    const [isBuilding, setIsBuilding] = useState(false);
    const [buildStatus, setBuildStatus] = useState('idle'); // idle, building, success, error
    const [logs, setLogs] = useState([
        { type: 'info', msg: 'KCL Compiler v2.4.0 ready.' },
        { type: 'info', msg: 'Target: BRICK-RISC-V64' }
    ]);
    const logsEndRef = useRef(null);

    const scrollToBottom = () => {
        logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [logs]);

    const handleBuild = () => {
        if (isBuilding) return;
        setIsBuilding(true);
        setBuildStatus('building');
        setLogs([{ type: 'info', msg: 'Starting build process...' }]);

        const steps = [
            { msg: 'Lexing source code...', delay: 500 },
            { msg: 'Parsing AST...', delay: 1200 },
            { msg: 'Optimizing (Level O2)...', delay: 2000 },
            { msg: 'Generating IR...', delay: 2800 },
            { msg: 'Linking std.geometry...', delay: 3500 },
            { msg: 'Build SUCCESS. (3.6s)', delay: 3800, type: 'success' }
        ];

        steps.forEach(step => {
            setTimeout(() => {
                setLogs(prev => [...prev, { type: step.type || 'info', msg: step.msg }]);
                if (step.type === 'success') {
                    setIsBuilding(false);
                    setBuildStatus('success');
                }
            }, step.delay);
        });
    };

    const renderAST = (node, depth = 0) => {
        return (
            <div style={{ marginLeft: depth * 12 }} className="font-mono text-xs">
                {Array.isArray(node) ? (
                    node.map((child, i) => <div key={i}>{renderAST(child, depth)}</div>)
                ) : typeof node === 'object' ? (
                    <div>
                        <span style={{ color: theme.colors.accent.secondary }}>{node.type}</span>
                        {node.name && <span style={{ color: theme.colors.text.secondary }}> {node.name}</span>}
                        {node.module && <span style={{ color: theme.colors.status.success }}> '{node.module}'</span>}
                        {node.value && <div>{renderAST(node.value, depth + 1)}</div>}
                        {node.body && <div>{renderAST(node.body, depth + 1)}</div>}
                        {node.args && (
                            <span style={{ color: theme.colors.text.muted }}> ({node.args.map(a => `'${a}'`).join(', ')})</span>
                        )}
                    </div>
                ) : null}
            </div>
        );
    };

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
                        <span className="font-mono" style={{ color: theme.colors.status.warning }}>RISC-V64</span>
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
                    {isBuilding ? 'BUILDING...' : 'BUILD KERNEL'}
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
                        {renderAST(MOCK_AST)}
                    </div>
                )}
                {activeTab === 'asm' && (
                    <pre className="font-mono text-xs whitespace-pre leading-relaxed" style={{ color: theme.colors.status.info }}>
                        {buildStatus === 'idle' ? <span className="italic" style={{ color: theme.colors.text.muted }}>; Waiting for build...</span> : MOCK_ASM}
                    </pre>
                )}
                {activeTab === 'symbols' && (
                    <div className="font-mono text-xs space-y-2">
                        <div className="flex text-[10px] pb-1 border-b" style={{ borderColor: theme.colors.border.secondary, color: theme.colors.text.muted }}>
                            <span className="w-16">ADDR</span>
                            <span className="flex-1">SYMBOL</span>
                            <span className="w-12 text-right">SIZE</span>
                        </div>
                        {buildStatus !== 'idle' && (
                            <>
                                <div className="flex" style={{ color: theme.colors.text.primary }}>
                                    <span className="w-16" style={{ color: theme.colors.status.info }}>0x4000</span>
                                    <span className="flex-1">sketch_1</span>
                                    <span className="w-12 text-right" style={{ color: theme.colors.text.muted }}>128B</span>
                                </div>
                                <div className="flex" style={{ color: theme.colors.text.primary }}>
                                    <span className="w-16" style={{ color: theme.colors.status.info }}>0x5000</span>
                                    <span className="flex-1">main_body</span>
                                    <span className="w-12 text-right" style={{ color: theme.colors.text.muted }}>2KB</span>
                                </div>
                            </>
                        )}
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
