import React, { useState, useEffect, useRef } from 'react';
import {
    History, MessageSquarePlus, X, Send, Loader2, Undo2, Trash2,
    SlidersHorizontal, ShieldCheck, Sparkles, ChevronDown, Pencil, Download
} from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';
import { useSettings } from '../../contexts/SettingsContext';
import { useDesign } from '../../contexts/DesignContext';
import { useSimulation } from '../../contexts/SimulationContext';

const Modal = ({ isOpen, onClose, title, content }) => {
    if (!isOpen) return null;
    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="w-[600px] max-h-[80vh] flex flex-col rounded-lg border shadow-2xl animate-in zoom-in-95 duration-200"
                style={{ backgroundColor: '#1A1A1A', borderColor: '#333' }}>
                <div className="flex items-center justify-between p-4 border-b" style={{ borderColor: '#333' }}>
                    <div className="font-bold text-sm tracking-wide text-white">{title}</div>
                    <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
                        <X size={16} />
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto p-6 text-xs font-mono leading-relaxed text-gray-300 whitespace-pre-wrap">
                    {content}
                </div>
                <div className="p-4 border-t flex justify-end" style={{ borderColor: '#333' }}>
                    <button
                        onClick={onClose}
                        className="px-4 py-2 rounded text-xs font-bold hover:bg-white/10 transition-colors text-white"
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
};

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import IntentInput from './IntentInput';

const ControlDeck = ({
    width,
    sessions = [],
    currentSession = null,
    onNewSession,
    onLoadSession,
    onSendIntent,
    onRollback,
    onDeleteSession,
    isProcessing = false,
    onSetReasoningTab,
    reasoningStream = [],
    // Customizable text strings
    placeholderText = "Describe hardware intent...",
    newBranchLabel = "New Branch",
    historyLabel = "Historical Branches",
    untitledLabel = "Untitled Intent",
    processingMessage = "Agents Negotiating Constraints...",
    aiLabel = "Gemini 2.5 Logic",
    // Input config
    minRows = 2,
    maxRows = 6
}) => {
    const { theme } = useTheme();
    const { aiModel, setAiModel } = useSettings();
    const { pendingPlanId, tabs, approvePlan, rejectPlan, setActiveTabId, reopenArtifact } = useDesign();
    const { sketchMode, setSketchMode, viewMode, setViewMode, reasoningStream: contextReasoningStream } = useSimulation();

    // Merge props stream with context stream
    const activeReasoningStream = [...reasoningStream, ...(contextReasoningStream || [])];
    const [showHistory, setShowHistory] = useState(false);
    const [showModelMenu, setShowModelMenu] = useState(false);
    const [showExportMenu, setShowExportMenu] = useState(false); // New Export Menu
    const [showViewMenu, setShowViewMenu] = useState(false); // Phase 10

    // Modal State
    const [inputText, setInputText] = useState('');
    const [modalOpen, setModalOpen] = useState(false);
    const [modalContent, setModalContent] = useState({ title: '', content: '' });
    const [textareaRows, setTextareaRows] = useState(minRows);
    const scrollRef = useRef(null);

    // Find pending plan tab
    const pendingPlanTab = tabs.find(t => t.artifactId === pendingPlanId);
    const artifactTabs = tabs.filter(t => t.type === 'artifact');

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [currentSession?.messages]);

    // Auto-grow textarea based on content
    useEffect(() => {
        const lineCount = inputText.split('\n').length;
        const newRows = Math.min(Math.max(lineCount, minRows), maxRows);
        setTextareaRows(newRows);
    }, [inputText, minRows, maxRows]);

    const handleSend = () => {
        if (!inputText.trim() || isProcessing) return;
        onSendIntent(inputText);
        setInputText('');

        // Auto-switch to reasoning tab when processing starts
        if (onSetReasoningTab) {
            setTimeout(() => onSetReasoningTab('reasoning'), 100);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    // Export Handler
    const handleExport = async (format) => {
        setShowExportMenu(false);
        // Get the latest geometry tree from context (assuming it's in reasoningStream or session)
        // For now, we'll try to find it in the last assistant message or query global state
        // In a real app, DesignContext should hold `currentGeometry`

        // Mock geometry fetch for now - we need DesignContext to expose this!
        // Assuming user has a valid design loaded.

        const tree = currentSession?.messages?.findLast(m => m.geometry_tree)?.geometry_tree || [];

        if (!tree || tree.length === 0) {
            alert("No geometry found to export.");
            return;
        }

        try {
            const res = await fetch('http://localhost:8000/api/geometry/compile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    geometry_tree: tree,
                    format: format,
                    mode: format === 'step' ? 'export' : 'standard'
                })
            });

            const data = await res.json();
            if (data.success && data.payload_base64) {
                // Trigger Download
                const binary = atob(data.payload_base64);
                const array = new Uint8Array(binary.length);
                for (let i = 0; i < binary.length; i++) array[i] = binary.charCodeAt(i);

                const blob = new Blob([array], { type: 'application/octet-stream' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `design_export.${format}`;
                a.click();
            } else {
                alert("Export Failed: " + (data.error || "Unknown Error"));
            }
        } catch (e) {
            console.error(e);
            alert("Export Error");
        }
    };

    if (width <= 0) return null;

    // Fallback if no session provided
    const activeSession = currentSession || {
        id: 'default',
        title: 'Default Session',
        messages: []
    };

    // Markdown Styling Components
    const MarkdownComponents = {
        // Headers - Dynamic based on theme but always bold/distinct
        h1: ({ node, ...props }) => <div className="text-[14px] font-bold mt-3 mb-2 pb-1 border-b" style={{ borderColor: theme.colors.border.secondary, color: theme.colors.text.primary }} {...props} />,
        h2: ({ node, ...props }) => <div className="text-[12px] font-bold mt-3 mb-1 uppercase tracking-wide" style={{ color: theme.colors.text.secondary }} {...props} />,
        h3: ({ node, ...props }) => <div className="text-[11px] font-bold mt-2 mb-1 underline decoration-current/30" style={{ color: theme.colors.text.tertiary }} {...props} />,

        // Lists - Proper indentation and bullets
        ul: ({ node, ...props }) => <ul className="list-disc pl-4 space-y-1 my-2 marker:opacity-50" {...props} />,
        ol: ({ node, ...props }) => <ol className="list-decimal pl-4 space-y-1 my-2 marker:opacity-50" {...props} />,
        li: ({ node, ...props }) => <li className="pl-1" {...props} />,

        // Code Blocks - Distinctive background
        code: ({ node, inline, className, children, ...props }) => {
            const match = /language-(\w+)/.exec(className || '');
            return !inline ? (
                <div className="my-2 rounded overflow-hidden border" style={{ backgroundColor: theme.colors.bg.tertiary, borderColor: theme.colors.border.secondary }}>
                    {match && (
                        <div className="px-2 py-1 text-[9px] font-mono border-b bg-white/5 opacity-70" style={{ borderColor: theme.colors.border.secondary }}>
                            {match[1].toUpperCase()}
                        </div>
                    )}
                    <code className="block p-2 text-[10px] font-mono overflow-x-auto whitespace-pre" {...props}>
                        {children}
                    </code>
                </div>
            ) : (
                <code className="px-1 py-0.5 rounded text-[10px] font-mono" style={{ backgroundColor: theme.colors.bg.tertiary }} {...props}>
                    {children}
                </code>
            );
        },

        // Paragraphs - Spacing
        p: ({ node, ...props }) => <p className="mb-2 last:mb-0 leading-relaxed" {...props} />,

        // Links
        a: ({ node, ...props }) => <a className="underline decoration-1 underline-offset-2 hover:opacity-80 transition-opacity" style={{ color: theme.colors.accent.primary }} {...props} />,

        // Blockquotes
        blockquote: ({ node, ...props }) => <blockquote className="border-l-2 pl-3 my-2 italic opacity-80" style={{ borderColor: theme.colors.accent.primary }} {...props} />
    };

    return (
        <aside
            className="h-full flex flex-col shrink-0 overflow-hidden relative"
            style={{
                width,
                backgroundColor: theme.colors.bg.secondary + '66',
                borderLeft: `1px solid ${theme.colors.border.primary}`
            }}
        >
            {/* Header with Session Management */}
            <div
                className="h-10 flex items-center justify-between px-3 shrink-0"
                style={{
                    backgroundColor: theme.colors.bg.primary,
                    borderBottom: `1px solid ${theme.colors.border.primary}`
                }}
            >
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setShowHistory(!showHistory)}
                        className="p-1.5 rounded-md transition-all"
                        style={{
                            color: showHistory ? theme.colors.accent.primary : theme.colors.text.muted,
                            backgroundColor: showHistory ? theme.colors.accent.primary + '1A' : 'transparent'
                        }}
                    >
                        <History size={16} />
                    </button>
                    <span
                        className="text-[10px] font-bold uppercase font-mono tracking-widest"
                        style={{ color: theme.colors.text.muted }}
                    >
                        Control Deck
                    </span>
                </div>
                {onNewSession && (
                    <button
                        onClick={onNewSession}
                        className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[9px] font-bold uppercase transition-all"
                        style={{
                            backgroundColor: theme.colors.accent.primary + '1A',
                            color: theme.colors.accent.primary,
                            border: `1px solid ${theme.colors.accent.primary}33`
                        }}
                    >
                        <MessageSquarePlus size={14} /> {newBranchLabel}
                    </button>
                )}
            </div>

            {/* History Sidebar Overlay */}
            {showHistory && (
                <div
                    className="absolute inset-y-10 left-0 w-full backdrop-blur-md z-[60] p-4 overflow-y-auto animate-in slide-in-from-left duration-200"
                    style={{
                        backgroundColor: theme.colors.bg.primary + 'F5',
                        borderRight: `1px solid ${theme.colors.border.primary}`
                    }}
                >
                    <div className="text-[10px] font-bold uppercase tracking-[0.2em] mb-4 flex justify-between" style={{ color: theme.colors.text.muted }}>
                        <span>{historyLabel}</span>
                        <X
                            size={14}
                            className="cursor-pointer transition-colors"
                            onClick={() => setShowHistory(false)}
                            style={{ color: theme.colors.text.muted }}
                        />
                    </div>
                    <div className="space-y-2">
                        {sessions.map(s => (
                            <div
                                key={s.id}
                                onClick={() => {
                                    if (onLoadSession) onLoadSession(s.id);
                                    setShowHistory(false);
                                }}
                                className="p-3 rounded cursor-pointer transition-all relative group"
                                style={{
                                    backgroundColor: s.id === activeSession.id
                                        ? theme.colors.accent.primary + '1A'
                                        : theme.colors.bg.secondary,
                                    border: `1px solid ${s.id === activeSession.id
                                        ? theme.colors.accent.primary + '4D'
                                        : theme.colors.border.primary}`
                                }}
                            >
                                <div className="text-[11px] font-bold truncate mb-1" style={{ color: theme.colors.text.primary }}>
                                    {s.title || untitledLabel}
                                </div>
                                <div className="flex items-center justify-between text-[9px] font-mono" style={{ color: theme.colors.text.muted }}>
                                    <span>{new Date(s.timestamp).toLocaleTimeString()}</span>
                                    <span>{s.messages?.length || 0} Nodes</span>
                                </div>
                                <button
                                    className="absolute top-2 right-2 p-1 opacity-0 group-hover:opacity-100 hover:text-red-400 transition-all"
                                    onClick={(e) => onDeleteSession && onDeleteSession(s.id, e)}
                                >
                                    <Trash2 size={12} />
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}


            {/* Plan Review Section - MOVED TO ABOVE INPUT */}
            {false && pendingPlanTab && (
                <div
                    className="mx-4 mt-4 p-4 rounded-lg border-2 animate-in slide-in-from-top duration-300"
                    style={{
                        backgroundColor: theme.colors.accent.primary + '0D',
                        borderColor: theme.colors.accent.primary + '66'
                    }}
                >
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                            <ShieldCheck size={14} style={{ color: theme.colors.accent.primary }} />
                            <span className="text-xs font-bold" style={{ color: theme.colors.text.primary }}>
                                Plan Review Required
                            </span>
                        </div>
                        <button
                            onClick={() => setActiveTabId(pendingPlanTab.id)}
                            className="text-[10px] px-2 py-1 rounded hover:bg-white/10 transition-colors"
                            style={{ color: theme.colors.accent.primary }}
                        >
                            Open Plan →
                        </button>
                    </div>

                    <div className="text-[10px] mb-3" style={{ color: theme.colors.text.muted }}>
                        {pendingPlanTab.name}
                    </div>

                    <div className="flex gap-2">
                        <button
                            onClick={async () => {
                                const result = await approvePlan(pendingPlanId, currentSession?.messages?.[0]?.text || '');
                                if (result) {
                                    // Execution started
                                    console.log('Plan approved, execution started');
                                }
                            }}
                            className="flex-1 py-2 rounded text-[10px] font-bold transition-all hover:scale-105"
                            style={{
                                backgroundColor: theme.colors.accent.primary,
                                color: '#000'
                            }}
                        >
                            ✓ Approve & Execute
                        </button>
                        <button
                            onClick={async () => {
                                await rejectPlan(pendingPlanId);
                            }}
                            className="px-4 py-2 rounded text-[10px] font-bold border transition-all hover:bg-red-500/20"
                            style={{
                                borderColor: theme.colors.border.primary,
                                color: theme.colors.text.muted
                            }}
                        >
                            ✗ Reject
                        </button>
                    </div>
                </div>
            )}

            {/* Chat / Intent Stream */}
            <div
                className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin"
                ref={scrollRef}
                style={{ backgroundColor: theme.colors.bg.primary + '40' }}
            >
                {activeSession.messages?.map((msg, i) => (
                    <div
                        key={i}
                        className={`flex flex-col gap-2 animate-in fade-in slide-in-from-bottom-2 ${msg.role === 'user' ? 'items-end' : 'items-start'
                            }`}
                    >
                        <div
                            onClick={(e) => {
                                // If message has artifactId, open that tab
                                if (msg.artifactId || (msg.text && msg.text.includes('Generated:'))) {
                                    e.stopPropagation();

                                    let artifactId = msg.artifactId;

                                    // Fallback: extract from title
                                    if (!artifactId && msg.text) {
                                        const titleMatch = msg.text.match(/Generated: (.+)/);
                                        if (titleMatch) {
                                            const title = titleMatch[1];
                                            const tab = tabs.find(t => t.type === 'artifact' && t.name === title);
                                            artifactId = tab?.artifactId;
                                        }
                                    }

                                    if (artifactId) {
                                        reopenArtifact(artifactId);
                                    }
                                }
                            }}
                            className={`max-w-[90%] p-3 rounded-md text-[11px] leading-relaxed font-mono border ${(msg.artifactId || (msg.text && msg.text.includes('Generated:'))) ? 'cursor-pointer hover:bg-white/5 transition-all' : ''}`}
                            style={{
                                backgroundColor: msg.role === 'user'
                                    ? theme.colors.bg.secondary
                                    : theme.colors.accent.primary + '0D',
                                borderColor: msg.role === 'user'
                                    ? theme.colors.border.primary
                                    : theme.colors.accent.primary + '33',
                                color: theme.colors.text.primary
                            }}
                        >
                            {msg.role === 'assistant' ? (
                                <ReactMarkdown remarkPlugins={[remarkGfm]} components={MarkdownComponents}>
                                    {msg.text}
                                </ReactMarkdown>
                            ) : (
                                msg.text
                            )}

                            {msg.role === 'user' && onRollback && (
                                <button
                                    onClick={() => {
                                        // Edit-Undo Logic:
                                        // 1. Populate input with this message's text
                                        setInputText(msg.text);
                                        // 2. Rollback to BEFORE this message (delete it from history) via excludeTarget=true
                                        if (onRollback) onRollback(i, true);
                                    }}
                                    className="mt-2 flex items-center gap-1 text-[9px] font-bold uppercase transition-colors"
                                    style={{ color: theme.colors.accent.primary }}
                                >
                                    <Undo2 size={12} /> Edit & Retry
                                </button>
                            )}
                        </div>
                        {msg.agent && (
                            <div className="flex items-center gap-1.5 px-1">
                                <div className="w-1 h-1 rounded-full" style={{ backgroundColor: theme.colors.accent.primary }} />
                                <span className="text-[9px] font-bold uppercase tracking-tighter" style={{ color: theme.colors.accent.primary }}>
                                    {msg.agent}
                                </span>
                            </div>
                        )}

                        {/* Artifact Card (Approval Workflow) */}
                        {msg.type === 'artifact' && (
                            <div
                                className="mt-1 w-full p-3 rounded-lg border flex flex-col gap-2 animate-in zoom-in-95 duration-300"
                                style={{
                                    backgroundColor: theme.colors.bg.primary,
                                    borderColor: theme.colors.border.secondary
                                }}
                            >
                                <div className="flex items-start justify-between">
                                    <div className="flex items-center gap-2">
                                        <div className="p-1.5 rounded bg-gray-800 text-white">
                                            <span className="text-xs">📄</span>
                                        </div>
                                        <div>
                                            <div className="text-[11px] font-bold text-white">{msg.title}</div>
                                            <div className="text-[9px] text-gray-400">Ready for review</div>
                                        </div>
                                    </div>
                                    <div className="text-[8px] font-mono px-1.5 py-0.5 rounded border border-gray-700 text-gray-400">
                                        MD
                                    </div>
                                </div>

                                <div className="h-px w-full bg-gray-800" />

                                <div className="flex gap-2">
                                    <button
                                        className="flex-1 py-1.5 rounded text-[10px] font-bold border border-gray-700 hover:bg-gray-800 transition-colors text-gray-300"
                                        onClick={() => {
                                            setModalContent({ title: msg.title, content: msg.content });
                                            setModalOpen(true);
                                        }}
                                    >
                                        Open
                                    </button>
                                    <button
                                        className="flex-1 py-1.5 rounded text-[10px] font-bold hover:brightness-110 transition-all text-black"
                                        style={{ backgroundColor: theme.colors.accent.primary }}
                                        onClick={async () => {
                                            // Call Approve API
                                            try {
                                                const res = await fetch('http://localhost:8000/api/approve', {
                                                    method: 'POST',
                                                    headers: { 'Content-Type': 'application/json' },
                                                    body: JSON.stringify({
                                                        plan_id: msg.id,
                                                        user_intent: "resume" // Simplified for now
                                                    })
                                                });
                                                const data = await res.json();
                                                // Handle success (maybe add a system message)
                                                console.log("Approved:", data);
                                            } catch (e) {
                                                console.error("Approval failed", e);
                                            }
                                        }}
                                    >
                                        Proceed
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                ))}

                {/* Generated Documents / Artifacts */}
                {artifactTabs.length > 0 && (
                    <div className="space-y-2">
                        <div className="text-[9px] font-mono font-bold opacity-50" style={{ color: theme.colors.text.muted }}>
                            GENERATED DOCUMENTS
                        </div>
                        {artifactTabs.map((tab) => (
                            <div
                                key={tab.id}
                                onClick={() => tab.artifactId && reopenArtifact(tab.artifactId)}
                                className="flex items-center justify-between p-2 rounded border cursor-pointer hover:bg-white/5 transition-all"
                                style={{
                                    backgroundColor: theme.colors.bg.secondary,
                                    borderColor: theme.colors.border.primary
                                }}
                            >
                                <span className="text-[10px] font-mono truncate flex-1" style={{ color: theme.colors.text.primary }}>
                                    📄 {tab.name}
                                </span>
                                <button
                                    className="text-[9px] px-2 py-0.5 rounded hover:bg-white/10"
                                    style={{ color: theme.colors.accent.primary }}
                                >
                                    Open →
                                </button>
                            </div>
                        ))}
                    </div>
                )}

                {/* Reasoning Stream / Intent Logs */}
                {activeReasoningStream.length > 0 && (
                    <div className="space-y-1 opacity-70">
                        {activeReasoningStream.map((log, i) => (
                            <div
                                key={i}
                                className="flex items-start gap-2 text-[9px] font-mono p-2 rounded"
                                style={{
                                    backgroundColor: theme.colors.bg.secondary,
                                    color: theme.colors.text.muted
                                }}
                            >
                                <span style={{ color: theme.colors.accent.primary }}>{log.time}</span>
                                <span className="font-bold" style={{
                                    color: log.agent === 'MANUFACTURING' ? '#f59e0b' :
                                        log.agent === 'PHYSICS' ? '#3b82f6' :
                                            theme.colors.text.tertiary
                                }}>
                                    [{log.agent}]
                                </span>
                                <span className="flex-1">{log.thought}</span>
                            </div>
                        ))}
                    </div>
                )}

                {isProcessing && (
                    <div className="flex flex-col gap-2 items-start opacity-50 italic font-mono text-[10px] animate-pulse" style={{ color: theme.colors.accent.primary }}>
                        <div className="flex items-center gap-2">
                            <Loader2 size={14} className="animate-spin" />
                            <span>{processingMessage}</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Compact Plan Review - Above Input */}
            {pendingPlanTab && (
                <div
                    className="px-3 pb-2 shrink-0"
                    style={{ backgroundColor: theme.colors.bg.primary }}
                >
                    <div
                        className="p-2 rounded border flex items-center justify-between gap-2 text-[9px]"
                        style={{
                            backgroundColor: theme.colors.accent.primary + '0A',
                            borderColor: theme.colors.accent.primary + '40'
                        }}
                    >
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                            <ShieldCheck size={11} style={{ color: theme.colors.accent.primary }} />
                            <span className="font-bold truncate" style={{ color: theme.colors.text.primary }}>
                                {pendingPlanTab.name}
                            </span>
                        </div>
                        <div className="flex items-center gap-1 shrink-0">
                            <button
                                onClick={() => setActiveTabId(pendingPlanTab.id)}
                                className="px-2 py-1 rounded hover:bg-white/10 transition-colors font-mono"
                                style={{ color: theme.colors.accent.primary }}
                            >
                                View
                            </button>
                            <button
                                onClick={async () => {
                                    const result = await approvePlan(pendingPlanId, currentSession?.messages?.[0]?.text || '');
                                    if (result) console.log('Plan approved');
                                }}
                                className="px-2 py-1 rounded font-bold transition-all"
                                style={{
                                    backgroundColor: theme.colors.accent.primary,
                                    color: '#000'
                                }}
                            >
                                ✓ Approve
                            </button>
                            <button
                                onClick={async () => await rejectPlan(pendingPlanId)}
                                className="px-2 py-1 rounded hover:bg-red-500/20 transition-all"
                                style={{ color: theme.colors.text.muted }}
                            >
                                ✗
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Input Area */}
            <IntentInput
                value={inputText}
                onChange={setInputText}
                onSend={handleSend}
                isProcessing={isProcessing}
                aiModel={aiModel}
                setAiModel={setAiModel}
                sketchMode={sketchMode}
                setSketchMode={setSketchMode}
            />
            <Modal
                isOpen={modalOpen}
                onClose={() => setModalOpen(false)}
                title={modalContent.title}
                content={modalContent.content}
            />
        </aside>
    );
};

export default ControlDeck;
