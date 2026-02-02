import React, { useState, useRef, useEffect } from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { usePanel } from '../../contexts/PanelContext';
import DraggablePanel from '../shared/DraggablePanel';
import { X, ChevronDown, ChevronUp, Box, FileText, CheckCircle2, History, List } from 'lucide-react';

const ThoughtBlock = ({ thought, theme }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    return (
        <div className="my-2 border-l-2 pl-4 py-1" style={{ borderColor: theme.colors.accent.primary + '40' }}>
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-wider opacity-60 hover:opacity-100 transition-opacity"
                style={{ color: theme.colors.text.secondary }}
            >
                {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                Internal Reasoning
            </button>
            {isExpanded && (
                <div className="mt-2 text-xs leading-relaxed opacity-80 italic" style={{ color: theme.colors.text.secondary }}>
                    {thought}
                </div>
            )}
        </div>
    );
};

const ArtifactBlock = ({ artifact, theme }) => {
    return (
        <div
            className="my-4 rounded-xl border p-4 bg-black/20 shadow-lg group hover:border-white/20 transition-all cursor-pointer"
            style={{ borderColor: theme.colors.border.secondary }}
        >
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <div className="p-2 rounded-lg bg-white/5">
                        <Box size={18} color={theme.colors.accent.primary} />
                    </div>
                    <div>
                        <div className="text-xs font-bold" style={{ color: theme.colors.text.primary }}>{artifact.title}</div>
                        <div className="text-[10px] opacity-50 uppercase tracking-widest font-black" style={{ color: theme.colors.text.secondary }}>{artifact.type}</div>
                    </div>
                </div>
                <CheckCircle2 size={16} className="text-green-500/50" />
            </div>
            <div className="text-xs opacity-70 line-clamp-2" style={{ color: theme.colors.text.secondary }}>
                {artifact.summary}
            </div>
            <div className="mt-4 flex items-center justify-between pt-3 border-t border-white/5">
                <span className="text-[10px] uppercase font-bold tracking-tighter opacity-40">Artifact Generated</span>
                <button className="text-[10px] font-bold text-blue-400 hover:text-blue-300 transition-colors">VIEW DETAILS →</button>
            </div>
        </div>
    );
};

const MainPanel = () => {
    const { theme } = useTheme();
    const {
        activeSession,
        togglePanel,
        PANEL_IDS,
        setIsHistoryModalOpen
    } = usePanel();
    const scrollRef = useRef(null);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [activeSession.history]);

    // Simplified Header Content
    const headerContent = (
        <div className="flex items-center justify-between w-full">
            <div className="flex items-center gap-3">
                <div className="p-1.5 rounded-md bg-white/5">
                    <History size={14} color={theme.colors.accent.primary} />
                </div>
                <div>
                    <div className="text-[10px] font-bold uppercase tracking-widest leading-none" style={{ color: theme.colors.text.primary }}>
                        {activeSession.title || 'Session History'}
                    </div>
                    <div className="text-[9px] opacity-40 uppercase tracking-tighter font-black mt-0.5" style={{ color: theme.colors.text.secondary }}>
                        {activeSession.branchName || 'Main'} Branch
                    </div>
                </div>
            </div>

            <div className="flex items-center gap-2">
                <button
                    onClick={() => setIsHistoryModalOpen(true)}
                    className="p-1.5 hover:bg-white/10 rounded transition-colors group"
                    title="Conversation List"
                >
                    <List size={16} color={theme.colors.text.secondary} />
                </button>
                <button
                    onClick={() => togglePanel(PANEL_IDS.MAIN)}
                    className="p-1 hover:bg-white/10 rounded transition-colors"
                >
                    <X size={16} color={theme.colors.text.secondary} />
                </button>
            </div>
        </div>
    );

    return (
        <DraggablePanel
            id={PANEL_IDS.MAIN}
            headerContent={headerContent}
            className="pointer-events-auto"
        >
            <div
                ref={scrollRef}
                className="h-full flex flex-col overflow-y-auto p-4 custom-scrollbar"
                style={{ backgroundColor: theme.colors.bg.secondary + '80' }} // Slight transparency
            >
                <div className="flex flex-col min-h-full">
                    {/* Spacer to push content to bottom when list is short */}
                    <div className="flex-grow" />

                    <div className="flex flex-col gap-6 py-4">
                        {activeSession.history.length === 0 ? (
                            <div className="text-center opacity-30 text-xs mb-10 italic" style={{ color: theme.colors.text.secondary }}>
                                No activity recorded. Send a prompt to begin.
                            </div>
                        ) : (
                            activeSession.history.map((msg) => {
                                if (msg.role === 'thought') {
                                    return <ThoughtBlock key={msg.id} thought={msg.content} theme={theme} />;
                                }
                                if (msg.role === 'artifact') {
                                    return <ArtifactBlock key={msg.id} artifact={msg} theme={theme} />;
                                }

                                return (
                                    <div
                                        key={msg.id}
                                        className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
                                    >
                                        <div
                                            className={`p-3 rounded-xl text-sm leading-relaxed shadow-sm ${msg.role === 'user'
                                                ? 'bg-white/10 rounded-tr-none'
                                                : 'bg-black/20 rounded-tl-none border border-white/5'
                                                }`}
                                            style={{ color: theme.colors.text.primary, maxWidth: '90%' }}
                                        >
                                            {msg.content}
                                        </div>
                                    </div>
                                );
                            })
                        )}
                    </div>
                </div>
            </div>
        </DraggablePanel>
    );
};

export default MainPanel;
