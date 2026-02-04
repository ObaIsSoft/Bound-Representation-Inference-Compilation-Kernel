import React, { useState, useRef, useEffect } from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { usePanel } from '../../contexts/PanelContext';
import DraggablePanel from '../shared/DraggablePanel';
import { X, ChevronDown, ChevronUp, Box, FileText, CheckCircle2, History, List } from 'lucide-react';
import ChatMessage from '../shared/ChatMessage';
import MarkdownViewer from '../viewers/MarkdownViewer';
import GenomeViewer from '../viewers/GenomeViewer';

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

const ArtifactBlock = ({ artifact, theme, onView }) => {
    return (
        <div
            onClick={() => onView({ id: artifact.id, name: artifact.title, type: artifact.type })}
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
        setIsHistoryModalOpen,
        activeTab,
        activeArtifact,
        viewArtifact,
        openTabs
    } = usePanel();
    const scrollRef = useRef(null);

    // Auto-scroll to bottom when new messages arrive (only if in chat tab)
    useEffect(() => {
        if (scrollRef.current && activeTab === 'chat') {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [activeSession?.history, activeTab]);

    // Find active tab metadata
    const activeTabData = openTabs.find(t => t.id === activeTab);

    // Render Content Based on Active Tab
    const renderContent = () => {
        if (activeTab === 'chat' || !activeTab) {
            return (
                <div className="flex flex-col min-h-full p-4">
                    <div className="flex-grow" />
                    <div className="flex flex-col gap-6 py-4">
                        {activeSession?.history.length === 0 ? (
                            <div className="text-center opacity-30 text-xs mb-10 italic" style={{ color: theme.colors.text.secondary }}>
                                No activity recorded. Send a prompt to begin.
                            </div>
                        ) : (
                            activeSession?.history.map((msg, idx) => {
                                if (msg.role === 'thought') {
                                    return <ThoughtBlock key={idx} thought={msg.content} theme={theme} />;
                                }
                                if (msg.role === 'artifact') {
                                    return <ArtifactBlock key={idx} artifact={msg} theme={theme} onView={viewArtifact} />;
                                }
                                return <ChatMessage key={idx} msg={msg} />;
                            })
                        )}
                    </div>
                </div>
            );
        }

        // Artifact/File Rendering
        if (activeTab.endsWith('.md') || activeTabData?.type === 'document') {
            return <MarkdownViewer path={activeTab} fileName={activeTabData?.name || 'Document'} />;
        }

        if (activeTab.includes('genome') || activeTabData?.type === 'artifact') {
            return <GenomeViewer
                path={activeTab}
                fileName={activeTabData?.name || 'Design'}
                modelId={activeTabData?.metadata?.model_id}
            />;
        }

        return (
            <div className="h-full flex items-center justify-center opacity-30 text-xs italic">
                Unsupported file type: {activeTab}
            </div>
        );
    };

    // Simplified Header Content
    const headerContent = (
        <div className="flex items-center justify-between w-full">
            <div className="flex items-center gap-3">
                <div className="p-1.5 rounded-md bg-white/5">
                    <History size={14} color={theme.colors.accent.primary} />
                </div>
                <div>
                    <div className="text-[10px] font-bold uppercase tracking-widest leading-none" style={{ color: theme.colors.text.primary }}>
                        {activeTabData?.name || activeSession?.title || 'Session'}
                    </div>
                    <div className="text-[9px] opacity-40 uppercase tracking-tighter font-black mt-0.5" style={{ color: theme.colors.text.secondary }}>
                        {activeTab === 'chat' ? (activeSession?.branchName || 'Main') + ' Branch' : activeTabData?.type || 'File'}
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
                className="h-full flex flex-col overflow-y-auto custom-scrollbar relative"
                style={{ backgroundColor: theme.colors.bg.secondary + '80' }} // Slight transparency
            >
                {renderContent()}
            </div>
        </DraggablePanel>
    );
};

export default MainPanel;
