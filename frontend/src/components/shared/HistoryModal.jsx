import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePanel } from '../../contexts/PanelContext';
import { useTheme } from '../../contexts/ThemeContext';
import { X, Search, Trash2, Clock, ShieldAlert, History, Plus, GitMerge } from 'lucide-react';

const HistoryModal = () => {
    const { theme } = useTheme();
    const {
        isHistoryModalOpen,
        setIsHistoryModalOpen,
        sessions,
        activeSessionId,
        switchSession,
        deleteSession,
        mergeSession,
        createNewSession
    } = usePanel();
    const [searchQuery, setSearchQuery] = useState('');
    const [expandedSections, setExpandedSections] = useState({});

    if (!isHistoryModalOpen) return null;

    const toggleSection = (sectionId) => {
        setExpandedSections(prev => ({
            ...prev,
            [sectionId]: !prev[sectionId]
        }));
    };

    const filteredSessions = sessions.filter(s =>
        s.title.toLowerCase().includes(searchQuery.toLowerCase())
    );

    // Grouping sessions
    const groups = {
        current: filteredSessions.filter(s => s.status === 'active'),
        blocked: filteredSessions.filter(s => s.status === 'blocked'),
        recent: filteredSessions.filter(s => s.status === 'recent'),
        other: filteredSessions.filter(s => s.status === 'other')
    };

    const renderSessionItem = (session, showWorkspace = false) => {
        const isActive = session.id === activeSessionId;

        // Simple relative time calculation (mock for now)
        const getRelativeTime = (isoString) => {
            const diff = Date.now() - new Date(isoString).getTime();
            if (diff < 3600000) return `${Math.floor(diff / 60000)} min ago`;
            if (diff < 86400000) return `${Math.floor(diff / 3600000)} hrs ago`;
            if (diff < 604800000) return `${Math.floor(diff / 86400000)} days ago`;
            if (diff < 1209600000) return `1 wk ago`;
            return `${Math.floor(diff / 604800000)} wks ago`;
        };

        return (
            <div
                key={session.id}
                onClick={() => switchSession(session.id)}
                className={`group flex items-center justify-between p-3 rounded-lg cursor-pointer transition-all ${isActive ? 'bg-white/10' : 'hover:bg-white/10'
                    }`}
                style={{
                    backgroundColor: isActive ? theme.colors.bg.tertiary : 'transparent',
                }}
            >
                <div className="flex items-center gap-3">
                    <div className="flex-1">
                        <div className="flex items-center gap-2">
                            <span className="text-sm font-medium" style={{ color: theme.colors.text.primary }}>
                                {session.title}
                            </span>
                            {session.status === 'blocked' && <div className="w-1.5 h-1.5 rounded-full bg-orange-500" />}
                            {showWorkspace && session.branchName && (
                                <span className="text-[10px] opacity-30 font-medium px-2 py-0.5 rounded-md" style={{ color: theme.colors.text.secondary }}>
                                    {session.branchName}
                                </span>
                            )}
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    <span className="text-[10px] opacity-40 font-medium" style={{ color: theme.colors.text.secondary }}>
                        {getRelativeTime(session.lastModified)}
                    </span>
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            mergeSession(session.id);
                        }}
                        className={`p-1.5 rounded hover:bg-white/10 transition-all flex items-center gap-1.5 ${session.parentId ? 'opacity-40 group-hover:opacity-100' : 'hidden'
                            }`}
                        title="Merge into Parent"
                    >
                        <GitMerge size={14} style={{ color: theme.colors.accent.primary }} />
                        <span className="text-[9px] font-bold uppercase tracking-tighter">Merge</span>
                    </button>
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            deleteSession(session.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1.5 rounded hover:bg-red-500/20 transition-all"
                    >
                        <Trash2 size={14} className="text-red-400" />
                    </button>
                </div>
            </div>
        );
    };

    const renderGroup = (id, label, sessionsList, showWorkspace = false) => {
        if (sessionsList.length === 0) return null;

        const limit = 3;
        const isExpanded = expandedSections[id];
        const visibleSessions = isExpanded ? sessionsList : sessionsList.slice(0, limit);
        const remainingCount = sessionsList.length - limit;

        return (
            <div className="mb-6">
                <h3 className="px-3 mb-2 text-[10px] uppercase tracking-[0.05em] font-bold opacity-30" style={{ color: theme.colors.text.secondary }}>
                    {label}
                </h3>
                {visibleSessions.map(s => renderSessionItem(s, showWorkspace))}
                {!isExpanded && remainingCount > 0 && (
                    <button
                        onClick={() => toggleSection(id)}
                        className="w-full text-left px-3 py-2 text-[11px] font-medium opacity-40 hover:opacity-100 transition-opacity"
                        style={{ color: theme.colors.text.primary }}
                    >
                        Show {remainingCount} more...
                    </button>
                )}
            </div>
        );
    }

    return (
        <AnimatePresence>
            <div className="fixed inset-0 z-[200] flex items-center justify-center p-4">
                {/* Backdrop */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    onClick={() => setIsHistoryModalOpen(false)}
                    className="absolute inset-0 bg-black/40 backdrop-blur-sm"
                />

                {/* Modal Content */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.98, y: 10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.98, y: 10 }}
                    className="relative w-full max-w-2xl rounded-xl shadow-2xl overflow-hidden flex flex-col border backdrop-blur-2xl"
                    style={{
                        maxHeight: '68vh',
                        backgroundColor: theme.colors.bg.secondary + 'F2',
                        borderColor: theme.colors.border.primary
                    }}
                >
                    {/* Header/Search */}
                    <div className="p-6 border-b flex items-center justify-between" style={{ borderColor: theme.colors.border.primary }}>
                        <div className="relative flex-1">
                            <input
                                autoFocus
                                type="text"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                placeholder="Select a conversation"
                                className="w-full bg-transparent text-xl font-medium outline-none border-none placeholder:opacity-20"
                                style={{
                                    color: theme.colors.text.primary,
                                }}
                            />
                        </div>
                        <button
                            onClick={() => {
                                createNewSession();
                                setIsHistoryModalOpen(false);
                            }}
                            className="p-2 rounded-lg hover:bg-white/10 transition-all group flex items-center gap-2"
                            title="New Conversation"
                        >
                            <span className="text-[10px] uppercase tracking-widest font-bold opacity-0 group-hover:opacity-60 transition-opacity" style={{ color: theme.colors.text.primary }}>New Chat</span>
                            <div className="p-1.5 rounded-md bg-white/5 group-hover:bg-white/10 transition-colors">
                                <Plus size={20} style={{ color: theme.colors.accent.primary }} />
                            </div>
                        </button>
                    </div>

                    {/* Scrollable List */}
                    <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                        {renderGroup('current', 'Current', groups.current)}
                        {renderGroup('blocked', 'Blocked on Your Input in brick', groups.blocked)}
                        {renderGroup('recent', 'Recent in brick', groups.recent)}
                        {renderGroup('other', 'Other Conversations', groups.other, true)}

                        {filteredSessions.length === 0 && (
                            <div className="flex flex-col items-center justify-center py-20 opacity-20">
                                <History size={48} className="mb-4" />
                                <p className="text-sm font-mono tracking-tight">No matching sessions found</p>
                            </div>
                        )}
                    </div>

                    {/* Footer */}
                    <div className="p-4 flex items-center justify-between border-t" style={{ backgroundColor: 'rgba(0,0,0,0.1)', borderColor: theme.colors.border.primary }}>
                        <div className="text-[9px] font-mono opacity-40 uppercase tracking-widest" style={{ color: theme.colors.text.secondary }}>
                            {sessions.length} TOTAL SESSIONS • ESCAPE TO CLOSE
                        </div>
                        <button
                            onClick={() => setIsHistoryModalOpen(false)}
                            className="px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-widest transition-all hover:brightness-110"
                            style={{
                                backgroundColor: theme.colors.bg.tertiary,
                                color: theme.colors.text.primary,
                                border: `1px solid ${theme.colors.border.primary}`
                            }}
                        >
                            Close
                        </button>
                    </div>
                </motion.div>
            </div>
        </AnimatePresence>
    );
};

export default HistoryModal;
