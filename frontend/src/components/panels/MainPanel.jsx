import React, { useState } from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { usePanel } from '../../contexts/PanelContext';
import DraggablePanel from '../shared/DraggablePanel';
import { X, Maximize2, Minimize2, History, Brain, Box } from 'lucide-react';

const TABS = [
    { id: 'history', label: 'History', icon: History },
    { id: 'thoughts', label: 'Thoughts', icon: Brain },
    { id: 'artifacts', label: 'Artifacts', icon: Box },
];

const MainPanel = () => {
    const { theme } = useTheme();
    const { PANEL_IDS, togglePanel, activeSession } = usePanel();
    const [activeTab, setActiveTab] = useState('history');

    // Header Content with Tabs
    const headerContent = (
        <div className="flex items-center justify-between w-full">
            <div className="flex bg-black/20 rounded-lg p-1 gap-1">
                {TABS.map(tab => {
                    const Icon = tab.icon;
                    const isActive = activeTab === tab.id;
                    return (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${isActive
                                ? 'bg-white/10 text-white shadow-sm'
                                : 'text-white/40 hover:text-white/70 hover:bg-white/5'
                                }`}
                        >
                            <Icon size={14} />
                            <span>{tab.label}</span>
                        </button>
                    );
                })}
            </div>

            <div className="flex items-center gap-2 pl-4">
                <button
                    onClick={() => togglePanel(PANEL_IDS.MAIN)}
                    className="p-1 hover:bg-white/10 rounded text-white/40 hover:text-white transition-colors"
                >
                    <X size={16} />
                </button>
            </div>
        </div>
    );

    return (
        <DraggablePanel
            id={PANEL_IDS.MAIN}
            headerContent={headerContent}
            className="pointer-events-auto flex flex-col"
        >
            <div
                className="flex-1 overflow-y-auto p-4 custom-scrollbar"
                style={{ backgroundColor: theme.colors.bg.secondary + '80' }} // Slight transparency
            >
                {activeTab === 'history' ? (
                    <div className="space-y-4">
                        <div className="flex flex-col gap-4">
                            {activeSession.history.length === 0 ? (
                                <div className="text-center opacity-50 text-sm mt-10" style={{ color: theme.colors.text.secondary }}>
                                    No messages yet. Start a conversation!
                                </div>
                            ) : (
                                activeSession.history.map((msg) => (
                                    <div
                                        key={msg.id}
                                        className={`p-3 rounded-lg text-sm ${msg.role === 'user' ? 'self-end bg-white/10' : 'self-start bg-black/20'}`}
                                        style={{ color: theme.colors.text.primary, maxWidth: '85%' }}
                                    >
                                        {msg.content}
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                ) : (
                    <div className="space-y-4">
                        <h2 className="text-xl font-bold" style={{ color: theme.colors.text.primary }}>
                            {TABS.find(t => t.id === activeTab)?.label}
                        </h2>

                        <p style={{ color: theme.colors.text.secondary }} className="text-sm leading-relaxed">
                            Placeholder content for {activeTab}.
                        </p>
                    </div>
                )}
            </div>
        </DraggablePanel>
    );
};

export default MainPanel;
