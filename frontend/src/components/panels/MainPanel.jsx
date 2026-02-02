import React, { useState } from 'react';
import { usePanel } from '../../contexts/PanelContext';
import { useTheme } from '../../contexts/ThemeContext';
import { Layers, Clock, FileText, Activity } from 'lucide-react';

const MainPanel = () => {
    const { activeSession } = usePanel();
    const { theme } = useTheme();
    const [activeTab, setActiveTab] = useState('stream');

    const tabs = [
        { id: 'stream', label: 'Stream', icon: Activity },
        { id: 'history', label: 'History', icon: Clock },
        { id: 'artifacts', label: 'Artifacts', icon: FileText },
    ];

    return (
        <div className="flex flex-col h-full bg-[#0a0a0a]/90 backdrop-blur-xl text-white">
            {/* Header / Tabs */}
            <div className="flex items-center border-b border-white/10 px-2 pt-2">
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`
                            flex items-center gap-2 px-4 py-2 text-xs font-medium uppercase tracking-wider transition-all
                            border-b-2 
                            ${activeTab === tab.id
                                ? 'border-blue-500 text-blue-400'
                                : 'border-transparent text-gray-500 hover:text-gray-300'}
                        `}
                    >
                        <tab.icon size={12} />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Content Area */}
            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                {activeTab === 'stream' && (
                    <div className="space-y-4">
                        <div className="text-xs text-gray-500 font-mono text-center my-4">
                            -- Session Started: {activeSession.branchName} --
                        </div>
                        {/* Mock Stream Content */}
                        <div className="p-3 rounded bg-white/5 border border-white/10 text-sm">
                            <div className="text-blue-300 font-mono text-xs mb-1">System</div>
                            <p className="text-gray-300 leading-relaxed">
                                Initializing session context for <strong>{activeSession.branchName}</strong>.
                                Ready for input.
                            </p>
                        </div>
                    </div>
                )}

                {activeTab === 'history' && (
                    <div className="text-center text-gray-500 text-sm mt-10">
                        No previous history found for this session.
                    </div>
                )}

                {activeTab === 'artifacts' && (
                    <div className="text-center text-gray-500 text-sm mt-10">
                        No artifacts generated yet.
                    </div>
                )}
            </div>

            {/* Status Bar */}
            <div className="border-t border-white/10 px-3 py-1.5 flex justify-between items-center text-[10px] text-gray-500 font-mono">
                <span>ID: {activeSession.id}</span>
                <span>STATUS: IDLE</span>
            </div>
        </div>
    );
};

export default MainPanel;
