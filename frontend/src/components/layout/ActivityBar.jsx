import React from 'react';
import {
    Search, Package, Play, Layers, Code2, Wrench,
    GitBranch, Download, FileText, Settings, Cpu, Server
} from 'lucide-react';
import { ACTIVITY_BAR_WIDTH } from '../../utils/constants';
import { useTheme } from '../../contexts/ThemeContext';

const ActivityBar = ({ activeTab, setActiveTab }) => {
    const { theme } = useTheme();

    const topActions = [
        { id: 'search', icon: Search, label: 'Global Search' },
        { id: 'design', icon: Package, label: 'Design Explorer' },
        { id: 'components', icon: Cpu, label: 'Universal Catalog' },
        { id: 'run', icon: Play, label: 'Run and Debug' },
        { id: 'agents', icon: Layers, label: 'Agent Pods' },
        { id: 'isa', icon: Server, label: 'Recursive ISA Structure' }, // Phase 9
        { id: 'compile', icon: Code2, label: 'Compile ISA' },
        { id: 'mfg', icon: Wrench, label: 'Manufacturing' },
        { id: 'fork', icon: GitBranch, label: 'Version Control (Fork)' },
        { id: 'export', icon: Download, label: 'Export Data' },
    ];

    const bottomActions = [
        { id: 'docs', icon: FileText, label: 'Documentation' },
        { id: 'settings', icon: Settings, label: 'Kernel Settings' },
    ];

    return (
        <aside
            className="h-full flex flex-col items-center py-4 shrink-0 z-50 select-none"
            style={{
                width: ACTIVITY_BAR_WIDTH,
                backgroundColor: theme.colors.bg.primary,
                borderRight: `1px solid ${theme.colors.border.primary}`
            }}
        >
            <div className="flex-1 flex flex-col gap-2 w-full items-center">
                {topActions.map((item) => (
                    <button
                        key={item.id}
                        onClick={() => setActiveTab(item.id)}
                        title={item.label}
                        className="p-2.5 transition-all group relative rounded-md"
                        style={{
                            color: activeTab === item.id ? theme.colors.accent.primary : theme.colors.text.muted,
                            backgroundColor: activeTab === item.id ? theme.colors.accent.primary + '0D' : 'transparent'
                        }}
                    >
                        <item.icon size={20} strokeWidth={activeTab === item.id ? 2.5 : 2} />
                        {activeTab === item.id && (
                            <div
                                className="absolute left-0 top-1/4 bottom-1/4 w-[2px] rounded-r-full"
                                style={{
                                    backgroundColor: theme.colors.accent.primary,
                                    boxShadow: `0 0 8px ${theme.colors.accent.glow}`
                                }}
                            />
                        )}
                    </button>
                ))}
            </div>
            <div className="flex flex-col gap-2 w-full items-center mb-2">
                {bottomActions.map((item) => (
                    <button
                        key={item.id}
                        title={item.label}
                        onClick={() => setActiveTab(item.id)}
                        className="p-2.5 transition-all group relative rounded-md"
                        style={{
                            color: activeTab === item.id ? theme.colors.accent.primary : theme.colors.text.muted,
                            backgroundColor: activeTab === item.id ? theme.colors.accent.primary + '0D' : 'transparent'
                        }}
                    >
                        <item.icon size={20} strokeWidth={activeTab === item.id ? 2.5 : 2} />
                    </button>
                ))}
            </div>
        </aside>
    );
};

export default ActivityBar;
