import React, { useState } from 'react';
import { GitBranch, GitCommit, GitMerge, Clock, User } from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';

const ForkPanel = ({ width }) => {
    const { theme } = useTheme();

    const [commits] = useState([
        { hash: '9a2f3b1', message: 'Updated shroud diameter to 7.62m', author: 'AERO_PHYSICS', time: '2m ago', branch: 'main' },
        { hash: '7c4e8d2', message: 'Added thermal analysis constraints', author: 'PROP_THERMAL', time: '15m ago', branch: 'main' },
        { hash: '5f1a9c3', message: 'Optimized propulsion module', author: 'USER', time: '1h ago', branch: 'main' },
        { hash: '2d6b4e7', message: 'Initial project structure', author: 'USER', time: '3h ago', branch: 'main' }
    ]);

    const [branches] = useState([
        { name: 'main', commits: 12, active: true },
        { name: 'feature/aero-optimization', commits: 3, active: false },
        { name: 'experimental/new-kernel', commits: 7, active: false }
    ]);

    if (width <= 0) return null;

    return (
        <aside
            className="h-full flex flex-col shrink-0 overflow-hidden"
            style={{
                width,
                backgroundColor: theme.colors.bg.secondary + '66',
                borderRight: `1px solid ${theme.colors.border.primary}`
            }}
        >
            <PanelHeader title="Version Control" icon={GitBranch} />

            <div className="p-3" style={{ borderBottom: `1px solid ${theme.colors.border.primary}80` }}>
                <div className="flex items-center justify-between mb-2">
                    <span className="text-[9px] uppercase font-mono font-semibold" style={{ color: theme.colors.text.muted }}>
                        Current Branch
                    </span>
                    <span className="text-[10px] font-mono font-bold" style={{ color: theme.colors.accent.primary }}>
                        main
                    </span>
                </div>
                <div className="flex gap-2">
                    <button
                        className="flex-1 flex items-center justify-center gap-1 py-1.5 rounded text-[9px] font-mono font-bold uppercase"
                        style={{
                            background: `linear-gradient(to right, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`,
                            color: theme.colors.bg.primary
                        }}
                    >
                        <GitCommit size={12} /> Commit
                    </button>
                    <button
                        className="px-3 py-1.5 rounded text-[9px] font-mono uppercase"
                        style={{
                            backgroundColor: theme.colors.bg.tertiary,
                            border: `1px solid ${theme.colors.border.primary}`,
                            color: theme.colors.text.tertiary
                        }}
                    >
                        <GitMerge size={12} />
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-4">
                <div>
                    <h3 className="text-[10px] uppercase font-mono mb-2 flex items-center gap-2" style={{ color: theme.colors.text.muted }}>
                        <GitBranch size={12} /> Branches ({branches.length})
                    </h3>
                    <div className="space-y-1">
                        {branches.map((branch, i) => (
                            <div
                                key={i}
                                className="p-2 rounded flex items-center justify-between cursor-pointer"
                                style={{
                                    backgroundColor: branch.active ? theme.colors.accent.primary + '1A' : theme.colors.bg.primary,
                                    border: `1px solid ${branch.active ? theme.colors.accent.primary + '33' : theme.colors.border.primary}`
                                }}
                            >
                                <div className="flex items-center gap-2">
                                    <GitBranch size={10} style={{ color: branch.active ? theme.colors.accent.primary : theme.colors.text.muted }} />
                                    <span className="text-[10px] font-mono" style={{ color: theme.colors.text.primary }}>
                                        {branch.name}
                                    </span>
                                </div>
                                <span className="text-[9px] font-mono" style={{ color: theme.colors.text.tertiary }}>
                                    {branch.commits} commits
                                </span>
                            </div>
                        ))}
                    </div>
                </div>

                <div>
                    <h3 className="text-[10px] uppercase font-mono mb-2 flex items-center gap-2" style={{ color: theme.colors.text.muted }}>
                        <GitCommit size={12} /> Recent Commits
                    </h3>
                    <div className="space-y-2">
                        {commits.map((commit, i) => (
                            <div
                                key={i}
                                className="p-2 rounded"
                                style={{
                                    backgroundColor: theme.colors.bg.primary,
                                    border: `1px solid ${theme.colors.border.primary}`
                                }}
                            >
                                <div className="flex items-start gap-2 mb-1">
                                    <code className="text-[9px] font-mono px-1 py-0.5 rounded" style={{
                                        backgroundColor: theme.colors.bg.tertiary,
                                        color: theme.colors.accent.primary
                                    }}>
                                        {commit.hash}
                                    </code>
                                    <div className="flex-1">
                                        <div className="text-[10px] font-mono" style={{ color: theme.colors.text.primary }}>
                                            {commit.message}
                                        </div>
                                        <div className="flex items-center gap-2 mt-1 text-[9px] font-mono" style={{ color: theme.colors.text.muted }}>
                                            <span className="flex items-center gap-1">
                                                <User size={8} /> {commit.author}
                                            </span>
                                            <span className="flex items-center gap-1">
                                                <Clock size={8} /> {commit.time}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div
                className="p-2 text-[9px] font-mono"
                style={{
                    backgroundColor: theme.colors.bg.primary,
                    borderTop: `1px solid ${theme.colors.border.primary}`,
                    color: theme.colors.text.muted
                }}
            >
                Last sync: 30s ago
            </div>
        </aside>
    );
};

export default ForkPanel;
