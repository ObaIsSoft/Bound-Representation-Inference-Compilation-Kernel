import React, { useState } from 'react';
import { GitBranch, GitCommit, GitMerge, Clock, User } from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';
import { useSimulation } from '../../contexts/SimulationContext';

const ForkPanel = ({ width }) => {
    const { theme } = useTheme();
    const simulation = useSimulation();

    // Safely extract design state (with fallbacks for undefined)

    // Safely extract design state (with fallbacks for undefined)
    const isaTree = simulation?.isaTree || null;
    const geometryTree = simulation?.geometryTree || [];
    const sketchPoints = simulation?.sketchPoints || [];

    const [commits, setCommits] = useState([]);
    const [branches, setBranches] = useState([]);
    const [currentBranch, setCurrentBranch] = useState("main");

    // Fetch Version History
    const fetchHistory = React.useCallback(async () => {
        try {
            const res = await fetch(`http://localhost:8000/api/version/history?branch=${currentBranch}`);
            const data = await res.json();
            if (data.commits) setCommits(data.commits);
            if (data.branches) setBranches(data.branches);
        } catch (e) {
            console.error("Failed to fetch version history:", e);
            setCommits([]);
        }
    }, [currentBranch]);

    React.useEffect(() => {
        fetchHistory();
        const interval = setInterval(fetchHistory, 5000);
        return () => clearInterval(interval);
    }, [fetchHistory]);

    const handleCommit = async () => {
        const message = prompt("Enter commit message:", "Checkpoint");
        if (!message) return;

        try {
            // Capture REAL design state from SimulationContext
            const projectSnapshot = {
                manifest: {
                    author: "User",
                    timestamp: new Date().toISOString(),
                    branch: currentBranch
                },
                isa_tree: isaTree,
                geometry: geometryTree,
                sketch: sketchPoints
            };

            const res = await fetch('http://localhost:8000/api/version/commit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    project_data: projectSnapshot,
                    branch: currentBranch
                })
            });

            const data = await res.json();
            if (data.success) {
                fetchHistory();
                // Also trigger a save to the main project file
                await fetch('http://localhost:8000/api/project/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        data: projectSnapshot,
                        filename: `autosave_${currentBranch}.brick`,
                        branch: currentBranch
                    })
                });
            }
        } catch (e) {
            alert("Commit failed: " + e.message);
        }
    };

    const handleCreateBranch = async () => {
        const name = prompt("New branch name:", `feat-${Math.floor(Math.random() * 1000)}`);
        if (!name) return;

        try {
            const res = await fetch('http://localhost:8000/api/version/branch/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name,
                    source: currentBranch
                })
            });
            const data = await res.json();
            if (data.success) {
                alert(`Branch '${name}' created! Folder: backend/projects/${name}/`);
                setCurrentBranch(name); // Auto-switch
                fetchHistory();
            }
        } catch (e) {
            alert("Create Branch failed: " + e.message);
        }
    };

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
                        {currentBranch}
                    </span>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={handleCommit}
                        className="flex-1 flex items-center justify-center gap-1 py-1.5 rounded text-[9px] font-mono font-bold uppercase hover:brightness-110 active:scale-95 transition-all"
                        style={{
                            background: `linear-gradient(to right, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`,
                            color: theme.colors.bg.primary,
                            boxShadow: `0 0 10px ${theme.colors.accent.primary}40`
                        }}
                    >
                        <GitCommit size={12} /> Commit
                    </button>
                    <button
                        className="px-3 py-1.5 rounded text-[9px] font-mono uppercase hover:bg-white/5 transition-colors"
                        style={{
                            backgroundColor: theme.colors.bg.tertiary,
                            border: `1px solid ${theme.colors.border.primary}`,
                            color: theme.colors.text.tertiary
                        }}
                        onClick={handleCreateBranch}
                        title="New Branch"
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
                                className="p-2 rounded flex items-center justify-between cursor-pointer hover:bg-white/5 transition-colors"
                                style={{
                                    backgroundColor: branch.name === currentBranch ? theme.colors.accent.primary + '1A' : theme.colors.bg.primary,
                                    border: `1px solid ${branch.name === currentBranch ? theme.colors.accent.primary + '33' : theme.colors.border.primary}`
                                }}
                                onClick={() => setCurrentBranch(branch.name)}
                            >
                                <div className="flex items-center gap-2">
                                    <GitBranch size={10} style={{ color: branch.name === currentBranch ? theme.colors.accent.primary : theme.colors.text.muted }} />
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
                                className="p-2 rounded group hover:bg-white/5 transition-colors cursor-pointer"
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
                Last sync: Just now
            </div>
        </aside>
    );
};

export default ForkPanel;
