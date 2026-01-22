import React, { useState, useEffect } from 'react';
import { Layers, ChevronRight, ChevronDown, Box, Cpu, Server } from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';
import { useSimulation } from '../../contexts/SimulationContext';

const ISABrowserPanel = ({ width }) => {
    const { theme } = useTheme();
    const { focusedPodId, setFocusedPodId, isaTree, refreshIsaTree } = useSimulation();
    const [expanded, setExpanded] = useState(new Set());

    // Auto-expand root when tree loads
    useEffect(() => {
        if (isaTree) {
            setExpanded(prev => {
                const next = new Set(prev);
                next.add(isaTree.id);
                return next;
            });
        } else {
            refreshIsaTree();
        }
    }, [isaTree]);

    const toggleExpand = (id, e) => {
        e.stopPropagation();
        setExpanded(prev => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    };

    const handleCheckout = (pod) => {
        setFocusedPodId(pod.id);
        console.log(`Checked out: ${pod.name} (${pod.id})`);
    };

    const handleCheckoutRoot = () => {
        setFocusedPodId(null);
    };

    const renderNode = (node, depth = 0) => {
        const isExpanded = expanded.has(node.id);
        const hasChildren = node.children && node.children.length > 0;
        const isActive = focusedPodId === node.id;
        // Root matches if focusedPodId is null AND node is root? 
        // No, 'null' usually means global, which effectively is root context but visually different.
        // Let's treat 'Root' node as ID match if focusedPodId matches, otherwise 'Global' button handles clearing.

        return (
            <div key={node.id} style={{ marginLeft: depth * 12 }}>
                <div
                    className="flex items-center p-1.5 rounded cursor-pointer group mb-0.5 select-none transition-colors"
                    style={{
                        backgroundColor: isActive
                            ? theme.colors.accent.primary + '20'
                            : 'transparent',
                        border: `1px solid ${isActive
                            ? theme.colors.accent.primary
                            : 'transparent'}`,
                        color: isActive ? theme.colors.accent.primary : theme.colors.text.secondary
                    }}
                    onClick={() => handleCheckout(node)}
                    onMouseEnter={(e) => !isActive && (e.currentTarget.style.backgroundColor = theme.colors.bg.secondary)}
                    onMouseLeave={(e) => !isActive && (e.currentTarget.style.backgroundColor = 'transparent')}
                >
                    <div
                        className="w-4 h-4 flex items-center justify-center mr-1 rounded hover:bg-white/10"
                        onClick={(e) => hasChildren && toggleExpand(node.id, e)}
                    >
                        {hasChildren && (
                            isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />
                        )}
                    </div>

                    <Box size={12} className="mr-2" style={{ color: isActive ? theme.colors.accent.primary : theme.colors.text.muted }} />

                    <div className="flex-1 min-w-0">
                        <div className="text-[11px] font-mono truncate">{node.name}</div>
                        {/* Constraints Preview (Micro text) */}
                        {node.constraints && Object.keys(node.constraints).length > 0 && (
                            <div className="text-[9px] opacity-60 truncate">
                                {Object.keys(node.constraints)[0]}: {JSON.stringify(Object.values(node.constraints)[0])}
                            </div>
                        )}
                    </div>
                </div>

                {isExpanded && hasChildren && (
                    <div className="border-l ml-2.5 pl-1" style={{ borderColor: theme.colors.border.secondary + '40' }}>
                        {node.children.map(child => renderNode(child, depth + 1))}
                    </div>
                )}
            </div>
        );
    };

    if (width <= 0) return null;

    return (
        <div
            className="flex flex-col h-full border-r relative"
            style={{
                width: width,
                borderColor: theme.colors.border.primary,
                backgroundColor: theme.colors.bg.primary
            }}
        >
            <PanelHeader title="Hardware ISA" icon={Layers} />

            <div className="flex-1 p-3 overflow-y-auto scrollbar-thin">
                {/* Global Context Button */}
                <div
                    className="mb-2 p-1.5 rounded cursor-pointer flex items-center border border-dashed hover:bg-white/5 transition-colors"
                    style={{
                        borderColor: theme.colors.border.secondary,
                        color: !focusedPodId ? theme.colors.text.primary : theme.colors.text.muted
                    }}
                    onClick={handleCheckoutRoot}
                >
                    <Server size={12} className="mr-2" />
                    <span className="text-[10px] font-bold uppercase tracking-wider">Global Scope</span>
                    {!focusedPodId && <div className="ml-auto w-1.5 h-1.5 rounded-full bg-green-500" />}
                </div>

                <div className="h-px w-full my-2" style={{ backgroundColor: theme.colors.border.primary + '40' }} />

                {/* Recursive Tree */}
                {isaTree ? renderNode(isaTree) : (
                    <div className="p-4 text-center text-xs opacity-50">Loading structure...</div>
                )}
            </div>

            {/* Phase 10: Agent Results / Properties Pane */}
            {focusedPodId && (() => {
                // Find active node in tree
                const findNode = (node) => {
                    if (node.id === focusedPodId) return node;
                    if (node.children) {
                        for (let child of node.children) {
                            const found = findNode(child);
                            if (found) return found;
                        }
                    }
                    return null;
                };
                const activeNode = isaTree ? findNode(isaTree) : null;

                if (!activeNode) return null;

                const constraints = activeNode.constraints || {};
                const exports = activeNode.exports || {};

                return (
                    <div className="shrink-0 border-t flex flex-col" style={{ borderColor: theme.colors.border.primary, height: '35%' }}>
                        <div className="p-2 text-[10px] font-bold uppercase tracking-wider bg-white/5 text-center"
                            style={{ color: theme.colors.text.secondary }}>
                            {activeNode.name} Properties
                        </div>
                        <div className="flex-1 overflow-y-auto p-2 space-y-3">

                            {/* Inputs / Constraints */}
                            <div>
                                <div className="text-[9px] uppercase opacity-50 mb-1">Constraints (Input)</div>
                                {Object.keys(constraints).length > 0 ? (
                                    <div className="grid grid-cols-2 gap-1">
                                        {Object.entries(constraints).map(([k, v]) => (
                                            <div key={k} className="bg-white/5 p-1 rounded">
                                                <div className="text-[9px] opacity-70 truncate" title={k}>{k}</div>
                                                <div className="text-[10px] font-mono truncate" title={v} style={{ color: theme.colors.accent.secondary }}>
                                                    {Array.isArray(v) ? `[${v.length}]` : v}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : <div className="text-[10px] italic opacity-40">No constraints defined.</div>}
                            </div>

                            {/* Outputs / Exports */}
                            <div>
                                <div className="text-[9px] uppercase opacity-50 mb-1">Agent Results (Output)</div>
                                {Object.keys(exports).length > 0 ? (
                                    <div className="space-y-1">
                                        {Object.entries(exports).map(([k, v]) => (
                                            <div key={k} className="flex justify-between items-center text-[10px] font-mono border-b border-white/5 pb-0.5">
                                                <span className="opacity-70">{k}</span>
                                                <span style={{ color: theme.colors.status.success }}>{typeof v === 'number' ? v.toFixed(2) : v}</span>
                                            </div>
                                        ))}
                                    </div>
                                ) : <div className="text-[10px] italic opacity-40">Run agents to see results.</div>}
                            </div>

                        </div>
                    </div>
                );
            })()}

            <div className="p-1 border-t text-[9px] text-center opacity-30 font-mono" style={{ borderColor: theme.colors.border.primary }}>
                {focusedPodId ? `ID: ${focusedPodId.substring(0, 8)}` : "Type: System"}
            </div>
        </div>
    );
};

export default ISABrowserPanel;
