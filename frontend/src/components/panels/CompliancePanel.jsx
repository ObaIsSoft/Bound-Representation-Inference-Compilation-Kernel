import React, { useState, useEffect } from 'react';
import {
    ShieldCheck, AlertCircle, CheckCircle2, Circle,
    ExternalLink, ChevronDown, ChevronUp, BookOpen,
    RefreshCw, Info, Scale
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTheme } from '../../contexts/ThemeContext';
import PanelHeader from '../shared/PanelHeader';

/**
 * CompliancePanel - Interactive Regulatory Checklist
 */
const CompliancePanel = ({ width, designParams = {} }) => {
    const { theme } = useTheme();
    const [loading, setLoading] = useState(false);
    const [checklist, setChecklist] = useState([]);
    const [regime, setRegime] = useState('AERIAL');
    const [expandedItem, setExpandedItem] = useState(null);
    const [overallStatus, setOverallStatus] = useState('unknown');

    const fetchCompliance = async () => {
        setLoading(true);
        try {
            // Mocking the call to the actual endpoint we just created
            // In a real environment, we'd use the centralized API utility
            const response = await fetch('http://localhost:8000/api/compliance/check', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    regime: regime,
                    design_params: {
                        mass_kg: 2.8, // Mocked for now, should come from simulation state
                        is_fcc_certified: true,
                        emc_test_passed: false,
                        ...designParams
                    }
                })
            });
            const data = await response.json();
            setChecklist(data.checklist || []);
            setOverallStatus(data.status);
        } catch (error) {
            console.error("Compliance Check Failed:", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchCompliance();
    }, [regime]);

    if (width <= 0) return null;

    return (
        <aside
            className="h-full flex flex-col shrink-0 overflow-hidden"
            style={{
                width,
                backgroundColor: theme.colors.bg.secondary + '66',
                borderRight: `1px solid ${theme.colors.border.primary}`,
                color: theme.colors.text.primary
            }}
        >
            <PanelHeader title="Compliance" icon={Scale}>
                <button
                    onClick={fetchCompliance}
                    className="p-1 rounded hover:bg-white/10 transition-colors"
                    disabled={loading}
                >
                    <RefreshCw size={14} className={loading ? "animate-spin" : ""} style={{ color: theme.colors.text.muted }} />
                </button>
            </PanelHeader>

            {/* Regime Selector */}
            <div className="p-3 border-b" style={{ borderColor: theme.colors.border.primary }}>
                <div className="flex bg-black/20 rounded-lg p-1">
                    {['AERIAL', 'TERRESTRIAL', 'MEDICAL'].map((r) => (
                        <button
                            key={r}
                            onClick={() => setRegime(r)}
                            className={`flex-1 text-[9px] font-bold py-1 rounded transition-all ${regime === r ? 'shadow-sm' : 'opacity-40'}`}
                            style={{
                                backgroundColor: regime === r ? theme.colors.accent.primary : 'transparent',
                                color: regime === r ? theme.colors.bg.primary : theme.colors.text.primary
                            }}
                        >
                            {r}
                        </button>
                    ))}
                </div>
            </div>

            {/* Overall Status Banner */}
            <div className="px-3 pt-4">
                <div className="p-3 rounded-xl border flex items-center justify-between"
                    style={{
                        backgroundColor: theme.colors.bg.primary,
                        borderColor: overallStatus === 'compliant' ? theme.colors.status.success + '40' : (overallStatus === 'non_compliant' ? theme.colors.status.error + '40' : theme.colors.border.primary)
                    }}>
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg" style={{ backgroundColor: overallStatus === 'compliant' ? theme.colors.status.success + '20' : theme.colors.status.error + '20' }}>
                            {overallStatus === 'compliant' ?
                                <ShieldCheck size={20} style={{ color: theme.colors.status.success }} /> :
                                <AlertCircle size={20} style={{ color: theme.colors.status.error }} />
                            }
                        </div>
                        <div>
                            <div className="text-[10px] uppercase font-bold tracking-widest opacity-50">System Status</div>
                            <div className="text-sm font-bold font-mono">
                                {overallStatus.replace('_', ' ').toUpperCase()}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Checklist items */}
            <div className="flex-1 overflow-y-auto p-3 space-y-2 custom-scrollbar">
                <div className="text-[10px] font-bold uppercase tracking-widest opacity-30 mb-2 px-1">Standards Checklist</div>

                {checklist.map((item) => (
                    <div
                        key={item.id}
                        className="rounded-xl border transition-all overflow-hidden"
                        style={{
                            backgroundColor: theme.colors.bg.primary,
                            borderColor: theme.colors.border.primary
                        }}
                    >
                        <button
                            onClick={() => setExpandedItem(expandedItem === item.id ? null : item.id)}
                            className="w-full text-left p-3 flex items-center justify-between group"
                        >
                            <div className="flex items-center gap-3">
                                {item.status === 'passed' ? (
                                    <CheckCircle2 size={16} style={{ color: theme.colors.status.success }} />
                                ) : item.status === 'failed' ? (
                                    <AlertCircle size={16} style={{ color: theme.colors.status.error }} />
                                ) : (
                                    <Circle size={16} className="opacity-20" />
                                )}
                                <span className="text-[11px] font-bold truncate pr-2">{item.name}</span>
                            </div>
                            <div className="flex items-center gap-2 opacity-50 group-hover:opacity-100 transition-opacity">
                                <span className="text-[9px] font-mono">{item.citation}</span>
                                {expandedItem === item.id ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                            </div>
                        </button>

                        <AnimatePresence>
                            {expandedItem === item.id && (
                                <motion.div
                                    initial={{ height: 0 }}
                                    animate={{ height: 'auto' }}
                                    exit={{ height: 0 }}
                                    className="overflow-hidden"
                                >
                                    <div className="px-3 pb-3 pt-1 space-y-3" style={{ borderTop: `1px solid ${theme.colors.border.primary}40` }}>
                                        <div className="p-2 rounded-lg bg-black/10 text-[10px] leading-relaxed italic opacity-80"
                                            style={{ borderLeft: `2px solid ${theme.colors.accent.primary}` }}>
                                            "{item.regulation_text}"
                                        </div>

                                        {item.message && (
                                            <div className="flex items-start gap-2 p-2 rounded-lg bg-red-500/10 text-[10px]">
                                                <AlertCircle size={12} className="shrink-0 mt-0.5" style={{ color: theme.colors.status.error }} />
                                                <span style={{ color: theme.colors.status.error }}>{item.message}</span>
                                            </div>
                                        )}

                                        <div className="flex justify-between items-center pt-1">
                                            <a
                                                href={item.official_link}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="text-[9px] flex items-center gap-1 hover:underline"
                                                style={{ color: theme.colors.accent.primary }}
                                                onClick={(e) => e.stopPropagation()}
                                            >
                                                <BookOpen size={10} /> View Full Regulation <ExternalLink size={10} />
                                            </a>
                                            <div className="text-[8px] font-mono opacity-30">ID: {item.id}</div>
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                ))}

                {checklist.length === 0 && !loading && (
                    <div className="text-center py-12 opacity-30">
                        <Info size={24} className="mx-auto mb-2" />
                        <div className="text-[10px] font-bold">No standards found for this regime</div>
                    </div>
                )}
            </div>

            {/* Footer Summary */}
            <div className="p-3 border-t text-[10px] font-medium italic opacity-40 text-center" style={{ borderColor: theme.colors.border.primary }}>
                Compliance Agent V2.1 • All rules locally validated by UPK
            </div>
        </aside>
    );
};

export default CompliancePanel;
