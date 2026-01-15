import React, { useState } from 'react';
import { Wrench, Package, Truck, DollarSign, AlertTriangle, RefreshCw } from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';
import { compileDesign } from '../../utils/api';

const ManufacturingPanel = ({ width }) => {
    const { theme } = useTheme();
    const [loading, setLoading] = useState(false);
    const [analysis, setAnalysis] = useState(null);
    const [components, setComponents] = useState([]);

    const handleGenerateAnalysis = async () => {
        setLoading(true);
        try {
            // In a real scenario, we'd pass the actual design intent or file content
            const result = await compileDesign("Analyze manufacturing requirements for current assembly");
            if (result && result.bom_analysis) {
                setAnalysis(result.bom_analysis);
                setComponents(result.components || []);
            }
        } catch (error) {
            console.error("Failed to generate analysis", error);
        } finally {
            setLoading(false);
        }
    };

    if (width <= 0) return null;

    // Fallback/Initial Data
    const totalCost = analysis ? analysis.total_cost : 0;
    const leadTime = analysis ? analysis.lead_time_days : 0;

    return (
        <aside
            className="h-full flex flex-col shrink-0 overflow-hidden"
            style={{
                width,
                backgroundColor: theme.colors.bg.secondary + '66',
                borderRight: `1px solid ${theme.colors.border.primary}`
            }}
        >
            <PanelHeader title="Manufacturing" icon={Wrench} >
                <button
                    onClick={handleGenerateAnalysis}
                    className="p-1 rounded hover:bg-white/10 transition-colors"
                    disabled={loading}
                    title="Refresh Analysis"
                >
                    <RefreshCw size={14} className={loading ? "animate-spin" : ""} style={{ color: theme.colors.text.muted }} />
                </button>
            </PanelHeader>

            <div className="p-3" style={{ borderBottom: `1px solid ${theme.colors.border.primary}80` }}>
                <div className="grid grid-cols-2 gap-2">
                    <div className="p-2 rounded text-center" style={{ backgroundColor: theme.colors.bg.primary }}>
                        <div className="text-lg font-bold font-mono" style={{ color: theme.colors.accent.primary }}>
                            ${totalCost.toLocaleString()}
                        </div>
                        <div className="text-[8px] uppercase" style={{ color: theme.colors.text.muted }}>Total Cost</div>
                    </div>
                    <div className="p-2 rounded text-center" style={{ backgroundColor: theme.colors.bg.primary }}>
                        <div className="text-lg font-bold font-mono" style={{ color: theme.colors.text.primary }}>
                            {leadTime}d
                        </div>
                        <div className="text-[8px] uppercase" style={{ color: theme.colors.text.muted }}>Lead Time</div>
                    </div>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-4">
                {components.length === 0 && !loading && (
                    <div className="text-center py-8 text-[10px]" style={{ color: theme.colors.text.muted }}>
                        Click refresh to generate BOM
                    </div>
                )}

                {components.length > 0 && (
                    <div>
                        <h3 className="text-[10px] uppercase font-mono mb-2 flex items-center gap-2" style={{ color: theme.colors.text.muted }}>
                            <Wrench size={12} /> Components
                        </h3>
                        <div className="space-y-2">
                            {components.map((comp, i) => (
                                <div
                                    key={i}
                                    className="p-2 rounded"
                                    style={{
                                        backgroundColor: theme.colors.bg.primary,
                                        border: `1px solid ${theme.colors.border.primary}`
                                    }}
                                >
                                    <div className="flex justify-between items-start mb-1">
                                        <span className="text-[10px] font-mono font-semibold" style={{ color: theme.colors.text.primary }}>
                                            {comp.name}
                                        </span>
                                        <span
                                            className="text-[8px] px-1.5 py-0.5 rounded uppercase font-mono"
                                            style={{
                                                backgroundColor: theme.colors.status.success + '1A',
                                                color: theme.colors.status.success
                                            }}
                                        >
                                            {comp.process || "Machining"}
                                        </span>
                                    </div>
                                    <div className="flex justify-between text-[9px] font-mono">
                                        <span style={{ color: theme.colors.text.muted }}>
                                            {comp.material}
                                        </span>
                                        <span style={{ color: theme.colors.text.tertiary }}>
                                            <DollarSign size={8} className="inline" /> {comp.cost}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            <div
                className="p-2 flex items-center gap-2 text-[9px] font-mono"
                style={{
                    backgroundColor: theme.colors.status.warning + '1A',
                    borderTop: `1px solid ${theme.colors.status.warning}33`,
                    color: theme.colors.status.warning
                }}
            >
                <AlertTriangle size={12} />
                <span>Review tolerances before production</span>
            </div>
        </aside>
    );
};

export default ManufacturingPanel;
