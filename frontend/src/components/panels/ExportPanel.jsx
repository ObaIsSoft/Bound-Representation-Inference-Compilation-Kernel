import React, { useState } from 'react';
import { Download, FileText, Package, Database, CheckCircle } from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';

const ExportPanel = ({ width }) => {
    const { theme } = useTheme();

    const [exportFormats] = useState([
        { name: 'STEP', description: 'CAD interchange format', size: '2.4 MB', icon: Package },
        { name: 'STL', description: '3D printing mesh', size: '1.8 MB', icon: Package },
        { name: 'JSON', description: 'ISA source code', size: '124 KB', icon: FileText },
        { name: 'PDF', description: 'Technical documentation', size: '3.2 MB', icon: FileText },
        { name: 'CSV', description: 'Bill of materials', size: '18 KB', icon: Database }
    ]);

    const [recentExports] = useState([
        { name: 'main_assembly.step', time: '5m ago', size: '2.4 MB' },
        { name: 'propulsion_module.stl', time: '1h ago', size: '1.8 MB' },
        { name: 'project_data.json', time: '2h ago', size: '124 KB' }
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
            <PanelHeader title="Export Data" icon={Download} />

            <div className="p-3" style={{ borderBottom: `1px solid ${theme.colors.border.primary}80` }}>
                <button
                    className="w-full flex items-center justify-center gap-2 py-2 rounded font-mono text-xs font-bold transition-all"
                    style={{
                        background: `linear-gradient(to right, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`,
                        color: theme.colors.bg.primary
                    }}
                >
                    <Download size={14} /> Export All
                </button>
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-4">
                <div>
                    <h3 className="text-[10px] uppercase font-mono mb-2 flex items-center gap-2" style={{ color: theme.colors.text.muted }}>
                        <Package size={12} /> Available Formats
                    </h3>
                    <div className="space-y-2">
                        {exportFormats.map((format, i) => (
                            <div
                                key={i}
                                className="p-2 rounded cursor-pointer transition-all"
                                style={{
                                    backgroundColor: theme.colors.bg.primary,
                                    border: `1px solid ${theme.colors.border.primary}`
                                }}
                                onMouseEnter={(e) => e.currentTarget.style.borderColor = theme.colors.accent.primary + '80'}
                                onMouseLeave={(e) => e.currentTarget.style.borderColor = theme.colors.border.primary}
                            >
                                <div className="flex items-start justify-between mb-1">
                                    <div className="flex items-center gap-2">
                                        <format.icon size={12} style={{ color: theme.colors.accent.primary }} />
                                        <span className="text-[10px] font-mono font-bold" style={{ color: theme.colors.text.primary }}>
                                            {format.name}
                                        </span>
                                    </div>
                                    <span className="text-[9px] font-mono" style={{ color: theme.colors.text.tertiary }}>
                                        {format.size}
                                    </span>
                                </div>
                                <div className="text-[9px] font-mono" style={{ color: theme.colors.text.tertiary }}>
                                    {format.description}
                                </div>
                                <button
                                    className="mt-2 w-full py-1 rounded text-[9px] font-mono uppercase transition-all"
                                    style={{
                                        backgroundColor: theme.colors.bg.tertiary,
                                        border: `1px solid ${theme.colors.border.primary}`,
                                        color: theme.colors.text.tertiary
                                    }}
                                >
                                    <Download size={10} className="inline mr-1" /> Export
                                </button>
                            </div>
                        ))}
                    </div>
                </div>

                <div>
                    <h3 className="text-[10px] uppercase font-mono mb-2 flex items-center gap-2" style={{ color: theme.colors.text.muted }}>
                        Recent Exports
                    </h3>
                    <div className="space-y-1">
                        {recentExports.map((exp, i) => (
                            <div
                                key={i}
                                className="p-2 rounded flex items-center justify-between"
                                style={{
                                    backgroundColor: theme.colors.bg.primary,
                                    border: `1px solid ${theme.colors.border.primary}`
                                }}
                            >
                                <div className="flex items-center gap-2">
                                    <CheckCircle size={10} style={{ color: theme.colors.status.success }} />
                                    <div>
                                        <div className="text-[10px] font-mono" style={{ color: theme.colors.text.primary }}>
                                            {exp.name}
                                        </div>
                                        <div className="text-[9px] font-mono" style={{ color: theme.colors.text.muted }}>
                                            {exp.time} • {exp.size}
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
                Export location: ~/Downloads/BRICK
            </div>
        </aside>
    );
};

export default ExportPanel;
