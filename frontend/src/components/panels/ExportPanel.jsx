import React, { useState } from 'react';
import { Download, FileText, Package, Database, CheckCircle } from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';
import { useSimulation } from '../../contexts/SimulationContext';

const ExportPanel = ({ width }) => {
    const { theme } = useTheme();
    const { isaTree } = useSimulation(); // Correctly access isaTree from SimulationContext

    // Only show implemented formats
    const [exportFormats] = useState([
        { name: 'STL', id: 'stl', description: '3D printing mesh', size: 'Unknown', icon: Package },
        { name: 'JSON', id: 'json', description: 'ISA source code', size: '124 KB', icon: FileText },
        { name: 'PDF', id: 'pdf', description: 'Technical documentation', size: 'Unknown', icon: FileText },
        { name: 'CSV', id: 'csv', description: 'Bill of materials', size: '18 KB', icon: Database }
    ]);

    const [recentExports, setRecentExports] = useState([]);
    const [isExporting, setIsExporting] = useState(false);

    const handleExport = async (formatId) => {
        setIsExporting(true);
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, "");
        const filename = `brick_export_${timestamp}.${formatId}`;

        try {
            if (formatId === 'stl') {
                // Call Backend Phase 13 Global Export
                const response = await fetch('http://localhost:8000/api/geometry/export/stl', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        geometry_tree: isaTree ? (isaTree.children || []) : [], // Flatten or pass root? Assuming array for now based on endpoint
                        resolution: 64, // Default quality
                        format: 'stl'
                    })
                });

                if (!response.ok) throw new Error("Export failed");

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                a.remove();

                setRecentExports(prev => [{ name: filename, time: 'Just now', size: 'BIN' }, ...prev]);

            } else if (formatId === 'json') {
                // Client-side JSON dump
                const data = JSON.stringify(isaTree || {}, null, 2);
                const blob = new Blob([data], { type: 'application/json' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                a.click();
                setRecentExports(prev => [{ name: filename, time: 'Just now', size: 'JSON' }, ...prev]);
            } else if (formatId === 'pdf') {
                try {
                    const { jsPDF } = await import('jspdf');
                    const autoTable = (await import('jspdf-autotable')).default;

                    const doc = new jsPDF();

                    // Header
                    doc.setFontSize(22);
                    doc.setTextColor(theme.colors.accent.primary);
                    doc.text("BRICK OS Technical Report", 14, 20);

                    doc.setFontSize(10);
                    doc.setTextColor('#555');
                    doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 28);

                    doc.setLineWidth(0.5);
                    doc.setDrawColor('#ccc');
                    doc.line(14, 32, 196, 32);

                    // Project Summary (Stubbed for now, eventually from DesignContext)
                    doc.setFontSize(14);
                    doc.setTextColor('#000');
                    doc.text("1. Project Overview", 14, 42);
                    doc.setFontSize(10);
                    doc.setTextColor('#333');
                    const desc = "This document contains technical specifications, dimensions, and manufacturing data for the current design configuration.";
                    doc.text(doc.splitTextToSize(desc, 180), 14, 50);

                    // 2. Bill of Materials (BOM) Table from ISA Tree
                    doc.setFontSize(14);
                    doc.setTextColor('#000');
                    doc.text("2. Bill of Materials", 14, 70);

                    // Flatten Tree for BOM
                    const rows = [];
                    const traverse = (nodes) => {
                        if (!nodes) return;
                        nodes.forEach(node => {
                            if (node.type !== 'group') {
                                rows.push([
                                    node.name || node.type || "Part",
                                    node.type,
                                    JSON.stringify(node.params || {}).substring(0, 30) + "...",
                                    (node.mass_kg || 0).toFixed(2) + " kg"
                                ]);
                            }
                            if (node.children) traverse(node.children);
                        });
                    };

                    if (isaTree) {
                        traverse(isaTree.children || []);
                    }

                    autoTable(doc, {
                        startY: 75,
                        head: [['Name', 'Type', 'Params / Dims', 'Mass']],
                        body: rows,
                        theme: 'grid',
                        headStyles: { fillColor: [40, 40, 40] },
                        styles: { fontSize: 8, font: 'courier' }
                    });

                    doc.save(filename);
                    setRecentExports(prev => [{ name: filename, time: 'Just now', size: 'PDF' }, ...prev]);

                } catch (e) {
                    console.error("PDF Gen Error", e);
                    alert("Failed to generate PDF: " + e.message);
                }
            } else {
                alert("Format not implemented yet.");
            }

        } catch (err) {
            console.error("Export Error:", err);
            alert("Export Failed: " + err.message);
        } finally {
            setIsExporting(false);
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
            <PanelHeader title="Export Data" icon={Download} />

            <div className="p-3" style={{ borderBottom: `1px solid ${theme.colors.border.primary}80` }}>
                <button
                    className="w-full flex items-center justify-center gap-2 py-2 rounded font-mono text-xs font-bold transition-all"
                    style={{
                        background: `linear-gradient(to right, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`,
                        color: theme.colors.bg.primary,
                        opacity: isExporting ? 0.5 : 1,
                        cursor: isExporting ? 'wait' : 'pointer'
                    }}
                    onClick={() => handleExport('stl')}
                >
                    <Download size={14} /> {isExporting ? "Exporting..." : "Export STL"}
                </button>
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-4">
                <div>
                    <h3 className="text-[10px] uppercase font-mono mb-2 flex items-center gap-2" style={{ color: theme.colors.text.muted }}>
                        <Package size={12} /> Live Formats
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
                                    onClick={() => handleExport(format.id)}
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
                        {recentExports.length === 0 && <div className="text-[9px] opacity-30 italic">No recent exports.</div>}
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
                Exports download to browser defaults.
            </div>
        </aside>
    );
};

export default ExportPanel;
