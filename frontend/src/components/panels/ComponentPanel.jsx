import React, { useState, useEffect } from 'react';
import { Package, Download, Cpu, Activity, Clock, Search } from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';
import { useDesign } from '../../contexts/DesignContext';

const ComponentPanel = ({ width }) => {
    const { theme } = useTheme();
    const [catalog, setCatalog] = useState([]);
    const [loading, setLoading] = useState(true);
    const [installingId, setInstallingId] = useState(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [customUrl, setCustomUrl] = useState('');
    const [isCustomExpanded, setIsCustomExpanded] = useState(false);

    useEffect(() => {
        const timer = setTimeout(() => {
            fetchCatalog();
        }, 500); // Debounce search
        return () => clearTimeout(timer);
    }, [searchQuery]);

    const [inspectionResult, setInspectionResult] = useState(null);
    const [loadingInspection, setLoadingInspection] = useState(false);

    const handleCustomInstall = async () => {
        if (!customUrl) return;

        const customId = `custom_${Date.now()}`;
        const name = customUrl.split('/').pop().split('?')[0] || "Custom Component";

        await handleInstall({
            id: customId,
            name: name,
            mesh_url: customUrl
        });

        setCustomUrl('');
        setInspectionResult(null);
    };

    const handleInspect = async () => {
        setLoadingInspection(true);
        try {
            const res = await fetch('http://localhost:8000/api/components/inspect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: customUrl })
            });
            const data = await res.json();
            setInspectionResult(data);
        } catch (err) {
            setInspectionResult({ valid: false, error: err.message });
        } finally {
            setLoadingInspection(false);
        }
    };

    const fetchCatalog = async () => {
        setLoading(true);
        try {
            const res = await fetch(`http://localhost:8000/api/components/catalog?search=${encodeURIComponent(searchQuery)}`);
            const data = await res.json();
            setCatalog(data.catalog || []);
        } catch (err) {
            console.error("Failed to fetch catalog:", err);
        } finally {
            setLoading(false);
        }
    };

    const { activeTabId, updateTabMetadata } = useDesign();

    if (width <= 0) return null;

    const handleInstall = async (component) => {
        if (!activeTabId) {
            // alert("Please open a design first.");
            return;
        }

        setInstallingId(component.id);
        try {
            const res = await fetch('http://localhost:8000/api/components/install', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    component_id: component.id,
                    mesh_url: component.mesh_url
                })
            });

            if (!res.ok) throw new Error('Installation failed');

            const result = await res.json();
            console.log("Installed:", result);

            // Inject into active design
            // We append to the manifest if it exists, or create new one
            // Ideally we should merge textures, but for now we might overwrite 
            // or rely on the backend sending the FULL texture/manifest for the scene.
            // Note: Current backend `install` returns a FRESH texture for just that component.
            // To support multiple, backend needs to be smarter or frontend needs to merge.
            // For Phase 8.2 MVP, let's just show the installed part (Single Object / Atlas).

            updateTabMetadata(activeTabId, {
                mesh_sdf_data: result.texture_data,
                manifest: result.manifest || result.metadata, // API compat
                sdf_resolution: result.resolution,
                sdf_bounds: result.bounds,
                sdf_range: result.sdf_range
            });

        } catch (err) {
            console.error("Install error:", err);
        } finally {
            setInstallingId(null);
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
            <PanelHeader title="Universal Catalog" icon={Cpu} />

            <div className="p-3" style={{ borderBottom: `1px solid ${theme.colors.border.primary}80` }}>
                <div className="relative mb-2">
                    <input
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full rounded px-3 py-2 pl-8 text-xs font-mono outline-none transition-all"
                        placeholder="Search COTS parts..."
                        style={{
                            backgroundColor: theme.colors.bg.primary,
                            border: `1px solid ${theme.colors.border.primary}`,
                            color: theme.colors.text.primary
                        }}
                        onFocus={(e) => e.target.style.borderColor = theme.colors.accent.primary}
                        onBlur={(e) => e.target.style.borderColor = theme.colors.border.primary}
                    />
                    <Search
                        size={12}
                        className="absolute left-2.5 top-2.5"
                        style={{ color: theme.colors.text.tertiary }}
                    />
                </div>

                <div
                    className="flex flex-col gap-2 p-2 rounded"
                    style={{ backgroundColor: theme.colors.bg.primary }}
                >
                    <div className="flex items-center justify-between cursor-pointer" onClick={() => setIsCustomExpanded(!isCustomExpanded)}>
                        <span className="text-[10px] font-bold uppercase opacity-70">Import from URL</span>
                        <span className="text-[10px]">{isCustomExpanded ? '−' : '+'}</span>
                    </div>

                    {isCustomExpanded && (
                        <div className="flex flex-col gap-2 animate-in fade-in slide-in-from-top-1">
                            <div className="flex gap-1">
                                <input
                                    value={customUrl}
                                    onChange={(e) => {
                                        setCustomUrl(e.target.value);
                                        setInspectionResult(null); // Reset on change
                                    }}
                                    className="flex-1 rounded px-2 py-1 text-[10px] font-mono outline-none"
                                    placeholder="http://example.com/part.stl"
                                    style={{
                                        backgroundColor: theme.colors.bg.secondary,
                                        border: `1px solid ${theme.colors.border.primary}`,
                                        color: theme.colors.text.primary
                                    }}
                                />
                                <button
                                    onClick={handleInspect}
                                    disabled={!customUrl || loadingInspection}
                                    className="px-2 py-1 rounded border text-[10px] font-bold"
                                    style={{
                                        borderColor: theme.colors.border.primary,
                                        backgroundColor: theme.colors.bg.secondary,
                                        opacity: (!customUrl || loadingInspection) ? 0.5 : 1,
                                        color: theme.colors.text.primary
                                    }}
                                >
                                    {loadingInspection ? '...' : 'Check'}
                                </button>
                            </div>

                            {inspectionResult && (
                                <div className="text-[9px] font-mono p-1.5 rounded border flex justify-between items-center"
                                    style={{
                                        borderColor: inspectionResult.valid ? theme.colors.accent.primary : 'red',
                                        backgroundColor: inspectionResult.valid ? theme.colors.accent.primary + '10' : '#ff000010',
                                        color: theme.colors.text.primary
                                    }}
                                >
                                    {inspectionResult.valid ? (
                                        <div className="flex flex-col overflow-hidden max-w-[120px]">
                                            <span className="font-bold truncate" title={inspectionResult.filename}>{inspectionResult.filename}</span>
                                            <span className="opacity-70">{inspectionResult.size_fmt} • {inspectionResult.type.split('/')[1] || 'unknown'}</span>
                                        </div>
                                    ) : (
                                        <span style={{ color: 'red' }}>Error: {inspectionResult.error}</span>
                                    )}

                                    {inspectionResult.valid && (
                                        <button
                                            onClick={handleCustomInstall}
                                            className="px-2 py-1 rounded text-[9px] font-bold"
                                            style={{
                                                backgroundColor: theme.colors.accent.primary,
                                                color: '#fff'
                                            }}
                                        >
                                            IMPORT
                                        </button>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-2 space-y-2">
                {loading ? (
                    <div className="text-center py-4 text-[10px] font-mono opacity-50">Loading Catalog...</div>
                ) : (
                    catalog.map(item => (
                        <div
                            key={item.id}
                            className="p-3 rounded border flex flex-col gap-2 group"
                            style={{
                                backgroundColor: theme.colors.bg.primary,
                                borderColor: theme.colors.border.primary
                            }}
                        >
                            <div className="flex justify-between items-start">
                                <div>
                                    <div className="text-[11px] font-bold" style={{ color: theme.colors.text.primary }}>
                                        {item.name}
                                    </div>
                                    <div className="text-[9px] font-mono opacity-70" style={{ color: theme.colors.text.secondary }}>
                                        {item.source} • {item.category}
                                    </div>
                                </div>
                                <Cpu size={14} style={{ color: theme.colors.accent.primary }} />
                            </div>

                            <button
                                onClick={() => handleInstall(item)}
                                disabled={installingId === item.id}
                                className="mt-1 w-full flex items-center justify-center gap-2 py-1.5 rounded text-[9px] font-mono uppercase tracking-wider transition-all"
                                style={{
                                    backgroundColor: installingId === item.id
                                        ? theme.colors.bg.secondary
                                        : theme.colors.accent.primary + '15',
                                    color: installingId === item.id
                                        ? theme.colors.text.muted
                                        : theme.colors.accent.primary,
                                    border: `1px solid ${installingId === item.id
                                        ? 'transparent'
                                        : theme.colors.accent.primary + '40'}`
                                }}
                            >
                                {installingId === item.id ? (
                                    <>
                                        <Activity size={10} className="animate-spin" />
                                        Baking SDF...
                                    </>
                                ) : (
                                    <>
                                        <Download size={10} />
                                        Prepare for Machining
                                    </>
                                )}
                            </button>
                        </div>
                    ))
                )}

                {!loading && catalog.length === 0 && (
                    <div className="text-center py-8 opacity-50 text-[10px] font-mono">
                        No components found.
                    </div>
                )}
            </div>

            <div className="p-2 border-t" style={{ borderColor: theme.colors.border.primary }}>
                <div className="text-[9px] font-mono opacity-50 text-center">
                    Connected to Global Supply Chain
                </div>
            </div>
        </aside>
    );
};

export default ComponentPanel;
