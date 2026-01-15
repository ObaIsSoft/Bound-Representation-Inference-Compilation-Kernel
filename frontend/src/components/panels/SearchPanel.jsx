import React, { useState, useEffect } from 'react';
import { Search, File, Folder, Box, Clock } from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { useTheme } from '../../contexts/ThemeContext';
import { useDesign } from '../../contexts/DesignContext';
import { ASSET_REGISTRY } from '../../utils/assetRegistry';

const SearchPanel = ({ width }) => {
    const { theme } = useTheme();
    const { files, openFile, createNewFile, renameItem, updateTabContent } = useDesign();
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState([]);

    // Live Search Effect
    useEffect(() => {
        if (!searchQuery.trim()) {
            setSearchResults([]);
            return;
        }

        const lowerQuery = searchQuery.toLowerCase();

        // 1. Search Files in Project
        const matchedFiles = files.filter(f =>
            f.name.toLowerCase().includes(lowerQuery)
        ).map(f => ({
            ...f,
            resultType: f.type, // 'file' or 'folder'
            location: '/root'   // Simplified path for now
        }));

        // 2. Search Asset Registry
        const matchedAssets = ASSET_REGISTRY.filter(a =>
            a.name.toLowerCase().includes(lowerQuery) ||
            a.keywords.some(k => k.includes(lowerQuery))
        ).map(a => ({
            ...a,
            resultType: 'asset',
            location: 'Registry',
            name: a.name
        }));

        setSearchResults([...matchedFiles, ...matchedAssets]);

    }, [searchQuery, files]);

    const handleResultClick = (result) => {
        if (result.resultType === 'file') {
            openFile(result);
        } else if (result.resultType === 'asset') {
            // Instantiate Asset: Create new file with asset content
            const newTab = createNewFile();
            // Rename to Asset Name
            // We use setTimeout to ensure state updates don't clash, though React batching handles most
            setTimeout(() => {
                renameItem(newTab.fileId, `${result.name}.brick`);
                // Remove ID/Keywords for clean definition
                const { id, keywords, ...cleanDef } = result;
                updateTabContent(newTab.id, JSON.stringify(cleanDef, null, 2));
            }, 0);
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
            <PanelHeader title="Global Search" icon={Search} />

            <div className="p-3" style={{ borderBottom: `1px solid ${theme.colors.border.primary}80` }}>
                <div className="relative">
                    <input
                        autoFocus
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full rounded px-3 py-2 pl-8 text-xs font-mono outline-none transition-all"
                        placeholder="Search..."
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

                {/* Search Stats */}
                <div className="flex justify-between items-center mt-2 px-1">
                    <span className="text-[9px] uppercase font-mono" style={{ color: theme.colors.text.muted }}>
                        {searchResults.length} found
                    </span>
                    <span className="text-[9px] font-mono flex items-center gap-1" style={{ color: theme.colors.text.tertiary }}>
                        <Clock size={8} /> Live
                    </span>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-2 space-y-1">
                {searchResults.map((result, i) => (
                    <div
                        key={i}
                        onClick={() => handleResultClick(result)}
                        className="p-2 rounded cursor-pointer group flex flex-col gap-1 select-none"
                        style={{
                            backgroundColor: 'transparent',
                            border: '1px solid transparent'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = theme.colors.bg.secondary;
                            e.currentTarget.style.borderColor = theme.colors.border.primary;
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = 'transparent';
                            e.currentTarget.style.borderColor = 'transparent';
                        }}
                    >
                        <div className="flex items-center gap-2">
                            {result.resultType === 'file' && <File size={12} style={{ color: theme.colors.text.secondary }} />}
                            {result.resultType === 'folder' && <Folder size={12} style={{ color: theme.colors.text.secondary }} />}
                            {result.resultType === 'asset' && <Box size={12} style={{ color: theme.colors.accent.primary }} />}

                            <span className="text-[11px] font-mono leading-none truncate" style={{ color: theme.colors.text.primary }}>
                                {result.name}
                            </span>
                        </div>

                        <div className="flex items-center justify-between pl-5">
                            <span className="text-[9px] font-mono truncate opacity-60" style={{ color: theme.colors.text.secondary }}>
                                {result.location}
                            </span>
                            {result.resultType === 'asset' && (
                                <span className="text-[8px] uppercase tracking-wider px-1 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                                    style={{ backgroundColor: theme.colors.accent.primary + '20', color: theme.colors.accent.primary }}>
                                    Insert
                                </span>
                            )}
                        </div>
                    </div>
                ))}

                {searchQuery && searchResults.length === 0 && (
                    <div className="text-center py-8 opacity-50">
                        <span className="text-[10px] font-mono" style={{ color: theme.colors.text.muted }}>
                            No results found.
                        </span>
                    </div>
                )}

                {!searchQuery && (
                    <div className="text-center py-8 opacity-30">
                        <span className="text-[10px] font-mono" style={{ color: theme.colors.text.muted }}>
                            Start typing to search...
                        </span>
                    </div>
                )}
            </div>
        </aside>
    );
};

export default SearchPanel;
