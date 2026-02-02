import React from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { FileText, Box, FileCode, ChevronLeft, ChevronRight } from 'lucide-react';

const Sidebar = ({ artifacts = [], width = 260, collapsed = false, onToggle }) => {
    const { theme } = useTheme();

    // Mock artifacts if none provided
    const displayArtifacts = artifacts.length > 0 ? artifacts : [
        { id: 1, name: 'requirements.md', type: 'doc' },
        { id: 2, name: 'architecture_diagram.png', type: 'image' },
        { id: 3, name: 'api_specs.json', type: 'code' },
    ];

    return (
        <div
            className="h-full border-r transition-all duration-300 flex flex-col relative"
            style={{
                width: collapsed ? 60 : width,
                backgroundColor: theme.colors.bg.secondary,
                borderColor: theme.colors.border.secondary
            }}
        >
            {/* Toggle Button */}
            <button
                onClick={onToggle}
                className="absolute -right-3 top-6 bg-blue-600 rounded-full p-1 text-white shadow-md z-10 hover:bg-blue-500"
            >
                {collapsed ? <ChevronRight size={12} /> : <ChevronLeft size={12} />}
            </button>

            {/* Header */}
            <div className={`p-4 border-b h-16 flex items-center ${collapsed ? 'justify-center' : ''}`}
                style={{ borderColor: theme.colors.border.secondary }}>
                {collapsed ? (
                    <Box size={24} className="text-blue-500" />
                ) : (
                    <span className="font-bold tracking-wide text-sm opacity-80">PROJECT ARTIFACTS</span>
                )}
            </div>

            {/* List */}
            <div className="flex-1 overflow-y-auto p-2">
                {displayArtifacts.map(art => (
                    <div
                        key={art.id}
                        className={`
                            flex items-center gap-3 p-2 rounded-lg cursor-pointer transition-colors mb-1
                            hover:bg-white/5
                        `}
                    >
                        <div className="text-gray-400">
                            {art.type === 'doc' && <FileText size={18} />}
                            {art.type === 'image' && <Box size={18} />}
                            {art.type === 'code' && <FileCode size={18} />}
                        </div>

                        {!collapsed && (
                            <div className="flex-1 overflow-hidden">
                                <div className="text-sm truncate">{art.name}</div>
                            </div>
                        )}
                    </div>
                ))}
            </div>

            {/* Footer */}
            {!collapsed && (
                <div className="p-4 border-t text-xs opacity-50 text-center" style={{ borderColor: theme.colors.border.secondary }}>
                    v0.1.0-alpha
                </div>
            )}
        </div>
    );
};

export default Sidebar;
