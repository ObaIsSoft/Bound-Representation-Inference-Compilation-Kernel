import React, { useState } from 'react';
import { Search, FolderPlus, FilePlus, File, BoxSelect, ChevronRight, ChevronDown } from 'lucide-react';
import PanelHeader from '../shared/PanelHeader';
import { Package } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';
import { useDesign } from '../../contexts/DesignContext';
import SidebarContextMenu from './SidebarContextMenu';

const DesignLibrary = ({ width }) => {
    const { theme } = useTheme();
    const { files, createNewFile, createNewFolder, openFile, deleteItem, renameItem, moveItem } = useDesign();

    // UI State
    const [contextMenu, setContextMenu] = useState(null);
    const [renamingId, setRenamingId] = useState(null);
    const [renameValue, setRenameValue] = useState('');
    const [expandedFolders, setExpandedFolders] = useState(new Set());
    const [dragOverId, setDragOverId] = useState(null);

    const toggleFolder = (folderId) => {
        setExpandedFolders(prev => {
            const next = new Set(prev);
            if (next.has(folderId)) next.delete(folderId);
            else next.add(folderId);
            return next;
        });
    };

    // Drag and Drop Handlers
    const handleDragStart = (e, item) => {
        e.dataTransfer.setData('text/plain', item.id);
        e.dataTransfer.effectAllowed = 'move';
    };

    const handleDragOver = (e, item) => {
        e.preventDefault();
        e.stopPropagation();

        // Allow drop on folders or empty space (root)
        if (!item || item.type === 'folder') {
            setDragOverId(item ? item.id : 'root');
            e.dataTransfer.dropEffect = 'move';
        }
    };

    const handleRenameSubmit = () => {
        if (!renamingId) return;

        if (renameValue.trim()) {
            renameItem(renamingId, renameValue.trim());
        }
        setRenamingId(null);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragOverId(null);
    };

    const handleDrop = (e, targetItem) => {
        e.preventDefault();
        e.stopPropagation();
        setDragOverId(null);

        const draggedId = e.dataTransfer.getData('text/plain');
        if (!draggedId) return;

        // Target is either a folder ID, or null (root)
        const targetId = targetItem ? targetItem.id : null;

        moveItem(draggedId, targetId);

        // Auto-expand target folder if it wasn't expanded
        if (targetId) {
            setExpandedFolders(prev => new Set(prev).add(targetId));
        }
    };

    const handleContextMenu = (e, item) => {
        e.preventDefault();
        e.stopPropagation();
        setContextMenu({
            x: e.clientX,
            y: e.clientY,
            type: item ? item.type : 'background',
            targetId: item ? item.id : null
        });
    };

    const handleAction = (action) => {
        if (!contextMenu) return;

        const { targetId } = contextMenu;

        switch (action) {
            case 'new_file':
                createNewFile();
                break;
            case 'new_folder':
                createNewFolder();
                break;
            case 'delete':
                if (targetId) deleteItem(targetId);
                break;
            case 'rename':
                if (targetId) {
                    const item = files.find(f => f.id === targetId);
                    if (item) {
                        setRenamingId(targetId);
                        setRenameValue(item.name);
                    }
                }
                break;
            default:
                break;
        }
    };

    const renderTree = (parentId, depth = 0) => {
        const children = files.filter(f => f.parentId === parentId);

        return children.map(item => {
            const isFolder = item.type === 'folder';
            const isExpanded = expandedFolders.has(item.id);
            const isRenaming = renamingId === item.id;

            return (
                <div key={item.id} style={{ marginLeft: depth * 12 }}>
                    {isRenaming ? (
                        <input
                            autoFocus
                            value={renameValue}
                            onChange={(e) => setRenameValue(e.target.value)}
                            onBlur={handleRenameSubmit}
                            onKeyDown={(e) => e.key === 'Enter' && handleRenameSubmit()}
                            className="w-full p-2 rounded text-[10px] font-mono outline-none mb-1"
                            style={{
                                backgroundColor: theme.colors.bg.primary,
                                border: `1px solid ${theme.colors.accent.primary}`,
                                color: theme.colors.text.primary
                            }}
                            onClick={(e) => e.stopPropagation()}
                        />
                    ) : (
                        <div
                            draggable
                            onDragStart={(e) => handleDragStart(e, item)}
                            onDragOver={(e) => isFolder ? handleDragOver(e, item) : null}
                            onDragLeave={handleDragLeave}
                            onDrop={(e) => isFolder ? handleDrop(e, item) : null}
                            onClick={(e) => {
                                e.stopPropagation();
                                if (isFolder) toggleFolder(item.id);
                                else openFile(item);
                            }}
                            onContextMenu={(e) => handleContextMenu(e, item)}
                            className="p-1.5 rounded text-[10px] cursor-pointer flex items-center justify-between group font-mono transition-all mb-0.5 select-none"
                            style={{
                                backgroundColor: dragOverId === item.id
                                    ? theme.colors.accent.primary + '20'
                                    : (theme.colors.bg.primary + '80'),
                                border: `1px solid ${dragOverId === item.id
                                    ? theme.colors.accent.primary
                                    : (theme.colors.border.primary + '80')}`
                            }}
                            onMouseEnter={(e) => {
                                if (dragOverId !== item.id) {
                                    e.currentTarget.style.borderColor = theme.colors.accent.primary + '80';
                                }
                            }}
                            onMouseLeave={(e) => {
                                if (dragOverId !== item.id) {
                                    e.currentTarget.style.borderColor = theme.colors.border.primary + '80';
                                }
                            }}
                        >
                            <div className="flex items-center gap-2 truncate pr-2">
                                {isFolder && (
                                    <div className="w-3 h-3 flex items-center justify-center">
                                        {isExpanded ?
                                            <ChevronDown size={10} style={{ color: theme.colors.text.muted }} /> :
                                            <ChevronRight size={10} style={{ color: theme.colors.text.muted }} />
                                        }
                                    </div>
                                )}
                                {!isFolder && <div className="w-3" />}

                                {isFolder ?
                                    <FolderPlus size={10} style={{ color: theme.colors.accent.primary }} /> :
                                    <File size={10} style={{ color: theme.colors.text.muted }} />
                                }
                                <span style={{ color: theme.colors.text.tertiary }}>{item.name}</span>
                            </div>
                        </div>
                    )}
                    {isFolder && isExpanded && renderTree(item.id, depth + 1)}
                </div>
            );
        });
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
            <PanelHeader title="Explorer" icon={Package} />

            <div className="p-3" style={{ borderBottom: `1px solid ${theme.colors.border.primary}80` }}>
                <div className="relative">
                    <Search size={12} className="absolute left-2 top-2.5" style={{ color: theme.colors.text.muted }} />
                    <input
                        className="w-full rounded px-8 py-1.5 text-[10px] font-mono outline-none"
                        placeholder="Filter files..."
                        style={{
                            backgroundColor: theme.colors.bg.primary,
                            border: `1px solid ${theme.colors.border.primary}`,
                            color: theme.colors.text.primary
                        }}
                        onFocus={(e) => e.target.style.borderColor = theme.colors.accent.primary + '80'}
                        onBlur={(e) => e.target.style.borderColor = theme.colors.border.primary}
                    />
                </div>
            </div>

            <div
                className="flex-1 p-3 overflow-y-auto space-y-4"
                onContextMenu={(e) => handleContextMenu(e, null)}
            >
                <div onClick={() => setContextMenu(null)}>

                    <div className="space-y-1 min-h-[100px]"
                        onDragOver={(e) => handleDragOver(e, null)}
                        onDrop={(e) => handleDrop(e, null)}
                        style={{
                            backgroundColor: dragOverId === 'root' ? theme.colors.accent.primary + '10' : 'transparent',
                            borderRadius: '4px'
                        }}
                    >
                        {renderTree(null)}
                    </div>
                </div>
            </div>

            <div
                className="p-2 grid grid-cols-2 gap-2 shrink-0"
                style={{
                    backgroundColor: theme.colors.bg.primary,
                    borderTop: `1px solid ${theme.colors.border.primary}`
                }}
            >
                <button
                    onClick={createNewFile}
                    className="flex items-center justify-center gap-1.5 p-1.5 rounded text-[9px] font-mono transition-colors uppercase"
                    style={{
                        backgroundColor: theme.colors.bg.secondary,
                        border: `1px solid ${theme.colors.border.primary}`,
                        color: theme.colors.text.tertiary
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = theme.colors.accent.primary + '20';
                        e.currentTarget.style.borderColor = theme.colors.accent.primary;
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = theme.colors.bg.secondary;
                        e.currentTarget.style.borderColor = theme.colors.border.primary;
                    }}
                >
                    <FilePlus size={12} /> New File
                </button>
                <button
                    onClick={createNewFolder}
                    className="flex items-center justify-center gap-1.5 p-1.5 rounded text-[9px] font-mono transition-colors uppercase"
                    style={{
                        backgroundColor: theme.colors.bg.secondary,
                        border: `1px solid ${theme.colors.border.primary}`,
                        color: theme.colors.text.tertiary
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = theme.colors.accent.primary + '20';
                        e.currentTarget.style.borderColor = theme.colors.accent.primary;
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = theme.colors.bg.secondary;
                        e.currentTarget.style.borderColor = theme.colors.border.primary;
                    }}
                >
                    <FolderPlus size={12} /> New Folder
                </button>
            </div>
            {contextMenu && (
                <SidebarContextMenu
                    x={contextMenu.x}
                    y={contextMenu.y}
                    type={contextMenu.type}
                    onClose={() => setContextMenu(null)}
                    onAction={handleAction}
                />
            )}
        </aside>
    );
};

export default DesignLibrary;
