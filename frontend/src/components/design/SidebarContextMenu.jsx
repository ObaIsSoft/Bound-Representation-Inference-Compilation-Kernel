import React, { useEffect, useRef } from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { FilePlus, FolderPlus, Edit2, Trash2, Scissors, Copy, Clipboard } from 'lucide-react';

const SidebarContextMenu = ({ x, y, type, onClose, onAction }) => {
    const { theme } = useTheme();
    const menuRef = useRef(null);

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                onClose();
            }
        };

        const handleEscape = (e) => {
            if (e.key === 'Escape') onClose();
        };

        document.addEventListener('mousedown', handleClickOutside);
        document.addEventListener('keydown', handleEscape);
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
            document.removeEventListener('keydown', handleEscape);
        };
    }, [onClose]);

    // Adjust position to stay within viewport
    const style = {
        top: y,
        left: x,
        backgroundColor: theme.colors.bg.secondary,
        border: `1px solid ${theme.colors.border.primary}`,
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
        zIndex: 1000
    };

    const MenuItem = ({ icon: Icon, label, action, danger = false }) => (
        <button
            onClick={() => { onAction(action); onClose(); }}
            className="w-full text-left px-3 py-1.5 flex items-center gap-2 text-[11px] font-mono hover:bg-white/5 transition-colors"
            style={{
                color: danger ? theme.colors.status.error : theme.colors.text.primary
            }}
        >
            {Icon && <Icon size={12} style={{ opacity: 0.7 }} />}
            <span>{label}</span>
        </button>
    );

    const Separator = () => (
        <div className="h-px my-1 w-full" style={{ backgroundColor: theme.colors.border.primary }} />
    );

    return (
        <div
            ref={menuRef}
            className="fixed min-w-[160px] py-1 rounded-sm flex flex-col"
            style={style}
        >
            <MenuItem icon={FilePlus} label="New File" action="new_file" />
            <MenuItem icon={FolderPlus} label="New Folder" action="new_folder" />

            {(type === 'file' || type === 'folder') && (
                <>
                    <Separator />
                    <MenuItem icon={Scissors} label="Cut" action="cut" />
                    <MenuItem icon={Copy} label="Copy" action="copy" />
                    <MenuItem icon={Clipboard} label="Paste" action="paste" />
                    <Separator />
                    <MenuItem icon={Edit2} label="Rename" action="rename" />
                    <MenuItem icon={Trash2} label="Delete" action="delete" danger />
                </>
            )}
        </div>
    );
};

export default SidebarContextMenu;
