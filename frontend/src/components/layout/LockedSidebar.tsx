import React, { useState, useRef, useEffect } from 'react';
import { PanelLeft, Plus, Search, Download, Settings, User, BookOpen, Boxes, Play, GitBranch, Factory, FileCheck } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

interface LockedSidebarProps {
    activePanel: string;
    onPanelChange: (panel: string) => void;
    onNewChat: () => void;
    onSettingsClick: () => void;
    onAccountClick: () => void;
    onDocsClick: () => void;
}

import { useSidebar } from '../../contexts/SidebarContext';

export default function LockedSidebar({ activePanel, onPanelChange, onNewChat, onSettingsClick, onAccountClick, onDocsClick }: LockedSidebarProps) {
    const { isCollapsed, setIsCollapsed } = useSidebar();
    const [sidebarWidth, setSidebarWidth] = useState(280); // Default width in pixels
    const [isResizing, setIsResizing] = useState(false);
    const sidebarRef = useRef<HTMLDivElement>(null);
    const { theme } = useTheme();

    const MIN_WIDTH = 200;
    const MAX_WIDTH = 500;

    // Handle resize drag
    const handleMouseDown = (e: React.MouseEvent) => {
        setIsResizing(true);
        e.preventDefault();
    };

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (!isResizing) return;

            const newWidth = e.clientX;
            if (newWidth >= MIN_WIDTH && newWidth <= MAX_WIDTH) {
                setSidebarWidth(newWidth);
            }
        };

        const handleMouseUp = () => {
            setIsResizing(false);
        };

        if (isResizing) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
        }

        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isResizing]);

    // Keyboard shortcut for New Chat (Cmd+K)
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                onNewChat();
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [onNewChat]);

    const iconButtons = [
        { id: 'search', icon: Search, label: 'Search', functional: true },
        { id: 'export', icon: Download, label: 'Export', functional: true },
        { id: 'agent-pods', icon: Boxes, label: 'Agent Pods', functional: false },
        { id: 'compile', icon: FileCheck, label: 'Compile ISA', functional: false },
        { id: 'run-debug', icon: Play, label: 'Run & Debug', functional: false },
        { id: 'manufacturing', icon: Factory, label: 'Manufacturing', functional: false },
        { id: 'version-control', icon: GitBranch, label: 'Version Control', functional: false },
    ];

    return (
        <>
            {/* Sidebar */}
            <div
                ref={sidebarRef}
                className="fixed top-0 left-0 h-screen flex transition-all duration-300 ease-in-out z-40"
                style={{
                    width: isCollapsed ? '0px' : `${sidebarWidth}px`,
                    backgroundColor: theme.colors.bg.secondary,
                    borderRight: isCollapsed ? 'none' : `1px solid ${theme.colors.border.primary}`,
                }}
            >
                {!isCollapsed && (
                    <>
                        {/* Sidebar Content */}
                        <div className="flex-1 flex flex-col overflow-hidden">
                            {/* Header - New Chat Button */}
                            <div className="p-4 border-b" style={{ borderColor: theme.colors.border.primary }}>
                                <button
                                    onClick={onNewChat}
                                    className="w-full flex items-center gap-3 px-4 py-3 rounded-lg font-bold transition-all hover:scale-105 active:scale-95"
                                    style={{
                                        background: `linear-gradient(135deg, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`,
                                        color: theme.colors.bg.primary,
                                    }}
                                >
                                    <Plus size={20} />
                                    <span>New Chat</span>
                                    <div
                                        className="ml-auto px-2 py-0.5 rounded text-xs font-mono opacity-70"
                                        style={{ backgroundColor: theme.colors.bg.primary + '40' }}
                                    >
                                        ⌘K
                                    </div>
                                </button>
                            </div>

                            {/* Icon Stack */}
                            <div className="flex-1 py-4 overflow-y-auto">
                                {iconButtons.map((button) => {
                                    const Icon = button.icon;
                                    const isActive = activePanel === button.id;

                                    return (
                                        <button
                                            key={button.id}
                                            onClick={() => button.functional && onPanelChange(button.id)}
                                            className="w-full flex items-center gap-3 px-6 py-3 transition-all hover:bg-opacity-10 relative group"
                                            style={{
                                                backgroundColor: isActive ? theme.colors.accent.primary + '15' : 'transparent',
                                                borderLeft: isActive ? `3px solid ${theme.colors.accent.primary}` : '3px solid transparent',
                                                color: isActive ? theme.colors.accent.primary : theme.colors.text.primary,
                                                cursor: button.functional ? 'pointer' : 'not-allowed',
                                                opacity: button.functional ? 1 : 0.5,
                                            }}
                                            title={button.label}
                                        >
                                            <Icon size={20} />
                                            <span className="text-sm font-medium">{button.label}</span>
                                            {!button.functional && (
                                                <div className="ml-auto text-xs opacity-60">(Soon)</div>
                                            )}
                                        </button>
                                    );
                                })}
                            </div>

                            {/* Settings, Account & Docs at Bottom */}
                            <div className="p-4 border-t" style={{ borderColor: theme.colors.border.primary }}>
                                <div className="grid grid-cols-3 gap-2">
                                    <button
                                        onClick={onSettingsClick}
                                        className="flex flex-col items-center gap-2 px-2 py-3 rounded-lg transition-all hover:bg-opacity-10"
                                        style={{
                                            backgroundColor: activePanel === 'settings' ? theme.colors.accent.primary + '15' : theme.colors.bg.tertiary,
                                            color: theme.colors.text.primary,
                                        }}
                                    >
                                        <Settings size={20} />
                                        <span className="text-xs font-medium">Settings</span>
                                    </button>
                                    <button
                                        onClick={onAccountClick}
                                        className="flex flex-col items-center gap-2 px-2 py-3 rounded-lg transition-all hover:bg-opacity-10"
                                        style={{
                                            backgroundColor: activePanel === 'account' ? theme.colors.accent.primary + '15' : theme.colors.bg.tertiary,
                                            color: theme.colors.text.primary,
                                        }}
                                    >
                                        <User size={20} />
                                        <span className="text-xs font-medium">Account</span>
                                    </button>
                                    <button
                                        onClick={onDocsClick}
                                        className="flex flex-col items-center gap-2 px-2 py-3 rounded-lg transition-all hover:bg-opacity-10"
                                        style={{
                                            backgroundColor: activePanel === 'docs' ? theme.colors.accent.primary + '15' : theme.colors.bg.tertiary,
                                            color: theme.colors.text.primary,
                                        }}
                                    >
                                        <BookOpen size={20} />
                                        <span className="text-xs font-medium">Docs</span>
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* Resize Handle */}
                        <div
                            onMouseDown={handleMouseDown}
                            className="w-1 cursor-col-resize hover:bg-opacity-100 transition-colors"
                            style={{
                                backgroundColor: isResizing ? theme.colors.accent.primary : theme.colors.border.primary,
                            }}
                        />
                    </>
                )}
            </div>

            {/* Toggle Button (always visible) */}
            <button
                onClick={() => setIsCollapsed(!isCollapsed)}
                className="fixed top-4 z-[999] px-3 py-2 rounded-lg transition-all hover:scale-105 active:scale-95 flex items-center gap-2 shadow-lg"
                style={{
                    left: isCollapsed ? '16px' : `${sidebarWidth + 16}px`,
                    backgroundColor: theme.colors.bg.tertiary,
                    border: `2px solid ${theme.colors.border.primary}`,
                    color: theme.colors.text.primary,
                }}
                title={isCollapsed ? 'Show sidebar' : 'Hide sidebar'}
            >
                <PanelLeft size={24} style={{ transform: isCollapsed ? 'rotate(180deg)' : 'none' }} />
                {isCollapsed && <span className="text-sm font-bold">Sidebar</span>}
            </button>
        </>
    );
}
