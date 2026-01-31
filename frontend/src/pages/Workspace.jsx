import React, { useState } from 'react';
import { useTheme } from '../contexts/ThemeContext';
import { X, FileText, Code, Package } from 'lucide-react';
import LockedSidebar from '../components/layout/LockedSidebar';
import { useSidebar } from '../contexts/SidebarContext';

export default function Workspace() {
    const { theme } = useTheme();
    const { isCollapsed: isSidebarCollapsed } = useSidebar();
    const [openTabs, setOpenTabs] = useState([
        { id: 'proj-1', name: 'Drone_v1.brick', type: 'project', icon: Package },
        { id: 'proj-2', name: 'Robotic_Arm.brick', type: 'project', icon: Package },
        { id: 'file-1', name: 'design_plan.md', type: 'document', icon: FileText },
        { id: 'file-2', name: 'geometry.kcl', type: 'code', icon: Code },
        { id: 'file-3', name: 'bom.md', type: 'document', icon: FileText },
    ]);
    const [activeTab, setActiveTab] = useState('proj-1');

    const handleCloseTab = (tabId, e) => {
        e.stopPropagation();
        setOpenTabs(openTabs.filter(tab => tab.id !== tabId));
        if (activeTab === tabId && openTabs.length > 1) {
            const currentIndex = openTabs.findIndex(tab => tab.id === tabId);
            const newActiveTab = openTabs[currentIndex + 1] || openTabs[currentIndex - 1];
            setActiveTab(newActiveTab.id);
        }
    };

    return (
        <div className="flex h-screen" style={{ backgroundColor: theme.colors.bg.primary }}>
            {/* Sidebar */}
            <LockedSidebar
                activePanel=""
                onPanelChange={() => { }}
                onNewChat={() => { }}
                onSettingsClick={() => { }}
                onAccountClick={() => { }}
                onDocsClick={() => { }}
            />

            {/* Floating Vertical Tabs - Only visible when sidebar is collapsed */}
            {isSidebarCollapsed && (
                <div className="fixed left-2 top-20 z-40 flex flex-col gap-1">
                    {openTabs.map((tab) => {
                        const Icon = tab.icon;
                        const isActive = activeTab === tab.id;

                        return (
                            <div
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className="group relative flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-all hover:scale-105 shadow-lg"
                                style={{
                                    backgroundColor: isActive ? theme.colors.accent.primary : theme.colors.bg.tertiary,
                                    border: `2px solid ${isActive ? theme.colors.accent.primary : theme.colors.border.primary}`,
                                    color: isActive ? theme.colors.bg.primary : theme.colors.text.primary,
                                    minWidth: '200px',
                                }}
                            >
                                <Icon size={14} />
                                <span className="flex-1 text-xs font-medium truncate">{tab.name}</span>
                                <button
                                    onClick={(e) => handleCloseTab(tab.id, e)}
                                    className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 rounded hover:bg-black/20"
                                    style={{
                                        color: isActive ? theme.colors.bg.primary : theme.colors.text.muted,
                                    }}
                                >
                                    <X size={12} />
                                </button>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col">
                {/* Content Header */}
                <div
                    className="px-6 py-4 border-b"
                    style={{
                        backgroundColor: theme.colors.bg.secondary,
                        borderColor: theme.colors.border.primary,
                    }}
                >
                    <h1 className="text-2xl font-bold" style={{ color: theme.colors.text.primary }}>
                        {openTabs.find(tab => tab.id === activeTab)?.name || 'Workspace'}
                    </h1>
                </div>

                {/* Content Body */}
                <div className="flex-1 p-6 overflow-y-auto">
                    {activeTab ? (
                        <div style={{ color: theme.colors.text.primary }}>
                            <p className="text-lg">Content for: <strong>{openTabs.find(tab => tab.id === activeTab)?.name}</strong></p>
                            <p className="mt-4 text-sm" style={{ color: theme.colors.text.muted }}>
                                This is where the actual design work happens (3D viewer, code editor, ISA browser, etc.)
                            </p>
                        </div>
                    ) : (
                        <div className="flex items-center justify-center h-full" style={{ color: theme.colors.text.muted }}>
                            <p>No file selected</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
