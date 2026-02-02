import React from 'react';
import { useTheme } from '../contexts/ThemeContext';
// import { X, FileText, Code, Package } from 'lucide-react'; 
import LockedSidebar from '../components/layout/LockedSidebar';
// import { useSidebar } from '../contexts/SidebarContext';
import { usePanel } from '../contexts/PanelContext';

export default function Workspace() {
    const { theme } = useTheme();
    // const { isCollapsed: isSidebarCollapsed } = useSidebar();
    const { openTabs, activeTab } = usePanel();

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
