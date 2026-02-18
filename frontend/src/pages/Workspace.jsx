import React, { useState, useRef } from 'react';
import { useTheme } from '../contexts/ThemeContext';
import { usePanel } from '../contexts/PanelContext';
import LockedSidebar from '../components/layout/LockedSidebar';
import Omniviewport from '../workspace/Omniviewport';
import ThoughtStream from '../components/xai/ThoughtStream';

/**
 * Workspace Page - Main 3D design environment
 * 
 * Follows the same layout pattern as Landing.tsx:
 * - LockedSidebar on the left (collapsible)
 * - Main content area with Omniviewport 3D workspace
 * - Floating ThoughtStream HUD
 * - GlobalOverlay provides chat panels (from App.jsx)
 */
export default function Workspace() {
    const { theme } = useTheme();
    const { openTabs, activeTab } = usePanel();
    const [activePanel, setActivePanel] = useState('');
    const viewportContainerRef = useRef(null);

    // Get current project from active tab or use default
    const currentProject = openTabs.find(tab => tab.id === activeTab)?.projectId || 'default';

    const handlePanelChange = (panel) => {
        setActivePanel(panel === activePanel ? '' : panel);
    };

    const handleNewChat = () => {
        // Reset or create new session
        window.location.reload();
    };

    const handleSettingsClick = () => {
        setActivePanel('settings');
    };

    const handleAccountClick = () => {
        setActivePanel('account');
    };

    const handleDocsClick = () => {
        setActivePanel('docs');
    };

    return (
        <div 
            className="flex h-screen w-full overflow-hidden"
            style={{ backgroundColor: theme.colors.bg.primary }}
        >
            {/* Locked Sidebar - Same as Landing */}
            <LockedSidebar
                activePanel={activePanel}
                onPanelChange={handlePanelChange}
                onNewChat={handleNewChat}
                onSettingsClick={handleSettingsClick}
                onAccountClick={handleAccountClick}
                onDocsClick={handleDocsClick}
            />

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col relative">
                {/* Header */}
                <div 
                    className="h-12 flex items-center justify-between px-6 border-b shrink-0"
                    style={{ 
                        backgroundColor: theme.colors.bg.secondary,
                        borderColor: theme.colors.border.primary 
                    }}
                >
                    <div className="flex items-center gap-3">
                        <span 
                            className="text-sm font-bold uppercase tracking-wider"
                            style={{ color: theme.colors.text.primary }}
                        >
                            {openTabs.find(tab => tab.id === activeTab)?.name || 'Workspace'}
                        </span>
                        <span 
                            className="text-xs px-2 py-0.5 rounded"
                            style={{ 
                                backgroundColor: theme.colors.status.success + '20',
                                color: theme.colors.status.success 
                            }}
                        >
                            Live
                        </span>
                    </div>
                    
                    <div className="flex items-center gap-4">
                        <span 
                            className="text-xs font-mono"
                            style={{ color: theme.colors.text.muted }}
                        >
                            Omniviewport Active
                        </span>
                    </div>
                </div>

                {/* 3D Workspace Container */}
                <div 
                    ref={viewportContainerRef}
                    className="flex-1 relative overflow-hidden"
                    style={{ backgroundColor: theme.colors.bg.primary }}
                >
                    <Omniviewport projectId={currentProject} />
                </div>
            </div>

            {/* Thought Stream HUD - Fixed position like PlanningPage */}
            <div className="fixed bottom-24 right-8 max-w-[280px] z-50 pointer-events-none">
                <div 
                    className="pointer-events-auto rounded-xl border overflow-hidden shadow-2xl"
                    style={{ 
                        backgroundColor: theme.colors.bg.secondary + 'CC',
                        borderColor: theme.colors.border.primary,
                        backdropFilter: 'blur(12px)'
                    }}
                >
                    <ThoughtStream compact={true} />
                </div>
            </div>
        </div>
    );
}
