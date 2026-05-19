import React, { useState } from 'react';
import { useTheme } from '../contexts/ThemeContext';
import { usePanel } from '../contexts/PanelContext';
import LockedSidebar from '../components/layout/LockedSidebar';
import Omniviewport from '../workspace/Omniviewport';

/**
 * Workspace Page - Main 3D design environment
 * 
 * Follows the same layout pattern as Landing.tsx:
 * - LockedSidebar on the left (collapsible)
 * - Main content area with Omniviewport 3D workspace
 * - GlobalOverlay provides chat panels, tabs, input console (from App.jsx)
 */
export default function Workspace() {
    const { theme } = useTheme();
    const { openTabs, activeTab, activeProjectId, createNewSession } = usePanel();
    const [activePanel, setActivePanel] = useState('');

    // Prefer the actively running project_id; fall back to tab projectId
    const currentProject = activeProjectId || openTabs.find(tab => tab.id === activeTab)?.projectId || 'default';

    const handlePanelChange = (panel) => {
        setActivePanel(panel === activePanel ? '' : panel);
    };

    const handleNewChat = async () => {
        await createNewSession();
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

            {/* Main Content Area - Just the 3D viewport */}
            <div className="flex-1 relative">
                <Omniviewport projectId={currentProject} />
            </div>
        </div>
    );
}
