import React from 'react';
import { usePanel } from '../../contexts/PanelContext';
import { useTheme } from '../../contexts/ThemeContext';
import { useSidebar } from '../../contexts/SidebarContext';
import { X, FileText, Code, Package } from 'lucide-react';

const GlobalTabs = () => {
    const { theme } = useTheme();
    const { openTabs, activeTab, setActiveTab, setOpenTabs } = usePanel();
    const { isCollapsed } = useSidebar();

    // Icon mapping since we stored strings in Context
    const getIcon = (iconName) => {
        const icons = { Package, FileText, Code };
        return icons[iconName] || FileText;
    };

    const handleCloseTab = (tabId, e) => {
        e.stopPropagation();
        const newTabs = openTabs.filter(tab => tab.id !== tabId);
        setOpenTabs(newTabs);

        if (activeTab === tabId && newTabs.length > 0) {
            const currentIndex = openTabs.findIndex(tab => tab.id === tabId);
            const newActiveTab = newTabs[currentIndex - 1] || newTabs[0]; // Safer fallback
            setActiveTab(newActiveTab.id);
        } else if (newTabs.length === 0) {
            setActiveTab(null);
        }
    };

    // Only show if sidebar is collapsed (to avoid overlap)
    if (!isCollapsed) return null;

    return (
        <div className="fixed left-2 top-20 z-40 flex flex-col gap-1 pointer-events-auto">
            {openTabs.map((tab) => {
                const Icon = getIcon(tab.icon);
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
    );
};

export default GlobalTabs;
