import React, { createContext, useContext, useState, useEffect } from 'react';

const PanelContext = createContext();

export const PANEL_IDS = {
    INPUT: 'inputConsole',
    MAIN: 'mainPanel'
};

const DEFAULT_PANEL_STATE = {
    [PANEL_IDS.INPUT]: {
        isOpen: true,
        position: { x: window.innerWidth / 2 - 250, y: window.innerHeight - 100 }, // Bottom center-ish
        size: { width: 500, height: 60 },
        isMinimized: false
    },
    [PANEL_IDS.MAIN]: {
        isOpen: false,
        position: { x: window.innerWidth - 420, y: 20 }, // Right side
        size: { width: 400, height: window.innerHeight - 40 },
        mode: 'unified' // 'unified' | 'history' | 'artifacts'
    }
};

export const PanelProvider = ({ children }) => {
    // Load persisted state or use defaults
    const [panels, setPanels] = useState(() => {
        const saved = localStorage.getItem('brick_panel_state_v1');
        return saved ? JSON.parse(saved) : DEFAULT_PANEL_STATE;
    });

    // Mock Session State for Branching Logic
    const [activeSession, setActiveSession] = useState({
        id: 'session-main',
        branchName: 'Main',
        parentId: null,
        history: [] // This would connect to backend later
    });

    // Persist on change
    useEffect(() => {
        localStorage.setItem('brick_panel_state_v1', JSON.stringify(panels));
    }, [panels]);

    const updatePanel = (id, updates) => {
        setPanels(prev => ({
            ...prev,
            [id]: { ...prev[id], ...updates }
        }));
    };

    const togglePanel = (id, forceState = null) => {
        setPanels(prev => ({
            ...prev,
            [id]: {
                ...prev[id],
                isOpen: forceState !== null ? forceState : !prev[id].isOpen
            }
        }));
    };

    const setPosition = (id, position) => {
        updatePanel(id, { position });
    };

    const setSize = (id, size) => {
        updatePanel(id, { size });
    };

    return (
        <PanelContext.Provider value={{
            panels,
            updatePanel,
            togglePanel,
            setPosition,
            setSize,
            activeSession,
            setActiveSession,
            PANEL_IDS
        }}>
            {children}
        </PanelContext.Provider>
    );
};

export const usePanel = () => {
    const context = useContext(PanelContext);
    if (!context) {
        throw new Error('usePanel must be used within a PanelProvider');
    }
    return context;
};
