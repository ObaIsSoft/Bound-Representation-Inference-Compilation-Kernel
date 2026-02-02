import React, { createContext, useContext, useState } from 'react';

const PanelContext = createContext();

export const PANEL_IDS = {
    INPUT: 'inputConsole',
    MAIN: 'mainPanel'
};

export const PanelProvider = ({ children }) => {
    // Minimal mock state for now to prevent crashes
    const [panels, setPanels] = useState({
        [PANEL_IDS.INPUT]: { isOpen: true, position: { x: window.innerWidth - 500, y: window.innerHeight - 200 }, size: { width: 450, height: 120 } },
        [PANEL_IDS.MAIN]: { isOpen: true, position: { x: window.innerWidth - 500, y: 100 }, size: { width: 450, height: 720 } }
    });

    const [activeSession, setActiveSession] = useState({
        id: 'session-main',
        branchName: 'Main',
        history: []
    });

    // Global Floating Tabs State
    const [openTabs, setOpenTabs] = useState([
        { id: 'proj-1', name: 'Drone_v1.brick', type: 'project', icon: 'Package' },
        { id: 'proj-2', name: 'Robotic_Arm.brick', type: 'project', icon: 'Package' },
        { id: 'file-1', name: 'design_plan.md', type: 'document', icon: 'FileText' },
        { id: 'file-2', name: 'geometry.kcl', type: 'code', icon: 'Code' },
        { id: 'file-3', name: 'bom.md', type: 'document', icon: 'FileText' },
    ]);
    const [activeTab, setActiveTab] = useState('proj-1');

    const [activeArtifact, setActiveArtifact] = useState(null);

    const updatePanel = (id, updates) => {
        setPanels(prev => ({
            ...prev,
            [id]: { ...prev[id], ...updates }
        }));
    };

    const addMessageToSession = (message) => {
        setActiveSession(prev => ({
            ...prev,
            history: [...prev.history, {
                id: Date.now(),
                role: 'user',
                content: message,
                timestamp: new Date().toISOString()
            }]
        }));
    };

    const togglePanel = (id) => {
        setPanels(prev => ({
            ...prev,
            [id]: { ...prev[id], isOpen: !prev[id].isOpen }
        }));
    };

    const setPosition = (id, position) => updatePanel(id, { position });
    const setSize = (id, size) => updatePanel(id, { size });

    const viewArtifact = (artifact) => {
        setActiveArtifact(artifact);
        updatePanel(PANEL_IDS.MAIN, { isOpen: true });
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
            activeArtifact,
            activeArtifact,
            setActiveArtifact,
            viewArtifact,
            addMessageToSession,
            openTabs,
            setOpenTabs,
            activeTab,
            setActiveTab,
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
