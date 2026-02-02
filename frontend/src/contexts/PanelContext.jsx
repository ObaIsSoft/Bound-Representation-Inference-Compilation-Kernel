import React, { createContext, useContext, useState } from 'react';

const PanelContext = createContext();

export const PANEL_IDS = {
    INPUT: 'inputConsole',
    MAIN: 'mainPanel'
};

export const PanelProvider = ({ children }) => {
    // Minimal mock state for now to prevent crashes
    const [panels, setPanels] = useState({
        [PANEL_IDS.INPUT]: { isOpen: true, position: { x: window.innerWidth - 650, y: window.innerHeight - 250 }, size: { width: 600, height: 180 } },
        [PANEL_IDS.MAIN]: { isOpen: true, position: { x: window.innerWidth - 500, y: 50 }, size: { width: 450, height: 800 } }
    });

    const [sessions, setSessions] = useState([
        {
            id: 'session-main',
            title: 'Implementing Conversation History',
            branchName: 'Main',
            lastModified: new Date(Date.now() - 1 * 60000).toISOString(),
            status: 'active',
            history: []
        },
        {
            id: 'session-2',
            title: 'Integrating Forensic Agent',
            branchName: 'brick',
            lastModified: new Date(Date.now() - 3 * 3600000).toISOString(),
            status: 'blocked',
            history: []
        },
        {
            id: 'session-3',
            title: 'Adding User Icon',
            branchName: 'brick',
            lastModified: new Date(Date.now() - 4 * 86400000).toISOString(),
            status: 'recent',
            history: []
        },
        {
            id: 'session-4',
            title: 'Evolving Tier 4 Agent Capabilities',
            branchName: 'Main',
            lastModified: new Date(Date.now() - 7 * 86400000).toISOString(),
            status: 'recent',
            history: []
        },
        {
            id: 'session-6',
            title: 'Fixing Light Pen Persistence',
            branchName: 'Main',
            lastModified: new Date(Date.now() - 7 * 86400000).toISOString(),
            status: 'recent',
            history: []
        },
        {
            id: 'session-5',
            title: 'Debugging Simulation UI',
            branchName: 'agent',
            lastModified: new Date(Date.now() - 7 * 86400000).toISOString(),
            status: 'other',
            history: []
        },
        {
            id: 'session-7',
            title: 'Deploying Static Portfolio',
            branchName: 'portfolio',
            lastModified: new Date(Date.now() - 7 * 86400000).toISOString(),
            status: 'other',
            history: []
        },
        {
            id: 'session-8',
            title: 'Refining Technical Report',
            branchName: 'os',
            lastModified: new Date(Date.now() - 14 * 86400000).toISOString(),
            status: 'other',
            history: []
        }
    ]);

    const [activeSessionId, setActiveSessionId] = useState('session-main');
    const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);

    const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];

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
        setSessions(prev => prev.map(session => {
            if (session.id === activeSessionId) {
                return {
                    ...session,
                    lastModified: new Date().toISOString(),
                    history: [...session.history, {
                        id: Date.now(),
                        role: 'user',
                        content: message,
                        timestamp: new Date().toISOString()
                    }]
                };
            }
            return session;
        }));
    };

    const switchSession = (sessionId) => {
        setActiveSessionId(sessionId);
        setIsHistoryModalOpen(false);
        updatePanel(PANEL_IDS.MAIN, { isOpen: true });
    };

    const deleteSession = (sessionId) => {
        setSessions(prev => prev.filter(s => s.id !== sessionId));
        if (activeSessionId === sessionId) {
            setActiveSessionId(null);
        }
    };

    const togglePanel = (id) => {
        setPanels(prev => ({
            ...prev,
            [id]: { ...prev[id], isOpen: !prev[id].isOpen }
        }));
    };

    const setPosition = (id, position) => updatePanel(id, { position });
    const setSize = (id, size) => updatePanel(id, { size });

    const branchSession = () => {
        const newSession = {
            id: `session-branch-${Date.now()}`,
            title: `Branch: ${activeSession.title}`,
            branchName: activeSession.branchName,
            lastModified: new Date().toISOString(),
            status: 'active',
            parentId: activeSession.id, // Track parent
            history: [...activeSession.history]
        };
        setSessions(prev => [newSession, ...prev]);
        setActiveSessionId(newSession.id);
        updatePanel(PANEL_IDS.MAIN, { isOpen: true });
    };

    const mergeSession = (sessionId) => {
        const branch = sessions.find(s => s.id === sessionId);
        if (!branch || !branch.parentId) return;

        const parent = sessions.find(s => s.id === branch.parentId);
        if (!parent) return;

        // Take only the new messages from the branch
        const newMessages = branch.history.slice(parent.history.length);

        setSessions(prev => prev
            .map(s => {
                if (s.id === parent.id) {
                    return {
                        ...s,
                        lastModified: new Date().toISOString(),
                        history: [...s.history, ...newMessages]
                    };
                }
                return s;
            })
            .filter(s => s.id !== sessionId) // Delete the branch after merge
        );

        setActiveSessionId(parent.id);
        setIsHistoryModalOpen(false);
    };

    const createNewSession = () => {
        const newSession = {
            id: `session-${Date.now()}`,
            title: 'New Conversation',
            branchName: 'Main',
            lastModified: new Date().toISOString(),
            status: 'active',
            history: []
        };
        setSessions(prev => [newSession, ...prev]);
        setActiveSessionId(newSession.id);
        updatePanel(PANEL_IDS.MAIN, { isOpen: true });
    };

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
            sessions,
            activeSession,
            setActiveSessionId,
            activeSessionId,
            switchSession,
            deleteSession,
            branchSession,
            mergeSession,
            createNewSession,
            isHistoryModalOpen,
            setIsHistoryModalOpen,
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
