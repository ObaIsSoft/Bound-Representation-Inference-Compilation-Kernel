import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

const PanelContext = createContext();

const API_BASE_URL = 'http://localhost:8000';

export const PANEL_IDS = {
    INPUT: 'inputConsole',
    MAIN: 'mainPanel'
};

export const PanelProvider = ({ children }) => {
    const [panels, setPanels] = useState({
        [PANEL_IDS.INPUT]: { isOpen: true, position: { x: window.innerWidth - 650, y: window.innerHeight - 250 }, size: { width: 600, height: 180 } },
        [PANEL_IDS.MAIN]: { isOpen: true, position: { x: window.innerWidth - 500, y: 50 }, size: { width: 450, height: 800 } }
    });

    const [sessions, setSessions] = useState([]);
    const [activeSessionId, setActiveSessionId] = useState(null);
    const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
    const [activeArtifact, setActiveArtifact] = useState(null);
    const [openTabs, setOpenTabs] = useState([
        { id: 'proj-1', name: 'Drone_v1.brick', type: 'project', icon: 'Package' },
        { id: 'proj-2', name: 'Robotic_Arm.brick', type: 'project', icon: 'Package' },
        { id: 'file-1', name: 'design_plan.md', type: 'document', icon: 'FileText' },
        { id: 'file-2', name: 'geometry.kcl', type: 'code', icon: 'Code' },
        { id: 'file-3', name: 'bom.md', type: 'document', icon: 'FileText' },
    ]);
    const [activeTab, setActiveTab] = useState('proj-1');

    const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];

    const fetchSessions = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/sessions`);
            if (!response.ok) throw new Error('Failed to fetch sessions');
            const data = await response.json();
            // Transform backend snake_case to frontend camelCase if necessary, 
            // but our backend classes use the same names mostly.
            const formattedSessions = data.sessions.map(s => ({
                id: s.conversation_id,
                title: s.title,
                branchName: s.branch_name,
                parentId: s.parent_id,
                lastModified: s.updated_at,
                history: s.messages.map(m => ({
                    role: m.role,
                    content: m.content,
                    timestamp: m.timestamp,
                    metadata: m.metadata
                })),
                status: s.ready_for_planning ? 'active' : 'recent', // Simple mapping
                gatheredRequirements: s.gathered_requirements
            }));

            setSessions(formattedSessions);
            if (!activeSessionId && formattedSessions.length > 0) {
                setActiveSessionId(formattedSessions[0].id);
            }
        } catch (err) {
            console.error('Session fetch error:', err);
        }
    }, [activeSessionId]);

    useEffect(() => {
        fetchSessions();
    }, []);

    const updatePanel = (id, updates) => {
        setPanels(prev => ({
            ...prev,
            [id]: { ...prev[id], ...updates }
        }));
    };

    const addMessageToSession = (sessionId, role, content, metadata = {}) => {
        // Optimistic UI update or full sync from backend
        setSessions(prev => prev.map(s => {
            if (s.id === sessionId) {
                return {
                    ...s,
                    lastModified: new Date().toISOString(),
                    history: [...s.history, { role, content, timestamp: new Date().toISOString(), metadata }]
                };
            }
            return s;
        }));
    };

    const switchSession = (sessionId) => {
        setActiveSessionId(sessionId);
        setIsHistoryModalOpen(false);
        updatePanel(PANEL_IDS.MAIN, { isOpen: true });
    };

    const deleteSession = async (sessionId) => {
        try {
            await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`, { method: 'DELETE' });
            setSessions(prev => prev.filter(s => s.id !== sessionId));
            if (activeSessionId === sessionId) {
                setActiveSessionId(null);
            }
        } catch (err) {
            console.error('Delete session error:', err);
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

    const branchSession = async () => {
        if (!activeSessionId) return;
        try {
            const response = await fetch(`${API_BASE_URL}/api/sessions/${activeSessionId}/branch`, { method: 'POST' });
            if (!response.ok) throw new Error('Branch failed');
            await fetchSessions();
            updatePanel(PANEL_IDS.MAIN, { isOpen: true });
        } catch (err) {
            console.error('Branch error:', err);
        }
    };

    const mergeSession = async (sessionId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/merge`, { method: 'POST' });
            if (!response.ok) throw new Error('Merge failed');
            await fetchSessions();
            setIsHistoryModalOpen(false);
        } catch (err) {
            console.error('Merge error:', err);
        }
    };

    const createNewSession = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/sessions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: 'New Conversation' })
            });
            const data = await response.json();
            await fetchSessions();
            setActiveSessionId(data.conversation_id);
            updatePanel(PANEL_IDS.MAIN, { isOpen: true });
        } catch (err) {
            console.error('Create session error:', err);
        }
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
            fetchSessions,
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
