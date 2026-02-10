import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import apiClient from '../utils/apiClient';
import { useThoughtStream } from '../hooks/useThoughtStream';

const PanelContext = createContext();

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
    const [openTabs, setOpenTabs] = useState([]);
    const [activeTab, setActiveTab] = useState('chat');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [leftPanelRequest, setLeftPanelRequest] = useState(null);

    const [isAgentProcessing, setIsAgentProcessing] = useState(false);

    // XAI: Thought Streaming Hook
    // Only poll when agent is processing
    const { thoughts, isStreaming, clearThoughts } = useThoughtStream(isAgentProcessing, 1000);

    const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];

    // ... (fetchSessions restored) ...
    const fetchSessions = useCallback(async () => {
        try {
            const data = await apiClient.get('/sessions');
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
            await apiClient.delete(`/sessions/${sessionId}`);
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
            await apiClient.post(`/sessions/${activeSessionId}/branch`);
            await fetchSessions();
            updatePanel(PANEL_IDS.MAIN, { isOpen: true });
        } catch (err) {
            console.error('Branch error:', err);
        }
    };

    const mergeSession = async (sessionId) => {
        try {
            await apiClient.post(`/sessions/${sessionId}/merge`);
            await fetchSessions();
            setIsHistoryModalOpen(false);
        } catch (err) {
            console.error('Merge error:', err);
        }
    };

    const createNewSession = async () => {
        try {
            const data = await apiClient.post('/sessions', { title: 'New Conversation' });
            await fetchSessions();
            setActiveSessionId(data.conversation_id);
            updatePanel(PANEL_IDS.MAIN, { isOpen: true });
            return data.conversation_id;
        } catch (err) {
            console.error('Create session error:', err);
            return null;
        }
    };

    const startNewSession = async (initialData = {}) => {
        // Create session first
        const sessionId = await createNewSession();
        if (sessionId) {
            // Optionally update with initial data if backend supports it update
            // For now, we just ensure it's active
            setActiveSessionId(sessionId);
            return sessionId;
        }
        return null;
    };

    const viewArtifact = (artifact) => {
        // artifact should have { id, name, type, ... }
        if (!artifact || !artifact.id) return;

        setActiveArtifact(artifact);

        // Tab Lifecycle Wiring
        setOpenTabs(prev => {
            const exists = prev.find(t => t.id === artifact.id);
            if (exists) return prev;

            // Map artifact type to icon
            let icon = 'Box';
            if (artifact.type === 'document' || artifact.id.endsWith('.md')) icon = 'FileText';
            if (artifact.type === 'code' || artifact.id.endsWith('.py') || artifact.id.endsWith('.js')) icon = 'Code';

            return [...prev, {
                id: artifact.id,
                name: artifact.name || artifact.title || artifact.id.split('/').pop(),
                type: artifact.type || 'artifact',
                icon: icon
            }];
        });

        setActiveTab(artifact.id);
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
            startNewSession,
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
            isSubmitting,
            setIsSubmitting,
            isAgentProcessing,
            setIsAgentProcessing,
            thoughts,
            isStreaming,
            clearThoughts,
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
