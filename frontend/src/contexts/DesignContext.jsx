import React, { createContext, useContext, useState, useCallback } from 'react';
import { DEFAULT_FILE_CONTENT } from '../utils/constants';

const DesignContext = createContext();

export const useDesign = () => {
    const context = useContext(DesignContext);
    if (!context) {
        throw new Error('useDesign must be used within a DesignProvider');
    }
    return context;
};

export const DesignProvider = ({ children }) => {
    const [tabs, setTabs] = useState([]);
    const [activeTabId, setActiveTabId] = useState(null);
    const [files, setFiles] = useState([]);

    // Counter for untitled designs
    const [untitledCounter, setUntitledCounter] = useState(1);
    const [isEditorVisible, setIsEditorVisible] = useState(true);

    // Comment management
    const [comments, setComments] = useState({}); // {artifactId: [comments]}
    const [pendingPlanId, setPendingPlanId] = useState(null); // Plan awaiting approval
    const [artifacts, setArtifacts] = useState({}); // {artifactId: {id, title, content}} - Permanent storage

    // Initial File Fetch from NexusAgent
    React.useEffect(() => {
        const fetchFiles = async () => {
            try {
                const res = await fetch('http://localhost:8000/api/agents/nexus/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ operation: 'list', path: '/' })
                });
                const response = await res.json();

                if (response.status === 'success' && response.result.data) {
                    // Merge remote files with current local mock files if needed
                    // For now, we overwrite or append
                    // Ensure unique IDs
                    const remoteFiles = response.result.data;
                    setFiles(prev => {
                        // Avoid duplicates
                        const newFiles = remoteFiles.filter(rf => !prev.find(p => p.id === rf.id));
                        return [...prev, ...newFiles];
                    });
                }
            } catch (e) {
                console.error("Failed to fetch file list from Nexus:", e);
            }
        };
        fetchFiles();
    }, []);

    const createNewFile = useCallback(() => {
        const fileId = `file-${Date.now()}`;
        const fileName = `Untitled Design ${untitledCounter}.brick`;

        const newFile = {
            id: fileId,
            name: fileName,
            type: 'file',
            parentId: null
        };

        const newTab = {
            id: `tab-${Date.now()}`,
            fileId: fileId,
            name: fileName,
            type: 'design',
            content: DEFAULT_FILE_CONTENT,
            modified: true
        };

        setFiles(prev => [...prev, newFile]);
        setTabs(prev => [...prev, newTab]);
        setActiveTabId(newTab.id);
        setUntitledCounter(prev => prev + 1);

        return newTab;
    }, [untitledCounter]);

    const toggleEditor = useCallback(() => {
        setIsEditorVisible(prev => !prev);
    }, []);

    const createNewFolder = useCallback(() => {
        const folderCounter = files.filter(f => f.type === 'folder' && f.name.startsWith('New Folder')).length;
        const folderName = folderCounter === 0 ? 'New Folder' : `New Folder ${folderCounter + 1}`;

        const newFolder = {
            id: `folder-${Date.now()}`,
            name: folderName,
            type: 'folder',
            parentId: null
        };

        setFiles(prev => [...prev, newFolder]);
        return newFolder;
    }, [files]);

    const deleteItem = useCallback((itemId) => {
        setFiles(prev => prev.filter(f => f.id !== itemId));
        // If it's a folder, we might want to delete children too, but flat list for now
        // Also close associated tab if it's a file
        setTabs(prev => prev.filter(t => t.fileId !== itemId));
        if (activeTabId) {
            // Check if active tab was the deleted file
            const activeTabObj = tabs.find(t => t.id === activeTabId);
            if (activeTabObj && activeTabObj.fileId === itemId) {
                setActiveTabId(null);
            }
        }
    }, [tabs, activeTabId]);

    const renameItem = useCallback((itemId, newName) => {
        setFiles(prev => prev.map(f =>
            f.id === itemId ? { ...f, name: newName } : f
        ));
        // Update tab name if open
        setTabs(prev => prev.map(t =>
            t.fileId === itemId ? { ...t, name: newName } : t
        ));
    }, []);

    const moveItem = useCallback((itemId, newParentId) => {
        setFiles(prev => {
            const item = prev.find(f => f.id === itemId);
            if (!item) return prev;
            if (item.id === newParentId) return prev;

            // Cycle detection: Ensure newParentId is not a descendant of itemId
            let currentParentId = newParentId;
            while (currentParentId) {
                if (currentParentId === itemId) return prev; // Cycle detected
                const parent = prev.find(f => f.id === currentParentId);
                currentParentId = parent ? parent.parentId : null;
            }

            return prev.map(f =>
                f.id === itemId ? { ...f, parentId: newParentId } : f
            );
        });
    }, []);

    const closeTab = useCallback((tabId) => {
        setTabs(prev => {
            const filtered = prev.filter(tab => tab.id !== tabId);

            // If closing active tab, switch to another tab
            if (activeTabId === tabId && filtered.length > 0) {
                const currentIndex = prev.findIndex(tab => tab.id === tabId);
                const newActiveIndex = currentIndex > 0 ? currentIndex - 1 : 0;
                setActiveTabId(filtered[newActiveIndex]?.id || null);
            } else if (filtered.length === 0) {
                setActiveTabId(null);
            }

            return filtered;
        });
    }, [activeTabId]);

    const renameTab = useCallback((tabId, newName) => {
        setTabs(prev => prev.map(tab =>
            tab.id === tabId ? { ...tab, name: newName } : tab
        ));
    }, []);

    const updateTabContent = useCallback((tabId, content) => {
        setTabs(prev => prev.map(tab =>
            tab.id === tabId ? { ...tab, content, modified: true } : tab
        ));
    }, []);

    const updateTabMetadata = useCallback((tabId, metadata) => {
        setTabs(prev => prev.map(tab =>
            tab.id === tabId ? { ...tab, ...metadata, modified: true } : tab
        ));
    }, []);

    const openFile = useCallback((file) => {
        // Check if file is already open
        const existingTab = tabs.find(tab => tab.fileId === file.id);
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return;
        }

        // Create new tab for file
        const newTab = {
            id: `tab-${Date.now()}`,
            fileId: file.id,
            name: file.name,
            type: 'file',
            content: null,
            modified: false
        };

        setTabs(prev => [...prev, newTab]);
        setActiveTabId(newTab.id);
    }, [tabs]);

    const createArtifactTab = useCallback((artifact) => {
        // Store artifact permanently (even if tab is closed)
        setArtifacts(prev => ({
            ...prev,
            [artifact.id]: {
                id: artifact.id,
                title: artifact.title || 'Untitled Document',
                content: artifact.content
            }
        }));

        // Check if artifact tab already exists
        const existingTab = tabs.find(tab => tab.artifactId === artifact.id);
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return existingTab;
        }

        // Create new tab for artifact
        const newTab = {
            id: `tab-${Date.now()}`,
            artifactId: artifact.id,
            name: artifact.title || 'Untitled Document',
            type: 'artifact',
            content: artifact.content,
            modified: false,
            readOnly: true
        };

        setTabs(prev => [...prev, newTab]);
        setActiveTabId(newTab.id);
        return newTab;
    }, [tabs]);

    const reopenArtifact = useCallback((artifactId) => {
        // Check if tab already exists
        const existingTab = tabs.find(tab => tab.artifactId === artifactId);
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return existingTab;
        }

        // Recreate tab from stored artifact
        const artifact = artifacts[artifactId];
        if (artifact) {
            return createArtifactTab(artifact);
        }

        console.warn('Artifact not found in storage:', artifactId);
        return null;
    }, [tabs, artifacts, createArtifactTab]);

    const addComment = useCallback(async (artifactId, selection, content) => {
        try {
            const response = await fetch(`http://localhost:8000/api/plans/${artifactId}/comments`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    artifact_id: artifactId,
                    selection,
                    content
                })
            });

            const result = await response.json();

            // Update local state
            setComments(prev => ({
                ...prev,
                [artifactId]: [...(prev[artifactId] || []), result.comment]
            }));

            return result.comment;
        } catch (error) {
            console.error('Failed to add comment:', error);
            return null;
        }
    }, []);

    const getComments = useCallback((artifactId) => {
        return comments[artifactId] || [];
    }, [comments]);

    const requestReview = useCallback(async (planId, userIntent) => {
        try {
            const response = await fetch(`http://localhost:8000/api/plans/${planId}/review`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    plan_id: planId,
                    user_intent: userIntent
                })
            });

            const result = await response.json();

            // Update comments with agent responses
            if (result.responses) {
                setComments(prev => {
                    const updated = { ...prev };
                    result.responses.forEach(resp => {
                        const commentList = updated[planId] || [];
                        const commentIndex = commentList.findIndex(c => c.id === resp.comment_id);
                        if (commentIndex >= 0) {
                            commentList[commentIndex].agent_response = resp.response;
                        }
                    });
                    return updated;
                });
            }

            return result;
        } catch (error) {
            console.error('Failed to request review:', error);
            return null;
        }
    }, []);

    const approvePlan = useCallback(async (planId, userIntent) => {
        try {
            const response = await fetch(`http://localhost:8000/api/plans/${planId}/approve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    plan_id: planId,
                    user_intent: userIntent
                })
            });

            const result = await response.json();

            // Clear pending plan
            setPendingPlanId(null);

            // If KCL code was generated, create/update file in simulation bay
            if (result.kcl_code) {
                // Check if there's an existing KCL file open
                const existingKclTab = tabs.find(t => t.name.endsWith('.kcl'));

                if (existingKclTab) {
                    // Update existing file
                    updateTabContent(existingKclTab.id, result.kcl_code);
                    setActiveTabId(existingKclTab.id);
                } else {
                    // Create new KCL file
                    const newFile = {
                        id: `file-${Date.now()}-kcl`,
                        name: 'design.kcl',
                        type: 'file',
                        parentId: null,
                        isDirectory: false
                    };

                    setFiles(prev => [...prev, newFile]);
                    // Only open if no GLSL (visual pref) or handle multiple
                    openFile(newFile);
                    // Need to actually set content, openFile just creates tab with null?
                    // openFile implementation (line 148) creates tab with content=null. 
                    // We need to update content immediately.
                    // Wait, openFile in my mock above was simpler.
                    // Let's rely on standard flow: create file -> openFile -> updateTabContent
                    // But openFile doesn't take content arg in line 148 definition.
                    // I'll manually set content after opening.
                }
            }

            // Handle GLSL Code (HWC Kernel)
            if (result.glsl_code) {
                const existingGlslTab = tabs.find(t => t.name.endsWith('.glsl'));

                if (existingGlslTab) {
                    updateTabContent(existingGlslTab.id, result.glsl_code);
                    // Set active if it's the primary view
                    setActiveTabId(existingGlslTab.id);
                } else {
                    const newFile = {
                        id: `file-${Date.now()}-glsl`,
                        name: 'design.glsl',
                        type: 'file',
                        parentId: null,
                        isDirectory: false
                    };
                    setFiles(prev => [...prev, newFile]);

                    // Open the file and set content
                    // We need to simulate openFile then update content
                    // Simplified:
                    const newTab = {
                        id: `tab-${Date.now()}-glsl`,
                        fileId: newFile.id,
                        name: newFile.name,
                        type: 'file',
                        content: result.glsl_code,
                        modified: false
                    };
                    setTabs(prev => [...prev, newTab]);
                    setActiveTabId(newTab.id);
                }
            }

            return result;
        } catch (error) {
            console.error('Failed to approve plan:', error);
            return null;
        }
    }, [tabs, updateTabContent, setActiveTabId, openFile, setFiles]);

    const rejectPlan = useCallback(async (planId) => {
        try {
            const response = await fetch(`http://localhost:8000/api/plans/${planId}/reject`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const result = await response.json();
            setPendingPlanId(null);
            return result;
        } catch (error) {
            console.error('Failed to reject plan:', error);
            return null;
        }
    }, []);

    const value = {
        tabs,
        activeTabId,
        setActiveTabId,
        isEditorVisible,
        toggleEditor,
        files,
        setFiles,
        createNewFile,
        createNewFolder,
        deleteItem,
        renameItem,
        moveItem,
        closeTab,
        renameTab,
        updateTabContent,
        updateTabMetadata,
        openFile,
        createArtifactTab,
        reopenArtifact,
        addComment,
        getComments,
        requestReview,
        approvePlan,
        rejectPlan,
        pendingPlanId,
        setPendingPlanId,
        activeTab: tabs.find(tab => tab.id === activeTabId) || null
    };

    return (
        <DesignContext.Provider value={value}>
            {children}
        </DesignContext.Provider>
    );
};
