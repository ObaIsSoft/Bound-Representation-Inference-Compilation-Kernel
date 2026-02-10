import React, { useState, useRef, useCallback } from 'react';
import { useLocation } from 'react-router-dom';
import { useTheme } from '../../contexts/ThemeContext';
import { usePanel } from '../../contexts/PanelContext';
import DraggablePanel from '../shared/DraggablePanel';
import { Image, Pencil, Mic, History, GitGraph, X, Paperclip } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { LLM_PROVIDERS } from '../../utils/constants';
import apiClient from '../../utils/apiClient';

const GlobalInputConsole = () => {
    const { theme } = useTheme();
    const location = useLocation();
    const {
        PANEL_IDS,
        panels,
        togglePanel,
        addMessageToSession,
        isHistoryModalOpen,
        setIsHistoryModalOpen,
        branchSession,
        activeSessionId,
        setActiveSessionId,
        fetchSessions,
        activeTab,
        activeArtifact,
        isSubmitting,
        setIsSubmitting,
        thoughts,
        clearThoughts
    } = usePanel();

    const [message, setMessage] = useState('');
    const [llmProvider, setLlmProvider] = useState('groq');
    const [attachedFiles, setAttachedFiles] = useState([]);
    const [isRecording, setIsRecording] = useState(false);
    const [isTranscribing, setIsTranscribing] = useState(false);
    const [showAttachMenu, setShowAttachMenu] = useState(false);

    const isMainPanelOpen = panels[PANEL_IDS.MAIN]?.isOpen;

    const textareaRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    // File Handling Logic
    const onDrop = useCallback((acceptedFiles) => {
        setAttachedFiles(prev => [...prev, ...acceptedFiles]);
    }, []);

    const { getRootProps, getInputProps, isDragActive, open: openFileDialog } = useDropzone({
        onDrop,
        noClick: true, // We trigger manually via button
        noKeyboard: true
    });

    const removeFile = (index) => {
        setAttachedFiles(prev => prev.filter((_, i) => i !== index));
    };

    const handleSubmit = async () => {
        if ((!message.trim() && attachedFiles.length === 0) || isSubmitting) return;

        const userMsg = message;
        const filesToSend = [...attachedFiles];

        setMessage(''); // Clear input early for responsiveness
        setAttachedFiles([]);
        setIsSubmitting(true);

        try {
            // 1. Gather View Context
            const context = {
                pathname: location.pathname,
                activeTab: activeTab,
                activeArtifact: activeArtifact?.id || null,
                timestamp: new Date().toISOString()
            };

            // 2. Update local state optimistically
            addMessageToSession(activeSessionId, 'user', userMsg, {
                context,
                attachments: filesToSend.map(f => ({ name: f.name, type: f.type, size: f.size }))
            });

            // 3. Prepare Payload (Multipart if files exist)
            let responseData;

            if (filesToSend.length > 0) {
                const formData = new FormData();
                formData.append('message', userMsg);
                formData.append('session_id', activeSessionId);
                formData.append('ai_model', llmProvider);
                formData.append('context', JSON.stringify(context));
                filesToSend.forEach(file => formData.append('files', file));

                // Assuming backend supports multipart/form-data on /chat or a specialized endpoint
                // Ideally, we'd upload files first then send message, but specific implementation depends on backend
                // For now, let's assume standard JSON chat for text, separate for files or mixed if backend supports
                // Fallback to text-only if file upload unimplemented on backend:
                responseData = await apiClient.post('/chat', {
                    message: userMsg,
                    session_id: activeSessionId,
                    ai_model: llmProvider,
                    context: context
                });
            } else {
                responseData = await apiClient.post('/chat', {
                    message: userMsg,
                    session_id: activeSessionId,
                    ai_model: llmProvider,
                    context: context
                });
            }

            // 4. Update sessions
            if (responseData.session_id && responseData.session_id !== activeSessionId) {
                setActiveSessionId(responseData.session_id);
                await fetchSessions();
            }

            // 5. Sync accumulated thoughts to session history
            const targetSessionId = responseData.session_id || activeSessionId;
            if (thoughts && thoughts.length > 0) {
                thoughts.forEach(t => {
                    addMessageToSession(targetSessionId, 'thought', t.text || t.content, {
                        agent: t.agent,
                        timestamp: t.timestamp
                    });
                });
                clearThoughts();
            }

            // 6. Add agent response
            addMessageToSession(targetSessionId, 'agent', responseData.response, {
                intent: responseData.intent
            });

        } catch (err) {
            console.error('Chat error:', err);
            addMessageToSession(activeSessionId, 'agent', 'Sorry, I encountered an error processing your request.');
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    // Voice / STT Logic
    const toggleRecording = async () => {
        if (isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) audioChunksRef.current.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                stream.getTracks().forEach(track => track.stop());
                await transcribeAudio(audioBlob);
            };

            mediaRecorder.start();
            setIsRecording(true);
        } catch (err) {
            console.error('Mic access denied:', err);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            setIsTranscribing(true);
        }
    };

    const transcribeAudio = async (audioBlob) => {
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');
            formData.append('format', 'webm');

            const data = await apiClient.post('/stt/transcribe', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            if (data.transcript) {
                setMessage(prev => (prev ? prev + ' ' + data.transcript : data.transcript));
            }
        } catch (err) {
            console.error('Transcription error:', err);
        } finally {
            setIsTranscribing(false);
        }
    };

    return (
        <DraggablePanel
            id={PANEL_IDS.INPUT}
            headerContent={null}
            className="pointer-events-auto overflow-visible"
            zIndex={60}
        >
            <div
                {...getRootProps()}
                className={`flex flex-col h-full backdrop-blur-xl relative overflow-visible rounded-xl border shadow-2xl transition-colors
                    ${isDragActive ? 'border-accent-primary bg-accent-primary/10' : 'border-white/10 bg-white/10'}`}
            >
                <input {...getInputProps()} />

                {/* Drag Overlay */}
                {isDragActive && (
                    <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 rounded-xl backdrop-blur-sm">
                        <div className="text-white font-bold text-lg flex items-center gap-2">
                            <Paperclip size={24} />
                            Drop to Attach
                        </div>
                    </div>
                )}

                {/* Top Floating Controls */}
                <div className="absolute -top-16 left-0 right-0 flex items-center z-30 overflow-visible pointer-events-none">
                    <div className="flex-1 flex justify-start pointer-events-none">
                        <button
                            onClick={() => setIsHistoryModalOpen(true)}
                            className="flex items-center gap-2 px-3 py-1.5 rounded-lg backdrop-blur-xl border border-white/10 hover:bg-white/5 transition-all shadow-lg pointer-events-auto"
                            style={{ backgroundColor: theme.colors.bg.secondary + 'CC' }}
                        >
                            <History size={14} color={theme.colors.accent.primary} />
                            <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: theme.colors.text.primary }}>Recall</span>
                        </button>
                    </div>

                    <div className="flex justify-center pointer-events-none">
                        {!isMainPanelOpen && (
                            <button
                                onClick={() => togglePanel(PANEL_IDS.MAIN)}
                                className="flex items-center px-4 py-1.5 rounded-lg backdrop-blur-xl border border-white/20 hover:bg-white/10 hover:border-white/30 transition-all shadow-lg pointer-events-auto animate-in fade-in slide-in-from-bottom-2 duration-300"
                                style={{ backgroundColor: theme.colors.bg.secondary + 'E6' }}
                            >
                                <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: theme.colors.text.primary }}>Open Panel</span>
                            </button>
                        )}
                    </div>

                    <div className="flex-1 flex justify-end pointer-events-none">
                        <button
                            onClick={branchSession}
                            className="flex items-center gap-2 px-3 py-1.5 rounded-lg backdrop-blur-xl border border-white/10 hover:bg-white/5 transition-all shadow-lg pointer-events-auto"
                            style={{ backgroundColor: theme.colors.bg.secondary + 'CC' }}
                        >
                            <GitGraph size={14} color={theme.colors.accent.primary} />
                            <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: theme.colors.text.primary }}>Branch</span>
                        </button>
                    </div>
                </div>

                {/* Voice Button */}
                <button
                    onClick={toggleRecording}
                    className={`absolute top-2 right-2 p-2 rounded-full transition-all z-20 ${isRecording ? 'animate-pulse bg-red-500/20' : 'hover:bg-white/10'}`}
                    title={isRecording ? "Stop Recording" : "Start Voice Input"}
                >
                    <Mic
                        size={18}
                        color={isRecording ? '#ef4444' : (isTranscribing ? theme.colors.accent.primary : theme.colors.text.tertiary)}
                        className={isTranscribing ? 'animate-bounce' : ''}
                    />
                </button>

                {/* File Attachments Area (Inside Input) */}
                {attachedFiles.length > 0 && (
                    <div className="px-3 pt-3 flex gap-2 flex-wrap overflow-y-auto max-h-20" style={{ borderColor: theme.colors.border.primary }}>
                        {attachedFiles.map((file, index) => (
                            <div key={index} className="relative group flex items-center gap-2 px-2 py-1 rounded bg-black/20 border border-white/10">
                                {file.type.startsWith('image/') ? (
                                    <img src={URL.createObjectURL(file)} alt="preview" className="w-8 h-8 object-cover rounded" />
                                ) : (
                                    <Paperclip size={16} className="text-white/60" />
                                )}
                                <span className="text-xs max-w-[100px] truncate text-white/80">{file.name}</span>
                                <button
                                    onClick={(e) => { e.stopPropagation(); removeFile(index); }}
                                    className="ml-1 p-0.5 rounded-full hover:bg-red-500/20 text-white/40 hover:text-red-400 transition-colors"
                                >
                                    <X size={12} />
                                </button>
                            </div>
                        ))}
                    </div>
                )}

                {/* Text Area */}
                <div className="p-4 flex-1 pr-10">
                    <textarea
                        ref={textareaRef}
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={isRecording ? "Listening..." : (isTranscribing ? "Transcribing..." : "Type or drop files...")}
                        className="w-full h-full bg-transparent resize-none outline-none font-mono text-sm"
                        style={{ color: theme.colors.text.primary }}
                    />
                </div>

                {/* Toolbar */}
                <div className="px-3 py-2 flex items-center justify-between pr-4 border-t border-white/5 relative">
                    <div className="flex items-center gap-2">
                        <select
                            value={llmProvider}
                            onChange={(e) => setLlmProvider(e.target.value)}
                            className="bg-black/20 rounded px-2 py-1 text-xs outline-none cursor-pointer hover:bg-black/30 transition-colors"
                            style={{ color: theme.colors.text.secondary, border: `1px solid ${theme.colors.border.secondary}` }}
                        >
                            {LLM_PROVIDERS.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
                        </select>

                        {/* Attachment Menu */}
                        <div className="relative">
                            <button
                                onClick={() => setShowAttachMenu(!showAttachMenu)}
                                className={`p-1.5 rounded hover:bg-white/10 transition-colors ${attachedFiles.length > 0 ? 'text-accent-primary' : ''}`}
                                title="Attach..."
                            >
                                <Paperclip size={16} color={attachedFiles.length > 0 ? theme.colors.accent.primary : theme.colors.text.secondary} />
                            </button>

                            {showAttachMenu && (
                                <div
                                    className="absolute bottom-full left-0 mb-2 w-48 rounded-xl shadow-xl backdrop-blur-xl border overflow-hidden animate-in fade-in slide-in-from-bottom-2 z-50"
                                    style={{
                                        backgroundColor: theme.colors.bg.secondary,
                                        borderColor: theme.colors.border.primary
                                    }}
                                >
                                    <button
                                        className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-white/5 transition-colors"
                                        onClick={() => { openFileDialog(); setShowAttachMenu(false); }}
                                    >
                                        <div className="text-blue-400"><Paperclip size={16} /></div>
                                        <span style={{ color: theme.colors.text.primary }} className="text-xs font-bold uppercase tracking-wide">Any File</span>
                                    </button>
                                    <button
                                        className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-white/5 transition-colors"
                                        onClick={() => { /* TODO: Specific image trigger */ openFileDialog(); setShowAttachMenu(false); }}
                                    >
                                        <div className="text-green-400"><Image size={16} /></div>
                                        <span style={{ color: theme.colors.text.primary }} className="text-xs font-bold uppercase tracking-wide">Images</span>
                                    </button>
                                    <button
                                        className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-white/5 transition-colors"
                                        onClick={() => { /* TODO: App specific triggers */ setShowAttachMenu(false); }}
                                    >
                                        <div className="text-purple-400"><GitGraph size={16} /></div>
                                        <span style={{ color: theme.colors.text.primary }} className="text-xs font-bold uppercase tracking-wide">Repo Context</span>
                                    </button>
                                </div>
                            )}
                        </div>

                        <button className="p-1.5 rounded hover:bg-white/10 transition-colors">
                            <Pencil size={16} color={theme.colors.text.secondary} />
                        </button>
                    </div>

                    <button
                        onClick={handleSubmit}
                        disabled={(!message.trim() && attachedFiles.length === 0) || isSubmitting}
                        className="px-4 py-1.5 rounded text-xs font-bold uppercase tracking-wide transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                        style={{
                            backgroundColor: (message.trim() || attachedFiles.length > 0) ? theme.colors.accent.primary : theme.colors.bg.tertiary,
                            color: (message.trim() || attachedFiles.length > 0) ? theme.colors.bg.primary : theme.colors.text.muted
                        }}
                    >
                        {isSubmitting ? 'Sending...' : 'Submit'}
                    </button>
                </div>
            </div>
        </DraggablePanel>
    );
};

export default GlobalInputConsole;
