import React, { useState, useRef } from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { usePanel } from '../../contexts/PanelContext';
import DraggablePanel from '../shared/DraggablePanel';
import { Image, Pencil, Mic } from 'lucide-react';

const LLM_PROVIDERS = [
    { value: 'groq', label: 'Groq (Llama 3.3 70B)' },
    { value: 'openai', label: 'OpenAI (GPT-4 Turbo)' },
    { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
    { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
    { value: 'ollama', label: 'Ollama (Local)' },
];

const GlobalInputConsole = () => {
    const { theme } = useTheme();
    const { PANEL_IDS, addMessageToSession } = usePanel();
    const [message, setMessage] = useState('');
    const [llmProvider, setLlmProvider] = useState('groq');
    const [attachedImages, setAttachedImages] = useState([]);
    const [isRecording, setIsRecording] = useState(false);
    const [isTranscribing, setIsTranscribing] = useState(false);

    const fileInputRef = useRef(null);
    const textareaRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    const handleSubmit = () => {
        if (!message.trim()) return;
        console.log('Sending message:', message, { llmProvider, attachedImages });
        // TODO: Wire to actual backend or session context
        addMessageToSession(message);
        setMessage('');
        setAttachedImages([]);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
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
            // Optionally show error toast here
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

            const response = await fetch('http://localhost:8000/api/stt/transcribe', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error('Transcription failed');

            const data = await response.json();
            if (data.transcript) {
                setMessage(prev => (prev ? prev + ' ' + data.transcript : data.transcript));
            }
        } catch (err) {
            console.error('Transcription error:', err);
        } finally {
            setIsTranscribing(false);
        }
    };

    const handleImageAttach = () => fileInputRef.current?.click();

    const handleFileChange = (e) => {
        if (e.target.files) {
            const files = Array.from(e.target.files);
            setAttachedImages(prev => [...prev, ...files]);
        }
    };

    const removeImage = (index) => {
        setAttachedImages(prev => prev.filter((_, i) => i !== index));
    };

    // Styling derived from TextInput.tsx but adapted for JSX and drag wrapper
    return (
        <DraggablePanel
            id={PANEL_IDS.INPUT}
            headerContent={null}
            className="pointer-events-auto"
        >
            <div className="flex flex-col h-full bg-white/5 backdrop-blur-md relative">
                {/* Voice Button (Top Right Absolute) */}
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

                {/* Images Preview */}
                {attachedImages.length > 0 && (
                    <div className="p-3 flex gap-2 flex-wrap border-b" style={{ borderColor: theme.colors.border.primary }}>
                        {attachedImages.map((file, index) => (
                            <div key={index} className="relative group w-16 h-16 rounded-md overflow-hidden">
                                <img src={URL.createObjectURL(file)} alt="preview" className="w-full h-full object-cover" />
                                <button
                                    onClick={() => removeImage(index)}
                                    className="absolute top-0 right-0 bg-red-500 text-white w-4 h-4 flex items-center justify-center text-xs opacity-0 group-hover:opacity-100"
                                >
                                    ×
                                </button>
                            </div>
                        ))}
                    </div>
                )}

                {/* Text Area */}
                <div className="p-4 flex-1 pr-10"> {/* Add padding-right to avoid overlap with Mic */}
                    <textarea
                        ref={textareaRef}
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={isRecording ? "Listening..." : (isTranscribing ? "Transcribing..." : "Type to create...")}
                        className="w-full h-full bg-transparent resize-none outline-none font-mono text-sm"
                        style={{ color: theme.colors.text.primary }}
                    />
                </div>

                {/* Toolbar */}
                <div className="px-3 py-2 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <select
                            value={llmProvider}
                            onChange={(e) => setLlmProvider(e.target.value)}
                            className="bg-black/20 rounded px-2 py-1 text-xs outline-none cursor-pointer hover:bg-black/30 transition-colors"
                            style={{ color: theme.colors.text.secondary, border: `1px solid ${theme.colors.border.secondary}` }}
                        >
                            {LLM_PROVIDERS.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
                        </select>

                        <button onClick={handleImageAttach} className="p-1.5 rounded hover:bg-white/10 transition-colors">
                            <Image size={16} color={theme.colors.text.secondary} />
                        </button>
                        <input ref={fileInputRef} type="file" hidden multiple accept="image/*" onChange={handleFileChange} />

                        <button className="p-1.5 rounded hover:bg-white/10 transition-colors">
                            <Pencil size={16} color={theme.colors.text.secondary} />
                        </button>
                    </div>

                    <button
                        onClick={handleSubmit}
                        disabled={!message.trim()}
                        className="px-4 py-1.5 rounded text-xs font-bold uppercase tracking-wide transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                        style={{
                            backgroundColor: message.trim() ? theme.colors.accent.primary : theme.colors.bg.tertiary,
                            color: message.trim() ? theme.colors.bg.primary : theme.colors.text.muted
                        }}
                    >
                        Submit
                    </button>
                </div>
            </div>
        </DraggablePanel>
    );
};

export default GlobalInputConsole;
