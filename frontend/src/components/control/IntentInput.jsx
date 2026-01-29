import React, { useState, useRef, useEffect } from 'react';
import { Sparkles, Send, Mic, Square, Play, X, Check, Trash2, ChevronDown, Pencil, ShieldCheck } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';
import VoiceVisualizer from './VoiceVisualizer';

const IntentInput = ({
    value,
    onChange,
    onSend,
    isProcessing,
    aiModel,
    setAiModel,
    sketchMode,
    setSketchMode
}) => {
    const { theme } = useTheme();
    const [showModelMenu, setShowModelMenu] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [audioBlob, setAudioBlob] = useState(null);
    const [audioUrl, setAudioUrl] = useState(null);
    const [visualState, setVisualState] = useState('idle'); // idle, listening, processing
    const [audioStream, setAudioStream] = useState(null);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            setAudioStream(stream);
            mediaRecorderRef.current = new MediaRecorder(stream);
            audioChunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
                setAudioBlob(blob);
                setAudioUrl(URL.createObjectURL(blob));
                stream.getTracks().forEach(track => track.stop());
                setAudioStream(null);
                setVisualState('idle');
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);
            setVisualState('listening');
        } catch (err) {
            console.error("Error accessing microphone:", err);
            alert("Microphone access denied or not available.");
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const clearRecording = () => {
        setAudioBlob(null);
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        setAudioUrl(null);
        setVisualState('idle');
    };

    const handleSyncToSwarm = async () => {
        if (!audioBlob) return;
        setVisualState('processing');

        // Sync raw blob to the swarm orchestrator
        // This will be handled by the parent's onSend which now supports blobs
        onSend("", audioBlob);

        // Cleanup after short delay to allow visual to populate
        setTimeout(() => {
            clearRecording();
            setVisualState('idle');
        }, 500);
    };

    const playRecording = () => {
        if (audioUrl) {
            const audio = new Audio(audioUrl);
            audio.play();
        }
    };

    return (
        <div className="px-4 pb-4 shrink-0 flex flex-col relative" style={{ backgroundColor: theme.colors.bg.primary }}>
            <div className="relative group rounded-xl border border-white/5 bg-white/[0.02] backdrop-blur-md transition-all focus-within:border-white/20 shadow-2xl">
                {/* --- SEAMLESS VOICE OVERLAY --- */}
                {isRecording && (
                    <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-black/90 backdrop-blur-xl animate-in fade-in duration-500">
                        <VoiceVisualizer state={visualState} audioStream={audioStream} />
                        <div className="mt-6 flex flex-col items-center gap-3">
                            <div className="text-[10px] font-mono text-cyan-400 animate-pulse tracking-[0.3em] uppercase font-black">Biometric Sync Active</div>
                            <button
                                onClick={stopRecording}
                                className="px-5 py-2 rounded-full bg-red-500 text-white text-[10px] font-black tracking-widest flex items-center gap-2 hover:scale-105 active:scale-95 transition-all shadow-[0_0_20px_rgba(239,68,68,0.4)]"
                            >
                                <Square size={12} fill="currentColor" /> STOP CAPTURE
                            </button>
                        </div>
                    </div>
                )}

                <textarea
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            onSend();
                        }
                    }}
                    className="w-full p-4 text-[9px] font-mono resize-none outline-none h-40 transition-all scrollbar-hide pb-16 leading-relaxed"
                    placeholder="Describe your hardware intent..."
                    style={{
                        backgroundColor: 'transparent',
                        color: theme.colors.text.primary,
                        caretColor: theme.colors.accent.primary
                    }}
                />

                {/* --- UNIFIED PREMIUM STATUS DOCK --- */}
                <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between p-1 backdrop-blur-2xl rounded-lg border shadow-2xl transition-all group-focus-within:border-white/20"
                    style={{
                        backgroundColor: theme.colors.bg.secondary + 'CC',
                        borderColor: theme.colors.border.primary
                    }}>
                    <div className="flex items-center gap-1">
                        {/* Voice Control Node */}
                        <div className="flex items-center pr-2 mr-1 ml-1" style={{ borderRight: `1px solid ${theme.colors.border.primary}` }}>
                            {!audioUrl && !isRecording ? (
                                <button
                                    onClick={startRecording}
                                    className="p-2 hover:bg-white/10 rounded-md transition-all group/mic"
                                    style={{ color: theme.colors.text.muted }}
                                >
                                    <Mic size={16} className="group-hover/mic:scale-110 transition-transform" />
                                </button>
                            ) : audioUrl && !isRecording ? (
                                <div className="flex items-center gap-1 rounded-md p-1 border"
                                    style={{ backgroundColor: theme.colors.accent.primary + '1A', borderColor: theme.colors.accent.primary + '33' }}>
                                    <button onClick={playRecording} className="p-1.5 hover:bg-cyan-400/20 rounded transition-colors" style={{ color: theme.colors.accent.primary }}>
                                        <Play size={14} fill="currentColor" />
                                    </button>
                                    <button onClick={clearRecording} className="p-1.5 hover:bg-red-500/20 rounded text-red-400 transition-colors">
                                        <Trash2 size={14} />
                                    </button>
                                </div>
                            ) : null}
                        </div>

                        {/* Model Selector Hub */}
                        <div className="relative">
                            <button
                                onClick={() => setShowModelMenu(!showModelMenu)}
                                className="flex items-center gap-2 px-3 py-1.5 rounded-md hover:bg-white/5 transition-all group/model"
                            >
                                <Sparkles size={14} style={{ color: theme.colors.accent.primary }} className="group-hover/model:rotate-12 transition-transform" />
                                <span className="text-[9px] font-black font-mono tracking-tighter uppercase" style={{ color: theme.colors.text.primary + 'CC' }}>
                                    {aiModel === 'openai' ? 'OpenAI GPT-4' :
                                        aiModel === 'gemini-robotics' ? 'Gemini Bio' :
                                            aiModel?.includes('codex') ? 'Codex 5' :
                                                aiModel?.includes('huggingface') ? 'HuggingFace' :
                                                    aiModel?.includes('opus') ? 'Claude Opus' :
                                                        aiModel?.includes('2.5-pro') ? 'Gemini Pro' :
                                                            aiModel?.includes('2.5-flash') ? 'Gemini Flash' :
                                                                aiModel?.includes('groq') ? 'Groq Llama' :
                                                                    aiModel?.includes('claude') ? 'Claude 3.5' :
                                                                        'Logic Kernel'}
                                </span>
                                <ChevronDown size={8} className="opacity-30" style={{ color: theme.colors.text.primary }} />
                            </button>

                            {showModelMenu && (
                                <div className="absolute bottom-full left-0 mb-2 w-44 border rounded-lg shadow-[0_15px_40px_rgba(0,0,0,0.6)] overflow-y-auto max-h-[160px] animate-in slide-in-from-bottom-2 duration-200 z-50 p-1 scrollbar-thin scrollbar-thumb-white/10"
                                    style={{
                                        backgroundColor: theme.colors.bg.secondary,
                                        borderColor: theme.colors.border.primary
                                    }}>
                                    {[
                                        { id: 'openai', label: 'GPT-4 (OpenAI)', desc: 'Reasoning' },
                                        { id: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro', desc: 'Optimization' },
                                        { id: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash', desc: 'Real-time' },
                                        { id: 'gemini-robotics', label: 'Gemini 2.1 Bio', desc: 'Robotics' },
                                        { id: 'claude-3-5-sonnet', label: 'Claude 3.5 Sonnet', desc: 'Creative' },
                                        { id: 'claude-3-opus', label: 'Claude 3 Opus', desc: 'Symbolic' },
                                        { id: 'codex-5', label: 'OpenAI Codex 5', desc: 'Hardware DSL' },
                                        { id: 'huggingface-meta-llama', label: 'HuggingFace Llama', desc: 'Open Swarm' },
                                        { id: 'groq', label: 'Groq Llama 3', desc: 'Ultrafast' }
                                    ].map(m => (
                                        <button
                                            key={m.id}
                                            onClick={() => {
                                                setAiModel(m.id);
                                                setShowModelMenu(false);
                                            }}
                                            className="w-full h-9 px-2 text-left hover:bg-white/5 rounded-md transition-all flex flex-col justify-center gap-0"
                                        >
                                            <div className="flex items-center justify-between">
                                                <span className="text-[9px] font-bold font-mono tracking-tight"
                                                    style={{ color: aiModel === m.id ? theme.colors.accent.primary : theme.colors.text.primary }}>
                                                    {m.label}
                                                </span>
                                                {aiModel === m.id && <Check size={8} style={{ color: theme.colors.accent.primary }} />}
                                            </div>
                                            <div className="text-[7px] font-mono uppercase tracking-widest"
                                                style={{ color: theme.colors.text.muted }}>
                                                {m.desc}
                                            </div>
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Sketch Mode Valve */}
                        <button
                            onClick={() => setSketchMode(!sketchMode)}
                            className="flex items-center gap-2 px-3 py-1.5 rounded-md hover:bg-white/5 transition-all group/sketch"
                            style={{
                                color: sketchMode ? theme.colors.accent.primary : theme.colors.text.muted,
                                borderLeft: `1px solid ${theme.colors.border.primary}`
                            }}
                        >
                            <Pencil size={14} className={sketchMode ? 'animate-pulse' : 'opacity-40'} />
                            <span className="text-[9px] font-black font-mono tracking-widest uppercase">{sketchMode ? 'Sketching' : 'Linear'}</span>
                        </button>
                    </div>

                    <div className="flex items-center gap-2 pr-1">
                        {/* Action Trigger - Circular Arrow Style per Image 2 */}
                        <button
                            onClick={audioUrl && !isRecording ? handleSyncToSwarm : onSend}
                            className="w-8 h-8 rounded-full flex items-center justify-center transition-all active:scale-90 shadow-lg hover:brightness-110"
                            style={{
                                background: `linear-gradient(135deg, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`,
                                color: '#000'
                            }}
                        >
                            <Send size={16} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default IntentInput;
