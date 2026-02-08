import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Play, Pause, RotateCcw, Check, Loader2 } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';
import apiClient from '../../utils/apiClient';

interface VoiceRecorderProps {
    onTranscriptionComplete: (transcription: string) => void;
    onCancel: () => void;
}

type RecorderStatus = 'idle' | 'recording' | 'playback' | 'transcribing';

export default function VoiceRecorder({ onTranscriptionComplete, onCancel }: VoiceRecorderProps) {
    const [status, setStatus] = useState<RecorderStatus>('idle');
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [transcription, setTranscription] = useState<string>('');
    const [duration, setDuration] = useState<number>(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [error, setError] = useState<string>('');

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const audioElementRef = useRef<HTMLAudioElement | null>(null);
    const timerRef = useRef<NodeJS.Timeout>();
    const { theme } = useTheme();

    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
            if (mediaRecorderRef.current?.state === 'recording') {
                mediaRecorderRef.current.stop();
            }
        };
    }, []);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                setAudioBlob(blob);
                setStatus('playback');
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorderRef.current = mediaRecorder;
            mediaRecorder.start();
            setStatus('recording');
            setDuration(0);
            setError('');

            // Start timer
            timerRef.current = setInterval(() => {
                setDuration(prev => prev + 1);
            }, 1000);
        } catch (err) {
            setError('Microphone access denied. Please enable microphone permissions.');
            console.error('Error accessing microphone:', err);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
            if (timerRef.current) clearInterval(timerRef.current);
        }
    };

    const playAudio = () => {
        if (!audioBlob) return;

        const audio = new Audio(URL.createObjectURL(audioBlob));
        audioElementRef.current = audio;

        audio.onplay = () => setIsPlaying(true);
        audio.onended = () => setIsPlaying(false);
        audio.onerror = () => {
            setIsPlaying(false);
            setError('Failed to play audio');
        };

        audio.play();
    };

    const pauseAudio = () => {
        if (audioElementRef.current) {
            audioElementRef.current.pause();
            setIsPlaying(false);
        }
    };

    const reRecord = () => {
        setAudioBlob(null);
        setTranscription('');
        setDuration(0);
        setError('');
        startRecording();
    };

    const transcribeAudio = async () => {
        if (!audioBlob) return;

        setStatus('transcribing');
        setError('');

        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');
            formData.append('format', 'webm');

            const data = await apiClient.post('/stt/transcribe', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                }
            });





            setTranscription(data.transcript);
            setStatus('playback'); // Return to playback to show transcription
        } catch (err) {
            setError('Transcription failed. Please try again.');
            console.error('Transcription error:', err);
            setStatus('playback');
        }
    };

    const confirmTranscription = () => {
        if (transcription) {
            onTranscriptionComplete(transcription);
        }
    };

    const formatDuration = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div
            className="flex flex-col items-center gap-6 p-8 rounded-2xl backdrop-blur-sm"
            style={{
                backgroundColor: theme.colors.bg.secondary + 'CC',
                border: `1px solid ${theme.colors.border.primary}`,
            }}
        >
            {/* Status Indicator */}
            <div className="flex items-center gap-3">
                {status === 'recording' && (
                    <>
                        <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
                        <span style={{ color: theme.colors.text.primary }} className="font-mono">
                            Recording: {formatDuration(duration)}
                        </span>
                    </>
                )}
                {status === 'playback' && (
                    <span style={{ color: theme.colors.text.muted }} className="font-mono">
                        Duration: {formatDuration(duration)}
                    </span>
                )}
                {status === 'transcribing' && (
                    <>
                        <Loader2 size={16} className="animate-spin" style={{ color: theme.colors.accent.primary }} />
                        <span style={{ color: theme.colors.text.primary }} className="font-mono">
                            Transcribing...
                        </span>
                    </>
                )}
            </div>

            {/* Error Message */}
            {error && (
                <div className="text-sm text-red-400 bg-red-900/20 px-4 py-2 rounded-lg">
                    {error}
                </div>
            )}

            {/* Transcription Display */}
            {transcription && (
                <div
                    className="w-full p-4 rounded-lg font-mono text-sm"
                    style={{
                        backgroundColor: theme.colors.bg.tertiary,
                        color: theme.colors.text.primary,
                        border: `1px solid ${theme.colors.accent.primary}`,
                    }}
                >
                    "{transcription}"
                </div>
            )}

            {/* Controls */}
            <div className="flex gap-4">
                {status === 'idle' && (
                    <button
                        onClick={startRecording}
                        className="px-6 py-3 rounded-full transition-all duration-200 hover:scale-110 active:scale-95"
                        style={{
                            background: `linear-gradient(135deg, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`,
                            color: theme.colors.bg.primary,
                        }}
                    >
                        <Mic size={24} />
                    </button>
                )}

                {status === 'recording' && (
                    <button
                        onClick={stopRecording}
                        className="px-6 py-3 rounded-full transition-all duration-200 hover:scale-110 active:scale-95"
                        style={{
                            backgroundColor: theme.colors.status.error,
                            color: theme.colors.text.primary,
                        }}
                    >
                        <Square size={24} />
                    </button>
                )}

                {status === 'playback' && !transcription && (
                    <>
                        <button
                            onClick={isPlaying ? pauseAudio : playAudio}
                            className="px-4 py-2 rounded-lg transition-all hover:scale-105"
                            style={{
                                backgroundColor: theme.colors.bg.tertiary,
                                border: `1px solid ${theme.colors.border.primary}`,
                                color: theme.colors.text.primary,
                            }}
                        >
                            {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                        </button>
                        <button
                            onClick={reRecord}
                            className="px-4 py-2 rounded-lg transition-all hover:scale-105"
                            style={{
                                backgroundColor: theme.colors.bg.tertiary,
                                border: `1px solid ${theme.colors.border.primary}`,
                                color: theme.colors.text.primary,
                            }}
                        >
                            <RotateCcw size={20} />
                        </button>
                        <button
                            onClick={transcribeAudio}
                            className="px-6 py-2 rounded-lg font-bold transition-all hover:scale-105"
                            style={{
                                background: `linear-gradient(135deg, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`,
                                color: theme.colors.bg.primary,
                            }}
                        >
                            Transcribe
                        </button>
                    </>
                )}

                {status === 'playback' && transcription && (
                    <>
                        <button
                            onClick={reRecord}
                            className="px-4 py-2 rounded-lg transition-all hover:scale-105"
                            style={{
                                backgroundColor: theme.colors.bg.tertiary,
                                border: `1px solid ${theme.colors.border.primary}`,
                                color: theme.colors.text.primary,
                            }}
                        >
                            <RotateCcw size={20} />
                        </button>
                        <button
                            onClick={confirmTranscription}
                            className="px-6 py-2 rounded-lg font-bold transition-all hover:scale-105 flex items-center gap-2"
                            style={{
                                background: `linear-gradient(135deg, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`,
                                color: theme.colors.bg.primary,
                            }}
                        >
                            <Check size={20} />
                            Confirm
                        </button>
                    </>
                )}
            </div>

            {/* Cancel */}
            <button
                onClick={onCancel}
                className="text-sm opacity-60 hover:opacity-100 transition-opacity"
                style={{ color: theme.colors.text.muted }}
            >
                Cancel
            </button>
        </div>
    );
}
