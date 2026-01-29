import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTheme } from '../../contexts/ThemeContext';

const VoiceVisualizer = ({ state, audioStream }) => {
    const { theme } = useTheme();
    const [audioData, setAudioData] = useState(new Uint8Array(0));
    const analyserRef = useRef(null);
    const animationFrameRef = useRef(null);

    // 1. Audio Analysis Loop
    useEffect(() => {
        if (state === 'listening' && audioStream) {
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioCtx.createMediaStreamSource(audioStream);
            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 128;
            source.connect(analyser);
            analyserRef.current = analyser;

            const update = () => {
                const data = new Uint8Array(analyser.frequencyBinCount);
                analyser.getByteFrequencyData(data);
                setAudioData(data);
                animationFrameRef.current = requestAnimationFrame(update);
            };
            update();

            return () => {
                cancelAnimationFrame(animationFrameRef.current);
                audioCtx.close();
            };
        } else {
            setAudioData(new Uint8Array(0));
        }
    }, [state, audioStream]);

    // Render Logic
    return (
        <div className="relative w-full h-16 flex items-center justify-center overflow-hidden">
            <AnimatePresence mode="wait">
                {/* IDLE STATE: Slow, cyan pulsing ring */}
                {state === 'idle' && (
                    <motion.div
                        key="idle"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 1.2 }}
                        className="relative flex flex-col items-center gap-1"
                    >
                        <motion.div
                            className="w-8 h-8 rounded-full border-2"
                            style={{ borderColor: '#00f2ff' }} // Cyan
                            animate={{
                                scale: [1, 1.2, 1],
                                opacity: [0.3, 0.6, 0.3],
                            }}
                            transition={{
                                duration: 5, // 0.2Hz (1 / 5 = 0.2)
                                repeat: Infinity,
                                ease: "easeInOut"
                            }}
                        />
                        <span className="text-[9px] font-mono text-cyan-400 opacity-50 uppercase tracking-widest animate-pulse">
                            Speak or type to create...
                        </span>
                    </motion.div>
                )}

                {/* LISTENING STATE: Active Waveform */}
                {state === 'listening' && (
                    <motion.div
                        key="listening"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="flex items-center gap-[2px] h-8"
                    >
                        {Array.from({ length: 16 }).map((_, i) => {
                            const val = audioData[i * 2] || 0;
                            const height = Math.max(4, (val / 255) * 32);
                            return (
                                <motion.div
                                    key={i}
                                    className="w-1 rounded-full bg-cyan-400"
                                    animate={{ height }}
                                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                                />
                            );
                        })}
                    </motion.div>
                )}

                {/* PROCESSING STATE: Swarm Activity (57 Particles) */}
                {state === 'processing' && (
                    <motion.div
                        key="processing"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="relative w-full h-full flex items-center justify-center"
                    >
                        {/* Center Point */}
                        <div className="w-1 h-1 rounded-full bg-white animate-ping" />

                        {/* 57 Micro-particles representing agents */}
                        {Array.from({ length: 57 }).map((_, i) => {
                            const angle = (i / 57) * Math.PI * 2;
                            const radius = 20 + Math.random() * 30;
                            const delay = Math.random() * 2;
                            return (
                                <motion.div
                                    key={i}
                                    className="absolute w-[2px] h-[2px] rounded-full bg-cyan-300"
                                    animate={{
                                        x: [
                                            Math.cos(angle) * radius,
                                            Math.cos(angle + 2) * (radius + 10),
                                            Math.cos(angle) * radius
                                        ],
                                        y: [
                                            Math.sin(angle) * radius,
                                            Math.sin(angle + 2) * (radius - 10),
                                            Math.sin(angle) * radius
                                        ],
                                        opacity: [0.2, 1, 0.2],
                                        scale: [0.5, 1.2, 0.5]
                                    }}
                                    transition={{
                                        duration: 2 + Math.random() * 2,
                                        repeat: Infinity,
                                        delay: delay,
                                        ease: "linear"
                                    }}
                                />
                            );
                        })}
                        <span className="absolute bottom-1 text-[8px] font-mono text-white/40 uppercase tracking-tighter">
                            Swarm Analysis in progress...
                        </span>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default VoiceVisualizer;
