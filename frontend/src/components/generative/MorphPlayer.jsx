import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, FastForward } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

const MorphPlayer = ({ sequence, onFrameChange }) => {
    const { theme } = useTheme();
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [speed, setSpeed] = useState(1);
    const [isLooping, setIsLooping] = useState(true);
    const timerRef = useRef(null);

    // Initial load
    useEffect(() => {
        if (sequence && sequence.length > 0) {
            onFrameChange(sequence[0]);
            setIsPlaying(true); // Auto-play
        }
    }, [sequence]); // Removed onFrameChange dependency to avoid infinite loops if stable

    // Animation Loop
    useEffect(() => {
        if (isPlaying) {
            timerRef.current = setInterval(() => {
                setCurrentIndex(prev => {
                    const next = prev + 1;
                    if (next >= sequence.length) {
                        return isLooping ? 0 : prev; // Loop or Stop
                    }
                    return next;
                });
            }, 100 / speed); // Base 100ms per frame
        } else {
            clearInterval(timerRef.current);
        }
        return () => clearInterval(timerRef.current);
    }, [isPlaying, isLooping, sequence.length, speed]);

    // Sync Frame Update
    useEffect(() => {
        if (sequence && sequence[currentIndex]) {
            onFrameChange(sequence[currentIndex]);
        }
    }, [currentIndex, sequence, onFrameChange]);

    const handleScrub = (e) => {
        const val = parseInt(e.target.value);
        setCurrentIndex(val);
        setIsPlaying(false); // Pause on scrub
    };

    if (!sequence || sequence.length === 0) return null;

    return (
        <div
            className="w-full p-3 rounded-lg flex flex-col gap-2 backdrop-blur-md border"
            style={{
                backgroundColor: theme.colors.bg.secondary + '99',
                borderColor: theme.colors.accent.primary + '33'
            }}
        >
            {/* Header / Info */}
            <div className="flex justify-between items-center text-[10px] font-mono text-white/70">
                <span className="uppercase tracking-wider">Evolutionary Morph</span>
                <span>Frame {currentIndex + 1} / {sequence.length}</span>
            </div>

            {/* Scrubber */}
            <input
                type="range"
                min="0"
                max={sequence.length - 1}
                value={currentIndex}
                onChange={handleScrub}
                className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer hover:bg-white/20 accent-primary"
                style={{
                    accentColor: theme.colors.accent.primary
                }}
            />

            {/* Controls */}
            <div className="flex justify-between items-center mt-1">
                <div className="flex gap-2">
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="p-1.5 rounded hover:bg-white/10 transition-colors text-white"
                    >
                        {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                    </button>
                    <button
                        onClick={() => { setCurrentIndex(0); setIsPlaying(true); }}
                        className="p-1.5 rounded hover:bg-white/10 transition-colors text-white/70"
                    >
                        <RotateCcw size={14} />
                    </button>
                    <button
                        onClick={() => setIsLooping(!isLooping)}
                        className={`p-1.5 rounded hover:bg-white/10 transition-colors ${isLooping ? 'text-green-400' : 'text-white/30'}`}
                        title="Toggle Loop"
                    >
                        Loop
                    </button>
                </div>

                {/* Speed Toggle */}
                <div className="flex gap-1">
                    {[0.5, 1, 2].map(s => (
                        <button
                            key={s}
                            onClick={() => setSpeed(s)}
                            className={`px-2 py-0.5 rounded text-[9px] font-mono transition-colors ${speed === s ? 'bg-primary text-white' : 'text-white/50 hover:bg-white/10'}`}
                            style={{ backgroundColor: speed === s ? theme.colors.accent.primary : 'transparent' }}
                        >
                            {s}x
                        </button>
                    ))}
                </div>
            </div>

            {/* Stats Overlay (Scientific Insight) */}
            {sequence[currentIndex].fitness && (
                <div className="mt-1 pt-2 border-t border-white/5 flex gap-4 text-[9px] font-mono text-white/50">
                    <span>FITNESS: {sequence[currentIndex].fitness?.toFixed(2) || 'N/A'}</span>
                    <span>NODES: {sequence[currentIndex].nodes?.length}</span>
                </div>
            )}
        </div>
    );
};

export default MorphPlayer;
