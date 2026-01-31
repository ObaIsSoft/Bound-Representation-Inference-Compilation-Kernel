import React, { useEffect, useRef } from 'react';
import { useTheme } from '../../contexts/ThemeContext';

interface WaveformVisualizerProps {
    isActive: boolean;
    audioStream?: MediaStream;
}

export default function WaveformVisualizer({ isActive, audioStream }: WaveformVisualizerProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationRef = useRef<number>();
    const audioContextRef = useRef<AudioContext>();
    const analyserRef = useRef<AnalyserNode>();
    const dataArrayRef = useRef<Uint8Array>();
    const { theme } = useTheme();

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Set canvas size
        const size = Math.min(window.innerWidth * 0.4, 400);
        canvas.width = size;
        canvas.height = size;

        const centerX = size / 2;
        const centerY = size / 2;
        const baseRadius = size * 0.3;
        let time = 0;

        // Setup audio context if active and stream is provided
        if (isActive && audioStream) {
            const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 128; // 64 frequency bins
            const source = audioContext.createMediaStreamSource(audioStream);
            source.connect(analyser);

            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);

            audioContextRef.current = audioContext;
            analyserRef.current = analyser;
            dataArrayRef.current = dataArray;
        }

        const drawIdleWaveform = () => {
            ctx.clearRect(0, 0, size, size);

            // Slow pulsing effect (0.2Hz)
            time += 0.01;
            const pulse = Math.sin(time) * 0.15 + 1;
            const radius = baseRadius * pulse;

            // Radial gradient (black hole accretion disk)
            const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
            gradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
            gradient.addColorStop(0.5, theme.colors.accent.primary + '40');
            gradient.addColorStop(1, theme.colors.accent.primary);

            // Draw pulsing ring
            ctx.strokeStyle = gradient;
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
            ctx.stroke();

            // Inner glow
            ctx.strokeStyle = theme.colors.accent.primary + '20';
            ctx.lineWidth = 8;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius - 5, 0, Math.PI * 2);
            ctx.stroke();

            animationRef.current = requestAnimationFrame(drawIdleWaveform);
        };

        const drawActiveSpectrum = () => {
            if (!analyserRef.current || !dataArrayRef.current) {
                drawIdleWaveform();
                return;
            }

            ctx.clearRect(0, 0, size, size);

            const analyser = analyserRef.current;
            const dataArray = dataArrayRef.current;
            const bufferLength = analyser.frequencyBinCount;

            analyser.getByteFrequencyData(dataArray);

            // Subtle rotation for accretion disk effect
            time += 0.005;

            // Draw radial bars for each frequency bin
            const barCount = 64;
            const angleStep = (Math.PI * 2) / barCount;

            ctx.save();
            ctx.translate(centerX, centerY);
            ctx.rotate(time);

            for (let i = 0; i < barCount; i++) {
                const angle = i * angleStep;
                const frequencyIndex = Math.floor((i / barCount) * bufferLength);
                const amplitude = dataArray[frequencyIndex] / 255;

                // Bar length based on frequency amplitude
                const barLength = baseRadius * 0.5 * amplitude;
                const innerRadius = baseRadius * 0.7;
                const outerRadius = innerRadius + barLength;

                // Color based on amplitude (cyan to turquoise)
                const hue = amplitude > 0.5 ? 180 : 190;
                const color = `hsla(${hue}, 70%, ${50 + amplitude * 30}%, ${0.6 + amplitude * 0.4})`;

                ctx.strokeStyle = color;
                ctx.lineWidth = (size / barCount) * 0.8;
                ctx.beginPath();
                ctx.moveTo(
                    Math.cos(angle) * innerRadius,
                    Math.sin(angle) * innerRadius
                );
                ctx.lineTo(
                    Math.cos(angle) * outerRadius,
                    Math.sin(angle) * outerRadius
                );
                ctx.stroke();
            }

            // Draw center glow
            const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, baseRadius);
            gradient.addColorStop(0, theme.colors.accent.primary + '40');
            gradient.addColorStop(1, 'transparent');
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(0, 0, baseRadius, 0, Math.PI * 2);
            ctx.fill();

            ctx.restore();

            animationRef.current = requestAnimationFrame(drawActiveSpectrum);
        };

        // Start animation loop
        if (isActive) {
            drawActiveSpectrum();
        } else {
            drawIdleWaveform();
        }

        // Cleanup
        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
            if (audioContextRef.current) {
                audioContextRef.current.close();
            }
        };
    }, [isActive, audioStream, theme]);

    return (
        <div className="relative flex flex-col items-center justify-center">
            <canvas
                ref={canvasRef}
                className="transition-opacity duration-500"
                style={{ opacity: isActive || !audioStream ? 1 : 0.7 }}
            />
            {!isActive && (
                <p
                    className="absolute bottom-0 text-sm font-mono opacity-40 animate-pulse"
                    style={{ color: theme.colors.text.muted }}
                >
                    Speak or type to create...
                </p>
            )}
        </div>
    );
}
