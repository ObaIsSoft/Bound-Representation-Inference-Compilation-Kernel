
import React, { useRef, useState, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import { useTheme } from '../../contexts/ThemeContext';
import * as THREE from 'three';

/**
 * ZoomLevelMeter
 * 
 * Displays a vertical gauge indicating the current camera zoom level.
 * Style: Thin vertical ruler with sliding indicator (HUD style).
 * Position: Top Left (2D Overlay).
 */
export const ZoomLevelMeter = ({ baseDims = [1, 1, 1] }) => {
    const { camera, controls } = useThree();
    const { theme } = useTheme();
    const indicatorRef = useRef();

    // FIXED CONSTANTS DEBUG MODE - RETAINED FOR STABILITY
    // 10cm to 100m Range (Logarithmic)
    const MIN_DIST = 0.1;   // 10cm
    const MAX_DIST = 100.0; // 100m

    // Pre-calculate logs for linear interpolation
    const minLog = Math.log(MIN_DIST);
    const maxLog = Math.log(MAX_DIST);
    const scaleLog = maxLog - minLog;

    // Helper to format units
    const formatDist = (val) => {
        if (!isFinite(val)) return '---';
        const absVal = Math.abs(val);
        if (absVal >= 10) return `${val.toFixed(0)}m`;
        if (absVal >= 1) return `${val.toFixed(1)}m`;
        if (absVal >= 0.1) return `${(val * 100).toFixed(0)}cm`;
        return `${(val * 1000).toFixed(0)}mm`;
    };

    // Generate ticks (Logarithmically spaced)
    const TICK_COUNT = 11;
    const ticks = Array.from({ length: TICK_COUNT }, (_, i) => {
        const t = i / (TICK_COUNT - 1);
        const val = Math.exp(minLog + (t * scaleLog));
        return val;
    });

    // State for interactivity
    const isDragging = useRef(false);
    const containerRef = useRef();

    useFrame(() => {
        if (!controls || !camera || !indicatorRef.current) return;

        const target = controls.target || new THREE.Vector3(0, 0, 0);
        const currentDist = camera.position.distanceTo(target);

        // Reader Loop
        if (isDragging.current) return;

        const safeDist = Math.max(MIN_DIST, Math.min(MAX_DIST, currentDist));
        const logVal = Math.log(safeDist);
        let linearParams = (logVal - minLog) / scaleLog;

        // Invert for UI (0 at max/bottom, 1 at min/top)
        let norm = 1.0 - linearParams;
        norm = Math.max(0, Math.min(1, norm));

        indicatorRef.current.style.bottom = `${norm * 100}%`;
    });

    // Interaction Handlers (Writer)
    const handlePointerDown = (e) => {
        isDragging.current = true;
        // DISABLE Controls to avoid fighting
        if (controls) controls.enabled = false;

        handlePointerMove(e);
        e.stopPropagation();
        e.preventDefault();
        window.addEventListener('pointermove', handlePointerMove);
        window.addEventListener('pointerup', handlePointerUp);
    };

    const handlePointerUp = (e) => {
        isDragging.current = false;
        // RE-ENABLE Controls
        if (controls) {
            controls.enabled = true;
            controls.update();
        }

        if (e) { e.preventDefault(); e.stopPropagation(); }
        window.removeEventListener('pointermove', handlePointerMove);
        window.removeEventListener('pointerup', handlePointerUp);
    };

    const handlePointerMove = (e) => {
        if (!isDragging.current || !containerRef.current || !camera || !controls) return;

        const rect = containerRef.current.getBoundingClientRect();
        const y = rect.bottom - e.clientY;
        const height = rect.height;
        let norm = y / height;
        norm = Math.max(0, Math.min(1, norm));

        if (indicatorRef.current) indicatorRef.current.style.bottom = `${norm * 100}%`;

        const linearParams = 1.0 - norm;
        const targetDist = Math.exp(minLog + (linearParams * scaleLog));

        // Move Camera
        const target = controls.target || new THREE.Vector3(0, 0, 0);

        // Safety: Ensure we don't divide by zero or get NaNs
        const currentPos = camera.position.clone();
        const direction = new THREE.Vector3().subVectors(currentPos, target).normalize();

        if (direction.lengthSq() < 0.0001 || !isFinite(direction.x)) {
            direction.set(0, 0, 1); // Default backward if stacked on target
        }

        const newPos = target.clone().add(direction.multiplyScalar(targetDist));
        camera.position.copy(newPos);
        controls.object.position.copy(newPos); // Explicitly update control's object ref
    };

    // Manual Reset Handler
    const handleCenterView = (e) => {
        e.stopPropagation();
        if (controls && camera) {
            controls.target.set(0, 0, 0);
            camera.position.set(5, 5, 5);
            camera.lookAt(0, 0, 0);
            controls.update();
        }
    };

    // Theme Colors
    const textColor = theme?.colors?.text?.primary || '#ffffff';
    const tickColor = theme?.colors?.text?.tertiary || '#888888';
    const accentColor = theme?.colors?.accent?.primary || '#ffffff';
    const borderColor = theme?.colors?.border?.primary || 'rgba(255,255,255,0.2)';

    return (
        <Html
            fullscreen
            style={{
                pointerEvents: 'none',
                zIndex: 10
            }}
        >
            <div className="w-full h-full flex items-center justify-start pl-[2%] pointer-events-none">

                {/* Meter Container (Interactable) */}
                <div className="relative h-[60%] flex flex-col items-center gap-2 pointer-events-auto">

                    {/* Recenter Button (Target Icon) */}
                    <button
                        onClick={handleCenterView}
                        className="mb-1 p-1 rounded-full border border-opacity-30 hover:bg-white/10 active:bg-white/20 transition-all group"
                        style={{ borderColor: tickColor, color: textColor }}
                        title="Recenter View (Origin)"
                    >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="22" y1="12" x2="18" y2="12"></line>
                            <line x1="6" y1="12" x2="2" y2="12"></line>
                            <line x1="12" y1="6" x2="12" y2="2"></line>
                            <line x1="12" y1="22" x2="12" y2="18"></line>
                        </svg>
                    </button>

                    <div
                        ref={containerRef}
                        className="relative h-full flex flex-row items-center gap-2 cursor-ns-resize"
                        onPointerDown={handlePointerDown}
                    >

                        {/* The Scale (Ruler) */}
                        <div className="relative h-full w-2 border-r" style={{ borderColor: borderColor }}>
                            {/* Tick Marks */}
                            <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
                                {ticks.map((val, i) => (
                                    <div key={i} className="flex flex-row items-center justify-end w-24 -ml-20 pr-1">
                                        <span className="text-[9px] font-mono font-bold opacity-80" style={{ color: i % 5 === 0 ? textColor : tickColor }}>
                                            {formatDist(val)}
                                        </span>
                                        <div
                                            className={`ml-2 h-px ${i === 0 || i === TICK_COUNT - 1 ? 'w-3 opacity-100' : 'w-2 opacity-60'}`}
                                            style={{ backgroundColor: tickColor }}
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* The Indicator */}
                        <div className="relative h-full w-4 -ml-px">
                            <div
                                ref={indicatorRef}
                                className="absolute left-0 w-0 h-0 border-y-[5px] border-y-transparent border-l-[6px] transition-none pointer-events-none"
                                style={{
                                    borderLeftColor: accentColor,
                                    transform: 'translateY(5px)'
                                }}
                            />
                        </div>

                        {/* Vertical Label */}
                        <div className="absolute -right-8 top-1/2 -translate-y-1/2 -rotate-90 text-[9px] font-mono font-bold tracking-widest opacity-40 whitespace-nowrap pointer-events-none" style={{ color: tickColor }}>
                            DISTANCE
                        </div>
                    </div>
                </div>
            </div>
        </Html>
    );
};
