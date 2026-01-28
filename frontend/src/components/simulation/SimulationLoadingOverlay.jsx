import React from 'react';
import { Html } from '@react-three/drei';

/**
 * Shared Loading Overlay for Simulation Views
 * Provides a consistent visual style for both OpenSCAD compilation (Mesh) and SDF Generation.
 * 
 * Supports:
 * - Indeterminate Loading (Spinner)
 * - Determinate Loading (Progress Bar)
 */
export const SimulationLoadingOverlay = ({
    message = "PROCESSING...",
    subMessage,
    progress = 0,
    showProgress = false
}) => {
    return (
        <Html center>
            <div className="flex flex-col items-center justify-center p-6 bg-gray-900/90 rounded-xl backdrop-blur-md shadow-2xl border border-white/10 min-w-[240px] transform transition-all duration-300">

                {/* Visual Indicator: Progress Bar or Spinner */}
                {showProgress ? (
                    <div className="w-48 mb-4">
                        <div className="h-1.5 w-full bg-gray-700 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-blue-500 transition-all duration-300 ease-out shadow-[0_0_10px_rgba(59,130,246,0.5)]"
                                style={{ width: `${Math.max(5, progress * 100)}%` }}
                            />
                        </div>
                    </div>
                ) : (
                    <div className="w-8 h-8 border-4 border-t-blue-500 border-b-blue-500 border-r-transparent border-l-transparent rounded-full animate-spin mb-4 shadow-[0_0_15px_rgba(59,130,246,0.3)]"></div>
                )}

                {/* Main Message */}
                <span className="text-white font-mono text-sm tracking-widest font-bold uppercase animate-pulse">
                    {message}
                </span>

                {/* Sub Message or Percentage */}
                {showProgress ? (
                    <span className="text-blue-400 text-xs mt-2 font-mono">
                        {Math.round(progress * 100)}%
                    </span>
                ) : (
                    subMessage && (
                        <span className="text-gray-400 text-xs mt-2 font-sans">
                            {subMessage}
                        </span>
                    )
                )}
            </div>
        </Html>
    );
};
