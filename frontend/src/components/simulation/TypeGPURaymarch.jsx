import React, { useEffect, useState } from 'react';
import RaymarchScene from './RaymarchScene';

/**
 * BRICK OS: TypeGPU/WebGPU Bridge with GLSL Fallback
 * 
 * Strategy:
 * 1. Attempt to initialize WebGPU via browser flags
 * 2. If unavailable, gracefully fall back to proven GLSL kernel
 * 3. Log capability status for debugging
 */

export default function TypeGPURaymarch({ design, isaState }) {
    const [webgpuAvailable, setWebgpuAvailable] = useState(null);
    const [initError, setInitError] = useState(null);

    useEffect(() => {
        async function detectWebGPU() {
            try {
                // Check if WebGPU is available
                if (!navigator.gpu) {
                    console.warn('[TypeGPU Bridge] WebGPU not available in this WebView.');
                    console.warn('[TypeGPU Bridge] Ensure Tauri config has --enable-unsafe-webgpu flag.');
                    console.warn('[TypeGPU Bridge] Falling back to GLSL kernel.');
                    setWebgpuAvailable(false);
                    return;
                }

                // Attempt to request adapter
                const adapter = await navigator.gpu.requestAdapter();
                if (!adapter) {
                    console.warn('[TypeGPU Bridge] WebGPU adapter request failed.');
                    console.warn('[TypeGPU Bridge] Falling back to GLSL kernel.');
                    setWebgpuAvailable(false);
                    return;
                }

                console.log('[TypeGPU Bridge] ✅ WebGPU is available!');
                console.log('[TypeGPU Bridge] Adapter:', adapter);
                setWebgpuAvailable(true);

            } catch (error) {
                console.error('[TypeGPU Bridge] WebGPU initialization error:', error);
                setInitError(error.message);
                setWebgpuAvailable(false);
            }
        }

        detectWebGPU();
    }, []);

    // Show loading state during detection
    if (webgpuAvailable === null) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-black">
                <div className="text-white text-sm font-mono">
                    Detecting WebGPU capability...
                </div>
            </div>
        );
    }

    // Fallback to GLSL if WebGPU unavailable
    if (!webgpuAvailable) {
        return (
            <div className="w-full h-full relative">
                <RaymarchScene design={design} isaState={isaState} />
                <div className="absolute bottom-2 left-2 bg-yellow-900/80 text-yellow-200 px-2 py-1 rounded text-xs font-mono">
                    ⚠️ WebGPU unavailable - using GLSL fallback
                    {initError && <div className="text-[10px] mt-1">Error: {initError}</div>}
                </div>
            </div>
        );
    }

    // TODO: Implement actual TypeGPU rendering when WebGPU is available
    // For now, show success message and use GLSL
    return (
        <div className="w-full h-full relative">
            <RaymarchScene design={design} isaState={isaState} />
            <div className="absolute bottom-2 left-2 bg-green-900/80 text-green-200 px-2 py-1 rounded text-xs font-mono">
                ✅ WebGPU available - TypeGPU ready (using GLSL preview)
            </div>
        </div>
    );
}
