
import { useState, useEffect, useRef, useCallback } from 'react';
import apiClient from '../utils/apiClient';

/**
 * Hook to poll the backend for agent thoughts (XAI).
 * 
 * @param {boolean} isEnabled - Whether polling is active
 * @param {number} intervalMs - Polling interval in ms (default 1000)
 * @returns {Object} { thoughts, isStreaming, clearThoughts }
 */
export const useThoughtStream = (isEnabled = false, intervalMs = 1000) => {
    const [thoughts, setThoughts] = useState([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [error, setError] = useState(null);

    // Use ref for isEnabled to avoid effect re-triggering issues if needed, 
    // but here we want effect to react to isEnabled changes.
    const pollInterval = useRef(null);

    const fetchThoughts = useCallback(async () => {
        try {
            // "Destructive read" - backend clears buffer after sending
            const data = await apiClient.get('/agents/thoughts');

            if (data && data.thoughts && data.thoughts.length > 0) {
                setThoughts(prev => {
                    // Avoid duplicates if backend doesn't clear (safety check)
                    // But backend DOES clear, so we just append.
                    // For safety against React StrictMode double-invokes, we could check timestamps
                    // but simple append is usually fine for a log stream.

                    // Let's add simple deduping based on timestamp + agent just in case
                    const newThoughts = data.thoughts.filter(newT =>
                        !prev.some(oldT => oldT.timestamp === newT.timestamp && oldT.text === newT.text)
                    );

                    if (newThoughts.length === 0) return prev;

                    return [...prev, ...newThoughts];
                });
                setIsStreaming(true);
            }
        } catch (err) {
            console.error("Failed to poll thoughts:", err);
            setError(err);
        }
    }, []);

    useEffect(() => {
        if (isEnabled) {
            // Initial fetch
            fetchThoughts();

            // Start polling
            pollInterval.current = setInterval(fetchThoughts, intervalMs);
            setIsStreaming(true);
        } else {
            // Stop polling
            if (pollInterval.current) {
                clearInterval(pollInterval.current);
                pollInterval.current = null;
            }
            setIsStreaming(false);
        }

        return () => {
            if (pollInterval.current) {
                clearInterval(pollInterval.current);
            }
        };
    }, [isEnabled, intervalMs, fetchThoughts]);

    const clearThoughts = useCallback(() => {
        setThoughts([]);
    }, []);

    return { thoughts, isStreaming, error, clearThoughts };
};
