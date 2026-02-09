
import React, { useEffect, useRef } from 'react';
import { usePanel } from '../../contexts/PanelContext';
import { useTheme } from '../../contexts/ThemeContext';
import { Terminal, Cpu, Zap, Brain } from 'lucide-react';

/**
 * Renders a live stream of agent thoughts (XAI).
 * Designed to look like a system console or "Matrix" stream.
 */
export default function ThoughtStream({ compact = false }) {
    const { thoughts, isStreaming } = usePanel();
    const { theme } = useTheme();
    const scrollRef = useRef(null);

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [thoughts]);

    if (!thoughts || thoughts.length === 0) {
        if (!isStreaming) return null;
        return (
            <div className={`text-xs p-2 italic opacity-50 ${compact ? '' : 'text-center'}`} style={{ color: theme.colors.text.muted }}>
                Waiting for agent thoughts...
            </div>
        );
    }

    const getAgentIcon = (name) => {
        if (name.includes('Physics')) return <Zap size={12} />;
        if (name.includes('Cost')) return <Terminal size={12} />;
        if (name.includes('Geometry')) return <Cpu size={12} />;
        return <Brain size={12} />;
    };

    const getAgentColor = (name) => {
        if (name.includes('Physics')) return '#a855f7'; // Purple
        if (name.includes('Cost')) return '#22c55e'; // Green
        if (name.includes('Geometry')) return '#3b82f6'; // Blue
        if (name.includes('Error')) return '#ef4444'; // Red
        return theme.colors.accent.primary; // Default Gold
    };

    return (
        <div
            className={`flex flex-col ${compact ? 'h-32 text-xs' : 'h-full text-sm'}`}
            style={{
                fontFamily: 'monospace',
                backgroundColor: compact ? 'transparent' : theme.colors.bg.secondary,
                borderRadius: compact ? '4px' : '0',
            }}
        >
            {!compact && (
                <div className="p-2 border-b flex items-center gap-2 text-xs font-bold uppercase tracking-wider opacity-70"
                    style={{ borderColor: theme.colors.border.primary, color: theme.colors.text.secondary }}>
                    <Terminal size={14} />
                    Agent Neural Stream
                    {isStreaming && <span className="animate-pulse ml-auto text-[10px] text-green-500">● LIVE</span>}
                </div>
            )}

            <div
                ref={scrollRef}
                className={`flex-1 overflow-y-auto space-y-1 p-2 ${compact ? 'scrollbar-hide' : 'scrollbar-thin'}`}
            >
                {thoughts.map((t, idx) => (
                    <div key={idx} className="flex gap-2 animate-fadeIn opacity-90 hover:opacity-100 transition-opacity">
                        <span style={{ color: theme.colors.text.muted, minWidth: '60px' }}>
                            {new Date(t.timestamp).toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                        </span>

                        <div className="flex-1">
                            <span
                                className="font-bold mr-2 inline-flex items-center gap-1"
                                style={{ color: getAgentColor(t.agent) }}
                            >
                                [{t.agent}]
                            </span>
                            <span style={{ color: theme.colors.text.primary }}>
                                {t.text}
                            </span>
                        </div>
                    </div>
                ))}
                <div className="h-2" /> {/* Bottom spacer */}
            </div>
        </div>
    );
}
