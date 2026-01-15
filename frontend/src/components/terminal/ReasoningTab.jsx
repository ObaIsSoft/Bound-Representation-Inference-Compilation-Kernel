import React from 'react';
import { BrainCircuit } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

const ReasoningTab = ({ reasoningStream = [] }) => {
    const { theme } = useTheme();

    if (reasoningStream.length === 0) {
        return (
            <div
                className="h-32 flex flex-col items-center justify-center opacity-40"
                style={{ color: theme.colors.text.tertiary }}
            >
                <BrainCircuit size={40} className="mb-2" />
                <span>Waiting for Agentic Logic Pass...</span>
            </div>
        );
    }

    return (
        <div className="space-y-3">
            {reasoningStream.map((log, i) => (
                <div
                    key={i}
                    className="flex gap-4 pl-4 py-1 animate-in fade-in slide-in-from-left duration-300"
                    style={{ borderLeft: `2px solid ${theme.colors.border.primary}` }}
                >
                    <div
                        className="shrink-0 text-[9px] w-16"
                        style={{ color: theme.colors.text.tertiary }}
                    >
                        {log.time}
                    </div>
                    <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-2">
                            <span
                                className="font-black uppercase text-[9px] tracking-widest"
                                style={{ color: theme.colors.accent.primary }}
                            >
                                {log.agent}
                            </span>
                            <div
                                className="h-px flex-1"
                                style={{ backgroundColor: theme.colors.border.secondary }}
                            />
                        </div>
                        <div
                            className="italic"
                            style={{ color: theme.colors.text.secondary }}
                        >
                            "{log.thought}"
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default ReasoningTab;
