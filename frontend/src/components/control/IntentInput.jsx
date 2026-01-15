import React from 'react';
import { Sparkles, Send } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

const IntentInput = ({ value, onChange, onSend }) => {
    const { theme } = useTheme();

    return (
        <div
            className="p-3 shrink-0"
            style={{
                backgroundColor: theme.colors.bg.primary,
                borderTop: `1px solid ${theme.colors.border.primary}`
            }}
        >
            <div className="flex items-center justify-between mb-2">
                <h3 className="text-[10px] uppercase font-mono flex items-center gap-2" style={{ color: theme.colors.text.muted }}>
                    <Sparkles size={12} /> Intent
                </h3>
                <Sparkles size={12} className="animate-pulse" style={{ color: theme.colors.accent.primary }} />
            </div>
            <div className="relative">
                <textarea
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            onSend();
                        }
                    }}
                    className="w-full rounded p-2 text-xs font-mono resize-none outline-none h-24 transition-all scrollbar-hide pb-10"
                    placeholder="Hardware intent..."
                    style={{
                        backgroundColor: theme.colors.bg.primary,
                        border: `1px solid ${theme.colors.border.primary}`,
                        color: theme.colors.text.primary
                    }}
                    onFocus={(e) => e.target.style.borderColor = theme.colors.accent.primary + '80'}
                    onBlur={(e) => e.target.style.borderColor = theme.colors.border.primary}
                />
                <button
                    onClick={onSend}
                    className="absolute bottom-2 right-2 px-4 py-2 rounded transition-colors shadow-lg font-bold flex items-center gap-2"
                    style={{
                        background: `linear-gradient(to right, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`,
                        color: theme.colors.bg.primary
                    }}
                >
                    <span className="text-[10px] font-bold">SEND</span>
                    <Send size={14} />
                </button>
            </div>
        </div>
    );
};

export default IntentInput;
