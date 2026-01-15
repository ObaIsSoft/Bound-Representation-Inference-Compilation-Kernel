import React from 'react';
import { useTheme } from '../../contexts/ThemeContext';

const ConsoleTab = ({ commandHistory, commandInput, setCommandInput, handleCommand }) => {
    const { theme } = useTheme();

    return (
        <div className="space-y-1 not-italic font-mono">
            {/* Command History */}
            {commandHistory.map((entry, i) => (
                <div
                    key={i}
                    style={{
                        color: entry.type === 'cmd' ? theme.colors.text.secondary :
                            entry.type === 'err' ? theme.colors.status.error :
                                entry.type === 'sys' ? theme.colors.accent.primary :
                                    theme.colors.text.primary
                    }}
                >
                    {entry.text}
                </div>
            ))}

            {/* Input Line */}
            <div className="flex items-center gap-2">
                <span style={{ color: theme.colors.accent.primary }}>brick:~ %</span>
                <input
                    type="text"
                    value={commandInput}
                    onChange={(e) => setCommandInput(e.target.value)}
                    onKeyDown={handleCommand}
                    className="flex-1 bg-transparent outline-none"
                    style={{ color: theme.colors.text.primary }}
                    placeholder="Enter command..."
                    autoFocus
                />
            </div>
        </div>
    );
};

export default ConsoleTab;
