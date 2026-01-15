import React, { useEffect, useRef } from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { useSettings } from '../../contexts/SettingsContext';

const TerminalTab = ({ commandHistory, commandInput, setCommandInput, handleCommand }) => {
    const { theme } = useTheme();
    const { fontSize } = useSettings();

    return (
        <div className="flex flex-col min-h-full">
            <div className="flex-1 space-y-1" style={{ fontSize: `${fontSize}px` }}>
                {commandHistory.map((cmd, i) => (
                    <div
                        key={i}
                        style={{
                            color: cmd.type === 'cmd' ? theme.colors.text.primary :
                                cmd.type === 'err' ? theme.colors.status.error :
                                    cmd.type === 'sys' ? theme.colors.status.success :
                                        theme.colors.text.secondary
                        }}
                    >
                        {cmd.text}
                    </div>
                ))}
            </div>
            <div
                className="flex items-center gap-2 mt-4 sticky bottom-0 backdrop-blur-sm py-1"
                style={{
                    color: theme.colors.status.success,
                    backgroundColor: theme.colors.bg.secondary + 'CC',
                    borderTop: `1px solid ${theme.colors.border.primary}`
                }}
            >
                <span>brick:~ %</span>
                <input
                    autoFocus
                    className="bg-transparent border-none outline-none flex-1 font-mono"
                    placeholder="Enter command..."
                    value={commandInput}
                    onChange={e => setCommandInput(e.target.value)}
                    onKeyDown={handleCommand}
                    style={{
                        color: theme.colors.text.primary,
                        fontSize: `${fontSize}px`,
                        '::placeholder': { color: theme.colors.text.muted }
                    }}
                />
            </div>
        </div>
    );
};

export default TerminalTab;
