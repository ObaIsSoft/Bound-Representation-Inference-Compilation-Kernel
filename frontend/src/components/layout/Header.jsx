import React from 'react';
import { Check, Cpu } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

const Header = ({ isRunning }) => {
    const { theme } = useTheme();

    return (
        <header
            className="h-10 flex items-center justify-between px-4 shrink-0 z-50"
            style={{
                backgroundColor: theme.colors.bg.primary,
                borderBottom: `1px solid ${theme.colors.border.primary}`
            }}
        >
            <div className="flex items-center gap-4">
                <div className="flex items-center gap-1 group cursor-pointer">
                    <div
                        className="w-6 h-6 rounded flex items-center justify-center font-black italic text-xs"
                        style={{
                            background: `linear-gradient(to bottom right, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`,
                            color: theme.colors.bg.primary,
                            boxShadow: `0 0 10px ${theme.colors.accent.glow}`
                        }}
                    >
                        B
                    </div>
                    <h1
                        className="text-sm font-black italic tracking-tighter ml-1 font-mono uppercase"
                        style={{ color: theme.colors.accent.primary }}
                    >
                        BRICK
                    </h1>
                </div>
                <div className="h-4 w-px" style={{ backgroundColor: theme.colors.border.primary }} />
                <div className="text-[10px] font-mono uppercase flex gap-4 items-center" style={{ color: theme.colors.text.muted }}>
                    <span className="flex items-center gap-1" style={{ color: theme.colors.status.success }}>
                        <Check size={10} /> KERNEL_OK
                    </span>
                    <span className="hidden sm:inline">SHA: 0X9A2F...3B12</span>
                </div>
            </div>
            <div className="flex items-center gap-4 sm:gap-6 text-[10px] font-mono">
                <span className="flex items-center gap-2">
                    <div
                        className={`w-2 h-2 rounded-full ${isRunning ? 'animate-pulse' : ''}`}
                        style={{ backgroundColor: theme.colors.status.success }}
                    />
                    {isRunning ? 'LIVESTREAM' : 'STANDBY'}
                </span>
                <span className="hidden sm:flex items-center gap-2" style={{ color: theme.colors.text.muted }}>
                    <Cpu size={12} /> 1.2GHz
                </span>
            </div>
        </header>
    );
};

export default Header;
