import React from 'react';
import { Check, Cpu, ShieldCheck, Network } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';
import { useSimulation } from '../../contexts/SimulationContext';

const Header = ({ isRunning }) => {
    const { theme } = useTheme();
    const { solverStatus } = useSimulation();

    // Helper for status color
    const getStatusColor = (status, defaultColor) => {
        if (status === 'OK' || status === 'CONVERGED') return theme.colors.status.success || '#10b981';
        if (status === 'OFFLINE' || status === 'ERROR') return theme.colors.status.error || '#ef4444';
        return defaultColor;
    };

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
                            boxShadow: `0 0 10px ${theme.colors.accent.glow}`,
                            fontSize: '14px'
                        }}
                    >
                        🧱
                    </div>
                    <h1
                        className="text-sm font-black italic tracking-tighter ml-1 font-mono uppercase"
                        style={{ color: theme.colors.accent.primary }}
                    >
                        BRICK
                    </h1>
                </div>
                <div className="h-4 w-px" style={{ backgroundColor: theme.colors.border.primary }} />

                {/* Logic-First Status Indicators */}
                <div className="flex gap-3 ml-2">
                    {/* ARES Solver Status */}
                    <div
                        className="flex items-center gap-1.5 px-2 py-1 rounded border hidden sm:flex"
                        style={{
                            backgroundColor: theme.colors.bg.secondary,
                            borderColor: theme.colors.border.primary
                        }}
                    >
                        <ShieldCheck size={12} style={{ color: getStatusColor(solverStatus?.ares, theme.colors.text.muted) }} />
                        <span className="text-[9px] font-mono font-bold" style={{ color: theme.colors.text.primary }}>
                            ARES: {solverStatus?.ares || 'INIT'}
                        </span>
                    </div>

                    {/* LDP Status */}
                    <div
                        className="flex items-center gap-1.5 px-2 py-1 rounded border hidden sm:flex"
                        style={{
                            backgroundColor: theme.colors.bg.secondary,
                            borderColor: theme.colors.border.primary
                        }}
                    >
                        <Network size={12} style={{ color: solverStatus?.ldp === 'CONVERGED' ? '#3b82f6' : theme.colors.text.muted }} />
                        <span className="text-[9px] font-mono font-bold" style={{ color: theme.colors.text.primary }}>
                            LDP: {solverStatus?.ldp || 'IDLE'}
                        </span>
                    </div>
                </div>

                <div className="text-[10px] font-mono uppercase flex gap-4 items-center" style={{ color: theme.colors.text.muted }}>
                    {/* Legacy Indicator (Keep or Remove? Keeping for now but de-emphasized) */}
                    {/*  <span className="flex items-center gap-1" style={{ color: theme.colors.status.success }}>
                        <Check size={10} /> KERNEL_OK
                    </span> */}
                    <span className="hidden md:inline opacity-50">SHA: 0X9A2F...3B12</span>
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
