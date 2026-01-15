import React from 'react';
import { Lock, Zap } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

const ConstraintCard = ({ label, value, unit, locked }) => {
    const { theme } = useTheme();

    return (
        <div
            className="p-2 px-3 rounded border font-mono flex justify-between items-center transition-all"
            style={{
                backgroundColor: locked ? theme.colors.bg.primary : theme.colors.accent.primary + '1A',
                borderColor: locked ? theme.colors.border.primary : theme.colors.accent.primary + '4D'
            }}
        >
            <div>
                <div className="text-[8px] uppercase font-bold" style={{ color: theme.colors.text.muted }}>
                    {label}
                </div>
                <div className="text-xs font-bold" style={{ color: theme.colors.text.primary }}>
                    {value}{unit}
                </div>
            </div>
            {locked ?
                <Lock size={10} style={{ color: theme.colors.border.secondary }} /> :
                <Zap size={10} style={{ color: theme.colors.accent.primary }} />
            }
        </div>
    );
};

export default ConstraintCard;
