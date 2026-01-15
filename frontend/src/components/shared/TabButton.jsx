import React from 'react';
import { useTheme } from '../../contexts/ThemeContext';

const TabButton = ({ id, label, icon: Icon, active, onClick }) => {
    const { theme } = useTheme();

    return (
        <button
            onClick={() => onClick(id)}
            className="px-3 py-1.5 text-[10px] font-mono uppercase tracking-wider flex items-center gap-1.5 transition-all shrink-0"
            style={{
                color: active ? theme.colors.text.primary : theme.colors.text.muted,
                backgroundColor: active ? theme.colors.bg.secondary : 'transparent',
                borderBottom: active ? `2px solid ${theme.colors.accent.primary}` : '2px solid transparent'
            }}
        >
            {Icon && <Icon size={12} />}
            {label}
        </button>
    );
};

export default TabButton;
