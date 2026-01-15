import React from 'react';
import { useTheme } from '../../contexts/ThemeContext';

const PanelHeader = ({ title, icon: Icon, action }) => {
    const { theme } = useTheme();

    return (
        <div
            className="flex items-center justify-between px-3 py-2 shrink-0 select-none"
            style={{
                backgroundColor: theme.colors.bg.primary,
                borderBottom: `1px solid ${theme.colors.border.primary}`
            }}
        >
            <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider font-mono" style={{ color: theme.colors.text.tertiary }}>
                <Icon size={12} style={{ color: theme.colors.accent.primary }} />
                {title}
            </div>
            {action && <div className="flex items-center">{action}</div>}
        </div>
    );
};

export default PanelHeader;
