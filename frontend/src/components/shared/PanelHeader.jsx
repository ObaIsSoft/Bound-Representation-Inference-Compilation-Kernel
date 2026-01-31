import React from 'react';
import { useTheme } from '../../contexts/ThemeContext';

export default function PanelHeader({ title, icon: Icon }) {
    const { theme } = useTheme();

    return (
        <div
            className="px-6 py-4 border-b flex items-center gap-3"
            style={{ borderColor: theme.colors.border.primary }}
        >
            {Icon && <Icon size={20} style={{ color: theme.colors.text.primary }} />}
            <h2
                className="text-lg font-bold"
                style={{ color: theme.colors.text.primary }}
            >
                {title}
            </h2>
        </div>
    );
}
