import React from 'react';
import { useTheme } from '../../contexts/ThemeContext';

export default function ISABrowserPanel({ width }) {
    const { theme } = useTheme();
    return (
        <div style={{ width, backgroundColor: theme.colors.bg.secondary }} className="h-full" />
    );
}
