import React from 'react';
import { useTheme } from '../contexts/ThemeContext';

export default function PlanningPage() {
    const { theme } = useTheme();

    return (
        <div
            className="h-screen w-full flex items-center justify-center"
            style={{
                backgroundColor: theme.colors.bg.primary,
                color: theme.colors.text.primary
            }}
        >
            <h1 className="text-3xl font-bold opacity-20">Planning Page</h1>
        </div>
    );
}
