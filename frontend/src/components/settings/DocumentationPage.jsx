import React from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { Book } from 'lucide-react';

export default function DocumentationPage() {
    const { theme } = useTheme();

    return (
        <div
            className="flex flex-col min-h-screen w-full items-center justify-center"
            style={{
                backgroundColor: theme.colors.bg.primary,
                color: theme.colors.text.primary,
            }}
        >
            <div className="flex flex-col items-center gap-4 opacity-50">
                <Book size={48} />
                <h1 className="text-2xl font-bold">Documentation</h1>
                <p>Refer to docs/BIBLE.md for developer documentation.</p>
            </div>
        </div>
    );
}
