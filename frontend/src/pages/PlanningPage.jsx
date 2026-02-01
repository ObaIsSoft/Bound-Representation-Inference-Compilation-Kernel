import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, MessageCircle } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

export default function PlanningPage() {
    const location = useLocation();
    const navigate = useNavigate();
    const { theme } = useTheme();

    // State placeholders
    const [loading, setLoading] = useState(false);

    return (
        <div
            className="min-h-screen w-full flex flex-col animate-slideUp"
            style={{
                backgroundColor: theme.colors.bg.primary,
                color: theme.colors.text.primary,
            }}
        >
            {/* Header with Back Button */}
            <div
                className="px-8 py-4 border-b flex items-center gap-4"
                style={{ borderColor: theme.colors.border.primary }}
            >
                <button
                    onClick={() => navigate('/requirements')}
                    className="p-2 rounded-lg hover:bg-opacity-10 transition-all"
                    style={{ backgroundColor: theme.colors.bg.tertiary }}
                >
                    <ArrowLeft size={20} style={{ color: theme.colors.text.primary }} />
                </button>
                <div className="flex items-center gap-3">
                    <MessageCircle size={28} style={{ color: theme.colors.accent.primary }} />
                    <h1 className="text-2xl font-bold">Planning & Review</h1>
                </div>
            </div>

            {/* Main Content Area - Empty as requested */}
            <div className="flex-1 overflow-y-auto px-8 py-6">
                <div className="max-w-3xl mx-auto">
                    <p style={{ color: theme.colors.text.secondary }}>
                        Planning page initialized. Waiting for content...
                    </p>
                </div>
            </div>

            {/* CSS Animations */}
            <style>{`
@keyframes slideUp {
    from {
        transform: translateY(100%);
        opacity: 0.8;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

.animate-slideUp {
    animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
`}</style>
        </div>
    );
}
