import React, { useRef, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';
import { usePanel } from '../contexts/PanelContext';
import { Box, ChevronDown, ChevronUp, Sparkles, Loader2 } from 'lucide-react';
import ChatMessage from '../components/shared/ChatMessage';
import ThoughtStream from '../components/xai/ThoughtStream';

const ThoughtBlock = ({ thought, theme }) => {
    const [isExpanded, setIsExpanded] = React.useState(true);
    return (
        <div className="my-4 border-l-2 pl-6 py-2" style={{ borderColor: theme.colors.accent.primary + '40' }}>
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.2em] opacity-40 hover:opacity-100 transition-opacity"
                style={{ color: theme.colors.text.secondary }}
            >
                {isExpanded ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
                Process Trace
            </button>
            {isExpanded && (
                <div className="mt-3 text-sm leading-relaxed opacity-80 italic font-mono" style={{ color: theme.colors.text.secondary }}>
                    {thought}
                </div>
            )}
        </div>
    );
};

const ArtifactBlock = ({ artifact, theme }) => {
    return (
        <div
            className="my-6 rounded-2xl border p-6 bg-black/20 shadow-2xl group hover:border-white/20 transition-all cursor-pointer backdrop-blur-md"
            style={{ borderColor: theme.colors.border.secondary }}
        >
            <div className="flex items-center gap-4 mb-4">
                <div className="p-3 rounded-xl bg-white/5">
                    <Box size={24} color={theme.colors.accent.primary} />
                </div>
                <div>
                    <div className="text-sm font-bold" style={{ color: theme.colors.text.primary }}>{artifact.title}</div>
                    <div className="text-[10px] opacity-50 uppercase tracking-widest font-black" style={{ color: theme.colors.text.secondary }}>{artifact.type}</div>
                </div>
            </div>
            <div className="text-sm opacity-70 line-clamp-3" style={{ color: theme.colors.text.secondary }}>
                {artifact.summary}
            </div>
        </div>
    );
};

const PlanningPage = () => {
    const location = useLocation();
    const { sessions, activeSessionId, isSubmitting, switchSession, togglePanel, PANEL_IDS } = usePanel();
    const { theme } = useTheme();
    const chatEndRef = useRef(null);

    // Activate the session from requirements gathering
    const incomingSessionId = location.state?.sessionId;
    useEffect(() => {
        if (incomingSessionId && incomingSessionId !== activeSessionId) {
            switchSession(incomingSessionId);
            // Open the panel so the conversation is visible
            togglePanel(PANEL_IDS.MAIN);
        }
    }, [incomingSessionId]);

    const session = sessions.find(s => s.id === activeSessionId);

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [session?.history, isSubmitting]);

    return (
        <div className="flex flex-col h-screen w-full" style={{ backgroundColor: theme.colors.bg.primary }}>
            {/* Real-time Thought Stream Overlay (HUD Style) */}
            <div className="fixed bottom-24 right-8 max-w-[280px] z-50 pointer-events-none">
                <div className="pointer-events-auto bg-black/40 backdrop-blur-md rounded-xl border border-white/10 overflow-hidden shadow-2xl">
                    <ThoughtStream compact={true} />
                </div>
            </div>

            <div className="flex-1 overflow-y-auto px-6 py-8 space-y-6 scrollbar-hide pb-32">
                {/* Main view shows only artifacts via tabs - conversation is in the floating panel */}
                <div className="h-full flex flex-col items-center justify-center opacity-20 py-20">
                    <div className="p-6 rounded-full bg-white/5 border border-white/10 mb-6">
                        <Sparkles size={48} />
                    </div>
                    <h2 className="text-2xl font-black uppercase tracking-widest mb-2">Planning Phase</h2>
                    <p className="text-xs font-mono mb-4">Plan artifacts will appear here when generated</p>
                    <p className="text-[10px] opacity-40 max-w-md text-center">
                        Use the floating panel to view conversation history and open artifact tabs
                    </p>
                </div>

                <div ref={chatEndRef} />
            </div>
        </div>
    );
};

export default PlanningPage;
