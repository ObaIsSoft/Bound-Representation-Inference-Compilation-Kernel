import React from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { usePanel } from '../../contexts/PanelContext';
import { FileText, Box, Code, ExternalLink, ChevronDown, ChevronUp } from 'lucide-react';

const ARTIFACT_REGEX = /\[(.*?)\]\(file:\/\/\/(.*?)\)/g;

const ThoughtBlock = ({ thought, theme }) => {
    const [isExpanded, setIsExpanded] = React.useState(true);
    return (
        <div className="my-2 border-l-2 pl-4 py-1" style={{ borderColor: theme.colors.accent.primary + '30' }}>
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.2em] opacity-40 hover:opacity-100 transition-opacity"
                style={{ color: theme.colors.text.secondary }}
            >
                {isExpanded ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
                Process Trace
            </button>
            {isExpanded && (
                <div className="mt-2 text-xs leading-relaxed opacity-60 italic font-mono" style={{ color: theme.colors.text.secondary }}>
                    {thought}
                </div>
            )}
        </div>
    );
};

const ArtifactPill = ({ name, path, theme, onClick }) => {
    const isMarkdown = path.endsWith('.md');
    const isGenome = path.includes('genome') || path.endsWith('.json');

    return (
        <button
            onClick={() => onClick({ id: path, name, type: isMarkdown ? 'document' : 'artifact' })}
            className="inline-flex items-center gap-2 px-3 py-1.5 mx-1 my-1 rounded-lg border bg-white/5 hover:bg-white/10 transition-all group border-white/10 active:scale-95"
            style={{ color: theme.colors.accent.primary }}
        >
            {isMarkdown ? <FileText size={14} /> : isGenome ? <Box size={14} /> : <Code size={14} />}
            <span className="text-xs font-bold whitespace-nowrap">{name}</span>
            <ExternalLink size={10} className="opacity-0 group-hover:opacity-100 transition-opacity" />
        </button>
    );
};

const ChatMessage = ({ msg, isPlanningPage = false }) => {
    const { theme } = useTheme();
    const { viewArtifact } = usePanel();

    if (msg.role === 'thought') {
        return <ThoughtBlock thought={msg.content} theme={theme} />;
    }

    const renderContent = (content) => {
        const parts = [];
        let lastIndex = 0;
        let match;

        while ((match = ARTIFACT_REGEX.exec(content)) !== null) {
            // Push text before match
            if (match.index > lastIndex) {
                parts.push(content.substring(lastIndex, match.index));
            }

            // Push Artifact Pill
            const [fullMatch, name, path] = match;
            parts.push(
                <ArtifactPill
                    key={match.index}
                    name={name}
                    path={path}
                    theme={theme}
                    onClick={viewArtifact}
                />
            );

            lastIndex = ARTIFACT_REGEX.lastIndex;
        }

        // Push remaining text
        if (lastIndex < content.length) {
            parts.push(content.substring(lastIndex));
        }

        return parts.length > 0 ? parts : content;
    };

    const isUser = msg.role === 'user';

    return (
        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
            {!isPlanningPage && (
                <div className="text-[9px] uppercase tracking-widest font-black opacity-30 mb-1 px-1">
                    {msg.role}
                </div>
            )}

            <div
                className={`text-sm leading-relaxed shadow-sm transition-all ${isPlanningPage
                    ? `p-6 rounded-3xl ${isUser ? 'bg-white/10 rounded-tr-none border border-white/5' : 'bg-black/40 rounded-tl-none border border-white/5 backdrop-blur-md shadow-2xl'}`
                    : `p-3 rounded-xl ${isUser ? 'bg-white/10 rounded-tr-none' : 'bg-black/20 rounded-tl-none border border-white/5'}`
                    }`}
                style={{
                    color: theme.colors.text.primary,
                    maxWidth: isPlanningPage ? '85%' : '90%',
                    border: (isPlanningPage && isUser) ? `1px solid ${theme.colors.accent.primary}20` : undefined
                }}
            >
                {renderContent(msg.content)}
            </div>

            {msg.timestamp && (
                <div className="text-[8px] opacity-20 mt-1 uppercase font-bold tracking-tighter">
                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
            )}
        </div>
    );
};

export default ChatMessage;
