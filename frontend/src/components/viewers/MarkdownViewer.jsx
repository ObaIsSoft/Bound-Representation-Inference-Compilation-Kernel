import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useTheme } from '../../contexts/ThemeContext';
import apiClient from '../../utils/apiClient';
import { Download, FileText, Loader2 } from 'lucide-react';

const MarkdownViewer = ({ path, fileName }) => {
    const { theme } = useTheme();
    const [content, setContent] = useState('');
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchContent = async () => {
            setIsLoading(true);
            try {
                // In a real tauri/web app, we might use tauri's fs or a backend endpoint
                // For now, we assume a backend endpoint /api/files/read exists or we mock it
                // For now, we assume a backend endpoint /api/files/read exists or we mock it
                const text = await apiClient.get('/files/read', {
                    params: { path },
                    responseType: 'text'
                });
                setContent(text);
            } catch (err) {
                console.error('Markdown load error:', err);
                setError(err.message);
                // Mock content for demo if file doesn't exist
                setContent(`# ${fileName}\n\nThis is a preview of ${fileName}. The actual file content at \`${path}\` could not be loaded.\n\n## Summary\n- Type: Technical Report\n- Status: Draft\n\n> [!NOTE]\n> This is a placeholder preview.`);
            } finally {
                setIsLoading(false);
            }
        };

        if (path) fetchContent();
    }, [path]);

    if (isLoading) {
        return (
            <div className="h-full flex flex-col items-center justify-center opacity-40">
                <Loader2 size={32} className="animate-spin mb-4" />
                <div className="text-xs font-black uppercase tracking-widest">Reading Artifact...</div>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col overflow-hidden">
            {/* Toolbar */}
            <div className="flex items-center justify-between p-4 border-b border-white/10 bg-black/20">
                <div className="flex items-center gap-3">
                    <FileText size={18} color={theme.colors.accent.primary} />
                    <div className="text-xs font-bold truncate max-w-[200px]" style={{ color: theme.colors.text.primary }}>
                        {fileName}
                    </div>
                </div>
                <button className="p-2 hover:bg-white/5 rounded-lg transition-colors group">
                    <Download size={16} className="opacity-50 group-hover:opacity-100" />
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-8 custom-scrollbar bg-black/10">
                <article className="prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {content}
                    </ReactMarkdown>
                </article>
            </div>
        </div>
    );
};

export default MarkdownViewer;
