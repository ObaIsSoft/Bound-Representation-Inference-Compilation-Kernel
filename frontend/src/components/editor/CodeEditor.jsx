import React, { useState, useRef } from 'react';
import { useDesign } from '../../contexts/DesignContext';
import { useTheme } from '../../contexts/ThemeContext';
import { MessageSquarePlus } from 'lucide-react';

const CodeEditor = () => {
    const { activeTab, updateTabContent, addComment, getComments } = useDesign();
    const { theme } = useTheme();
    const textareaRef = useRef(null);
    const [selection, setSelection] = useState(null);
    const [showCommentButton, setShowCommentButton] = useState(false);
    const [commentText, setCommentText] = useState('');
    const [showCommentInput, setShowCommentInput] = useState(false);

    if (!activeTab) {
        return (
            <div
                className="flex-1 flex items-center justify-center text-[10px] font-mono select-none"
                style={{ backgroundColor: theme.colors.bg.primary, color: theme.colors.text.muted }}
            >
                NO OPEN EDITOR
            </div>
        );
    }

    const handleChange = (e) => {
        updateTabContent(activeTab.id, e.target.value);
    };

    const handleTextSelection = () => {
        if (!activeTab.readOnly) return; // Only for artifact tabs

        const textarea = textareaRef.current;
        if (!textarea) return;

        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;

        if (start !== end) {
            const selectedText = textarea.value.substring(start, end);
            setSelection({ start, end, text: selectedText });
            setShowCommentButton(true);
        } else {
            setShowCommentButton(false);
        }
    };

    const handleAddComment = async () => {
        if (!selection || !commentText.trim()) return;

        await addComment(activeTab.artifactId, selection, commentText);
        setCommentText('');
        setShowCommentInput(false);
        setShowCommentButton(false);
    };

    const comments = activeTab.artifactId ? getComments(activeTab.artifactId) : [];

    return (
        <div className="flex-1 h-full flex flex-col min-w-0 border-r relative" style={{ borderColor: theme.colors.border.primary }}>
            {/* Comment count indicator for artifacts */}
            {activeTab.readOnly && comments.length > 0 && (
                <div
                    className="px-3 py-1 text-[9px] font-mono border-b flex items-center gap-2"
                    style={{
                        backgroundColor: theme.colors.bg.secondary,
                        borderColor: theme.colors.border.primary,
                        color: theme.colors.text.muted
                    }}
                >
                    <MessageSquarePlus size={10} />
                    <span>{comments.length} comment{comments.length !== 1 ? 's' : ''}</span>
                </div>
            )}

            {/* Editor container with comment markers */}
            <div className="flex-1 relative overflow-hidden">
                <textarea
                    ref={textareaRef}
                    value={activeTab.content || ''}
                    onChange={handleChange}
                    onMouseUp={handleTextSelection}
                    onKeyUp={handleTextSelection}
                    readOnly={activeTab.readOnly || false}
                    spellCheck={false}
                    className="w-full h-full resize-none outline-none p-4 font-mono text-xs leading-5"
                    style={{
                        backgroundColor: theme.colors.bg.primary,
                        color: theme.colors.text.primary,
                        fontFamily: "'Fira Code', 'Roboto Mono', monospace",
                        cursor: activeTab.readOnly ? 'default' : 'text'
                    }}
                />

                {/* Comment markers overlay */}
                {activeTab.readOnly && comments.length > 0 && (
                    <div className="absolute top-0 right-0 p-2 space-y-1 pointer-events-none">
                        {comments.map((comment, idx) => {
                            // Calculate approximate line number from character position
                            const content = activeTab.content || '';
                            const linesBefore = content.substring(0, comment.selection?.start || 0).split('\n').length;
                            const lineHeight = 20; // matches text-xs leading-5
                            const topPosition = (linesBefore - 1) * lineHeight + 16; // 16px for padding

                            return (
                                <div
                                    key={comment.id || idx}
                                    className="absolute right-2 flex items-center gap-1 px-2 py-0.5 rounded pointer-events-auto cursor-pointer hover:scale-110 transition-transform"
                                    style={{
                                        top: `${topPosition}px`,
                                        backgroundColor: theme.colors.accent.primary + '40',
                                        borderLeft: `2px solid ${theme.colors.accent.primary}`
                                    }}
                                    title={`${comment.content}\n${comment.agent_response ? '\nAgent: ' + comment.agent_response : ''}`}
                                >
                                    <MessageSquarePlus size={10} style={{ color: theme.colors.accent.primary }} />
                                    <span className="text-[8px] font-bold" style={{ color: theme.colors.accent.primary }}>
                                        {idx + 1}
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* Comment button (appears on selection) */}
            {showCommentButton && !showCommentInput && (
                <div className="absolute bottom-4 right-4">
                    <button
                        onClick={() => setShowCommentInput(true)}
                        className="flex items-center gap-2 px-3 py-2 rounded shadow-lg text-[10px] font-bold transition-all hover:scale-105"
                        style={{
                            backgroundColor: theme.colors.accent.primary,
                            color: '#000'
                        }}
                    >
                        <MessageSquarePlus size={12} />
                        Add Comment
                    </button>
                </div>
            )}

            {/* Comment input */}
            {showCommentInput && (
                <div
                    className="absolute bottom-4 right-4 w-80 p-3 rounded-lg shadow-2xl border"
                    style={{
                        backgroundColor: theme.colors.bg.primary,
                        borderColor: theme.colors.border.primary
                    }}
                >
                    <div className="text-[9px] mb-2 font-mono" style={{ color: theme.colors.text.muted }}>
                        Selected: "{selection?.text?.substring(0, 50)}{selection?.text?.length > 50 ? '...' : ''}"
                    </div>
                    <textarea
                        value={commentText}
                        onChange={(e) => setCommentText(e.target.value)}
                        placeholder="Add your comment or question..."
                        className="w-full p-2 rounded text-[10px] font-mono mb-2 resize-none"
                        rows={3}
                        style={{
                            backgroundColor: theme.colors.bg.secondary,
                            border: `1px solid ${theme.colors.border.primary}`,
                            color: theme.colors.text.primary
                        }}
                    />
                    <div className="flex gap-2">
                        <button
                            onClick={handleAddComment}
                            className="flex-1 py-1 rounded text-[9px] font-bold"
                            style={{
                                backgroundColor: theme.colors.accent.primary,
                                color: '#000'
                            }}
                        >
                            Save Comment
                        </button>
                        <button
                            onClick={() => {
                                setShowCommentInput(false);
                                setShowCommentButton(false);
                            }}
                            className="px-3 py-1 rounded text-[9px]"
                            style={{
                                backgroundColor: theme.colors.bg.secondary,
                                color: theme.colors.text.muted
                            }}
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default CodeEditor;
