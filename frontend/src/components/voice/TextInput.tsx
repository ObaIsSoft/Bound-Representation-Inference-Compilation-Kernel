import React, { useState, useRef, KeyboardEvent, useCallback } from 'react';
import { Image, Pencil, Paperclip, FileText, Box, X, Mic } from 'lucide-react'; // Added icons
import { useDropzone } from 'react-dropzone';
import { useTheme } from '../../contexts/ThemeContext';
import { LLM_PROVIDERS } from '../../utils/constants';

interface TextInputProps {
    onSubmit: (message: string, options: InputOptions) => void;
    onFocus: () => void;
}

interface InputOptions {
    llmProvider: string;
    attachedImages: File[]; // Keeping name for compatibility, but it holds all files now
    drawings: Blob[];
}

export default function TextInput({ onSubmit, onFocus }: TextInputProps) {
    const [message, setMessage] = useState('');
    const [llmProvider, setLlmProvider] = useState<string>('groq');
    const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
    const [showAttachMenu, setShowAttachMenu] = useState(false);

    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const { theme } = useTheme();

    // File Handling
    const onDrop = useCallback((acceptedFiles: File[]) => {
        setAttachedFiles(prev => [...prev, ...acceptedFiles]);
    }, []);

    const { getRootProps, getInputProps, isDragActive, open: openAllFiles } = useDropzone({
        onDrop,
        noClick: true,
        noKeyboard: true
    });

    // We can use separate invisible inputs or just filter programmatic opens if needed. 
    // For simplicity, we will just use the dropzone's open for "All Files" 
    // and manual inputs for specific types if we want strict filtering.
    const imageInputRef = useRef<HTMLInputElement>(null);
    const modelInputRef = useRef<HTMLInputElement>(null);

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    const handleSubmit = () => {
        if (!message.trim() && attachedFiles.length === 0) return;

        onSubmit(message, {
            llmProvider,
            attachedImages: attachedFiles, // Passing all files here
            drawings: [],
        });

        setMessage('');
        setAttachedFiles([]);
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || []);
        setAttachedFiles(prev => [...prev, ...files]);
        setShowAttachMenu(false);
    };

    const removeFile = (index: number) => {
        setAttachedFiles(prev => prev.filter((_, i) => i !== index));
    };

    const toggleAttachMenu = () => setShowAttachMenu(!showAttachMenu);

    return (
        <div
            {...getRootProps()}
            className="w-full max-w-3xl rounded-2xl shadow-2xl backdrop-blur-sm animate-fade-in relative transition-all duration-300"
            style={{
                backgroundColor: isDragActive ? `${theme.colors.accent.primary}15` : theme.colors.bg.secondary + 'CC',
                border: `1px solid ${isDragActive ? theme.colors.accent.primary : theme.colors.border.primary}`,
                boxShadow: `0 20px 60px ${theme.colors.accent.primary}15`,
            }}
        >
            <input {...getInputProps()} />

            {/* Drag Overlay */}
            {isDragActive && (
                <div className="absolute inset-0 z-50 flex items-center justify-center rounded-2xl backdrop-blur-sm bg-black/40">
                    <div className="text-white font-bold text-lg flex items-center gap-2 animate-bounce">
                        <Paperclip size={24} />
                        Drop to Attach
                    </div>
                </div>
            )}

            {/* Attached Files Preview (Inline) */}
            {attachedFiles.length > 0 && (
                <div className="px-6 pt-4 flex gap-2 flex-wrap max-h-32 overflow-y-auto">
                    {attachedFiles.map((file, index) => (
                        <div
                            key={index}
                            className="relative group flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-all hover:bg-white/5"
                            style={{
                                backgroundColor: theme.colors.bg.tertiary,
                                borderColor: theme.colors.border.primary
                            }}
                        >
                            {/* Icon based on type */}
                            {file.type.startsWith('image/') ? (
                                <img src={URL.createObjectURL(file)} alt="preview" className="w-6 h-6 object-cover rounded" />
                            ) : file.name.endsWith('.stl') || file.name.endsWith('.obj') ? (
                                <Box size={16} color={theme.colors.accent.primary} />
                            ) : (
                                <FileText size={16} color={theme.colors.text.secondary} />
                            )}

                            <span className="text-sm truncate max-w-[120px]" style={{ color: theme.colors.text.primary }}>
                                {file.name}
                            </span>

                            <button
                                onClick={(e) => { e.stopPropagation(); removeFile(index); }}
                                className="ml-1 p-0.5 rounded-full hover:bg-red-500/20 hover:text-red-400 transition-colors"
                                style={{ color: theme.colors.text.muted }}
                            >
                                <X size={14} />
                            </button>
                        </div>
                    ))}
                </div>
            )}

            {/* Text Input */}
            <div className="p-6">
                <textarea
                    ref={textareaRef}
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={handleKeyDown}
                    onFocus={onFocus}
                    placeholder={isDragActive ? "Drop files now..." : "Type to create..."}
                    className="w-full bg-transparent resize-none outline-none font-mono text-lg"
                    style={{
                        color: theme.colors.text.primary,
                        minHeight: '60px',
                    }}
                    rows={3}
                />

                <div className="mt-2 text-xs opacity-60" style={{ color: theme.colors.text.muted }}>
                    {message.length} characters • Press Enter to submit
                </div>
            </div>

            {/* Control Bar */}
            <div
                className="px-6 py-4 flex items-center justify-between border-t relative"
                style={{ borderColor: theme.colors.border.primary }}
            >
                {/* Left Controls */}
                <div className="flex items-center gap-3">
                    {/* LLM Provider Dropdown */}
                    <select
                        value={llmProvider}
                        onChange={(e) => setLlmProvider(e.target.value as any)}
                        className="px-3 py-1.5 rounded-lg text-sm font-bold uppercase tracking-wide outline-none cursor-pointer transition-all hover:scale-105"
                        style={{
                            backgroundColor: theme.colors.bg.tertiary,
                            border: `1px solid ${theme.colors.border.primary}`,
                            color: theme.colors.text.primary,
                        }}
                    >
                        {LLM_PROVIDERS.map(provider => (
                            <option key={provider.value} value={provider.value}>
                                {provider.label}
                            </option>
                        ))}
                    </select>

                    {/* Attachment Menu Container */}
                    <div className="relative">
                        <button
                            onClick={toggleAttachMenu}
                            className={`p-2 rounded-lg transition-all ${showAttachMenu ? 'bg-white/10' : ''}`}
                            style={{
                                color: attachedFiles.length > 0 ? theme.colors.accent.primary : theme.colors.text.primary,
                            }}
                            title="Attach..."
                        >
                            <Paperclip size={20} />
                        </button>

                        {/* Hover/Click Menu */}
                        {showAttachMenu && (
                            <div
                                className="absolute bottom-full left-0 mb-2 w-48 rounded-xl shadow-xl backdrop-blur-xl border overflow-hidden animate-in fade-in slide-in-from-bottom-2"
                                style={{
                                    backgroundColor: theme.colors.bg.secondary,
                                    borderColor: theme.colors.border.primary
                                }}
                            >
                                <button
                                    className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-white/5 transition-colors"
                                    onClick={() => { openAllFiles(); setShowAttachMenu(false); }}
                                >
                                    <FileText size={16} className="text-blue-400" />
                                    <span style={{ color: theme.colors.text.primary }} className="text-sm font-medium">Any File</span>
                                </button>
                                <button
                                    className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-white/5 transition-colors"
                                    onClick={() => { imageInputRef.current?.click(); setShowAttachMenu(false); }}
                                >
                                    <Image size={16} className="text-green-400" />
                                    <span style={{ color: theme.colors.text.primary }} className="text-sm font-medium">Images</span>
                                </button>
                                <button
                                    className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-white/5 transition-colors"
                                    onClick={() => { modelInputRef.current?.click(); setShowAttachMenu(false); }}
                                >
                                    <Box size={16} className="text-purple-400" />
                                    <span style={{ color: theme.colors.text.primary }} className="text-sm font-medium">3D Models</span>
                                </button>
                            </div>
                        )}

                        {/* Hidden Specific Inputs */}
                        <input ref={imageInputRef} type="file" accept="image/*" multiple hidden onChange={handleFileChange} />
                        <input ref={modelInputRef} type="file" accept=".stl,.obj,.step,.stp" multiple hidden onChange={handleFileChange} />
                    </div>

                    <button
                        className="p-2 rounded-lg transition-all hover:bg-white/5"
                        style={{ color: theme.colors.text.primary }}
                        title="Draw (Coming Soon)"
                    >
                        <Pencil size={20} />
                    </button>
                </div>

                {/* Submit Button */}
                <button
                    onClick={handleSubmit}
                    disabled={!message.trim() && attachedFiles.length === 0}
                    className="px-6 py-2.5 rounded-lg font-bold text-sm uppercase tracking-wide transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed hover:scale-105 active:scale-95 flex items-center gap-2"
                    style={{
                        background: (message.trim() || attachedFiles.length > 0)
                            ? `linear-gradient(135deg, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`
                            : theme.colors.bg.tertiary,
                        color: (message.trim() || attachedFiles.length > 0) ? theme.colors.bg.primary : theme.colors.text.muted,
                        boxShadow: (message.trim() || attachedFiles.length > 0) ? `0 4px 12px ${theme.colors.accent.glow}` : 'none',
                    }}
                >
                    Submit
                </button>
            </div>
        </div >
    );
}
