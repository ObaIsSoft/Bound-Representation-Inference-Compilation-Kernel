import React, { useState, useRef, KeyboardEvent } from 'react';
import { Image, Pencil } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

interface TextInputProps {
    onSubmit: (message: string, options: InputOptions) => void;
    onFocus: () => void;
}

interface InputOptions {
    llmProvider: string;
    attachedImages: File[];
    drawings: Blob[];
}

const LLM_PROVIDERS = [
    // Cloud Providers
    { value: 'groq', label: 'Groq (Llama 3.3 70B)' },
    { value: 'openai', label: 'OpenAI (GPT-4 Turbo)' },
    { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
    { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
    { value: 'gemini-3-flash', label: 'Gemini 3 Flash' },
    { value: 'gemini-3-pro', label: 'Gemini 3 Pro' },
    { value: 'huggingface', label: 'HuggingFace (Llama 3 8B)' },
    // Local Providers
    { value: 'ollama', label: 'Ollama (Local)' },
] as const;

export default function TextInput({ onSubmit, onFocus }: TextInputProps) {
    const [message, setMessage] = useState('');
    const [llmProvider, setLlmProvider] = useState<string>('groq');
    const [attachedImages, setAttachedImages] = useState<File[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const { theme } = useTheme();

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    const handleSubmit = () => {
        if (!message.trim()) return;

        onSubmit(message, {
            llmProvider,
            attachedImages,
            drawings: [],
        });

        setMessage('');
        setAttachedImages([]);
    };

    const handleImageAttach = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || []);
        setAttachedImages(prev => [...prev, ...files]);
    };

    const removeImage = (index: number) => {
        setAttachedImages(prev => prev.filter((_, i) => i !== index));
    };

    return (
        <div
            className="w-full max-w-3xl rounded-2xl shadow-2xl backdrop-blur-sm animate-fade-in"
            style={{
                backgroundColor: theme.colors.bg.secondary + 'CC',
                border: `1px solid ${theme.colors.border.primary}`,
                boxShadow: `0 20px 60px ${theme.colors.accent.primary}15`,
            }}
        >
            {/* Attached Images Preview */}
            {
                attachedImages.length > 0 && (
                    <div className="p-4 flex gap-2 flex-wrap border-b" style={{ borderColor: theme.colors.border.primary }}>
                        {attachedImages.map((file, index) => (
                            <div
                                key={index}
                                className="relative group rounded-lg overflow-hidden"
                                style={{ width: '80px', height: '80px' }}
                            >
                                <img
                                    src={URL.createObjectURL(file)}
                                    alt={file.name}
                                    className="w-full h-full object-cover"
                                />
                                <button
                                    onClick={() => removeImage(index)}
                                    className="absolute top-1 right-1 w-5 h-5 rounded-full bg-red-500 text-white text-xs opacity-0 group-hover:opacity-100 transition-opacity"
                                >
                                    ×
                                </button>
                            </div>
                        ))}
                    </div>
                )
            }

            {/* Text Input */}
            <div className="p-6">
                <textarea
                    ref={textareaRef}
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={handleKeyDown}
                    onFocus={onFocus}
                    placeholder="Type to create..."
                    className="w-full bg-transparent resize-none outline-none font-mono text-lg"
                    style={{
                        color: theme.colors.text.primary,
                        minHeight: '60px',
                    }}
                    rows={3}
                />

                <div className="mt-2 text-xs opacity-60" style={{ color: theme.colors.text.muted }}>
                    {message.length} characters • Press Enter to submit, Shift+Enter for new line
                </div>
            </div>

            {/* Control Bar */}
            <div
                className="px-6 py-4 flex items-center justify-between border-t"
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

                    {/* Image Attach Button */}
                    <button
                        onClick={handleImageAttach}
                        className="p-2 rounded-lg transition-all hover:scale-110 active:scale-95"
                        style={{
                            backgroundColor: theme.colors.bg.tertiary,
                            border: `1px solid ${theme.colors.border.primary}`,
                            color: theme.colors.text.primary,
                        }}
                        title="Attach image"
                    >
                        <Image size={20} />
                    </button>

                    {/* Draw Button */}
                    <button
                        onClick={() => {/* TODO: Open draw modal */ }}
                        className="p-2 rounded-lg transition-all hover:scale-110 active:scale-95"
                        style={{
                            backgroundColor: theme.colors.bg.tertiary,
                            border: `1px solid ${theme.colors.border.primary}`,
                            color: theme.colors.text.primary,
                        }}
                        title="Draw"
                    >
                        <Pencil size={20} />
                    </button>

                    {/* Hidden file input */}
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        multiple
                        onChange={handleFileChange}
                        className="hidden"
                    />
                </div>

                {/* Submit Button */}
                <button
                    onClick={handleSubmit}
                    disabled={!message.trim()}
                    className="px-6 py-2.5 rounded-lg font-bold text-sm uppercase tracking-wide transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed hover:scale-105 active:scale-95"
                    style={{
                        background: message.trim()
                            ? `linear-gradient(135deg, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`
                            : theme.colors.bg.tertiary,
                        color: message.trim() ? theme.colors.bg.primary : theme.colors.text.muted,
                        boxShadow: message.trim() ? `0 4px 12px ${theme.colors.accent.glow}` : 'none',
                    }}
                >
                    Submit →
                </button>
            </div>
        </div >
    );
}
