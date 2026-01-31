import React from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { Book, Code, Zap, Terminal, Mic, Type } from 'lucide-react';

export default function DocumentationPage() {
    const { theme } = useTheme();

    return (
        <div
            className="flex flex-col min-h-screen w-full"
            style={{
                backgroundColor: theme.colors.bg.primary,
                color: theme.colors.text.primary,
            }}
        >
            {/* Header */}
            <div
                className="px-8 py-6 border-b"
                style={{ borderColor: theme.colors.border.primary }}
            >
                <div className="flex items-center gap-3 mb-2">
                    <Book size={28} style={{ color: theme.colors.accent.primary }} />
                    <h1 className="text-3xl font-black">Documentation</h1>
                </div>
                <p style={{ color: theme.colors.text.muted }}>
                    Complete guide to BRICK OS features and capabilities
                </p>
            </div>

            {/* Centered Content */}
            <div className="flex-1 flex flex-col items-center justify-center px-8 py-6 w-full">
                <div className="w-full max-w-4xl">
                    {/* Overview */}
                    <section className="mb-8">
                        <h2 className="text-2xl font-bold mb-4" style={{ color: theme.colors.accent.primary }}>
                            Overview
                        </h2>
                        <p className="mb-4" style={{ color: theme.colors.text.secondary }}>
                            BRICK OS is a conversational hardware design platform that combines:
                        </p>
                        <ul className="space-y-2 ml-6" style={{ color: theme.colors.text.secondary }}>
                            <li>• Multi-modal input (voice and text)</li>
                            <li>• AI-powered design agents</li>
                            <li>• Real-time physics simulation</li>
                            <li>• Integrated CAD and manufacturing pipeline</li>
                        </ul>
                    </section>

                    {/* Voice Input */}
                    <section className="mb-8">
                        <div className="flex items-center gap-2 mb-4">
                            <Mic size={24} style={{ color: theme.colors.accent.secondary }} />
                            <h2 className="text-2xl font-bold" style={{ color: theme.colors.accent.primary }}>
                                Voice Input
                            </h2>
                        </div>
                        <p className="mb-3" style={{ color: theme.colors.text.secondary }}>
                            Click the waveform visualizer to activate voice mode:
                        </p>
                        <div
                            className="p-4 rounded-lg mb-3"
                            style={{ backgroundColor: theme.colors.bg.tertiary }}
                        >
                            <ol className="space-y-2" style={{ color: theme.colors.text.secondary }}>
                                <li>1. Click the circular waveform</li>
                                <li>2. Allow microphone permissions</li>
                                <li>3. Speak your design intent</li>
                                <li>4. Click "Stop Recording" when done</li>
                                <li>5. Review transcription and submit</li>
                            </ol>
                        </div>
                    </section>

                    {/* Text Input */}
                    <section className="mb-8">
                        <div className="flex items-center gap-2 mb-4">
                            <Type size={24} style={{ color: theme.colors.accent.secondary }} />
                            <h2 className="text-2xl font-bold" style={{ color: theme.colors.accent.primary }}>
                                Text Input
                            </h2>
                        </div>
                        <p className="mb-3" style={{ color: theme.colors.text.secondary }}>
                            Type your design requests in the text input area:
                        </p>
                        <div
                            className="p-4 rounded-lg mb-3"
                            style={{ backgroundColor: theme.colors.bg.tertiary }}
                        >
                            <p className="mb-2" style={{ color: theme.colors.text.secondary }}>
                                <strong>Keyboard Shortcuts:</strong>
                            </p>
                            <ul className="space-y-1 ml-4" style={{ color: theme.colors.text.secondary }}>
                                <li>• <code className="px-2 py-1 rounded" style={{ backgroundColor: theme.colors.bg.primary }}>Enter</code> - Submit message</li>
                                <li>• <code className="px-2 py-1 rounded" style={{ backgroundColor: theme.colors.bg.primary }}>Shift+Enter</code> - New line</li>
                                <li>• <code className="px-2 py-1 rounded" style={{ backgroundColor: theme.colors.bg.primary }}>Cmd+K</code> - New chat</li>
                            </ul>
                        </div>
                    </section>

                    {/* API Endpoints */}
                    <section className="mb-8">
                        <div className="flex items-center gap-2 mb-4">
                            <Terminal size={24} style={{ color: theme.colors.accent.secondary }} />
                            <h2 className="text-2xl font-bold" style={{ color: theme.colors.accent.primary }}>
                                API Endpoints
                            </h2>
                        </div>
                        <div className="space-y-4">
                            {/* Chat Endpoint */}
                            <div
                                className="p-4 rounded-lg"
                                style={{ backgroundColor: theme.colors.bg.tertiary }}
                            >
                                <div className="flex items-center gap-2 mb-2">
                                    <code
                                        className="px-2 py-1 rounded font-bold"
                                        style={{
                                            backgroundColor: theme.colors.accent.primary + '20',
                                            color: theme.colors.accent.primary,
                                        }}
                                    >
                                        POST
                                    </code>
                                    <code style={{ color: theme.colors.text.primary }}>
                                        /api/chat
                                    </code>
                                </div>
                                <p className="mb-2" style={{ color: theme.colors.text.secondary }}>
                                    Main conversational agent endpoint. Accepts text and voice inputs.
                                </p>
                                <p className="text-sm" style={{ color: theme.colors.text.muted }}>
                                    <strong>Parameters:</strong> message, llm_provider, source (text/voice), attachments
                                </p>
                            </div>

                            {/* STT Endpoint */}
                            <div
                                className="p-4 rounded-lg"
                                style={{ backgroundColor: theme.colors.bg.tertiary }}
                            >
                                <div className="flex items-center gap-2 mb-2">
                                    <code
                                        className="px-2 py-1 rounded font-bold"
                                        style={{
                                            backgroundColor: theme.colors.accent.primary + '20',
                                            color: theme.colors.accent.primary,
                                        }}
                                    >
                                        POST
                                    </code>
                                    <code style={{ color: theme.colors.text.primary }}>
                                        /api/stt/transcribe
                                    </code>
                                </div>
                                <p className="mb-2" style={{ color: theme.colors.text.secondary }}>
                                    Speech-to-text transcription endpoint.
                                </p>
                                <p className="text-sm" style={{ color: theme.colors.text.muted }}>
                                    <strong>Parameters:</strong> audio (file)
                                </p>
                            </div>
                        </div>
                    </section>

                    {/* LLM Providers */}
                    <section className="mb-8">
                        <div className="flex items-center gap-2 mb-4">
                            <Zap size={24} style={{ color: theme.colors.accent.secondary }} />
                            <h2 className="text-2xl font-bold" style={{ color: theme.colors.accent.primary }}>
                                LLM Providers
                            </h2>
                        </div>
                        <p className="mb-3" style={{ color: theme.colors.text.secondary }}>
                            Select your preferred language model from the dropdown:
                        </p>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div
                                className="p-4 rounded-lg"
                                style={{ backgroundColor: theme.colors.bg.tertiary }}
                            >
                                <h3 className="font-bold mb-1" style={{ color: theme.colors.text.primary }}>
                                    Groq
                                </h3>
                                <p className="text-sm" style={{ color: theme.colors.text.muted }}>
                                    Ultra-fast inference with Llama models
                                </p>
                            </div>
                            <div
                                className="p-4 rounded-lg"
                                style={{ backgroundColor: theme.colors.bg.tertiary }}
                            >
                                <h3 className="font-bold mb-1" style={{ color: theme.colors.text.primary }}>
                                    OpenAI
                                </h3>
                                <p className="text-sm" style={{ color: theme.colors.text.muted }}>
                                    GPT-4 for advanced reasoning
                                </p>
                            </div>
                            <div
                                className="p-4 rounded-lg"
                                style={{ backgroundColor: theme.colors.bg.tertiary }}
                            >
                                <h3 className="font-bold mb-1" style={{ color: theme.colors.text.primary }}>
                                    Anthropic
                                </h3>
                                <p className="text-sm" style={{ color: theme.colors.text.muted }}>
                                    Claude for long-context tasks
                                </p>
                            </div>
                        </div>
                    </section>

                    {/* Sidebar Features */}
                    <section className="mb-8">
                        <div className="flex items-center gap-2 mb-4">
                            <Code size={24} style={{ color: theme.colors.accent.secondary }} />
                            <h2 className="text-2xl font-bold" style={{ color: theme.colors.accent.primary }}>
                                Sidebar Features
                            </h2>
                        </div>
                        <div className="space-y-3">
                            <div
                                className="p-3 rounded-lg"
                                style={{ backgroundColor: theme.colors.bg.tertiary }}
                            >
                                <h3 className="font-bold mb-1" style={{ color: theme.colors.text.primary }}>
                                    Search
                                </h3>
                                <p className="text-sm" style={{ color: theme.colors.text.muted }}>
                                    Search through design history and conversations
                                </p>
                            </div>
                            <div
                                className="p-3 rounded-lg"
                                style={{ backgroundColor: theme.colors.bg.tertiary }}
                            >
                                <h3 className="font-bold mb-1" style={{ color: theme.colors.text.primary }}>
                                    Export
                                </h3>
                                <p className="text-sm" style={{ color: theme.colors.text.muted }}>
                                    Export designs to various formats (STEP, STL, etc.)
                                </p>
                            </div>
                            <div
                                className="p-3 rounded-lg"
                                style={{ backgroundColor: theme.colors.bg.tertiary }}
                            >
                                <h3 className="font-bold mb-1" style={{ color: theme.colors.text.primary }}>
                                    Settings
                                </h3>
                                <p className="text-sm" style={{ color: theme.colors.text.muted }}>
                                    Configure themes, LLM providers, and system preferences
                                </p>
                            </div>
                        </div>
                    </section>
                </div>
            </div>

            {/* Footer */}
            <div
                className="pt-8 border-t text-center"
                style={{ borderColor: theme.colors.border.primary }}
            >
                <p className="text-sm" style={{ color: theme.colors.text.muted }}>
                    BRICK OS © 2026 • For more information, visit the GitHub repository
                </p>
            </div>
        </div>
    );
}
