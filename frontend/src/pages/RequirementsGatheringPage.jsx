import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, MessageCircle, Send, Mic, Loader, CheckCircle, FileText } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

export default function RequirementsGatheringPage() {
    const location = useLocation();
    const navigate = useNavigate();
    const userIntent = location.state?.userIntent || '';
    const llmProvider = location.state?.llmProvider || 'groq';
    const { theme } = useTheme();

    // Phase management
    const [phase, setPhase] = useState('gathering'); // 'gathering' | 'summary' | 'planning'

    // Gathering phase states
    const [agentMessages, setAgentMessages] = useState([]);
    const [userInput, setUserInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [isTyping, setIsTyping] = useState(false);

    // Summary & Planning phase states
    const [requirements, setRequirements] = useState({});
    const [planArtifacts, setPlanArtifacts] = useState([]);
    const [planGenerating, setPlanGenerating] = useState(false);
    const [conversationComplete, setConversationComplete] = useState(false);

    // Initialize conversation with first agent question
    useEffect(() => {
        if (userIntent && agentMessages.length === 0) {
            setLoading(true);
            setIsTyping(true);
            // Simulate initial agent response
            setTimeout(() => {
                setAgentMessages([
                    `Thank you for sharing your intent: "${userIntent}".Let me ask you a few questions to better understand your requirements.`,
                    'What is the primary use case or application for this design?'
                ]);
                setLoading(false);
                setIsTyping(false);
            }, 1500);
        }
    }, [userIntent]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!userInput.trim()) return;

        // Add user message to conversation
        const newMessages = [...agentMessages, `You: ${userInput} `];
        setAgentMessages(newMessages);
        setUserInput('');
        setLoading(true);
        setIsTyping(true);

        try {
            // Call backend API for conversational agent
            const response = await fetch('http://localhost:8000/api/chat/requirements', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: userInput,
                    conversation_history: agentMessages,
                    user_intent: userIntent,
                    mode: 'requirements_gathering',
                    ai_model: llmProvider
                }),
            });

            if (!response.ok) {
                throw new Error(`API failed: ${response.statusText} `);
            }

            const data = await response.json();

            // Simulate typing delay for better UX
            await new Promise(resolve => setTimeout(resolve, 800));

            // Add agent response
            setAgentMessages([...newMessages, `Agent: ${data.response} `]);
            setIsTyping(false);

            // Check if conversation is complete
            if (data.requirements_complete) {
                setRequirements(data.requirements || {});
                setConversationComplete(true);

                // Check if artifacts are included in response
                const artifacts = data.messages?.filter(msg => msg.type === 'artifact') || [];

                if (artifacts.length > 0) {
                    // Artifacts already generated - go straight to planning
                    setPlanArtifacts(artifacts);
                    setTimeout(() => {
                        setPhase('planning');
                    }, 2000);
                } else {
                    // Show summary and generate plan
                    setTimeout(() => {
                        setPhase('summary');
                        generatePlan(data.requirements || {}, data.conversation_id);
                    }, 2000);
                }
            }
        } catch (error) {
            console.error('Failed to get agent response:', error);
            setAgentMessages([...newMessages, 'Agent: Sorry, I encountered an error. Please try again.']);
            setIsTyping(false);
        } finally {
            setLoading(false);
        }
    };

    const generatePlan = async (requirements, conversationId) => {
        setPlanGenerating(true);

        try {
            const response = await fetch('http://localhost:8000/api/orchestrator/plan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    requirements,
                    user_intent: userIntent,
                    conversation_id: conversationId,
                    mode: 'plan',
                    ai_model: llmProvider
                }),
            });

            if (!response.ok) {
                throw new Error(`Plan generation failed: ${response.statusText} `);
            }

            const data = await response.json();
            setPlanArtifacts(data.artifacts || []);

            // Transition to planning phase
            setTimeout(() => {
                setPhase('planning');
                setPlanGenerating(false);
            }, 1000);
        } catch (error) {
            console.error('Plan generation failed:', error);
            setPlanGenerating(false);
            // Show error message but stay in summary phase
            setAgentMessages([...agentMessages, 'Agent: Failed to generate plan. Please try again.']);
        }
    };

    const handleUserInput = (e) => {
        setUserInput(e.target.value);
    };

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
                    onClick={() => navigate('/')}
                    className="p-2 rounded-lg hover:bg-opacity-10 transition-all"
                    style={{ backgroundColor: theme.colors.bg.tertiary }}
                >
                    <ArrowLeft size={20} style={{ color: theme.colors.text.primary }} />
                </button>
                <div className="flex items-center gap-3">
                    <MessageCircle size={28} style={{ color: theme.colors.accent.primary }} />
                    <h1 className="text-2xl font-bold">Requirements Gathering</h1>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 overflow-y-auto px-8 py-6">
                {/* Phase 1: Gathering */}
                {phase === 'gathering' && (
                    <div className="max-w-3xl mx-auto">
                        {/* User Intent Display */}
                        <div
                            className="mb-6 p-4 rounded-lg"
                            style={{
                                backgroundColor: theme.colors.bg.tertiary,
                                borderLeft: `4px solid ${theme.colors.accent.primary}`
                            }}
                        >
                            <span className="font-semibold" style={{ color: theme.colors.accent.primary }}>
                                Your Intent:
                            </span>
                            <p className="mt-2" style={{ color: theme.colors.text.secondary }}>
                                {userIntent || 'No intent provided'}
                            </p>
                        </div>

                        {/* Conversation Messages */}
                        <div className="space-y-4 mb-6">
                            {agentMessages.map((msg, idx) => {
                                const isUser = msg.startsWith('You:');
                                return (
                                    <div
                                        key={idx}
                                        className={`p-4 rounded-lg ${isUser ? 'ml-8' : 'mr-8'}`}
                                        style={{
                                            backgroundColor: isUser ? theme.colors.bg.tertiary : theme.colors.bg.secondary,
                                        }}
                                    >
                                        <p style={{ color: theme.colors.text.primary }}>
                                            {msg}
                                        </p>
                                    </div>
                                );
                            })}

                            {(loading || isTyping) && (
                                <div
                                    className="flex items-center gap-2 p-4 rounded-lg mr-8"
                                    style={{ backgroundColor: theme.colors.bg.secondary }}
                                >
                                    <Loader
                                        className="animate-spin"
                                        size={18}
                                        style={{ color: theme.colors.accent.secondary }}
                                    />
                                    <span style={{ color: theme.colors.text.muted }}>
                                        {isTyping ? 'Agent is typing' : 'Agent is thinking'}
                                        <span className="animate-pulse">...</span>
                                    </span>
                                </div>
                            )}

                            {conversationComplete && (
                                <div
                                    className="p-4 rounded-lg text-center"
                                    style={{
                                        backgroundColor: theme.colors.accent.primary + '20',
                                        borderLeft: `3px solid ${theme.colors.accent.primary}`
                                    }}
                                >
                                    <span style={{ color: theme.colors.accent.primary }} className="font-bold">
                                        ✓ Requirements gathered! Preparing summary...
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Phase 2: Summary */}
                {phase === 'summary' && (
                    <div className="max-w-3xl mx-auto phase-enter">
                        {/* Success Message */}
                        <div
                            className="mb-6 p-6 rounded-lg"
                            style={{
                                backgroundColor: theme.colors.accent.primary + '20',
                                borderLeft: `4px solid ${theme.colors.accent.primary}`
                            }}
                        >
                            <div className="flex items-center gap-3 mb-2">
                                <CheckCircle size={28} style={{ color: theme.colors.accent.primary }} />
                                <h2 className="text-2xl font-bold">
                                    Requirements Gathered!
                                </h2>
                            </div>
                            <p style={{ color: theme.colors.text.secondary }}>
                                I have all the information needed to create your design plan.
                            </p>
                        </div>

                        {/* Requirements Summary Card */}
                        <div
                            className="mb-6 p-6 rounded-lg"
                            style={{ backgroundColor: theme.colors.bg.secondary }}
                        >
                            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                                <FileText size={20} style={{ color: theme.colors.accent.secondary }} />
                                Summary
                            </h3>
                            <div className="space-y-3">
                                {Object.entries(requirements).length > 0 ? (
                                    Object.entries(requirements).map(([key, value]) => (
                                        <div key={key} className="pb-2 border-b" style={{ borderColor: theme.colors.border.primary }}>
                                            <span className="font-semibold" style={{ color: theme.colors.accent.primary }}>
                                                {key}:
                                            </span>
                                            <span className="ml-2" style={{ color: theme.colors.text.secondary }}>
                                                {String(value)}
                                            </span>
                                        </div>
                                    ))
                                ) : (
                                    <p style={{ color: theme.colors.text.muted }}>
                                        No specific requirements captured. Using user intent: "{userIntent}"
                                    </p>
                                )}
                            </div>
                        </div>

                        {/* Generating Plan Indicator */}
                        {planGenerating && (
                            <div className="flex items-center justify-center gap-3 p-6">
                                <Loader className="animate-spin" size={28} style={{ color: theme.colors.accent.primary }} />
                                <span className="text-lg" style={{ color: theme.colors.text.primary }}>
                                    Generating your design plan...
                                </span>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Input Area (Fixed at Bottom) - Only in Gathering Phase */}
            {phase === 'gathering' && !conversationComplete && (
                <div
                    className="px-8 py-4 border-t"
                    style={{
                        borderColor: theme.colors.border.primary,
                        backgroundColor: theme.colors.bg.primary
                    }}
                >
                    <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex gap-2">
                        <input
                            type="text"
                            value={userInput}
                            className="flex-1 px-4 py-3 rounded-lg border outline-none"
                            style={{
                                backgroundColor: theme.colors.bg.secondary,
                                borderColor: theme.colors.border.primary,
                                color: theme.colors.text.primary,
                            }}
                            placeholder="Type your answer..."
                            onChange={handleUserInput}
                            autoFocus
                            disabled={loading}
                        />
                        <button
                            type="button"
                            className="p-3 rounded-lg transition-all hover:opacity-80"
                            style={{ backgroundColor: theme.colors.accent.secondary }}
                            aria-label="Voice Input"
                            disabled={loading}
                        >
                            <Mic size={22} style={{ color: theme.colors.text.primary }} />
                        </button>
                        <button
                            type="submit"
                            className="px-6 py-3 rounded-lg font-bold transition-all hover:opacity-90"
                            style={{
                                backgroundColor: theme.colors.accent.primary,
                                color: theme.colors.text.primary
                            }}
                            disabled={loading || !userInput.trim()}
                        >
                            Submit
                        </button>
                    </form>
                </div>
            )}

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

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate-slideUp {
    animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.phase-enter {
    animation: fadeIn 0.5s ease-out;
}
`}</style>
        </div>
    );
}
