import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';
import WaveformVisualizer from '../components/voice/WaveformVisualizer';
import VoiceRecorder from '../components/voice/VoiceRecorder';
import TextInput from '../components/voice/TextInput';
import LockedSidebar from '../components/layout/LockedSidebar';
import SearchPanel from '../components/panels/SearchPanel';
import ExportPanel from '../components/panels/ExportPanel';
import SettingsPage from '../components/settings/SettingsPage';
import AccountPage from '../components/settings/AccountPage';
import DocumentationPage from '../components/settings/DocumentationPage';
import { DEFAULT_PANEL_SIZES } from '../utils/constants';
import apiClient from '../utils/apiClient';
import { usePanel } from '../contexts/PanelContext';

type Mode = 'idle' | 'text' | 'voice';

interface InputOptions {
  llmProvider: string;
  attachedImages: File[];
  drawings: Blob[];
}

export default function Landing() {
  const { theme } = useTheme();
  const navigate = useNavigate();
  const [mode, setMode] = useState<Mode>('idle');
  const [audioStream, setAudioStream] = useState<MediaStream | undefined>(undefined);
  const [userIntent, setUserIntent] = useState<string>('');
  const [activePanel, setActivePanel] = useState<string | null>(null);
  const [fadeOut, setFadeOut] = useState(false);
  const [leftWidth] = useState(DEFAULT_PANEL_SIZES.left);
  const { startNewSession, addMessageToSession } = usePanel();

  const currentHour = new Date().getHours();
  const greeting = currentHour < 12 ? 'Good morning' : currentHour < 18 ? 'Good afternoon' : 'Good evening';

  const handleNewChat = () => {
    setMode('idle');
    setUserIntent('');
    setAudioStream(undefined);
    setActivePanel(null); // Close any active panels
  };

  const handleSettingsClick = () => {
    setActivePanel('settings');
  };

  const handleAccountClick = () => {
    setActivePanel('account');
  };

  const handleDocsClick = () => {
    setActivePanel('docs');
  };

  const handlePanelChange = (panel: string) => {
    setActivePanel(panel);
  };

  const handleWaveformClick = async () => {
    if (mode !== 'idle') return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setAudioStream(stream);
      setMode('voice');
    } catch (err) {
      console.error('Microphone access denied:', err);
      alert('Microphone permission is required for voice input. Falling back to text mode.');
      setMode('text');
    }
  };

  const handleTextFocus = () => {
    if (mode === 'idle') {
      setMode('text');
    }
  };

  const handleVoiceTranscription = (transcription: string) => {
    setUserIntent(transcription);
    submitToChat(transcription, 'voice', { llmProvider: 'groq', attachedImages: [], drawings: [] });

    // Fade out before navigation
    setFadeOut(true);
    setTimeout(() => {
      navigate('/requirements', {
        state: {
          userIntent: transcription,
          llmProvider: 'groq'
        }
      });
    }, 300);

    resetToIdle();
  };

  const handleTextSubmit = (message: string, options: InputOptions) => {
    setUserIntent(message);
    setUserIntent(message);
    // submitToChat(message, 'text', options); // Moved to RequirementsGatheringPage for better UX/State sync

    // Fade out before navigation
    setFadeOut(true);
    setTimeout(() => {
      navigate('/requirements', {
        state: {
          userIntent: message,
          llmProvider: options.llmProvider
        }
      });
    }, 300);
  };

  const submitToChat = async (message: string, source: 'text' | 'voice', options: InputOptions) => {
    const { llmProvider, attachedImages } = options;

    try {
      // 1. Start a new session via Context
      const sessionId = await startNewSession();
      if (!sessionId) throw new Error('Failed to start session');

      // 2. Optimistically add user message to the new session
      // (This will appear in history immediately)
      addMessageToSession(sessionId, 'user', message);

      // 3. Send initial message to Discovery/Requirements endpoint
      // Using JSON payload instead of FormData
      const payload = {
        message: message,
        user_intent: message,
        ai_model: llmProvider,
        mode: 'requirements_gathering',
        session_id: sessionId,
        conversation_history: []
        // Note: File uploads (attachedImages) need separate handling
      };

      // We fire and forget the actual API call here so navigation is instant,
      // OR we await it if we want the agent's first response to be ready.
      // For better UX, let's await it so the next page has data.
      const data = await apiClient.post('/chat/requirements', payload);

      // 4. Update session with agent response
      addMessageToSession(sessionId, 'agent', data.response);

    } catch (error) {
      console.error('Failed to submit chat:', error);
      alert('Failed to submit your request. Please check the console for details.');
    }
  };

  const resetToIdle = () => {
    if (audioStream) {
      audioStream.getTracks().forEach(track => track.stop());
      setAudioStream(undefined);
    }
    setMode('idle');
    setUserIntent('');
  };

  return (
    <div
      className={`flex h-screen ${fadeOut ? 'animate-fadeOut' : ''}`}
      style={{
        backgroundColor: theme.colors.bg.primary,
        backgroundImage: `
          radial-gradient(circle at 20% 50%, ${theme.colors.accent.primary}15 0%, transparent 50%),
          radial-gradient(circle at 80% 80%, ${theme.colors.accent.secondary}10 0%, transparent 50%)
        `,
      }}
    >
      {/* Sidebar */}
      <LockedSidebar
        activePanel={activePanel || ''}
        onPanelChange={handlePanelChange}
        onNewChat={handleNewChat}
        onSettingsClick={handleSettingsClick}
        onAccountClick={handleAccountClick}
        onDocsClick={handleDocsClick}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex">
        {/* Left Panel (when active) */}
        {activePanel && activePanel !== 'settings' && (
          <div style={{ width: `${leftWidth}px` }}>
            {activePanel === 'search' && <SearchPanel width={leftWidth} />}
            {activePanel === 'export' && <ExportPanel width={leftWidth} />}
          </div>
        )}

        {/* Center Content */}
        <div className="flex-1 flex flex-col">
          {activePanel === 'settings' ? (
            <SettingsPage />
          ) : activePanel === 'account' ? (
            <AccountPage />
          ) : activePanel === 'docs' ? (
            <DocumentationPage />
          ) : (
            <div
              className="flex-1 flex flex-col items-center justify-between px-4 py-8"
              style={{
                backgroundImage: `radial-gradient(circle at 20% 30%, ${theme.colors.accent.primary}08 0%, transparent 50%),
                                  radial-gradient(circle at 80% 70%, ${theme.colors.accent.secondary}05 0%, transparent 50%)`,
              }}
            >
              {/* Greeting Section */}
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center animate-fade-in">
                  <h1
                    className="text-4xl md:text-5xl lg:text-6xl font-black tracking-tight mb-3"
                    style={{ color: theme.colors.text.primary }}
                  >
                    {greeting},{' '}
                    <span
                      style={{
                        background: `linear-gradient(135deg, ${theme.colors.accent.primary}, ${theme.colors.accent.secondary})`,
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                      }}
                    >
                      obafemi
                    </span>
                  </h1>
                  <p
                    className="text-lg md:text-xl font-mono"
                    style={{ color: theme.colors.text.muted }}
                  >
                    The system is ready. What shall we build?
                  </p>
                </div>
              </div>

              {/* Main Interface Area */}
              <div className="flex-1 flex flex-col items-center justify-center gap-8 w-full max-w-4xl px-4">
                {/* Waveform (always visible, clickable for voice mode) */}
                <div
                  onClick={handleWaveformClick}
                  className={`cursor-pointer transition-all duration-500 ${mode === 'idle' ? 'hover:scale-105' : ''
                    }`}
                >
                  <WaveformVisualizer
                    isActive={mode === 'voice'}
                    audioStream={audioStream}
                  />
                </div>

                {/* Voice Recorder Modal Overlay (shown in voice mode) */}
                {mode === 'voice' && (
                  <div className="fixed inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm z-50">
                    <VoiceRecorder
                      onTranscriptionComplete={handleVoiceTranscription}
                      onCancel={resetToIdle}
                    />
                  </div>
                )}

                {/* Text Input (always visible below waveform) */}
                <div className="w-full max-w-3xl">
                  <TextInput
                    onSubmit={handleTextSubmit}
                    onFocus={handleTextFocus}
                  />
                </div>
              </div>

              {/* Footer */}
              <div className="text-center">
                <p
                  className="text-xs font-mono tracking-widest"
                  style={{ color: theme.colors.text.muted }}
                >
                  ● BRICK OS © 2026 • System Operational
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* CSS Animations */}
      <style>{`
        @keyframes fadeOut {
          from {
            opacity: 1;
            transform: scale(1);
          }
          to {
            opacity: 0;
            transform: scale(0.98);
          }
        }
        
        .animate-fadeOut {
          animation: fadeOut 0.3s ease-in-out forwards;
        }
      `}</style>
    </div>
  );
}
