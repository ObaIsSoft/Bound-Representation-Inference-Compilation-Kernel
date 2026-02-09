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
  attachedImages: File[]; // Holds all files from TextInput
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
  const [isUploading, setIsUploading] = useState(false);
  const { addMessageToSession } = usePanel(); // Removed unused startNewSession

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

  const uploadFiles = async (files: File[]): Promise<string[]> => {
    if (files.length === 0) return [];

    setIsUploading(true);
    try {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append('files', file);
      });

      const data = await apiClient.post('/files/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return data.file_ids || [];
    } catch (error) {
      console.error('Failed to upload files:', error);
      alert('Failed to upload some files. Continuing without file context.');
      return [];
    } finally {
      setIsUploading(false);
    }
  };

  const handleVoiceTranscription = async (transcription: string) => {
    setUserIntent(transcription);

    // No files from voice for now
    const fileIds: string[] = [];

    // Fade out before navigation
    setFadeOut(true);
    setTimeout(() => {
      navigate('/requirements', {
        state: {
          userIntent: transcription,
          llmProvider: 'groq',
          uploadedFiles: fileIds,
          fileNames: []
        }
      });
    }, 300);

    resetToIdle();
  };

  const handleTextSubmit = async (message: string, options: InputOptions) => {
    setUserIntent(message);

    // Upload files passed from TextInput
    // Note: TextInput passes 'attachedImages' which now contains all files
    const filesToUpload = options.attachedImages;
    const fileIds = filesToUpload.length > 0 ? await uploadFiles(filesToUpload) : [];

    // Fade out before navigation
    setFadeOut(true);
    setTimeout(() => {
      navigate('/requirements', {
        state: {
          userIntent: message,
          llmProvider: options.llmProvider,
          uploadedFiles: fileIds,
          fileNames: filesToUpload.map(f => f.name)
        }
      });
    }, 300);
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
                  {isUploading && (
                    <div
                      className="mt-2 text-center text-sm"
                      style={{ color: theme.colors.accent.primary }}
                    >
                      Uploading files...
                    </div>
                  )}
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
