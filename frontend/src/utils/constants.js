// Shared constants for BRICK OS

export const ACTIVITY_BAR_WIDTH = 48;

export const DEFAULT_PANEL_SIZES = {
  left: 240,
  right: 384,
  bottom: 280
};

export const ACTIVITY_TABS = {
  SEARCH: 'search',
  DESIGN: 'design',
  RUN: 'run',
  AGENTS: 'agents',
  COMPILE: 'compile',
  MFG: 'mfg',
  FORK: 'fork',
  EXPORT: 'export',
  DOCS: 'docs',
  SETTINGS: 'settings'
};

export const LOWER_PANEL_TABS = {
  TERMINAL: 'terminal',
  CONSOLE: 'console',
  OUTPUT: 'output',
  DIAGNOSTICS: 'diagnostics',
  KCL: 'kcl'
};

export const DEFAULT_FILE_CONTENT = `// Enter your OpenSCAD code here...
// BRICK OS v1.0
`;

export const LLM_PROVIDERS = [
  // Cloud Providers
  { value: 'groq', label: 'Groq (Llama 3.3 70B)' },
  { value: 'kimi', label: 'Kimi AI (Moonshot)' },
  { value: 'openai', label: 'OpenAI (GPT-4 Turbo)' },
  { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
  { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
  { value: 'gemini-3-flash', label: 'Gemini 3 Flash' },
  { value: 'gemini-3-pro', label: 'Gemini 3 Pro' },
  { value: 'huggingface', label: 'HuggingFace (Llama 3 8B)' },
  // Local Providers
  { value: 'ollama', label: 'Ollama (Local)' },
];
