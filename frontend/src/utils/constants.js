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

export const DEFAULT_FILE_CONTENT = JSON.stringify({
  type: 'primitive',
  geometry: 'box',
  args: [1, 1, 1], // Unit Cube
  material: {
    color: '#94a3b8',
    roughness: 0.5,
    metalness: 0.5
  }
}, null, 2);
