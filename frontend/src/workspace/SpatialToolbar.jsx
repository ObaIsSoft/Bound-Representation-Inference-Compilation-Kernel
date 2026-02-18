/**
 * Spatial Toolbar - Mode Switching for Omniviewport
 * 
 * Minimal toolbar for switching between:
 * - design: Standard 3D editing
 * - simulate: CFD/FEA visualization
 * - animate: Physics timeline
 * - review: Annotation mode
 */

import React from 'react';
import { Box, Play, Activity, PenTool, MousePointer2 } from 'lucide-react';

const MODES = [
  { id: 'design', label: 'Design', icon: Box },
  { id: 'simulate', label: 'Simulate', icon: Activity },
  { id: 'animate', label: 'Animate', icon: Play },
  { id: 'review', label: 'Review', icon: PenTool },
];

const TOOLS = [
  { id: 'select', label: 'Select', icon: MousePointer2 },
  { id: 'draw', label: 'Draw', icon: PenTool },
];

/**
 * Spatial Toolbar Component
 */
export default function SpatialToolbar({ 
  mode, 
  setMode, 
  selectedTool, 
  setSelectedTool,
  theme 
}) {
  return (
    <div 
      className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2 px-4 py-2 rounded-2xl z-50"
      style={{ 
        backgroundColor: theme.colors.bg.secondary + 'DD',
        backdropFilter: 'blur(12px)',
        border: `1px solid ${theme.colors.border.primary}`
      }}
    >
      {/* Mode Buttons */}
      <div className="flex items-center gap-1">
        {MODES.map((m) => {
          const Icon = m.icon;
          const isActive = mode === m.id;
          
          return (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              className="flex items-center gap-2 px-3 py-2 rounded-xl transition-all"
              style={{
                backgroundColor: isActive ? theme.colors.accent.primary + '20' : 'transparent',
                color: isActive ? theme.colors.accent.primary : theme.colors.text.secondary,
                border: isActive ? `1px solid ${theme.colors.accent.primary}40` : '1px solid transparent',
              }}
              title={m.label}
            >
              <Icon size={16} />
              <span className="text-xs font-medium hidden sm:inline">{m.label}</span>
            </button>
          );
        })}
      </div>

      {/* Divider */}
      <div 
        className="w-px h-6 mx-2"
        style={{ backgroundColor: theme.colors.border.secondary }}
      />

      {/* Tool Buttons */}
      <div className="flex items-center gap-1">
        {TOOLS.map((t) => {
          const Icon = t.icon;
          const isActive = selectedTool === t.id;
          
          return (
            <button
              key={t.id}
              onClick={() => setSelectedTool(t.id)}
              className="p-2 rounded-xl transition-all"
              style={{
                backgroundColor: isActive ? theme.colors.bg.tertiary : 'transparent',
                color: isActive ? theme.colors.text.primary : theme.colors.text.muted,
              }}
              title={t.label}
            >
              <Icon size={16} />
            </button>
          );
        })}
      </div>
    </div>
  );
}
